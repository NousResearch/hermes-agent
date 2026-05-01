"""Helpers for loading Hermes .env files consistently across entrypoints."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from utils import atomic_replace


logger = logging.getLogger(__name__)


# Env var name suffixes that indicate credential values.  These are the
# only env vars whose values we sanitize on load — we must not silently
# alter arbitrary user env vars, but credentials are known to require
# pure ASCII (they become HTTP header values).
_CREDENTIAL_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_KEY")

# Names we've already warned about during this process, so repeated
# load_hermes_dotenv() calls (user env + project env, gateway hot-reload,
# tests) don't spam the same warning multiple times.
_WARNED_KEYS: set[str] = set()


def _format_offending_chars(value: str, limit: int = 3) -> str:
    """Return a compact 'U+XXXX ('c'), ...' summary of non-ASCII codepoints."""
    seen: list[str] = []
    for ch in value:
        if ord(ch) > 127:
            label = f"U+{ord(ch):04X}"
            if ch.isprintable():
                label += f" ({ch!r})"
            if label not in seen:
                seen.append(label)
            if len(seen) >= limit:
                break
    return ", ".join(seen)


def _sanitize_loaded_credentials() -> None:
    """Strip non-ASCII characters from credential env vars in os.environ.

    Called after dotenv loads so the rest of the codebase never sees
    non-ASCII API keys.  Only touches env vars whose names end with
    known credential suffixes (``_API_KEY``, ``_TOKEN``, etc.).

    Emits a one-line warning to stderr when characters are stripped.
    Silent stripping would mask copy-paste corruption (Unicode lookalike
    glyphs from PDFs / rich-text editors, ZWSP from web pages) as opaque
    provider-side "invalid API key" errors (see #6843).
    """
    for key, value in list(os.environ.items()):
        if not any(key.endswith(suffix) for suffix in _CREDENTIAL_SUFFIXES):
            continue
        try:
            value.encode("ascii")
            continue
        except UnicodeEncodeError:
            pass
        cleaned = value.encode("ascii", errors="ignore").decode("ascii")
        os.environ[key] = cleaned
        if key in _WARNED_KEYS:
            continue
        _WARNED_KEYS.add(key)
        stripped = len(value) - len(cleaned)
        detail = _format_offending_chars(value) or "non-printable"
        print(
            f"  Warning: {key} contained {stripped} non-ASCII character"
            f"{'s' if stripped != 1 else ''} ({detail}) — stripped so the "
            f"key can be sent as an HTTP header.",
            file=sys.stderr,
        )
        print(
            "  This usually means the key was copy-pasted from a PDF, "
            "rich-text editor, or web page that substituted lookalike\n"
            "  Unicode glyphs for ASCII letters. If authentication fails "
            "(e.g. \"API key not valid\"), re-copy the key from the\n"
            "  provider's dashboard and run `hermes setup` (or edit the "
            ".env file in a plain-text editor).",
            file=sys.stderr,
        )


def _load_dotenv_with_fallback(path: Path, *, override: bool) -> None:
    try:
        load_dotenv(dotenv_path=path, override=override, encoding="utf-8")
    except UnicodeDecodeError:
        load_dotenv(dotenv_path=path, override=override, encoding="latin-1")
    # Strip non-ASCII characters from credential env vars that were just
    # loaded.  API keys must be pure ASCII since they're sent as HTTP
    # header values (httpx encodes headers as ASCII).  Non-ASCII chars
    # typically come from copy-pasting keys from PDFs or rich-text editors
    # that substitute Unicode lookalike glyphs (e.g. ʋ U+028B for v).
    _sanitize_loaded_credentials()


def _decode_vault_broker_env(stdout: str) -> dict[str, str]:
    """Decode hermes-vault broker env output, including double-encoded JSON."""
    payload = json.loads(stdout)
    if isinstance(payload, str):
        payload = json.loads(payload)
    env = payload.get("env") if isinstance(payload, dict) else None
    return env if isinstance(env, dict) else {}


_VAULT_RUNTIME_SERVICES: dict[str, dict[str, str]] = {
    "telegram": {"TELEGRAM_BOT_TOKEN": "TELEGRAM_BOT_TOKEN"},
}


_VAULT_INJECTED = False


def _inject_vault_runtime_env(home_path: Path) -> None:
    """Best-effort runtime bridge for vault-backed Hermes credentials.

    Hermes still has call sites that read os.getenv("TELEGRAM_BOT_TOKEN")
    directly.  When that secret is migrated out of ~/.hermes/.env and into
    hermes-vault, restarts lose it unless we materialize a short-lived
    environment variable for the current process at startup.  This function
    never writes secrets to disk and never prints secret values.
    """
    global _VAULT_INJECTED
    if _VAULT_INJECTED:
        return
    _VAULT_INJECTED = True

    default_home = Path.home() / ".hermes"
    if home_path.expanduser().resolve() != default_home.resolve() and os.getenv(
        "HERMES_VAULT_ENABLE_RUNTIME_INJECTION"
    ) != "1":
        return

    missing_services: dict[str, dict[str, str]] = {}
    for service, mapping in _VAULT_RUNTIME_SERVICES.items():
        if not any(os.getenv(target) for target in set(mapping.values())):
            missing_services[service] = mapping
    if not missing_services:
        return

    vault_bin = home_path / "hermes-agent" / "venv" / "bin" / "hermes-vault"
    if not vault_bin.exists():
        vault_bin = Path.home() / ".local" / "bin" / "hermes-vault"
    if not vault_bin.exists():
        return

    if sys.platform != "darwin":
        return

    login_keychain = Path.home() / "Library" / "Keychains" / "login.keychain-db"
    policy_path = Path(os.getenv("HERMES_VAULT_POLICY", Path.home() / ".config" / "hermes-vault" / "policy.yaml"))
    try:
        passphrase_proc = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-s",
                "hermes-vault",
                "-a",
                "hermes",
                "-w",
                str(login_keychain),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        passphrase = passphrase_proc.stdout.strip()
    except Exception as exc:
        print(f"[vault-inject] skipped: keychain read failed ({exc.__class__.__name__})", file=sys.stderr)
        return
    if not passphrase:
        print("[vault-inject] skipped: empty vault passphrase", file=sys.stderr)
        return

    broker_env = os.environ.copy()
    broker_env["HERMES_VAULT_PASSPHRASE"] = passphrase
    broker_env["HERMES_VAULT_POLICY"] = str(policy_path)

    injected: list[str] = []
    for service, mapping in missing_services.items():
        try:
            proc = subprocess.run(
                [str(vault_bin), "broker", "env", service, "--agent", "hermes", "--ttl", "60"],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
                env=broker_env,
            )
            env_values = _decode_vault_broker_env(proc.stdout)
            for source_name, target_name in mapping.items():
                value = env_values.get(source_name)
                if value and not os.getenv(target_name):
                    os.environ[target_name] = str(value)
                    injected.append(target_name)
                    logger.info(
                        "[vault-inject] injected %s for service %s from hermes-vault env field %s",
                        target_name,
                        service,
                        source_name,
                    )
                    break
            else:
                logger.warning(
                    "[vault-inject] %s returned no recognized env fields; expected one of %s",
                    service,
                    sorted(mapping),
                )
        except Exception as exc:
            logger.warning("[vault-inject] %s skipped (%s)", service, exc.__class__.__name__)
            print(f"[vault-inject] {service} skipped ({exc.__class__.__name__})", file=sys.stderr)

    if injected:
        message = f"[vault-inject] injected {sorted(set(injected))} from hermes-vault"
        logger.info(message)
        print(message, file=sys.stderr)
        _sanitize_loaded_credentials()


def _sanitize_env_file_if_needed(path: Path) -> None:
    """Pre-sanitize a .env file before python-dotenv reads it.

    python-dotenv does not handle corrupted lines where multiple
    KEY=VALUE pairs are concatenated on a single line (missing newline).
    This produces mangled values — e.g. a bot token duplicated 8×
    (see #8908).

    We delegate to ``hermes_cli.config._sanitize_env_lines`` which
    already knows all valid Hermes env-var names and can split
    concatenated lines correctly.
    """
    if not path.exists():
        return
    try:
        from hermes_cli.config import _sanitize_env_lines
    except ImportError:
        return  # early bootstrap — config module not available yet

    read_kw = {"encoding": "utf-8", "errors": "replace"}
    try:
        with open(path, **read_kw) as f:
            original = f.readlines()
        sanitized = _sanitize_env_lines(original)
        if sanitized != original:
            import tempfile
            fd, tmp = tempfile.mkstemp(
                dir=str(path.parent), suffix=".tmp", prefix=".env_"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.writelines(sanitized)
                    f.flush()
                    os.fsync(f.fileno())
                atomic_replace(tmp, path)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
    except Exception:
        pass  # best-effort — don't block gateway startup


def load_hermes_dotenv(
    *,
    hermes_home: str | os.PathLike | None = None,
    project_env: str | os.PathLike | None = None,
) -> list[Path]:
    """Load Hermes environment files with user config taking precedence.

    Behavior:
    - `~/.hermes/.env` overrides stale shell-exported values when present.
    - project `.env` acts as a dev fallback and only fills missing values when
      the user env exists.
    - if no user env exists, the project `.env` also overrides stale shell vars.
    """
    loaded: list[Path] = []

    home_path = Path(hermes_home or os.getenv("HERMES_HOME", Path.home() / ".hermes"))
    user_env = home_path / ".env"
    project_env_path = Path(project_env) if project_env else None

    # Fix corrupted .env files before python-dotenv parses them (#8908).
    if user_env.exists():
        _sanitize_env_file_if_needed(user_env)
    if project_env_path and project_env_path.exists():
        _sanitize_env_file_if_needed(project_env_path)

    if user_env.exists():
        _load_dotenv_with_fallback(user_env, override=True)
        loaded.append(user_env)

    if project_env_path and project_env_path.exists():
        _load_dotenv_with_fallback(project_env_path, override=not loaded)
        loaded.append(project_env_path)

    _inject_vault_runtime_env(home_path)

    return loaded

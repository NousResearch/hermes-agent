"""KeePass / KeePassXC (``keepassxc-cli``) secret source.

KeePassXC is the local-first manager: one encrypted ``.kdbx`` file, no server
and no account. ``keepassxc-cli`` takes the database password as a LINE ON
STDIN (``Utils::getPassword`` — a plain ``readLine``, no TTY required), which
is its non-interactive contract: the password is written to the child's stdin,
never to argv, never to either process's environment, never to disk.

Users map env-var names to entry paths in ``secrets.keepass.env``; each is
resolved with one ``keepassxc-cli show -a Password -- <db> <entry>``. The
database key comes from ``password_file``, a ``.env``-sourced variable
(``KEEPASS_PASSWORD`` by default), or a key file — a startup source never
prompts (``base.SecretSource``); unlock-on-prompt is the session-scoped
browser-vault path (``agent/vault_backends/keepass.py``).

Deliberately uncached: the database is a local file a user can edit between
runs, and a cached value would outlive a rotation in the file.
"""

from __future__ import annotations

import csv
import io
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from agent.secret_sources.base import (
    ErrorKind,
    ErrorRules,
    FetchResult,
    SecretSource,
    classify_cli_error,
    get_source_environment,
    is_valid_env_name,
)

logger = logging.getLogger(__name__)

_BINARY_NAME = "keepassxc-cli"
_RUN_TIMEOUT = 30.0
#: Env var searched for the database password when ``password_file`` is not set.
DEFAULT_PASSWORD_ENV = "KEEPASS_PASSWORD"

# Minimal allowlisted child env — never the full post-dotenv os.environ, which holds every
# provider credential. The CLI needs nothing from us but a home for its error log.
_ENV_ALLOWLIST = ("PATH", "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "SystemRoot",
                  "TMPDIR", "TMP", "TEMP", "XDG_CONFIG_HOME", "XDG_RUNTIME_DIR", "LANG", "LC_ALL")

# "Enter password to unlock <db>: " is the CLI's own prompt on stderr, not part of a failure.
_PROMPT_RE = re.compile(r"\AEnter password to unlock [^\n]*:\s*")

# First matching rule wins.
_KEEPASS_ERROR_RULES: ErrorRules = (
    (ErrorKind.TIMEOUT, ("timed out",)),
    (ErrorKind.BINARY_MISSING, ("failed to invoke", "not found on path", "unknown command")),
    (ErrorKind.AUTH_FAILED, ("invalid credentials", "invalid password", "wrong password")),
    (ErrorKind.REF_INVALID, ("could not find entry", "unknown attribute", "is ambiguous")),
    (ErrorKind.NOT_CONFIGURED, ("failed to open database file", "is not a plain file", "not readable")),
)


def _classify_keepass_error(message: str) -> ErrorKind:
    return classify_cli_error(message, _KEEPASS_ERROR_RULES)


def find_keepassxc(binary_path: str = "") -> Optional[Path]:
    """Resolve a usable ``keepassxc-cli``, or None. A pinned ``binary_path`` is used
    verbatim — pinned-but-missing returns None rather than falling back to PATH."""
    explicit = str(binary_path or "").strip()
    if explicit:
        path = Path(explicit)
        return path if path.is_file() else None
    found = shutil.which(_BINARY_NAME)
    return Path(found) if found else None


def _missing_binary_error(binary_path: str = "") -> str:
    if binary_path:
        return f"secrets.keepass.binary_path ({binary_path!r}) is not an executable keepassxc-cli."
    return ("keepassxc-cli was not found on PATH.  Install KeePassXC "
            "(https://keepassxc.org/download/) or set secrets.keepass.binary_path.")


def _scrub(text: str) -> str:
    """Strip ANSI (a control sequence must not hide text after a marker), the CLI's own
    unlock prompt, and surrounding whitespace."""
    from tools.ansi_strip import strip_ansi

    return _PROMPT_RE.sub("", strip_ansi(text or "").replace("\x1b", "").strip()).strip()


def _child_env() -> Dict[str, str]:
    source_env = get_source_environment()
    env = {k: source_env[k] for k in _ENV_ALLOWLIST if k in source_env}
    env["NO_COLOR"] = "1"
    return env


def _invoke(argv: Sequence[str], *, key: Optional[str],
            timeout: float = _RUN_TIMEOUT) -> subprocess.CompletedProcess:
    """``subprocess.run`` an argv list (never a shell). ``key`` — the database password, or
    None for a key-file-only database — goes on the CHILD's stdin, where keepassxc-cli reads
    it; it never reaches argv or either process's environment. Without a key, stdin is
    /dev/null so the CLI can never block waiting for a password."""
    opts: Dict[str, object] = dict(env=_child_env(), capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=timeout)
    if key is None:
        opts["stdin"] = subprocess.DEVNULL
    else:
        opts["input"] = key + "\n"
    try:
        return subprocess.run(list(argv), **opts)  # noqa: S603 — argv list, no shell
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"keepassxc-cli timed out after {timeout:.0f}s") from exc
    except OSError as exc:
        raise RuntimeError(f"failed to invoke keepassxc-cli: {exc}") from exc


def _failure(proc: subprocess.CompletedProcess, action: str, what: str = "") -> RuntimeError:
    err = _scrub(proc.stderr or "")[:200] or f"exited {proc.returncode}"
    return RuntimeError(f"keepassxc-cli {action} failed{(' for ' + what) if what else ''}: {err}")


# --- Database key -----------------------------------------------------------


def _key_options(keyfile: str) -> List[str]:
    return ["-k", str(Path(keyfile).expanduser())] if keyfile else []


def _read_password_file(path: str) -> str:
    """First line of ``path``: a trailing newline from ``echo`` must not become the password."""
    file = Path(path).expanduser()
    try:
        lines = file.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise RuntimeError(f"cannot read secrets.keepass.password_file {file}: {exc.strerror or exc}") from exc
    if not lines or not lines[0].strip():
        raise RuntimeError(f"secrets.keepass.password_file {file} is empty")
    return lines[0]


def resolve_password(cfg: dict, password_env: str = DEFAULT_PASSWORD_ENV, *,
                     env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """The database password for this run, or None when only a key file protects the database.

    ``password_file`` (an explicit file) outranks the ``password_env`` variable. ``env`` defaults
    to the per-fetch environment view, so a multiplex gateway profile cannot see a sibling's key;
    the vault backend passes its profile-scoped secret view instead.
    """
    password_file = str((cfg or {}).get("password_file") or "").strip()
    if password_file:
        return _read_password_file(password_file)
    source = get_source_environment() if env is None else env
    return (source.get(password_env) or "").strip() or None


# --- Database reads ---------------------------------------------------------


@dataclass(frozen=True)
class KeepassEntry:
    """Non-secret entry metadata. The password attribute is never part of this shape."""

    path: str       # CLI entry path, usable verbatim as ``show`` argument or env: reference
    title: str
    username: str = ""
    url: str = ""
    created_at: str = ""


def verify_database(*, db: str, key: Optional[str], keyfile: str = "", binary_path: str = "",
                    timeout: float = _RUN_TIMEOUT) -> None:
    """Open the database once (``db-info``) to prove the key works; raises ``RuntimeError`` on a
    wrong password, a missing database, or a missing CLI. The vault unlock prompt calls this so a
    bad master password is reported before it is remembered for the session."""
    binary = find_keepassxc(binary_path)
    if binary is None:
        raise RuntimeError(_missing_binary_error(binary_path))
    argv = [str(binary), "db-info", *_key_options(keyfile), "--", str(db)]
    proc = _invoke(argv, key=key, timeout=timeout)
    if proc.returncode != 0:
        raise _failure(proc, "db-info")


def show_attribute(*, db: str, entry: str, key: Optional[str], keyfile: str = "",
                   attribute: str = "Password", binary_path: str = "",
                   timeout: float = _RUN_TIMEOUT) -> str:
    """Resolve one entry attribute through ``keepassxc-cli show``.

    Raises ``RuntimeError`` on any failure, including an exit-0 empty value: applying "" to an
    env var would clobber a good credential. Naming the attribute explicitly is what makes
    keepassxc-cli print a protected value in clear text (the summary form says ``PROTECTED``).
    """
    binary = find_keepassxc(binary_path)
    if binary is None:
        raise RuntimeError(_missing_binary_error(binary_path))
    # `--` then positionals: an entry path starting with `-` can never parse as an option
    # (QCommandLineParser treats everything after `--` as positional).
    argv = [str(binary), "show", "-a", attribute, *_key_options(keyfile), "--", str(db), str(entry)]
    proc = _invoke(argv, key=key, timeout=timeout)
    if proc.returncode != 0:
        raise _failure(proc, "show", repr(entry))
    value = (proc.stdout or "").rstrip("\r\n")
    if not value.strip():
        raise RuntimeError(f"keepassxc-cli show returned an empty {attribute} for {entry!r}")
    return value


def export_entries(*, db: str, key: Optional[str], keyfile: str = "", binary_path: str = "",
                   timeout: float = _RUN_TIMEOUT) -> List[KeepassEntry]:
    """Every entry in ONE ``export --format csv`` call, metadata only.

    The CSV carries a Password column: it is dropped as each row is parsed and never stored,
    logged, or returned. Listing with ``show`` instead would run a full database decrypt per
    entry (seconds on a real database).
    """
    binary = find_keepassxc(binary_path)
    if binary is None:
        raise RuntimeError(_missing_binary_error(binary_path))
    argv = [str(binary), "export", "--format", "csv", *_key_options(keyfile), "--", str(db)]
    proc = _invoke(argv, key=key, timeout=timeout)
    if proc.returncode != 0:
        raise _failure(proc, "export")
    return _parse_export_csv(proc.stdout or "")


def _parse_export_csv(raw: str) -> List[KeepassEntry]:
    """Rows of KeePassXC's CSV export → CLI-compatible entry paths.

    The export's Group column is prefixed with the ROOT GROUP NAME, while ``show`` resolves
    paths relative to the root group (``Group::findEntryByPath``). The first component is
    therefore dropped, so a path from here can be handed back to keepassxc-cli verbatim.
    """
    reader = csv.reader(io.StringIO(raw))
    try:
        header = [h.strip().lower() for h in next(reader)]
    except StopIteration:
        return []
    column = {name: i for i, name in enumerate(header)}
    if "title" not in column:
        raise RuntimeError("keepassxc-cli export returned an unrecognized CSV header (no Title column)")

    out: List[KeepassEntry] = []
    for row in reader:
        title = _cell(row, column.get("title"))
        if not title:
            continue
        # Everything after the root group name is the CLI-visible group path.
        groups = [part for part in _cell(row, column.get("group")).split("/")[1:] if part]
        out.append(KeepassEntry(
            path="/" + "/".join([*groups, title]),
            title=title,
            username=_cell(row, column.get("username")),
            url=_cell(row, column.get("url")),
            created_at=_cell(row, column.get("created")),
        ))
    return out


def _cell(row: Sequence[str], index: Optional[int]) -> str:
    return row[index].strip() if index is not None and index < len(row) else ""


# --- Mapped fetch -----------------------------------------------------------


def _validate_references(references: Optional[Dict[str, str]]) -> Tuple[Dict[str, str], List[str]]:
    """``(valid_refs, warnings)``: keep valid env names bound to trimmed entry paths."""
    valid: Dict[str, str] = {}
    warnings: List[str] = []
    for name, ref in (references or {}).items():
        if not is_valid_env_name(name):
            warnings.append(f"Skipping {name!r}: not a valid env-var name")
        elif not isinstance(ref, str) or not ref.strip():
            warnings.append(f"Skipping {name!r}: entry path is empty")
        else:
            valid[name] = ref.strip()
    return valid, warnings


def fetch_keepass_secrets(*, db: str, references: Dict[str, str], key: Optional[str] = None,
                          keyfile: str = "", binary_path: str = "") -> Tuple[Dict[str, str], List[str]]:
    """Resolve ``references`` (env-var name → entry path) to ``(secrets, warnings)``.

    Raises ``RuntimeError`` only for a missing CLI or a wrong ``db``; per-entry failures become
    warnings so one renamed entry cannot take the whole startup down.
    """
    binary = find_keepassxc(binary_path)
    if binary is None:
        raise RuntimeError(_missing_binary_error(binary_path))

    secrets: Dict[str, str] = {}
    warnings: List[str] = []
    for name in sorted(references):
        try:
            secrets[name] = show_attribute(db=db, entry=references[name], key=key, keyfile=keyfile,
                                           binary_path=str(binary))
        except RuntimeError as exc:
            warnings.append(str(exc))
    return secrets, warnings


class KeePassSource(SecretSource):
    """KeePassXC as a registered **mapped** source: explicit VAR → entry-path bindings, so its
    claims outrank bulk sources on contested vars (same shape as 1Password)."""

    name = "keepass"
    label = "KeePassXC"
    shape = "mapped"
    scheme = "keepass"
    token_env_key = "password_env"
    default_token_env = DEFAULT_PASSWORD_ENV
    # The .kdbx is the source of truth and an explicit VAR→entry binding is the strongest
    # intent; a stale .env line must not silently defeat it.
    override_existing_default = True
    _KEY_HINT = ("Set secrets.keepass.password_file to a 0600 file holding the database "
                 f"password, put it in .env as {DEFAULT_PASSWORD_ENV}, or point "
                 "secrets.keepass.keyfile at the database's key file.")
    remediation_hints = {
        ErrorKind.AUTH_FAILED: ("KeePassXC rejected the database password.  " + _KEY_HINT),
        ErrorKind.NOT_CONFIGURED: _KEY_HINT,
        ErrorKind.BINARY_MISSING: ("Install KeePassXC (https://keepassxc.org/download/) or set "
                                   "secrets.keepass.binary_path, then re-run "
                                   "`hermes secrets keepass status`."),
        ErrorKind.REF_INVALID: ("An entry path in secrets.keepass.env does not exist.  List the "
                                "database with `hermes secrets keepass list` and re-run "
                                "`hermes secrets keepass set ENV_VAR \"Group/Entry\"`."),
    }

    def config_schema(self) -> dict:
        return {
            "enabled": {"description": "Master switch", "default": False},
            "db": {"description": "Absolute path (or ~/…) to the .kdbx database", "default": ""},
            "env": {"description": "Map of ENV_VAR -> entry path (\"Group/Entry\")", "default": {}},
            "password_env": {"description": "Env var holding the database password",
                             "default": DEFAULT_PASSWORD_ENV},
            "password_file": {"description": "File whose first line is the database password",
                              "default": ""},
            "keyfile": {"description": "Key file unlocking the database (-k)", "default": ""},
            "binary_path": {"description": "Pin the keepassxc-cli binary (empty = resolve via PATH)",
                            "default": ""},
            "timeout_seconds": {"description": "Wall-clock budget for the whole fetch", "default": 120},
            "override_existing": {"description": "Resolved values overwrite .env/shell values",
                                  "default": True},
        }

    def fetch(self, cfg: dict, home_path: Path) -> FetchResult:
        cfg = cfg if isinstance(cfg, dict) else {}
        result = FetchResult()

        env_map = cfg.get("env")
        valid, warnings = _validate_references(env_map if isinstance(env_map, dict) else None)
        result.warnings.extend(warnings)
        if not valid:
            if not warnings:
                result.fail("secrets.keepass.enabled is true but the env: map is empty.  Add "
                            "ENV_VAR: \"Group/Entry\" entries (see `hermes secrets keepass list`).",
                            ErrorKind.NOT_CONFIGURED)
            return result

        db = str(cfg.get("db") or "").strip()
        if not db:
            return result.fail("secrets.keepass.db is empty.  Point it at the .kdbx database "
                               "(e.g. ~/KeePass/Passwords.kdbx).", ErrorKind.NOT_CONFIGURED)
        db_path = Path(db).expanduser()
        if not db_path.is_file():
            return result.fail(f"secrets.keepass.db ({db_path}) is not a file.  Point it at the "
                               ".kdbx database.", ErrorKind.NOT_CONFIGURED)

        binary_path = str(cfg.get("binary_path") or "")
        binary = find_keepassxc(binary_path)
        result.binary_path = binary
        if binary is None:
            return result.fail(_missing_binary_error(binary_path), ErrorKind.BINARY_MISSING)

        keyfile = str(cfg.get("keyfile") or "").strip()
        try:
            key = resolve_password(cfg, self.token_env(cfg))
        except RuntimeError as exc:
            return result.fail(str(exc), ErrorKind.NOT_CONFIGURED)
        if key is None and not keyfile:
            return result.fail("secrets.keepass has no database key: " + self._KEY_HINT,
                               ErrorKind.NOT_CONFIGURED)

        try:
            secrets, fetch_warnings = fetch_keepass_secrets(
                db=str(db_path), references=valid, key=key, keyfile=keyfile, binary_path=str(binary))
        except RuntimeError as exc:
            return result.fail(str(exc), _classify_keepass_error(str(exc)))

        result.secrets = secrets
        result.warnings.extend(fetch_warnings)
        return result

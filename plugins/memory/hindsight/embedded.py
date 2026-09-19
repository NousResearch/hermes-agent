"""Local-embedded Hindsight runtime seam: side-env runtime probe, install hint, and
the per-profile env file the standalone ``hindsight-embed`` daemon consumes. The
heavy stack itself lives in the isolated side env owned by ``embedded_runtime`` —
nothing here imports it in-process."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

from agent.secret_scope import UnscopedSecretError, get_secret

from .settings import _DEFAULT_IDLE_TIMEOUT, _daemon_llm_provider, _parse_int_setting

# Stale embedded-daemon connection markers (client recreated, operation retried once).
_RETRIABLE_CONNECTION_MARKERS = (
    "cannot connect to host",
    # Connection-establishment / DNS failure message patterns. These surface when the exception TYPE is
    # generic (RuntimeError/Exception from a local shim, MCP bridge, subprocess wrapper, or an SDK that
    # re-raises without chaining) so the _TRANSPORT_ERROR_TYPES check never fires, and the error carries no
    # HTTP status. Without message-level matching they fall through to FailoverReason.unknown, which misses
    # the transport eager-fallback path in the retry loop (unknown retries the same dead endpoint for the
    # full budget before fallback). Ported from anomalyco/opencode#40707, which hit the same bug shape:
    # serialized midstream errors matched by type only. Deliberately EXCLUDES mid-stream disconnect strings
    # ("connection reset by peer", "peer closed connection", "unexpected eof", "socket hang up") — those
    # belong to _SERVER_DISCONNECT_PATTERNS, whose classification step runs later and routes large sessions
    # to context-overflow compression. A connection that was never established cannot be a server-side
    # overflow rejection, so these are safe to classify as plain retryable transport.
    "connection refused",
    "connect call failed",
    "clientconnectorerror",
)


def _check_local_runtime() -> tuple[bool, str | None]:
    """Whether the isolated side-env runtime imports cleanly (probed in the side
    interpreter — older CPUs: NumPy can raise at import, so Hermes degrades
    instead of retrying a broken backend; ``sentence_transformers`` is probed
    too: ``hindsight`` imports fine with a broken embedding stack, and the
    daemon would then abort on every retain/recall)."""
    from .embedded_runtime import check_local_runtime

    return check_local_runtime()


def _local_runtime_hint(reason: str | None) -> str:
    """Install/reinstall guidance when the local_embedded side runtime is missing
    or broken (probe reason appended when present)."""
    from .embedded_runtime import _local_runtime_hint as _hint

    return _hint(reason)


def _load_simple_env(path) -> dict[str, str]:
    """Parse a KEY=VALUE env file (comments/blank lines ignored). utf-8-sig: also used
    on the Hermes .env during post_setup, where a Notepad BOM would stick to the first key."""
    if not path.exists():
        return {}
    pairs = (line.split("=", 1) for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
             if line and not line.startswith("#") and "=" in line)
    return {key.strip(): value.strip() for key, value in pairs}


def _embedded_profile_env_path(config: dict[str, Any]) -> Path:
    profile = str(config.get("profile", "hermes") or "hermes")
    return Path.home() / ".hindsight" / "profiles" / f"{profile}.env"


def _on_disk_llm_api_key(config: dict[str, Any]) -> str:
    """The key currently persisted in the profile env file ("" when absent)."""
    with contextlib.suppress(Exception):
        return _load_simple_env(_embedded_profile_env_path(config)).get("HINDSIGHT_API_LLM_API_KEY", "") or ""
    return ""


def _embedded_llm_api_key(config: dict[str, Any]) -> str:
    """Resolve the LLM API key: explicit config first, then the profile secret
    scope, then the on-disk profile env as a last resort.

    The disk fallback is the durability core: the background daemon-start
    worker usually runs with no secret scope, and without it the client would
    be built keyless — its ``ensure_running(config)`` merge would then
    overwrite the file's good key with emptiness inside the upstream manager
    (``_register_profile`` → ``create_profile`` rewrite). Falling back to the
    persisted key keeps the in-process client, the file compare, and the
    daemon subprocess all keyed from the same surviving copy.
    """
    if config.get("llmApiKey") or config.get("llm_api_key"):
        return config.get("llmApiKey") or config.get("llm_api_key")
    # NOTE: the vault item is named HINDSIGHT_API_LLM_API_KEY (matching the
    # daemon's env var), not HINDSIGHT_LLM_API_KEY (the setup-wizard name).
    # Accept both so vault-fed scopes resolve regardless of which name the
    # secret source carries.
    try:
        scoped = get_secret("HINDSIGHT_API_LLM_API_KEY", "") or get_secret("HINDSIGHT_LLM_API_KEY", "")
    except UnscopedSecretError:
        # Multiplexed gateway with no profile scope on this thread: never let
        # a missing scope read os.environ (another profile's key may live
        # there). Fall through to the on-disk copy below.
        scoped = ""
    if scoped:
        return scoped
    return _on_disk_llm_api_key(config)


def _may_rewrite_profile_env(config: dict[str, Any]) -> bool:
    """Whether rewriting the profile env file is safe right now.

    False exactly when the build has no key (no scope, no config key) but the
    file holds one: a rewrite would clobber live credentials with emptiness.
    All other mismatches (model/provider/base-url/idle-timeout drift, missing
    file, key rotation to a new non-empty value) remain writable.
    """
    if _build_embedded_profile_env(config).get("HINDSIGHT_API_LLM_API_KEY"):
        return True
    return not _on_disk_llm_api_key(config)


def _build_embedded_profile_env(config: dict[str, Any], *, llm_api_key: str | None = None) -> dict[str, str]:
    """Build the profile-scoped env that standalone hindsight-embed consumes."""
    if llm_api_key is None:
        llm_api_key = _embedded_llm_api_key(config)
    env_values = {
        "HINDSIGHT_API_LLM_PROVIDER": str(_daemon_llm_provider(config.get("llm_provider", ""))),
        "HINDSIGHT_API_LLM_API_KEY": str(llm_api_key or ""),
        "HINDSIGHT_API_LLM_MODEL": str(config.get("llm_model", "")),
        "HINDSIGHT_API_LOG_LEVEL": "info",
    }
    # Base URL is per-profile like the key beside it (the scoped key must not go to the default's host);
    # on the scopeless daemon worker a miss is a miss, never os.environ (same rule as the key above).
    base_url = config.get("llm_base_url")
    if not base_url:
        try:
            base_url = get_secret("HINDSIGHT_API_LLM_BASE_URL", "") or ""
        except UnscopedSecretError:
            base_url = ""
    if base_url:
        env_values["HINDSIGHT_API_LLM_BASE_URL"] = str(base_url)
    if (idle_timeout := config.get("idle_timeout")) is None:
        idle_timeout = os.environ.get("HINDSIGHT_IDLE_TIMEOUT")
    if idle_timeout is not None and idle_timeout != "":
        env_values["HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT"] = str(_parse_int_setting(idle_timeout, _DEFAULT_IDLE_TIMEOUT))
    return env_values


def _secure_write_profile_env(profile_env: Path, content: str) -> None:
    """Create/overwrite *profile_env* owner-only (0600); a pre-existing file is
    tightened BEFORE the plaintext LLM API key is written."""
    if profile_env.exists():
        with contextlib.suppress(OSError):
            os.chmod(profile_env, 0o600)
    fd = os.open(str(profile_env), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(content)


def _validate_profile_env_permissions(profile_env: Path) -> None:
    """Post-write check: owner-only on POSIX (Windows ACLs aren't mode bits; skipped)."""
    if os.name != "posix":
        return
    import stat

    if stat.S_IMODE(profile_env.stat().st_mode) != 0o600:
        with contextlib.suppress(OSError):
            os.chmod(profile_env, 0o600)
        if stat.S_IMODE(profile_env.stat().st_mode) != 0o600:
            raise PermissionError(
                f"Embedded Hindsight profile environment is not owner-only: {profile_env}"
            )


def _materialize_embedded_profile_env(config: dict[str, Any], *, llm_api_key: str | None = None) -> Path:
    """Write the profile env file; never leave a plaintext key in a file whose
    permissions could not be verified."""
    profile_env = _embedded_profile_env_path(config)
    profile_env.parent.mkdir(parents=True, exist_ok=True)
    env_values = _build_embedded_profile_env(config, llm_api_key=llm_api_key)
    content = "".join(f"{key}={value}\n" for key, value in env_values.items())
    try:
        _secure_write_profile_env(profile_env, content)
        _validate_profile_env_permissions(profile_env)
    except BaseException:
        with contextlib.suppress(OSError):
            profile_env.unlink()
        raise
    return profile_env

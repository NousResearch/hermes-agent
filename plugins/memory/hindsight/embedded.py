"""Local-embedded Hindsight runtime: import probe, install hint, the per-profile env
file the standalone ``hindsight-embed`` daemon consumes, and the health-grace export."""

from __future__ import annotations

import contextlib
import importlib
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from agent.secret_scope import UnscopedSecretError, get_secret

from .settings import _DEFAULT_IDLE_TIMEOUT, _daemon_llm_provider, _parse_int_setting

logger = logging.getLogger(__name__.rpartition(".")[0])

# Read by hindsight_embed.daemon_embed_manager AT IMPORT TIME: how long to wait
# for a slow /health before killing the daemon as stale. Busy hosts exceed the
# upstream 2s check and get needlessly restarted, so it's plugin config.
# Env var the embedded daemon manager reads (at import time, as a module-level constant) to size the grace
# window it waits for a slow /health before declaring a daemon stale and killing it. We surface it as plugin
# config so users can raise it without hand-setting an env var, consistent with "config.json, not raw env
# vars". See #13125.
_PORT_HEALTH_GRACE_ENV = "HINDSIGHT_EMBED_PORT_HEALTH_GRACE_TIMEOUT"

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


def _export_port_health_grace_timeout(config: dict[str, Any]) -> None:
    """Export the daemon health grace timeout BEFORE ``daemon_embed_manager`` is
    imported. Only when configured; ``setdefault`` so an explicit env override wins."""
    raw = config.get("port_health_grace_timeout")
    if raw is None or raw == "":
        return
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return logger.warning("Invalid Hindsight port_health_grace_timeout %r; ignoring.", raw)
    if seconds < 0:
        return logger.warning("Negative Hindsight port_health_grace_timeout %r; ignoring.", raw)
    os.environ.setdefault(_PORT_HEALTH_GRACE_ENV, repr(seconds))


def _check_local_runtime() -> tuple[bool, str | None]:
    """Whether the local embedded stack imports cleanly (older CPUs: NumPy can raise
    at import, so Hermes degrades instead of retrying a broken backend).
    ``sentence_transformers`` is probed too: ``hindsight`` imports fine with a broken
    embedding stack, and the daemon would then abort on every retain/recall."""
    try:
        for module in ("hindsight", "hindsight_embed.daemon_embed_manager", "sentence_transformers"):
            importlib.import_module(module)
        return True, None
    except Exception as exc:
        return False, str(exc)


def _local_runtime_hint(reason: str | None) -> str:
    """Install guidance when the local_embedded runtime is missing: ``plugin.yaml``
    declares only ``hindsight-client``, so a hand-written config, the legacy
    ``"mode": "local"`` alias or a restored backup hits ``No module named 'hindsight'``.

    ``local_embedded`` imports ``from hindsight import HindsightEmbedded``, which is provided only by the
    ``hindsight-all`` package (its wheel ships the top-level ``hindsight`` module).
    NousResearch/hermes-agent#7718.
    """
    text = (reason or "").lower()
    if "no module named" in text and any(m in text for m in ("hindsight'", 'hindsight"', "hindsight_embed")):
        return (
            f" Install the embedded runtime with: uv pip install --python "
            f"{sys.executable} hindsight-all — or run 'hermes memory setup'. "
            "(local_embedded needs the 'hindsight-all' package, which provides the "
            "top-level 'hindsight' module; 'hindsight-client' alone only covers "
            "cloud / local_external.)"
        )
    return ""


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


_HERMES_MANAGED_KEYS = {
    "HINDSIGHT_API_LLM_PROVIDER",
    "HINDSIGHT_API_LLM_API_KEY",
    "HINDSIGHT_API_LLM_MODEL",
    "HINDSIGHT_API_LOG_LEVEL",
    "HINDSIGHT_API_LLM_BASE_URL",
    "HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT",
    "HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU",
    "HINDSIGHT_API_RERANKER_LOCAL_FORCE_CPU",
    "HF_HUB_OFFLINE",
    "HF_ENDPOINT",
}

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parse_bool_setting(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes", "on")


def _sanitize_env_pair(key: str, value: Any) -> tuple[str, str] | None:
    k = str(key).strip()
    if not _ENV_KEY_RE.match(k):
        return None
    v = str(value).replace("\r", "").replace("\n", "").strip()
    return k, v


def _export_daemon_offline_env(config: dict[str, Any]) -> None:
    """Export offline and mirror environment variables into os.environ before daemon spawn.
    Bypasses upstream daemon_embed_manager's filter which only forwards HINDSIGHT_* from .env."""
    if "hf_hub_offline" in config:
        os.environ["HF_HUB_OFFLINE"] = "true" if _parse_bool_setting(config["hf_hub_offline"]) else "false"
    if endpoint := (config.get("hf_endpoint") or os.environ.get("HF_ENDPOINT")):
        os.environ["HF_ENDPOINT"] = str(endpoint).strip()


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
        env_values["HINDSIGHT_API_LLM_BASE_URL"] = str(base_url).strip()
    if (idle_timeout := config.get("idle_timeout")) is None:
        idle_timeout = os.environ.get("HINDSIGHT_IDLE_TIMEOUT")
    if idle_timeout is not None and idle_timeout != "":
        env_values["HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT"] = str(_parse_int_setting(idle_timeout, _DEFAULT_IDLE_TIMEOUT))

    if "embeddings_local_force_cpu" in config:
        env_values["HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU"] = "true" if _parse_bool_setting(config["embeddings_local_force_cpu"]) else "false"
    if "reranker_local_force_cpu" in config:
        env_values["HINDSIGHT_API_RERANKER_LOCAL_FORCE_CPU"] = "true" if _parse_bool_setting(config["reranker_local_force_cpu"]) else "false"

    hf_offline = config.get("hf_hub_offline")
    if hf_offline is None and "HF_HUB_OFFLINE" in os.environ:
        hf_offline = os.environ["HF_HUB_OFFLINE"]
    if hf_offline is not None:
        env_values["HF_HUB_OFFLINE"] = "true" if _parse_bool_setting(hf_offline) else "false"

    hf_endpoint = config.get("hf_endpoint") or os.environ.get("HF_ENDPOINT")
    if hf_endpoint:
        sanitized = _sanitize_env_pair("HF_ENDPOINT", hf_endpoint)
        if sanitized:
            env_values[sanitized[0]] = sanitized[1]

    extra_env = config.get("extra_env") or config.get("env_extra")
    if isinstance(extra_env, dict):
        for raw_k, raw_v in extra_env.items():
            pair = _sanitize_env_pair(raw_k, raw_v)
            if pair and pair[0] not in _HERMES_MANAGED_KEYS:
                env_values[pair[0]] = pair[1]

    return env_values


def _compute_target_env(profile_env: Path, config: dict[str, Any], *, llm_api_key: str | None = None) -> dict[str, str]:
    """Target configuration combining clean managed keys and unmanaged external keys."""
    existing_env = _load_simple_env(profile_env)
    managed_env = _build_embedded_profile_env(config, llm_api_key=llm_api_key)
    preserved_unmanaged = {k: v for k, v in existing_env.items() if k not in _HERMES_MANAGED_KEYS and k not in managed_env}
    return {**preserved_unmanaged, **managed_env}


def _secure_write_profile_env(profile_env: Path, content: str) -> None:
    """Create/overwrite *profile_env* owner-only (0600) via atomic rename;
    a pre-existing file is never left truncated or partially written."""
    parent = profile_env.parent
    parent.mkdir(parents=True, exist_ok=True)
    prefix = f".{profile_env.name}."
    fd, tmp_path_str = tempfile.mkstemp(prefix=prefix, dir=parent)
    tmp_path = Path(tmp_path_str)
    try:
        os.chmod(tmp_path, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, profile_env)
    except BaseException:
        if tmp_path.exists():
            with contextlib.suppress(OSError):
                tmp_path.unlink()
        raise


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
    target_env = _compute_target_env(profile_env, config, llm_api_key=llm_api_key)
    content = "".join(f"{key}={value}\n" for key, value in target_env.items())
    try:
        _secure_write_profile_env(profile_env, content)
        _validate_profile_env_permissions(profile_env)
    except BaseException:
        with contextlib.suppress(OSError):
            profile_env.unlink()
        raise
    return profile_env

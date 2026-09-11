"""Local-embedded Hindsight runtime: import probe, install hint, the per-profile env
file the standalone ``hindsight-embed`` daemon consumes, and the health-grace export."""

from __future__ import annotations

import contextlib
import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable

from agent.secret_scope import get_secret
from hermes_constants import get_hermes_home

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


def _hermes_dotenv() -> dict[str, str]:
    """Read $HERMES_HOME/.env from disk. A long-lived gateway process env can be stale."""
    path = get_hermes_home() / ".env"
    return _load_simple_env(path) if path.exists() else {}


def _disk_secret(name: str, fallback: str = "") -> str:
    """Prefer the on-disk Hermes .env secret over initialize-time process env."""
    disk = _hermes_dotenv().get(name, "")
    if disk:
        return disk
    return get_secret(name, fallback) or os.environ.get(name, "") or fallback


def _put_env(env_values: dict[str, str], key: str, value: Any) -> None:
    if value is None:
        return
    text = str(value)
    if text == "":
        return
    env_values[key] = text


def _embedded_llm_api_key(config: dict[str, Any]) -> str:
    return config.get("llmApiKey") or config.get("llm_api_key") or _disk_secret("HINDSIGHT_LLM_API_KEY")


def _build_embedded_profile_env(config: dict[str, Any], *, llm_api_key: str | None = None) -> dict[str, str]:
    """Build the profile-scoped env that standalone hindsight-embed consumes.

    Must forward embeddings/reranker from hindsight/config.json. An LLM-only
    snapshot lets the daemon fall back to local BAAI/bge-small-en-v1.5 (384-d)
    and crash on a custom-dimension bank when HindsightEmbedded.ensure_running
    re-registers the profile.
    """
    if llm_api_key is None:
        llm_api_key = _embedded_llm_api_key(config)
    env_values: dict[str, str] = {
        "HINDSIGHT_API_LLM_PROVIDER": str(_daemon_llm_provider(config.get("llm_provider", ""))),
        "HINDSIGHT_API_LLM_API_KEY": str(llm_api_key or ""),
        "HINDSIGHT_API_LLM_MODEL": str(config.get("llm_model", "")),
        "HINDSIGHT_API_LOG_LEVEL": "info",
    }
    base_url = (
        config.get("llm_base_url")
        or _hermes_dotenv().get("HINDSIGHT_API_LLM_BASE_URL")
        or os.environ.get("HINDSIGHT_API_LLM_BASE_URL", "")
    )
    _put_env(env_values, "HINDSIGHT_API_LLM_BASE_URL", base_url)
    _put_env(env_values, "HINDSIGHT_API_PORT", config.get("api_port"))
    _put_env(env_values, "HINDSIGHT_API_LLM_TIMEOUT", config.get("llm_timeout"))
    _put_env(env_values, "HINDSIGHT_API_LLM_REASONING_EFFORT", config.get("llm_reasoning_effort"))
    _put_env(env_values, "HINDSIGHT_API_LLM_WIRE", config.get("llm_wire"))
    if (idle_timeout := config.get("idle_timeout")) is None:
        idle_timeout = _hermes_dotenv().get("HINDSIGHT_IDLE_TIMEOUT") or os.environ.get("HINDSIGHT_IDLE_TIMEOUT")
    if idle_timeout is not None and idle_timeout != "":
        env_values["HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT"] = str(_parse_int_setting(idle_timeout, _DEFAULT_IDLE_TIMEOUT))

    _put_env(env_values, "HINDSIGHT_API_EMBEDDINGS_PROVIDER", config.get("embeddings_provider"))
    _put_env(env_values, "HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL", config.get("embeddings_openai_model"))
    _put_env(env_values, "HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL", config.get("embeddings_openai_base_url"))
    _put_env(env_values, "HINDSIGHT_API_EMBEDDINGS_QUERY_PREFIX", config.get("embeddings_query_prefix"))
    _put_env(env_values, "HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL", config.get("embeddings_local_model"))
    _put_env(
        env_values,
        "HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY",
        _disk_secret("HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY"),
    )

    _put_env(env_values, "HINDSIGHT_API_RERANKER_PROVIDER", config.get("reranker_provider"))
    _put_env(env_values, "HINDSIGHT_API_RERANKER_SILICONFLOW_MODEL", config.get("reranker_siliconflow_model"))
    _put_env(env_values, "HINDSIGHT_API_RERANKER_SILICONFLOW_BASE_URL", config.get("reranker_siliconflow_base_url"))
    _put_env(env_values, "HINDSIGHT_API_RERANKER_SILICONFLOW_TIMEOUT", config.get("reranker_siliconflow_timeout"))
    _put_env(
        env_values,
        "HINDSIGHT_API_RERANKER_SILICONFLOW_API_KEY",
        _disk_secret("HINDSIGHT_API_RERANKER_SILICONFLOW_API_KEY"),
    )
    failover = config.get("reranker_failover_provider")
    _put_env(env_values, "HINDSIGHT_API_RERANKER_1_PROVIDER", failover)
    local_model = config.get("reranker_local_model")
    if str(failover or "") == "local":
        _put_env(env_values, "HINDSIGHT_API_RERANKER_1_LOCAL_MODEL", local_model)
    if str(config.get("reranker_provider") or "") == "local":
        _put_env(env_values, "HINDSIGHT_API_RERANKER_LOCAL_MODEL", local_model)
    # The embedded daemon is a separate process; inherit the parent proxy.
    _put_env(env_values, "HTTP_PROXY", os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy"))
    _put_env(env_values, "HTTPS_PROXY", os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"))
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
    # Keep unknown extra keys (HTTP_PROXY, operator-added HINDSIGHT_API_*) so a
    # partial builder cannot wipe them on the next cold start.
    if profile_env.exists():
        saved = _load_simple_env(profile_env)
        merged = dict(saved)
        merged.update(env_values)
        env_values = merged
    content = "".join(f"{key}={value}\n" for key, value in env_values.items())
    try:
        _secure_write_profile_env(profile_env, content)
        _validate_profile_env_permissions(profile_env)
    except BaseException:
        with contextlib.suppress(OSError):
            profile_env.unlink()
        raise
    return profile_env


def _install_profile_env_guard(
    client,
    config: dict[str, Any],
    *,
    load_config: Callable[[], dict[str, Any]] | None = None,
) -> None:
    """Keep embeddings/reranker active across official profile registration.

    HindsightEmbedded.ensure_running calls _register_profile even when the
    daemon is already up. render_config comments out every key missing from
    that dict, so a constructor-only LLM snapshot wipes custom embeddings.
    Last-writer rematerialize is not enough: the dict passed into
    ensure_running must already contain the full profile env.
    """
    if client is None or getattr(client, "_hermes_profile_env_guard", False):
        return

    def _current_config() -> dict[str, Any]:
        if load_config is not None:
            try:
                return load_config() or config
            except Exception:
                return config
        return config

    def _full_profile_env() -> dict[str, str]:
        return _build_embedded_profile_env(_current_config())

    manager = getattr(client, "_manager", None)
    orig_ensure_running = getattr(manager, "ensure_running", None) if manager is not None else None
    if callable(orig_ensure_running) and not getattr(manager, "_hermes_ensure_running_guard", False):
        def _ensure_running_with_full_env(cfg, profile, extra_args=None):
            target = str(_current_config().get("profile") or "hermes")
            merged = dict(cfg or {})
            if not profile or str(profile) == target:
                full = _full_profile_env()
                merged.update(full)
                if hasattr(client, "config") and isinstance(getattr(client, "config", None), dict):
                    client.config.update(full)
            if extra_args is None:
                return orig_ensure_running(merged, profile)
            return orig_ensure_running(merged, profile, extra_args)

        manager.ensure_running = _ensure_running_with_full_env
        manager._hermes_ensure_running_guard = True

    orig = getattr(client, "_ensure_started", None)
    if callable(orig):
        def _ensure_started_and_repair(*args, **kwargs):
            try:
                return orig(*args, **kwargs)
            finally:
                try:
                    _materialize_embedded_profile_env(_current_config())
                except Exception:
                    logger.debug("Could not rematerialize Hindsight profile env", exc_info=True)

        client._ensure_started = _ensure_started_and_repair

    client._hermes_profile_env_guard = True
    try:
        _materialize_embedded_profile_env(config)
    except Exception:
        logger.debug("Could not rematerialize Hindsight profile env", exc_info=True)

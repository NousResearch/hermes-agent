"""Bind authenticated event identity and a routed privacy snapshot for MCP dispatch."""

import dataclasses
import logging
from pathlib import Path

from gateway.session import SessionContext

logger = logging.getLogger(__name__)


def _read_yaml_mapping_strict(path: Path) -> dict:
    import yaml

    try:
        with path.open(encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
    except FileNotFoundError:
        return {}
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Expected a YAML mapping")
    return value


def _load_redact_pii_policy() -> bool:
    # The normal config loader intentionally swallows policy I/O/parse errors.
    from gateway.run import _gateway_config_home
    from hermes_cli.config import _deep_merge, _expand_env_vars
    from hermes_cli import managed_scope

    raw = _expand_env_vars(_read_yaml_mapping_strict(_gateway_config_home() / "config.yaml"))
    managed_dir = managed_scope.get_managed_dir()
    if managed_dir is not None:
        managed = _expand_env_vars(_read_yaml_mapping_strict(Path(managed_dir) / "config.yaml"))
        raw = _deep_merge(raw, managed)
    privacy = raw.get("privacy")
    if privacy is None:
        privacy = {}
    if not isinstance(privacy, dict):
        raise ValueError("privacy must be a YAML mapping")
    value = privacy.get("redact_pii", False)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().casefold()
        if text in {"true", "1", "yes", "on"}:
            return True
        if text in {"false", "0", "no", "off"}:
            return False
    raise ValueError("privacy.redact_pii must be true or false")


def bind_session_context_for_turn(runner, context: SessionContext) -> tuple[list, bool | None]:
    from gateway.run import _profile_runtime_scope

    try:
        if getattr(runner.config, "multiplex_profiles", False):
            profile_home = runner._resolve_profile_home_for_source(context.source, strict=True)
            with _profile_runtime_scope(profile_home):
                redact_pii = _load_redact_pii_policy()
        else:
            redact_pii = _load_redact_pii_policy()
    except Exception:
        # Do not log raw identifiers or config values while handling a privacy failure.
        logger.warning("Session privacy policy unavailable; omitting external MCP session metadata")
        redact_pii = None
    from gateway.session_context import _SESSION_REDACT_PII
    tokens = runner._set_session_env(context)
    _SESSION_REDACT_PII.set(redact_pii)
    return tokens, redact_pii


def source_with_trigger_message_id(event, source=None):
    source = source if source is not None else event.source
    message_id = str(getattr(event, "message_id", None) or "").strip()
    if message_id and message_id != source.message_id:
        source = dataclasses.replace(source, message_id=message_id)
        event.source = source
    return source


def bind_followup_event_context(runner, event, *, session_key: str, session_id: str):
    # Recursive queued turns execute in the original task, so they must replace
    # its identity before starting any work for the next authenticated event.
    source = source_with_trigger_message_id(event)
    context = SessionContext(source=source, connected_platforms=[], home_channels={},
                             session_key=session_key, session_id=session_id)
    bind_session_context_for_turn(runner, context)
    return source

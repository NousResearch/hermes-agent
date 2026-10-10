"""CLI-only management for the opt-in session topic projection."""
from contextlib import suppress


def handle_session_topic_command(cli, cmd_original: str) -> None:
    """Manage topic sub-contexts without changing Telegram's /topic routing."""
    from agent.session_topics import (
        TOPIC_SESSION_CONFIG_KEY, _sync_context_engine_topic, normalize_topic_title,
    )
    from hermes_cli.cli_commands_mixin import _cp, _command_arg, _db_unavailable_line

    db = getattr(cli, "_session_db", None)
    session_id = getattr(cli, "session_id", None)
    agent = getattr(cli, "agent", None)
    if db is None or not session_id:
        return _cp(_db_unavailable_line())
    if agent is not None:
        with suppress(Exception):
            agent._ensure_db_session()
    if not db.get_session(session_id):
        return _cp("  Session topics are available after the session has started.")

    arg = _command_arg(cmd_original).strip()
    parts = arg.split(None, 1)
    action = parts[0].lower() if parts else "list"
    value = parts[1].strip() if len(parts) > 1 else ""

    def _set_enabled(enabled: bool) -> None:
        db.patch_session_model_config(session_id, {TOPIC_SESSION_CONFIG_KEY: enabled})
        if agent is not None:
            agent._session_init_model_config[TOPIC_SESSION_CONFIG_KEY] = enabled
            agent._topic_segmentation_enabled = enabled
            if not enabled:
                agent._active_topic_id = None
            _sync_context_engine_topic(agent)

    if action in {"on", "enable"}:
        _set_enabled(True)
        return _cp("  Session topic segmentation enabled.")
    if action in {"off", "disable"}:
        _set_enabled(False)
        return _cp("  Session topic segmentation disabled; stored topics were kept.")
    if action == "new":
        title = normalize_topic_title(value)
        if not title:
            return _cp("  Usage: /session-topic new <name>")
        _set_enabled(True)
        topic_id = db.create_topic(session_id, title)
        if agent is not None:
            agent._active_topic_id = topic_id
            _sync_context_engine_topic(agent)
        return _cp(f"  Created and activated topic {topic_id}: {title}")
    if action == "switch":
        if not value.isdigit() or int(value) <= 0:
            return _cp("  Usage: /session-topic switch <id>")
        topic_id = int(value)
        if not db.set_active_topic(session_id, topic_id):
            return _cp(f"  Topic {topic_id} does not belong to this session.")
        _set_enabled(True)
        if agent is not None:
            agent._active_topic_id = topic_id
            _sync_context_engine_topic(agent)
        topic = next((item for item in db.get_topics(session_id) if item["id"] == topic_id), None)
        return _cp(f"  Activated topic {topic_id}: {(topic or {}).get('title', '')}")
    if action not in {"", "list", "status"}:
        return _cp("  Usage: /session-topic [on|off|list|new <name>|switch <id>]")

    override = db.get_session_model_config_value(session_id, TOPIC_SESSION_CONFIG_KEY, None)
    enabled = (
        bool(getattr(agent, "_topic_segmentation_enabled", False))
        if agent is not None else override is True
    )
    topics = db.get_topics(session_id)
    _cp(f"  Session topics: {'enabled' if enabled else 'disabled'}")
    if not topics:
        return _cp("  No topics yet; the first enabled turn creates one.")
    for topic in topics:
        marker = "*" if topic["state"] == "active" else " "
        _cp(f"  {marker} {topic['id']}: {topic['title']} "
            f"({topic.get('message_count', 0)} messages)")

"""Value checks for single config sections (voice, timezone, web backends), run by
``hermes_cli.config.validate_config_structure`` for doctor and the startup warnings."""

from typing import Any

from hermes_cli.config_issues import ConfigIssue, _issue


def _validate_voice(config: dict[str, Any], issues: list[ConfigIssue]) -> None:
    voice_cfg = config.get("voice")
    if not (isinstance(voice_cfg, dict) and "submit_mode" in voice_cfg):
        return
    submit_mode = voice_cfg.get("submit_mode")
    normalized = submit_mode.strip().lower() if isinstance(submit_mode, str) else None
    if normalized not in {"direct", "draft"}:
        _issue(issues, "error", f"voice.submit_mode must be 'direct' or 'draft', got {submit_mode!r}",
               "Set voice.submit_mode to direct (submit immediately) or draft (edit before sending)")


def _validate_mtls(config: dict[str, Any], issues: list[ConfigIssue]) -> None:
    """``agent.mtls`` paths must exist and agree with each other.

    A missing certificate is not fatal at runtime (the client logs a warning and
    connects without one), which is exactly why it needs to be visible here: the
    endpoint's TLS error would otherwise be the only signal.
    """
    from pathlib import Path

    agent_cfg = config.get("agent")
    block = agent_cfg.get("mtls") if isinstance(agent_cfg, dict) else None
    if block is None:
        return
    if not isinstance(block, dict):
        _issue(issues, "error", f"agent.mtls must be a mapping, got {type(block).__name__}",
               "Use: agent:\n  mtls:\n    cert_file: /path/client.pem\n    key_file: /path/client-key.pem")
        return

    def _path(field: str) -> str:
        raw = block.get(field)
        if not isinstance(raw, str) or not raw.strip():
            return ""
        return str(Path(raw.strip()).expanduser())

    cert, key, ca = _path("cert_file"), _path("key_file"), _path("ca_cert")
    hint = ("Set agent.mtls.cert_file / key_file to the client certificate and key "
            "(PKCS#12 bundles are accepted as cert_file), or clear the block to disable mTLS.")
    if cert and not Path(cert).is_file():
        _issue(issues, "error", f"agent.mtls.cert_file does not exist: {cert}", hint)
    if key:
        if not cert:
            _issue(issues, "error", "agent.mtls.key_file is set without agent.mtls.cert_file",
                   "Add the matching cert_file, or remove key_file.")
        elif not Path(key).is_file():
            _issue(issues, "error", f"agent.mtls.key_file does not exist: {key}", hint)
    if ca and not Path(ca).is_file():
        _issue(issues, "error", f"agent.mtls.ca_cert does not exist: {ca}",
               "Point ca_cert at the signing bundle for your private PKI, or clear it to use the OS store.")
    password_env = block.get("password_env")
    if isinstance(password_env, str) and password_env.strip():
        import os

        if not os.environ.get(password_env.strip()):
            _issue(issues, "warning",
                   f"agent.mtls.password_env names '{password_env.strip()}', which is not set",
                   "Add it to ~/.hermes/.env, or clear password_env when the key is unencrypted.")


def _validate_timezone(config: dict[str, Any], issues: list[ConfigIssue]) -> None:
    """``timezone`` must be an IANA name the runtime can load.

    ``hermes_time._get_zoneinfo()`` swallows an invalid name behind a single WARNING in the
    gateway log, then runs the agent clock AND every cron schedule on server-local time.
    Surface it here, where doctor and the startup check both look. Silent when the
    interpreter has no tz database at all (bare Windows without ``tzdata``) — nothing can be
    judged there.
    """
    if "timezone" not in config:
        return
    tz = config.get("timezone")
    hint = ("Use an IANA zone name such as America/New_York or Asia/Tokyo (see "
            "`timedatectl list-timezones`). With an invalid value the agent clock and cron "
            "schedules silently fall back to server-local time. HERMES_TIMEZONE overrides "
            "this key when set.")
    if tz is not None and not isinstance(tz, str):
        _issue(issues, "error", f"timezone must be an IANA zone name string, got {tz!r}", hint)
        return
    if not (isinstance(tz, str) and tz.strip()):
        return
    name = tz.strip()
    try:
        import zoneinfo
        zoneinfo.ZoneInfo("UTC")  # is a tz database available at all?
    except Exception:
        return
    try:
        zoneinfo.ZoneInfo(name)
    except Exception:
        _issue(issues, "error", f"timezone {name!r} is not a valid IANA zone name", hint)


def _validate_web_backends(config: dict[str, Any], issues: list[ConfigIssue]) -> None:
    """A stale web backend selection otherwise fails only at the first web_search/web_extract
    call with a generic "no registered provider" error; warn at startup instead."""
    # See #99199.
    web_cfg = config.get("web")
    if not isinstance(web_cfg, dict):
        return
    try:
        from tools.tool_backend_helpers import removed_backend_note
    except Exception:
        return
    seen: set = set()
    for _key in ("backend", "search_backend", "extract_backend"):
        _val = str(web_cfg.get(_key) or "").strip().lower()
        if not _val or _val in seen:
            continue
        seen.add(_val)
        note = removed_backend_note("web", _val)
        if note:
            _issue(issues, "warning",
                   f"web.{_key} is set to '{_val}', but {note} — "
                   "web_search/web_extract will fail until it is changed",
                   "Run 'hermes tools' and pick a different Web Search & Extract provider")


def _container_slots() -> dict[str, str]:
    """Dotted key -> ``"list"``/``"mapping"`` for every slot the schema fixes to a container:
    ``DEFAULT_CONFIG`` (sections included) plus the known-container table for roots it omits."""
    from hermes_cli.config import _KNOWN_CONTAINER_TYPES, DEFAULT_CONFIG  # late: the facade imports this module
    slots: dict[str, str] = {}

    def walk(node: dict[str, Any], prefix: str) -> None:
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                slots[path] = "mapping"
                walk(value, path)
            elif isinstance(value, list):
                slots[path] = "list"

    walk(DEFAULT_CONFIG, "")
    slots.update(_KNOWN_CONTAINER_TYPES)
    return slots


def _validate_quoted_containers(config: dict[str, Any], issues: list[ConfigIssue]) -> None:
    """A container slot holding ONE quoted string (``enabled: '["a","b"]'``) is skipped by every
    isinstance-gated reader while ``config get`` echoes it back, so plugins silently unmount and
    exclusions silently lapse (#83308, #105706). Finding only — the file is never rewritten."""
    import shlex

    import hermes_yaml as yaml
    from hermes_cli.config import _SCALAR_AS_ONE_ITEM_LIST_KEYS, _looks_structured_value, cfg_get
    for key, kind in _container_slots().items():
        # ``parse_config_string_list`` readers accept the quoted form; nothing is ignored there.
        if key in _SCALAR_AS_ONE_ITEM_LIST_KEYS:
            continue
        value = cfg_get(config, *key.split("."))
        if not isinstance(value, str) or not _looks_structured_value(value):
            continue
        try:
            parsed = yaml.safe_load(value)
        except yaml.YAMLError:
            continue
        if isinstance(parsed, (list, dict)):
            _issue(issues, "warning",
                   f"{key} is the quoted string {value!r} — Hermes expects a YAML {kind} here "
                   "and every reader ignores the string",
                   f"Run: hermes config set {key} {shlex.quote(value)}  (stores a real {kind}), "
                   "or remove the quotes in config.yaml")

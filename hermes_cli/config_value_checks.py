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


def _declared_web_backends() -> dict[str, str]:
    """``backend_name -> plugin_name`` for every web provider declared by a plugin manifest.

    Read from disk (bundled ``plugins/`` and the installed ``$HERMES_HOME/plugins``), never by
    loading plugin code: this runs during config validation, which happens before plugins load and
    must stay cheap. A manifest that ships a web provider declares it under
    ``provides_web_providers`` — the same field the plugin loader uses to register it.
    """
    import os
    from pathlib import Path

    roots = [Path(__file__).resolve().parents[1] / "plugins"]
    home = os.environ.get("HERMES_HOME")
    if home:
        roots.append(Path(home) / "plugins")
    declared: dict[str, str] = {}
    for root in roots:
        try:
            manifests = sorted(root.glob("*/plugin.yaml")) + sorted(root.glob("*/*/plugin.yaml"))
        except OSError:
            continue
        for manifest in manifests:
            try:
                text = manifest.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if "provides_web_providers" not in text:
                continue
            block = False
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("provides_web_providers:"):
                    block = True
                    continue
                if not block:
                    continue
                if stripped.startswith("- "):
                    name = stripped[2:].strip().strip("'\"")
                    if name:
                        declared[name.lower()] = manifest.parent.name
                elif stripped and not stripped.startswith("#"):
                    break
    return declared


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
            continue
        # A name that no manifest declares is a typo or a hand-written guess: it fails at the first
        # web_search/web_extract call with a generic "no registered provider" error. Say it here.
        declared = _declared_web_backends()
        if _val not in declared:
            _issue(issues, "warning",
                   f"web.{_key} is set to '{_val}', but no bundled or installed plugin provides "
                   "that backend — web_search/web_extract will fail until it is changed",
                   "Check the name against 'hermes plugins list', or run 'hermes tools'")
            continue
        # Declared by a plugin that this process has not registered. Plugins load after config
        # validation, so only judge when the registry is already live (a long-running service that
        # predates the plugin install) — otherwise this would warn on every cold start.
        import sys
        registry = sys.modules.get("agent.web_search_registry")
        get_provider = getattr(registry, "get_provider", None)
        if get_provider is None:
            continue  # registry not live yet: plugins load after config validation
        try:
            registered = get_provider(_val) is not None
        except Exception:
            registered = True  # a registry error is not evidence the pin is broken
        if not registered:
            _issue(issues, "warning",
                   f"web.{_key} is set to '{_val}', provided by plugin "
                   f"'{declared[_val]}', but that provider is not registered in this process — "
                   "the running service predates the plugin and must be restarted",
                   "Restart the serving process (WebUI/gateway), then re-run 'hermes doctor'")

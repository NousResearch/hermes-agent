"""TypeSafe Jev skill routing — names at most one skill before the chat model runs."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from . import jev_skill_router as router

logger = logging.getLogger(__name__)

_PLUGIN_ID = router.PLUGIN_NAME


def _hermes_home() -> Path:
    env = os.environ.get("HERMES_HOME")
    return Path(env).expanduser() if env else Path.home() / ".hermes"


def _log_path() -> Path:
    return _hermes_home() / "logs" / f"{_PLUGIN_ID}.log"


def _settings(ctx: Any) -> Dict[str, Any]:
    try:
        raw = {
            "mode": ctx.get_config("mode", "auto"),
            "gate": ctx.get_config("gate", router.GATE_THRESHOLD),
            "timeout_sec": ctx.get_config("timeout_sec", router.HOOK_BUDGET_SEC),
            "model": ctx.get_config("model", router.DEFAULT_MODEL),
            "base_url": ctx.get_config("base_url", ""),
            "suggest_chars": ctx.get_config("suggest_chars", router.SUGGEST_CHAR_LIMIT),
        }
    except Exception:
        raw = {}
    return router.resolve_settings(raw)


def _route_turn(ctx: Any, *, user_message: Any, platform: str = "") -> Optional[dict]:
    settings = _settings(ctx)
    api_key, default_base = router.resolve_api_key()
    if not router.routing_enabled(settings, api_key_present=bool(api_key)):
        return None
    if not isinstance(user_message, str):
        return None

    text = user_message
    if router.should_skip_request(text, suggest_chars=settings["suggest_chars"]):
        return None

    skills = router.load_live_roster(platform=platform)
    if len(skills) < 2:
        router.append_log(_log_path(), "skip", f"roster too small ({len(skills)} skills)")
        return None

    base_url = settings["base_url"] or default_base
    try:
        result = router.suggest_skill(
            text,
            skills,
            api_key=api_key,
            base_url=base_url,
            model=settings["model"],
            gate_threshold=settings["gate"],
            timeout_sec=settings["timeout_sec"],
        )
    except Exception as exc:
        router.append_log(_log_path(), "error", f"{type(exc).__name__}: {exc}")
        logger.debug("typesafe-jev-skill-routing failed", exc_info=True)
        return None

    if result is None:
        router.append_log(
            _log_path(),
            "suggest",
            f"- gate below {settings['gate']:.2f} or no choice ({len(skills)} skills)",
        )
        return None

    router.append_log(
        _log_path(),
        "suggest",
        f"{result.skill} gate={result.gate:.4f} p={result.probability:.4f} "
        f"{result.elapsed_sec:.3f}s model={result.model}",
    )
    return {"context": result.block()}


def _format_status(ctx: Any) -> str:
    settings = _settings(ctx)
    api_key, default_base = router.resolve_api_key()
    skills = router.load_live_roster()
    enabled = router.routing_enabled(settings, api_key_present=bool(api_key))
    base = settings["base_url"] or default_base
    lines = [
        f"TypeSafe Jev skill routing ({_PLUGIN_ID})",
        f"  mode      : {settings['mode']} ({'active' if enabled else 'quiet'})",
        f"  gate      : {settings['gate']:.2f}",
        f"  budget    : {settings['timeout_sec']:.1f}s per turn",
        f"  roster    : {len(skills)} skills",
        f"  model     : {settings['model']} @ {base}",
        f"  api key   : {'present' if api_key else 'MISSING (TYPESAFE_API_KEY / hermes auth jev)'}",
        f"  log       : {_log_path()}",
        "",
        "Subcommands: on | off | auto | status | suggest <text>",
        "Jev is judgment-only — do not set it as the session chat model.",
    ]
    return "\n".join(lines)


class _SlashCtx:
    """Minimal ctx for slash commands when PluginContext is unavailable."""

    def get_config(self, key: str, default: Any = None) -> Any:
        try:
            from hermes_cli.config import load_config_readonly
            from hermes_cli.plugins import _nested_plugin_value, _plugin_settings_entry

            entry = _plugin_settings_entry(load_config_readonly() or {}, _PLUGIN_ID)
            if entry is None:
                return default
            value = _nested_plugin_value(entry.get("settings"), (key,), object())
            return default if value is object() else value
        except Exception:
            return default

    def set_config(self, key: str, value: Any) -> None:
        from hermes_cli.config import read_user_config_raw, save_config
        from hermes_cli.plugins import _nested_plugin_mapping

        full_prefix = ("plugins", "entries", _PLUGIN_ID, "settings")
        partial = _nested_plugin_mapping(full_prefix, _nested_plugin_mapping((key,), value))
        read_user_config_raw()
        save_config(partial, preserve_keys={(*full_prefix, key)}, merge_existing=True)


def _handle_slash(raw_args: str) -> Optional[str]:
    argv = raw_args.strip().split()
    ctx = _SlashCtx()
    if not argv or argv[0] in {"help", "-h", "--help", "status"}:
        return _format_status(ctx)

    action = argv[0].lower()
    if action == "on":
        ctx.set_config("mode", "on")
        return "TypeSafe Jev skill routing: on"
    if action == "off":
        ctx.set_config("mode", "off")
        return "TypeSafe Jev skill routing: off"
    if action == "auto":
        ctx.set_config("mode", "auto")
        return "TypeSafe Jev skill routing: auto (on when TYPESAFE_API_KEY is set)"

    if action == "suggest":
        if len(argv) < 2:
            return "Usage: /jev suggest <request text>"
        text = raw_args.split(None, 1)[1].strip()
        payload = _route_turn(ctx, user_message=text)
        if payload and payload.get("context"):
            return payload["context"]
        return "No skill suggestion (gate too low, empty roster, or routing skipped)."

    return f"Unknown subcommand: {argv[0]}\n\n{_format_status(ctx)}"


def register(ctx) -> None:
    def on_pre_llm_call(user_message: Any = None, platform: str = "", **_: Any) -> Optional[dict]:
        return _route_turn(ctx, user_message=user_message, platform=platform or "")

    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_command(
        "jev",
        handler=_handle_slash,
        description="TypeSafe Jev skill routing status and controls (not a chat model).",
        args_hint="[on|off|auto|status|suggest]",
    )

    def cli_setup(parser) -> None:
        sub = parser.add_subparsers(dest="action", metavar="ACTION")
        sub.add_parser("on", help="enable skill routing")
        sub.add_parser("off", help="disable skill routing")
        sub.add_parser("auto", help="enable only when TYPESAFE_API_KEY is set")
        sub.add_parser("status", help="settings, roster, and key presence")
        suggest = sub.add_parser("suggest", help="route one request now")
        suggest.add_argument("text", help="user request to route")

    def cli_handler(args) -> int:
        action = getattr(args, "action", None) or "status"
        if action == "on":
            ctx.set_config("mode", "on")
            print("typesafe-jev-skill-routing: on")
            return 0
        if action == "off":
            ctx.set_config("mode", "off")
            print("typesafe-jev-skill-routing: off")
            return 0
        if action == "auto":
            ctx.set_config("mode", "auto")
            print("typesafe-jev-skill-routing: auto")
            return 0
        if action == "status":
            print(_format_status(ctx))
            return 0
        if action == "suggest":
            payload = _route_turn(ctx, user_message=args.text)
            if payload and payload.get("context"):
                print(payload["context"])
                return 0
            print("No skill suggestion.")
            return 1
        print(_format_status(ctx))
        return 0

    ctx.register_cli_command(
        "jev",
        "TypeSafe Jev skill routing (on/off/auto/status/suggest)",
        cli_setup,
        cli_handler,
        description="Route each turn to at most one installed skill via TypeSafe System One.",
    )

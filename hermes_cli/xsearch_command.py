"""Shared `/xsearch` command logic for CLI, gateway, and dashboard chat."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from typing import Any

from hermes_cli.config import get_env_value, load_config, save_config
from hermes_constants import display_hermes_home

_CONTROL_SUBCOMMANDS = {"status", "setup", "enable", "disable", "model"}
DEFAULT_X_SEARCH_MODEL = "grok-4.20-reasoning"
DEFAULT_X_SEARCH_TIMEOUT_SECONDS = 180
DEFAULT_X_SEARCH_RETRIES = 2


@dataclass(frozen=True)
class XSearchCommandResult:
    output: str
    reset_session: bool = False


def _x_search_config(cfg: dict[str, Any]) -> dict[str, Any]:
    block = cfg.get("x_search")
    return block if isinstance(block, dict) else {}


def _effective_enabled(cfg: dict[str, Any], platform: str) -> bool:
    return "x_search" in _get_platform_toolsets(cfg, platform)


def _oauth_status() -> dict[str, Any]:
    try:
        from hermes_cli.auth import get_xai_oauth_auth_status
    except Exception:
        return {}
    status = get_xai_oauth_auth_status() or {}
    return status if isinstance(status, dict) else {}


def _check_x_search_requirements() -> bool:
    try:
        from tools.x_search_tool import check_x_search_requirements
    except Exception:
        return False
    return bool(check_x_search_requirements())


def _run_x_search_tool(**kwargs: Any) -> str:
    from tools.x_search_tool import x_search_tool

    return x_search_tool(**kwargs)


def _get_platform_toolsets(cfg: dict[str, Any], platform: str) -> set[str]:
    from hermes_cli.tools_config import _get_platform_tools

    return set(_get_platform_tools(cfg, platform))


def _apply_toolset_toggle(
    cfg: dict[str, Any],
    platform: str,
    toolset_names: list[str],
    action: str,
) -> None:
    from hermes_cli.tools_config import _apply_toolset_change

    _apply_toolset_change(cfg, platform, toolset_names, action)


def _api_key_present() -> bool:
    return bool(str(get_env_value("XAI_API_KEY") or "").strip())


def _preferred_credential_source(
    oauth_status: dict[str, Any],
    *,
    api_key_present: bool,
) -> str:
    if bool(oauth_status.get("logged_in")):
        return "xai-oauth"
    if api_key_present:
        return "xai"
    return "unavailable"


def _bool_word(value: bool) -> str:
    return "yes" if value else "no"


def _coerce_int(value: Any, default: int, *, minimum: int = 0) -> int:
    try:
        return max(minimum, int(value))
    except Exception:
        return default


def _usage_text() -> str:
    return "\n".join(
        [
            "Usage:",
            "  /xsearch <query> [--from handles] [--exclude handles] [--since YYYY-MM-DD] [--until YYYY-MM-DD] [--images] [--videos]",
            "  /xsearch status",
            "  /xsearch setup",
            "  /xsearch enable",
            "  /xsearch disable",
            "  /xsearch model [name]",
        ]
    )


def _status_text(platform: str) -> str:
    cfg = load_config()
    x_cfg = _x_search_config(cfg)
    enabled = _effective_enabled(cfg, platform)
    oauth_status = _oauth_status()
    api_key_present = _api_key_present()
    preferred = _preferred_credential_source(
        oauth_status,
        api_key_present=api_key_present,
    )
    ready = enabled and preferred != "unavailable" and _check_x_search_requirements()
    lines = [
        f"X Search status ({platform})",
        f"- Ready: {_bool_word(ready)}",
        f"- Toolset enabled: {_bool_word(enabled)}",
        f"- Model: {str(x_cfg.get('model') or DEFAULT_X_SEARCH_MODEL).strip() or DEFAULT_X_SEARCH_MODEL}",
        f"- Timeout: {_coerce_int(x_cfg.get('timeout_seconds', DEFAULT_X_SEARCH_TIMEOUT_SECONDS), DEFAULT_X_SEARCH_TIMEOUT_SECONDS, minimum=30)}s",
        f"- Retries: {_coerce_int(x_cfg.get('retries', DEFAULT_X_SEARCH_RETRIES), DEFAULT_X_SEARCH_RETRIES)}",
        f"- xAI OAuth logged in: {_bool_word(bool(oauth_status.get('logged_in')))}",
        f"- XAI_API_KEY configured: {_bool_word(api_key_present)}",
        f"- Preferred credential source: {preferred}",
        f"- Config path: {display_hermes_home()}/config.yaml",
    ]
    if not ready:
        lines.append("- Next step: run `/xsearch setup`")
    return "\n".join(lines)


def _ensure_default_model(cfg: dict[str, Any]) -> bool:
    cfg.setdefault("x_search", {})
    block = _x_search_config(cfg)
    current = str(block.get("model") or "").strip()
    if current:
        return False
    block["model"] = DEFAULT_X_SEARCH_MODEL
    cfg["x_search"] = block
    return True


def _setup(platform: str) -> XSearchCommandResult:
    cfg = load_config()
    changed: list[str] = []
    reset_session = False

    if not _effective_enabled(cfg, platform):
        _apply_toolset_toggle(cfg, platform, ["x_search"], "enable")
        changed.append(f"enabled x_search for {platform}")
        reset_session = True

    if _ensure_default_model(cfg):
        changed.append(f"set x_search.model to {DEFAULT_X_SEARCH_MODEL}")

    if changed:
        save_config(cfg)

    oauth_status = _oauth_status()
    api_key_present = _api_key_present()
    preferred = _preferred_credential_source(
        oauth_status,
        api_key_present=api_key_present,
    )
    lines = ["X Search setup"]
    if changed:
        lines.extend(f"- {item}" for item in changed)
    else:
        lines.append("- no config changes were needed")

    if preferred == "unavailable":
        lines.extend(
            [
                "- Remaining auth step: sign in with `hermes auth add xai-oauth`",
                "- Alternative auth step: set `XAI_API_KEY` in ~/.hermes/.env",
                "- Then run `/xsearch status` or `/xsearch \"latest xai posts\"`",
            ]
        )
    else:
        lines.extend(
            [
                f"- Credential source ready: {preferred}",
                "- Next: run `/xsearch status` to verify or `/xsearch \"latest xai posts\"` to search now",
            ]
        )

    return XSearchCommandResult("\n".join(lines), reset_session=reset_session)


def _toggle_toolset(platform: str, action: str) -> XSearchCommandResult:
    cfg = load_config()
    enabled = _effective_enabled(cfg, platform)
    if action == "enable" and enabled:
        return XSearchCommandResult(f"x_search is already enabled for {platform}")
    if action == "disable" and not enabled:
        return XSearchCommandResult(f"x_search is already disabled for {platform}")

    _apply_toolset_toggle(cfg, platform, ["x_search"], action)
    save_config(cfg)
    verb = "Enabled" if action == "enable" else "Disabled"
    return XSearchCommandResult(
        f"{verb} x_search for {platform}.",
        reset_session=True,
    )


def _set_or_show_model(rest: str) -> XSearchCommandResult:
    value = rest.strip()
    cfg = load_config()
    block = _x_search_config(cfg)
    current = str(block.get("model") or DEFAULT_X_SEARCH_MODEL).strip() or DEFAULT_X_SEARCH_MODEL
    if not value:
        return XSearchCommandResult(f"x_search.model: {current}")
    block["model"] = value
    cfg["x_search"] = block
    save_config(cfg)
    return XSearchCommandResult(f"x_search.model -> {value}")


def _split_handles(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_search(rest: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        tokens = shlex.split(rest)
    except ValueError as exc:
        return None, f"xsearch parse error: {exc}"

    if not tokens:
        return None, _usage_text()

    args: dict[str, Any] = {
        "query": "",
        "allowed_x_handles": [],
        "excluded_x_handles": [],
        "from_date": "",
        "to_date": "",
        "enable_image_understanding": False,
        "enable_video_understanding": False,
    }
    query_tokens: list[str] = []
    idx = 0
    seen_flag = False
    while idx < len(tokens):
        token = tokens[idx]
        if token == "--from":
            seen_flag = True
            idx += 1
            if idx >= len(tokens):
                return None, "xsearch: --from requires a comma-separated handle list"
            args["allowed_x_handles"] = _split_handles(tokens[idx])
        elif token == "--exclude":
            seen_flag = True
            idx += 1
            if idx >= len(tokens):
                return None, "xsearch: --exclude requires a comma-separated handle list"
            args["excluded_x_handles"] = _split_handles(tokens[idx])
        elif token == "--since":
            seen_flag = True
            idx += 1
            if idx >= len(tokens):
                return None, "xsearch: --since requires YYYY-MM-DD"
            args["from_date"] = tokens[idx]
        elif token == "--until":
            seen_flag = True
            idx += 1
            if idx >= len(tokens):
                return None, "xsearch: --until requires YYYY-MM-DD"
            args["to_date"] = tokens[idx]
        elif token == "--images":
            seen_flag = True
            args["enable_image_understanding"] = True
        elif token == "--videos":
            seen_flag = True
            args["enable_video_understanding"] = True
        elif token.startswith("--"):
            return None, f"xsearch: unknown flag {token}"
        else:
            if seen_flag:
                return None, "xsearch: put the query before flags"
            query_tokens.append(token)
        idx += 1

    query = " ".join(query_tokens).strip()
    if not query:
        return None, "xsearch: query is required"
    args["query"] = query
    return args, None


def _citation_rows(payload: dict[str, Any]) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    combined = list(payload.get("citations") or []) + list(payload.get("inline_citations") or [])
    for item in combined:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        title = str(item.get("title") or "").strip()
        rows.append(f"- {title + ' — ' if title else ''}{url}")
        if len(rows) >= 5:
            break
    return rows


def _format_search_result(raw: str) -> str:
    try:
        payload = json.loads(raw)
    except Exception:
        return raw

    if not isinstance(payload, dict):
        return raw

    if not payload.get("success"):
        error = str(payload.get("error") or "unknown x_search failure").strip()
        lines = ["X Search error", error]
        if "x_search is not enabled for this model" in error:
            lines.append("Hint: try `/xsearch model grok-4.20-reasoning`.")
        elif "No xAI credentials available" in error:
            lines.append("Hint: run `/xsearch setup`.")
        return "\n".join(lines)

    answer = str(payload.get("answer") or "").strip() or "(no answer)"
    model = str(payload.get("model") or DEFAULT_X_SEARCH_MODEL).strip() or DEFAULT_X_SEARCH_MODEL
    source = str(payload.get("credential_source") or "unknown").strip() or "unknown"
    lines = [
        f"X Search via {model} ({source})",
        "",
        answer,
    ]
    citations = _citation_rows(payload)
    if citations:
        lines.extend(["", "Sources:", *citations])
    if bool(payload.get("degraded")):
        reason = str(payload.get("degraded_reason") or "no citations returned").strip()
        lines.extend(["", f"Warning: degraded result — {reason}"])
    return "\n".join(lines)


def _run_search(rest: str, platform: str) -> XSearchCommandResult:
    cfg = load_config()
    if not _effective_enabled(cfg, platform):
        return XSearchCommandResult(
            f"x_search is disabled for {platform}. Run `/xsearch enable` or `/xsearch setup` first."
        )
    parsed, error = _parse_search(rest)
    if error:
        return XSearchCommandResult(error)
    assert parsed is not None
    return XSearchCommandResult(_format_search_result(_run_x_search_tool(**parsed)))


def run_xsearch_command(command: str, *, platform: str = "cli") -> XSearchCommandResult:
    text = command.strip()
    if not text:
        return XSearchCommandResult(_usage_text())

    if text.startswith("/"):
        text = text[1:]
    if text.lower().startswith("xsearch"):
        text = text[7:].strip()
    elif text.lower().startswith("x-search"):
        text = text[8:].strip()

    if not text:
        return XSearchCommandResult(_usage_text())

    try:
        tokens = shlex.split(text)
    except ValueError as exc:
        return XSearchCommandResult(f"xsearch parse error: {exc}")

    if not tokens:
        return XSearchCommandResult(_usage_text())

    head = tokens[0].lower()
    if head in {"status", "setup", "enable", "disable"} and len(tokens) == 1:
        if head == "status":
            return XSearchCommandResult(_status_text(platform))
        if head == "setup":
            return _setup(platform)
        return _toggle_toolset(platform, head)

    if head == "model":
        remainder = text.split(None, 1)[1].strip() if len(tokens) > 1 else ""
        return _set_or_show_model(remainder)

    if head in _CONTROL_SUBCOMMANDS and len(tokens) > 1:
        # Allow natural language searches that happen to start with words
        # like "status" or "setup" as long as additional words are present.
        return _run_search(text, platform)

    return _run_search(text, platform)

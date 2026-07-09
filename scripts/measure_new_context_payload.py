#!/usr/bin/env python3
"""Measure first-request context around /new without calling a model.

The diagnostic builds request-shaped payloads through the real AIAgent prompt
path, estimates their component sizes, and exercises SessionStore.reset_session
against a temporary database. It never sends an inference request or mutates the
user's real session store.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence, cast


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _strip_kanban_env() -> None:
    """Avoid measuring the current worker's Kanban-only prompt/tool surface."""
    for key in list(os.environ):
        if key.startswith("HERMES_KANBAN_"):
            os.environ.pop(key, None)


def _quiet_call(func, *args, **kwargs):
    """Keep prompt status notices from corrupting JSON diagnostic output."""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def _estimate_text_tokens(text: str) -> int:
    from agent.model_metadata import estimate_messages_tokens_rough

    if not text:
        return 0
    return estimate_messages_tokens_rough([{"role": "system", "content": text}])


def _estimate_messages_tokens(messages: Sequence[dict[str, Any]]) -> int:
    from agent.model_metadata import estimate_messages_tokens_rough

    return estimate_messages_tokens_rough(list(messages))


def _estimate_request_tokens(
    messages: Sequence[dict[str, Any]],
    *,
    tools: Optional[Sequence[dict[str, Any]]] = None,
) -> int:
    from agent.model_metadata import estimate_request_tokens_rough

    return estimate_request_tokens_rough(
        list(messages), tools=list(tools) if tools else None
    )


def _parse_toolsets(raw: Optional[str]) -> Optional[list[str]]:
    if raw is None:
        return None
    if not raw.strip():
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _platform_toolsets(platform: str, explicit: Optional[str]) -> Optional[list[str]]:
    parsed = _parse_toolsets(explicit)
    if parsed is not None:
        return parsed
    if platform == "cli":
        return None

    try:
        from hermes_cli.config import load_config
        from hermes_cli.tools_config import _get_platform_tools

        config = load_config() or {}
        return sorted(_get_platform_tools(config, platform))
    except Exception:
        return None


def _gateway_context_prompt(platform: str) -> str:
    if platform == "cli":
        return ""

    from gateway.config import GatewayConfig, Platform
    from gateway.session import (
        SessionEntry,
        SessionSource,
        build_session_context,
        build_session_context_prompt,
        build_session_key,
    )

    platform_enum = Platform(platform)
    source = SessionSource(
        platform=platform_enum,
        chat_id="diagnostic-chat",
        chat_name="Diagnostic chat",
        chat_type="dm",
        user_id="diagnostic-user",
        user_name="Diagnostic User",
        thread_id="diagnostic-thread" if platform == "telegram" else None,
        chat_topic=(
            "Diagnostic thread for measuring /new context size"
            if platform == "telegram"
            else None
        ),
    )
    entry = SessionEntry(
        session_key=build_session_key(source),
        session_id="diagnostic-session",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        origin=source,
        platform=platform_enum,
        chat_type=source.chat_type,
    )
    context = build_session_context(source, GatewayConfig(), entry)
    return build_session_context_prompt(context)


def _build_agent(
    *,
    platform: str,
    enabled_toolsets: Optional[list[str]],
    skip_context_files: bool,
):
    from run_agent import AIAgent

    return AIAgent(
        model="gpt-5.5",
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="diagnostic-no-model-call",
        max_iterations=1,
        quiet_mode=True,
        verbose_logging=False,
        platform=platform,
        enabled_toolsets=cast(Any, enabled_toolsets),
        skip_context_files=skip_context_files,
        session_db=None,
    )


def _synthetic_history(turns: int, chars_per_message: int) -> list[dict[str, str]]:
    history = []
    payload = "x" * max(chars_per_message, 0)
    for index in range(turns):
        history.extend(
            [
                {"role": "user", "content": f"old user turn {index}\n{payload}"},
                {
                    "role": "assistant",
                    "content": f"old assistant turn {index}\n{payload}",
                },
            ]
        )
    return history


def _first_request_snapshot(
    *,
    label: str,
    agent,
    system_message: Optional[str],
    history: list[dict[str, Any]],
    user_message: str,
) -> dict[str, Any]:
    from agent.codex_responses_adapter import _summarize_user_message_for_log
    from agent.conversation_loop import _restore_or_build_system_prompt
    from agent.message_sanitization import _sanitize_surrogates
    from agent.process_bootstrap import _install_safe_stdio
    from agent.turn_context import build_turn_context
    from hermes_logging import set_session_context
    from tools.skill_provenance import set_current_write_origin

    context = build_turn_context(
        agent,
        user_message,
        system_message,
        history,
        label,
        None,
        None,
        None,
        restore_or_build_system_prompt=_restore_or_build_system_prompt,
        install_safe_stdio=_install_safe_stdio,
        sanitize_surrogates=_sanitize_surrogates,
        summarize_user_message_for_log=_summarize_user_message_for_log,
        set_session_context=set_session_context,
        set_current_write_origin=set_current_write_origin,
        ra=lambda: __import__("run_agent"),
    )

    api_messages = list(context.messages)
    effective_system = context.active_system_prompt or ""
    if agent.ephemeral_system_prompt:
        effective_system = (
            effective_system + "\n\n" + agent.ephemeral_system_prompt
        ).strip()
    if effective_system:
        api_messages.insert(0, {"role": "system", "content": effective_system})

    system_parts = agent._build_system_prompt_parts(system_message=system_message)
    component_tokens = {
        "tool schemas": _estimate_request_tokens([], tools=agent.tools or None),
        "stable skills/guidance": _estimate_text_tokens(system_parts.get("stable", "")),
        "context files/gateway metadata": _estimate_text_tokens(
            system_parts.get("context", "")
        ),
        "memory/profile/volatile": _estimate_text_tokens(
            system_parts.get("volatile", "")
        ),
        "session tail": _estimate_messages_tokens(history),
        "current user message": (
            _estimate_messages_tokens([context.messages[-1]])
            if context.messages
            else 0
        ),
    }
    return {
        "label": label,
        "message_count": len(api_messages),
        "history_message_count": len(history),
        "tool_count": len(agent.tools or []),
        "total_request_tokens": _estimate_request_tokens(
            api_messages, tools=agent.tools or None
        ),
        "component_tokens": component_tokens,
        "top_contributors": sorted(
            component_tokens.items(), key=lambda item: item[1], reverse=True
        ),
        "system_chars": {
            name: len(system_parts.get(name, ""))
            for name in ("stable", "context", "volatile")
        },
    }


def _sessionstore_reset_probe() -> dict[str, Any]:
    """Exercise the gateway /new primitive in an isolated session store."""
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource, SessionStore
    from hermes_state import SessionDB

    with tempfile.TemporaryDirectory(prefix="hermes-new-reset-probe-") as temp_dir:
        root = Path(temp_dir)
        db = SessionDB(db_path=root / "state.db")
        store = SessionStore(sessions_dir=root / "sessions", config=GatewayConfig())
        store._db = db
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="diagnostic-chat",
            chat_type="dm",
            user_id="diagnostic-user",
            thread_id="diagnostic-thread",
        )
        entry = store.get_or_create_session(source)
        old_session_id = entry.session_id
        db.append_message(old_session_id, "user", "old message before /new")
        db.append_message(old_session_id, "assistant", "old response before /new")
        old_before = store.load_transcript(old_session_id)

        new_entry = store.reset_session(entry.session_key)
        assert new_entry is not None
        new_after = store.load_transcript(new_entry.session_id)
        old_after = store.load_transcript(old_session_id)
        db.close()
        return {
            "old_session_id_changed": old_session_id != new_entry.session_id,
            "old_messages_before_reset": len(old_before),
            "new_messages_after_reset": len(new_after),
            "old_messages_retained_in_old_session": len(old_after),
            "new_session_is_empty": not new_after,
        }


def _print_human(report: dict[str, Any]) -> None:
    print("/new context payload diagnostic (no model call)\n")
    metadata = report["metadata"]
    print(
        f"platform={metadata['platform']} "
        f"toolsets={metadata['enabled_toolsets']} "
        f"skip_context_files={metadata['skip_context_files']}"
    )
    if report.get("reset_probe"):
        probe = report["reset_probe"]
        print(
            "reset_probe: "
            f"new_session_is_empty={probe['new_session_is_empty']} "
            f"old_messages_before={probe['old_messages_before_reset']} "
            f"new_messages_after={probe['new_messages_after_reset']} "
            f"old_messages_retained={probe['old_messages_retained_in_old_session']}"
        )
    print()

    for snapshot in report["snapshots"]:
        print(
            f"{snapshot['label']}: total≈{snapshot['total_request_tokens']:,} "
            f"tokens, messages={snapshot['message_count']}, "
            f"tools={snapshot['tool_count']}"
        )
        for name, tokens in snapshot["top_contributors"]:
            print(f"  - {name}: ≈{tokens:,}")
        print()

    print("comparison:")
    for key, value in report["comparison"].items():
        print(f"  - {key}: {value}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=["cli", "telegram"], default="telegram")
    parser.add_argument(
        "--enabled-toolsets",
        default=None,
        help=(
            "Comma-separated toolsets. Omit to use platform config; "
            "pass '' for no tools."
        ),
    )
    parser.add_argument("--skip-context-files", action="store_true")
    parser.add_argument("--include-kanban-env", action="store_true")
    parser.add_argument("--history-turns", type=int, default=3)
    parser.add_argument("--history-chars", type=int, default=4000)
    parser.add_argument("--message", default="diagnostic message after /new")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--no-reset-probe", action="store_true")
    args = parser.parse_args(argv)

    if not args.include_kanban_env:
        _strip_kanban_env()

    enabled_toolsets = _quiet_call(
        _platform_toolsets, args.platform, args.enabled_toolsets
    )
    context_prompt = _quiet_call(_gateway_context_prompt, args.platform)
    histories = {
        "true_empty_session": [],
        "post_new_first_message": [],
        "after_one_normal_message": [
            {"role": "user", "content": "normal message before current turn"},
            {"role": "assistant", "content": "normal response before current turn"},
        ],
        "old_session_without_new": _synthetic_history(
            args.history_turns, args.history_chars
        ),
    }

    snapshots = []
    for label, history in histories.items():
        agent = _quiet_call(
            _build_agent,
            platform=args.platform,
            enabled_toolsets=enabled_toolsets,
            skip_context_files=args.skip_context_files,
        )
        snapshots.append(
            _quiet_call(
                _first_request_snapshot,
                label=label,
                agent=agent,
                system_message=context_prompt or None,
                history=list(history),
                user_message=args.message,
            )
        )

    by_label = {snapshot["label"]: snapshot for snapshot in snapshots}
    empty = by_label["true_empty_session"]
    post_new = by_label["post_new_first_message"]
    normal = by_label["after_one_normal_message"]
    old = by_label["old_session_without_new"]
    report = {
        "metadata": {
            "platform": args.platform,
            "enabled_toolsets": enabled_toolsets,
            "skip_context_files": args.skip_context_files,
            "history_turns": args.history_turns,
            "history_chars": args.history_chars,
        },
        "reset_probe": (
            None
            if args.no_reset_probe
            else _quiet_call(_sessionstore_reset_probe)
        ),
        "snapshots": snapshots,
        "comparison": {
            "post_new_minus_true_empty_tokens": (
                post_new["total_request_tokens"] - empty["total_request_tokens"]
            ),
            "normal_minus_true_empty_tokens": (
                normal["total_request_tokens"] - empty["total_request_tokens"]
            ),
            "old_without_new_minus_post_new_tokens": (
                old["total_request_tokens"] - post_new["total_request_tokens"]
            ),
            "prior_history_detached_by_new": (
                post_new["component_tokens"]["session tail"] == 0
            ),
        },
    }

    if args.json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

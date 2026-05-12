from __future__ import annotations

import subprocess
import json
import os
import itertools
import threading
import mimetypes
import shutil
from datetime import datetime, timezone
from pathlib import Path

SCRIPT = Path(
    os.environ.get(
        "TOOL_SITE_STATUS_SCRIPT",
        "/Users/dom/agents/hermes-toolsite-monitor/monitor/toolsite-status.sh",
    )
)
ADVISOR = Path(
    os.environ.get(
        "TOOL_SITE_ADVISOR_SCRIPT",
        "/Users/dom/agents/hermes-toolsite-monitor/monitor/phase2-advisor.sh",
    )
)
REPO = Path(
    os.environ.get(
        "TOOL_FACTORY_REPO",
        "/Users/dom/Desktop/库/toolsite-agent-factory-main-monitor",
    )
)
DEFAULT_SITE_ID = "typing-test-online"
HERMES_HOME = Path(
    os.environ.get("HERMES_HOME", "/Users/dom/agents/hermes-toolsite-monitor/hermes-home")
)
STATE_DIR = Path(os.environ.get("TOOL_SITE_REMOTE_STATE_DIR", HERMES_HOME / "state"))
REMOTE_STATE = STATE_DIR / "toolsite-remote.json"
INBOX = STATE_DIR / "toolsite-inbox.jsonl"
ATTACHMENTS_DIR = STATE_DIR / "toolsite-attachments"
EXPLAIN_TRIGGERS = (
    "这里我看不懂",
    "这是什么意思",
    "什么意思",
    "解释一下",
    "帮我解释",
)

_REPLY_COUNTER = itertools.count(1)
_PENDING_REPLIES: dict[str, str] = {}
_PENDING_REPLIES_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _store_reply(message: str) -> dict[str, str]:
    with _PENDING_REPLIES_LOCK:
        key = f"reply-{next(_REPLY_COUNTER)}"
        _PENDING_REPLIES[key] = message
    return {"action": "rewrite", "text": f"/toolsite-reply {key}"}


def _handle_reply(raw_args: str) -> str:
    key = raw_args.strip().split(maxsplit=1)[0] if raw_args.strip() else ""
    with _PENDING_REPLIES_LOCK:
        message = _PENDING_REPLIES.pop(key, "")
    return message or "已处理。"


def _event_source(event) -> tuple[str, str, str]:
    source = getattr(event, "source", None)
    platform = getattr(source, "platform", "")
    platform_name = getattr(platform, "value", platform) or "unknown"
    chat_id = str(getattr(source, "chat_id", "") or "")
    message_id = str(getattr(event, "message_id", "") or "")
    return str(platform_name), chat_id, message_id


def _safe_path_part(value: str, fallback: str = "unknown") -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in str(value or ""))
    safe = safe.strip(".-")
    return safe or fallback


def _event_text(event) -> str:
    text = str(getattr(event, "text", "") or "")
    if text:
        return text
    raw_message = getattr(event, "raw_message", None)
    caption = getattr(raw_message, "caption", None)
    if caption:
        return str(caption)
    return ""


def _image_extension(path: Path, mime_type: str) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        return suffix
    guessed = mimetypes.guess_extension(mime_type or "")
    if guessed in {".jpe"}:
        return ".jpg"
    if guessed in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        return guessed
    return ".jpg"


def _raw_photo_meta(event) -> dict:
    raw_message = getattr(event, "raw_message", None)
    photos = list(getattr(raw_message, "photo", None) or [])
    if photos:
        photo = photos[-1]
        return {
            "telegram_file_id": str(getattr(photo, "file_id", "") or ""),
            "width": getattr(photo, "width", None),
            "height": getattr(photo, "height", None),
            "file_name": "",
        }
    document = getattr(raw_message, "document", None)
    if document is not None:
        return {
            "telegram_file_id": str(getattr(document, "file_id", "") or ""),
            "width": getattr(document, "width", None),
            "height": getattr(document, "height", None),
            "file_name": str(getattr(document, "file_name", "") or ""),
        }
    return {"telegram_file_id": "", "width": None, "height": None, "file_name": ""}


def _is_image_media(media_path: str, media_type: str) -> bool:
    mime_type = str(media_type or "").lower()
    if mime_type.startswith("image/"):
        return True
    suffix = Path(str(media_path or "")).suffix.lower()
    return suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif"}


def _event_has_image_media(event) -> bool:
    media_urls = list(getattr(event, "media_urls", None) or [])
    media_types = list(getattr(event, "media_types", None) or [])
    return any(
        _is_image_media(media_url, media_types[index] if index < len(media_types) else "")
        for index, media_url in enumerate(media_urls)
    )


def _collect_image_attachments(event) -> list[dict]:
    media_urls = list(getattr(event, "media_urls", None) or [])
    media_types = list(getattr(event, "media_types", None) or [])
    if not media_urls:
        return []

    platform_name, chat_id, message_id = _event_source(event)
    del platform_name
    safe_chat = _safe_path_part(chat_id, "chat")
    safe_message = _safe_path_part(message_id, "message")
    target_dir = ATTACHMENTS_DIR / safe_chat / safe_message
    target_dir.mkdir(parents=True, exist_ok=True)

    raw_meta = _raw_photo_meta(event)
    attachments: list[dict] = []
    for index, media_url in enumerate(media_urls):
        media_type = str(media_types[index] if index < len(media_types) else "")
        if not _is_image_media(media_url, media_type):
            continue

        source_path = Path(str(media_url)).expanduser()
        if not source_path.is_file():
            continue

        mime_type = media_type or mimetypes.guess_type(source_path.name)[0] or "image/jpeg"
        original_name = raw_meta.get("file_name") or source_path.name
        extension = _image_extension(source_path, mime_type)
        file_name = _safe_path_part(original_name or f"image-{index + 1}{extension}", f"image-{index + 1}{extension}")
        if Path(file_name).suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
            file_name = f"{file_name}{extension}"
        if len(media_urls) > 1:
            stem = Path(file_name).stem
            suffix = Path(file_name).suffix or extension
            file_name = f"{stem}-{index + 1}{suffix}"

        local_path = target_dir / file_name
        if source_path.resolve() != local_path.resolve():
            shutil.copy2(source_path, local_path)

        attachments.append(
            {
                "kind": "image",
                "telegram_file_id": raw_meta.get("telegram_file_id") or "",
                "local_path": str(local_path),
                "mime_type": mime_type,
                "file_name": file_name,
                "width": raw_meta.get("width"),
                "height": raw_meta.get("height"),
            }
        )
    return attachments


def _event_actor(event) -> str:
    platform_name, chat_id, _message_id = _event_source(event)
    return f"{platform_name}:{chat_id}" if chat_id else platform_name


def _read_remote_state() -> dict:
    try:
        data = json.loads(REMOTE_STATE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"remote_mode": False}
    except Exception:
        return {"remote_mode": False}
    if not isinstance(data, dict):
        return {"remote_mode": False}
    return data


def _write_remote_state(remote_mode: bool, updated_by: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "remote_mode": bool(remote_mode),
        "updated_at": _now_iso(),
        "updated_by": updated_by,
    }
    tmp = REMOTE_STATE.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(REMOTE_STATE)


def _remote_status_message() -> str:
    state = _read_remote_state()
    enabled = bool(state.get("remote_mode"))
    if enabled:
        return "远程模式当前：开启。后续人工审核事件可以主动通知你。"
    return "远程模式当前：关闭。Hermes 不会主动通知，只响应你的主动查询。"


def _parse_remote_command(text: str) -> str:
    stripped = text.strip()
    if stripped == "开启远程模式":
        return "on"
    if stripped == "关闭远程模式":
        return "off"
    if stripped == "查看远程模式":
        return "status"
    parts = stripped.split(maxsplit=1)
    if not parts:
        return ""
    command = parts[0].split("@", 1)[0].lower()
    if command not in {"/remote", "/toolsite-remote"}:
        return ""
    arg = parts[1].strip().lower() if len(parts) > 1 else "status"
    if arg in {"on", "open", "enable", "enabled"}:
        return "on"
    if arg in {"off", "close", "disable", "disabled"}:
        return "off"
    if arg in {"status", "state", "查看"}:
        return "status"
    return "usage"


def _handle_remote_action(action: str, event=None, updated_by: str = "") -> str:
    actor = updated_by or (_event_actor(event) if event is not None else "manual")
    if action == "on":
        _write_remote_state(True, actor)
        return "远程模式已开启。后续人工审核事件可以主动通知你。"
    if action == "off":
        _write_remote_state(False, actor)
        return "远程模式已关闭。Hermes 不会主动通知，只响应你的主动查询。"
    if action == "status":
        return _remote_status_message()
    return "用法：/remote on、/remote off 或 /remote status。"


def _handle_remote(raw_args: str) -> str:
    text = f"/remote {raw_args.strip()}".strip()
    action = _parse_remote_command(text)
    return _handle_remote_action(action, updated_by="manual")


def _append_inbox_message(event, text: str, attachments: list[dict] | None = None) -> None:
    platform_name, chat_id, message_id = _event_source(event)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "user_message",
        "source": platform_name,
        "chat_id": chat_id,
        "message_id": message_id,
        "text": text,
        "created_at": _now_iso(),
        "handled": False,
    }
    if attachments:
        payload["attachments"] = attachments
    with INBOX.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _is_explain_request(text: str) -> bool:
    return any(trigger in text for trigger in EXPLAIN_TRIGGERS)


def _site_id(raw_args: str) -> str:
    parts = raw_args.strip().split()
    if len(parts) != 1:
        return ""
    site_id = parts[0]
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    if not site_id or "/" in site_id or ".." in site_id or site_id.startswith("."):
        return ""
    if any(ch not in allowed for ch in site_id):
        return ""
    return site_id


def _parse_reviews_command(text: str) -> str | None:
    stripped = text.strip()
    if stripped in {"查看审核点", "查看人工审核点"}:
        return ""
    for prefix in ("查看审核点 ", "查看人工审核点 "):
        if stripped.startswith(prefix):
            return stripped[len(prefix) :].strip()
    parts = stripped.split(maxsplit=1)
    if not parts:
        return None
    command = parts[0].split("@", 1)[0].lower()
    if command not in {"/reviews", "/toolsite-reviews"}:
        return None
    return parts[1].strip() if len(parts) > 1 else ""


def _iter_review_files(site_id: str = "") -> tuple[bool, list[Path]]:
    runs_dir = REPO / "runs"
    if site_id:
        event_file = runs_dir / site_id / "human-review-events.jsonl"
        return event_file.is_file(), [event_file] if event_file.is_file() else []
    files = sorted(runs_dir.glob("*/human-review-events.jsonl"))
    return bool(files), files


def _open_review_events(path: Path) -> list[dict]:
    events: list[dict] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return events
    except Exception:
        return events
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except Exception:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "human_review" and event.get("status") == "open":
            event.setdefault("site_id", path.parent.name)
            events.append(event)
    return events


def _format_attachment(attachment: dict) -> str:
    label = str(attachment.get("label") or "attachment")
    path = str(attachment.get("path") or "")
    if path:
        return f"- {label}: {path}"
    return f"- {label}"


def _format_review_event(event: dict) -> str:
    lines = [
        f"site_id: {event.get('site_id') or 'unknown'}",
        f"title: {event.get('title') or 'Untitled review'}",
        "message:",
        str(event.get("message") or ""),
    ]
    expected_reply = str(event.get("expected_reply") or "").strip()
    if expected_reply:
        lines.extend(["expected_reply:", expected_reply])
    attachments = event.get("attachments")
    if isinstance(attachments, list) and attachments:
        lines.append("attachments:")
        for attachment in attachments:
            if isinstance(attachment, dict):
                lines.append(_format_attachment(attachment))
    return "\n".join(lines)


def _handle_reviews(raw_args: str) -> str:
    raw = raw_args.strip()
    site_id = ""
    if raw:
        site_id = _site_id(raw)
        if not site_id:
            return "用法：/reviews 或 /reviews <site-id>。"
    found_any_file, files = _iter_review_files(site_id)
    if not found_any_file:
        return "当前没有找到 human-review-events.jsonl。"
    events: list[dict] = []
    for path in files:
        events.extend(_open_review_events(path))
    if not events:
        return "当前没有待处理的人工审核点。"
    return "\n\n".join(_format_review_event(event) for event in events)


def _run_fixed(*args: str) -> str:
    result = subprocess.run(
        [str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=30,
        shell=False,
        check=False,
    )
    output = result.stdout.strip() or result.stderr.strip()
    if result.returncode != 0:
        return output or f"toolsite-status failed with code {result.returncode}"
    return output or "Command returned no output."


def _run_advisor(question: str) -> str:
    result = subprocess.run(
        [str(ADVISOR), question],
        capture_output=True,
        text=True,
        timeout=180,
        shell=False,
        check=False,
    )
    output = result.stdout.strip()
    if result.returncode != 0:
        return "AI 顾问暂时不可用。请在电脑端/Codex 中查看详细错误。"
    if len(output) > 1800:
        output = output[:1800].rstrip() + "\n\n[已截断]"
    return output or "AI 顾问没有返回内容。"


def _status_entries(raw_status: str) -> list[str]:
    entries: list[str] = []
    in_status = False
    for line in raw_status.splitlines():
        stripped = line.strip()
        if stripped == "## git status":
            in_status = True
            continue
        if in_status and stripped in {"## last commit", "## open PRs"}:
            break
        if not in_status or not stripped:
            continue
        if stripped.startswith("## "):
            continue
        entries.append(stripped)
    return entries


def _count_prs(raw_prs: str) -> int:
    text = raw_prs.strip()
    if not text:
        return 0
    return len([line for line in text.splitlines() if line.strip()])


def _agent6_report_state(site_id: str = DEFAULT_SITE_ID) -> str:
    raw = _run_fixed("agent6", site_id)
    marker = "## agent-6 launch-report.md"
    if marker not in raw:
        return "missing"
    report = raw.split(marker, 1)[1]
    if "\nmissing" in report:
        return "missing"
    if "full_launch_completed" in report:
        return "full_launch_completed"
    return "present_not_full_launch_completed"


def _json_status(path: Path) -> str:
    try:
        data = json.loads(path.read_text())
    except Exception as error:
        return f"unreadable ({error})"
    for key in ("status", "result", "gateStatus", "outcome"):
        if key in data:
            return str(data[key])
    for key in ("pass", "passed"):
        if key in data:
            return "pass" if data[key] is True else "fail" if data[key] is False else str(data[key])
    return "unknown"


def _handle_agent6_summary(site_id: str) -> str:
    run_dir = REPO / "runs" / site_id
    state_path = run_dir / "state.json"
    report_path = run_dir / "agent-6-output" / "launch-report.md"
    lines = [f"Agent6 摘要：{site_id}"]
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text())
            lines.extend(
                [
                    f"- run status: {state.get('status', 'unknown')}",
                    f"- approved_for_production: {state.get('approved_for_production', 'unknown')}",
                    f"- QA passed: {state.get('qa', {}).get('passed', 'unknown')}",
                    f"- agent_6 output: {state.get('agent_outputs', {}).get('agent_6') or 'missing'}",
                    f"- production_url: {state.get('launch', {}).get('production_url', 'unknown')}",
                ]
            )
        except Exception as error:
            lines.append(f"- state.json: unreadable ({error})")
    else:
        lines.append("- state.json: missing")
    if report_path.is_file():
        report = report_path.read_text(errors="replace")
        if "full_launch_completed" in report:
            status = "full_launch_completed"
        elif "partial_launch_blocked" in report:
            status = "partial_launch_blocked"
        else:
            status = "present, final status unclear"
        lines.append(f"- launch-report.md: {status}")
    else:
        lines.append("- launch-report.md: missing")
    return "\n".join(lines)


def _handle_gates_summary(site_id: str) -> str:
    gate_dir = REPO / "runs" / site_id / "gate-results"
    lines = [f"Gate 摘要：{site_id}"]
    if not gate_dir.is_dir():
        lines.append("- gate-results: missing")
        return "\n".join(lines)
    files = sorted(gate_dir.glob("*.json"))
    if not files:
        lines.append("- gate-results: no json files")
        return "\n".join(lines)
    for path in files:
        lines.append(f"- {path.name}: {_json_status(path)}")
    return "\n".join(lines)


def _format_yes_count(has_items: bool, count: int, note: str = "文件数") -> str:
    if not has_items:
        return "无"
    return f"有，{note} {count}"


def _friendly_status() -> str:
    raw_status = _run_fixed("status")
    branch = _run_fixed("branch").splitlines()[0].strip() or "unknown"
    last_commit = _run_fixed("last-commit").splitlines()[0].strip() or "unknown"
    prs = _run_fixed("prs")
    open_pr_count = _count_prs(prs)

    repo_path = "unknown"
    for line in raw_status.splitlines():
        if line.startswith("repo path:"):
            repo_path = line.split(":", 1)[1].strip()
            break

    entries = _status_entries(raw_status)
    runs_typing = [entry for entry in entries if "runs/typing-test-online/" in entry]
    codex = [entry for entry in entries if ".codex/" in entry or entry.endswith(".codex/")]
    workspace = [entry for entry in entries if "toolsite-agent-factory.code-workspace" in entry]
    known = set(runs_typing + codex + workspace)
    other = [entry for entry in entries if entry not in known]
    dirty = bool(entries)
    on_main = branch == "main"

    risks: list[str] = []
    if runs_typing or codex:
        risks.append("有 runs/ 或 .codex：不要 git add .")
    if not on_main:
        risks.append("这是本地 feature 分支状态，不代表 GitHub main 最新状态。")
    if not risks:
        risks.append("暂无明显提交风险")

    suggestions: list[str] = []
    if not on_main:
        suggestions.append("不要直接 push 当前分支，先确认是否需要干净 PR")
        suggestions.append("如果 main 已经有新 merge，建议在干净 main worktree 里 pull 最新 main")
    if open_pr_count > 0:
        suggestions.append("有 open PR：先 review / merge")
    agent6_state = _agent6_report_state()
    if agent6_state == "present_not_full_launch_completed":
        suggestions.append("Agent6 launch-report 存在但未 full_launch_completed：继续 Agent6 remaining gates")
    only_run_artifacts = bool(runs_typing) and not codex and not workspace and not other
    if only_run_artifacts:
        suggestions.append("工作区只有 run 产物：可以清理或备份")
    if not suggestions:
        suggestions.append("没有必须立即处理的下一步")

    return "\n".join(
        [
            "项目状态摘要：",
            f"- 本地 repo path: {repo_path}",
            f"- 本地当前分支: {branch}",
            "- GitHub 默认分支: main",
            f"- 本地分支是否为 main: {'是' if on_main else '否'}",
            f"- 最近本地 commit: {last_commit}",
            f"- open PR 数量: {open_pr_count}",
            *([] if on_main else ["- 说明: 这是本地 feature 分支状态，不代表 GitHub main 最新状态。"]),
            "",
            "本地工作区状态：",
            f"- {'dirty' if dirty else 'clean'}",
            f"- runs/typing-test-online 运行产物: {_format_yes_count(bool(runs_typing), len(runs_typing))}",
            f"- .codex 本地文件: {_format_yes_count(bool(codex), len(codex), '状态条目数，未读取目录内容')}",
            f"- workspace 本地文件: {'有' if workspace else '无'}",
            f"- 其他文件: {_format_yes_count(bool(other), len(other))}",
            "",
            "风险提示：",
            *[f"- {item}" for item in risks],
            "",
            "下一步建议：",
            *[f"- {item}" for item in suggestions],
        ]
    )


def _handle_agent6(raw_args: str) -> str:
    site_id = _site_id(raw_args)
    if not site_id:
        return "Usage: /agent6 <site-id>"
    return _handle_agent6_summary(site_id)


def _handle_gates(raw_args: str) -> str:
    site_id = _site_id(raw_args)
    if not site_id:
        return "Usage: /gates <site-id>"
    return _handle_gates_summary(site_id)


def _handle_status(raw_args: str) -> str:
    if raw_args.strip():
        return "Usage: /status"
    return _friendly_status()


def _handle_branch(raw_args: str) -> str:
    if raw_args.strip():
        return "Usage: /branch"
    return _run_fixed("branch")


def _handle_last_commit(raw_args: str) -> str:
    if raw_args.strip():
        return "Usage: /last-commit"
    return _run_fixed("last-commit")


def _handle_prs(raw_args: str) -> str:
    if raw_args.strip():
        return "Usage: /prs"
    return _run_fixed("prs")


def _handle_advisor(raw_args: str) -> str:
    question = raw_args.strip()
    if not question:
        return "请直接问一个项目状态问题。"
    return _run_advisor(question)


def _handle_denied(_raw_args: str) -> str:
    return (
        "只允许只读查询：/toolsite-status, /status, /agent6 <site-id>, "
        "/gates <site-id>, /reviews <site-id>, /prs, /branch, /last-commit, "
        "/remote on|off|status，"
        "或直接发送自然语言消息。"
    )


def _rewrite_status(event=None, **_kwargs):
    text = _event_text(event).strip()
    has_image_media = _event_has_image_media(event)
    allowed = (
        "/toolsite-status",
        "/agent6",
        "/gates",
        "/toolsite-prs",
        "/toolsite-branch",
        "/toolsite-last-commit",
        "/toolsite-advisor",
        "/toolsite-remote",
        "/remote",
        "/reviews",
        "/toolsite-reviews",
        "/toolsite-reply",
        "/toolsite-deny",
    )
    remote_action = _parse_remote_command(text)
    if remote_action:
        return _store_reply(_handle_remote_action(remote_action, event=event))
    reviews_args = _parse_reviews_command(text)
    if reviews_args is not None and not text.startswith("/"):
        command = "/toolsite-reviews"
        if reviews_args:
            command = f"{command} {reviews_args}"
        return {"action": "rewrite", "text": command}
    if text == "/status":
        return {"action": "rewrite", "text": "/toolsite-status"}
    if text == "/prs":
        return {"action": "rewrite", "text": "/toolsite-prs"}
    if text == "/branch":
        return {"action": "rewrite", "text": "/toolsite-branch"}
    if text == "/last-commit":
        return {"action": "rewrite", "text": "/toolsite-last-commit"}
    if has_image_media and not text.startswith("/"):
        try:
            attachments = _collect_image_attachments(event)
            if not attachments:
                return _store_reply("图片保存失败，请在电脑端检查 Hermes inbox。")
            _append_inbox_message(event, text, attachments=attachments)
            return _store_reply("已收到，已原文保存到 Hermes inbox。")
        except Exception:
            return _store_reply("保存失败，请在电脑端检查 Hermes inbox。")
    if text and not text.startswith("/"):
        if _is_explain_request(text):
            return {"action": "rewrite", "text": f"/toolsite-advisor {text}"}
        try:
            _append_inbox_message(event, text)
            return _store_reply("已收到，已原文保存到 Hermes inbox。")
        except Exception:
            return _store_reply("保存失败，请在电脑端检查 Hermes inbox。")
    if not any(text == cmd or text.startswith(f"{cmd} ") for cmd in allowed):
        return {"action": "rewrite", "text": "/toolsite-deny"}
    return None


def register(ctx) -> None:
    ctx.register_hook("pre_gateway_dispatch", _rewrite_status)
    ctx.register_command(
        "toolsite-reply",
        handler=_handle_reply,
        description="Return a prepared local plugin response.",
        args_hint="<reply-id>",
    )
    ctx.register_command(
        "remote",
        handler=_handle_remote,
        description="Toggle or inspect Toolsite remote mode.",
        args_hint="on|off|status",
    )
    ctx.register_command(
        "toolsite-remote",
        handler=_handle_remote,
        description="Toggle or inspect Toolsite remote mode.",
        args_hint="on|off|status",
    )
    ctx.register_command(
        "toolsite-status",
        handler=_handle_status,
        description="Read-only toolsite-agent-factory status.",
    )
    ctx.register_command(
        "agent6",
        handler=_handle_agent6,
        description="Read Agent6 launch state for a toolsite run.",
        args_hint="<site-id>",
    )
    ctx.register_command(
        "gates",
        handler=_handle_gates,
        description="Read gate-results JSON files for a toolsite run.",
        args_hint="<site-id>",
    )
    ctx.register_command(
        "reviews",
        handler=_handle_reviews,
        description="Read open Toolsite human review events.",
        args_hint="<site-id>",
    )
    ctx.register_command(
        "toolsite-reviews",
        handler=_handle_reviews,
        description="Read open Toolsite human review events.",
        args_hint="<site-id>",
    )
    ctx.register_command(
        "toolsite-prs",
        handler=_handle_prs,
        description="Read open pull requests for toolsite-agent-factory.",
    )
    ctx.register_command(
        "toolsite-branch",
        handler=_handle_branch,
        description="Read current branch for toolsite-agent-factory.",
    )
    ctx.register_command(
        "toolsite-last-commit",
        handler=_handle_last_commit,
        description="Read latest commit for toolsite-agent-factory.",
    )
    ctx.register_command(
        "toolsite-advisor",
        handler=_handle_advisor,
        description="Read-only AI project advisor for toolsite-agent-factory.",
        args_hint="<question>",
    )
    ctx.register_command(
        "toolsite-deny",
        handler=_handle_denied,
        description="Deny non-Phase-1 commands.",
    )

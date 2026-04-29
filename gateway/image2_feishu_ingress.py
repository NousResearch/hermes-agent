"""Feishu -> Hermes-owned Image2 ingress bridge.

The gateway imports this module from the hot message path.  It performs only
classification, durable enqueue into a Hermes-owned runtime, and optional
detached worker launch.  The historical marketing-hub execution sidecar is
fail-closed unless deliberately restored outside this default path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

from gateway.image2_print import parse_print_spec, should_handle_print_request
from gateway.image2_store import Image2JobStore

try:  # PyYAML is available in the Hermes runtime, but keep tests importable.
    import yaml  # type: ignore
except Exception:  # pragma: no cover - exercised only in stripped envs
    yaml = None

logger = logging.getLogger(__name__)

_VISUAL_KEYWORDS = (
    "海报",
    " poster",
    "poster",
    "主视觉",
    "设计稿",
    "生图",
    "出图",
    "图片",
    "配图",
    "宣传图",
    "小红书封面",
    "菜单图",
    "单品图",
    "餐饮图",
    "火宫殿",
    "华卓",
)
_NON_VISUAL_DATA_KEYWORDS = (
    "销售额",
    "客流",
    "账套",
    "考勤",
    "报表",
    "数据",
    "查询",
    "多少",
    "统计",
)
_EXPLICIT_PREFIX_RE = re.compile(r"^\s*/(?:image2|img2|生图|海报)(?:\s|$)", re.I)


@dataclass(frozen=True)
class Image2IngressSettings:
    enabled: bool = False
    db_path: Path = Path("image2_jobs.sqlite")
    runtime_root: Path = Path("runtime/image2")
    python_executable: str = sys.executable
    launch_worker: bool = False
    log_dir: Path = Path("runtime/image2/worker-logs")
    profile_home: Optional[Path] = None
    opencli_timeout_seconds: int = 240
    subprocess_timeout_seconds: int = 360
    split_image_lookback_seconds: int = 120
    # Historical field kept only so old callers/tests fail safely instead of
    # accidentally reviving the deprecated marketing-hub sidecar path.
    marketing_hub_root: Optional[Path] = None
    legacy_disabled_reason: str = ""


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "enabled"}


def _as_path(value: Any, default: Path) -> Path:
    if value is None or str(value).strip() == "":
        return default
    return Path(os.path.expanduser(str(value))).resolve()


def _looks_like_legacy_marketing_hub_path(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip().replace("\\", "/").lower()
    return bool(text) and (
        "/marketing-hub" in text
        or "marketing-hub/" in text
        or "runtime-pack/workspaces/marketing-hub" in text
    )


def _read_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists() or yaml is None:
        return {}
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("[Image2Ingress] Could not read config at %s: %s", config_path, exc)
        return {}


def load_image2_ingress_settings(
    *,
    profile_home: Optional[Path] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Image2IngressSettings:
    env = os.environ if environ is None else environ
    profile = Path(profile_home or env.get("HERMES_PROFILE_HOME") or (Path.home() / ".hermes/profiles/storemanager-lab"))
    cfg = _read_config(profile / "config.yaml")
    section = cfg.get("image2_feishu_ingress") if isinstance(cfg.get("image2_feishu_ingress"), dict) else {}

    profile_runtime = profile / "runtime" / "image2"
    legacy_root_value = env.get("IMAGE2_MARKETING_HUB_ROOT") or section.get("marketing_hub_root")
    runtime_root_value = env.get("IMAGE2_RUNTIME_ROOT") or section.get("runtime_root")
    db_path_value = env.get("IMAGE2_DB_PATH") or section.get("db_path")
    log_dir_value = env.get("IMAGE2_WORKER_LOG_DIR") or section.get("log_dir")
    legacy_requested = bool(str(legacy_root_value or "").strip()) or any(
        _looks_like_legacy_marketing_hub_path(value)
        for value in (legacy_root_value, runtime_root_value, db_path_value, log_dir_value)
    )
    allow_legacy = _truthy(env.get("IMAGE2_ALLOW_LEGACY_MARKETING_HUB_SIDECAR", section.get("allow_legacy_marketing_hub_sidecar", False)))
    legacy_reason = ""
    if legacy_requested and not allow_legacy:
        legacy_reason = "legacy marketing-hub Image2 sidecar config ignored; Hermes-owned runtime required"

    runtime_root = _as_path(None if legacy_reason else runtime_root_value, profile_runtime)
    db_path = _as_path(None if legacy_reason else db_path_value, runtime_root / "image2_jobs.sqlite")
    log_dir = _as_path(None if legacy_reason else log_dir_value, runtime_root / "worker-logs")
    enabled = _truthy(env.get("IMAGE2_FEISHU_INGRESS_ENABLED", section.get("enabled", False)))
    launch_worker = _truthy(env.get("IMAGE2_FEISHU_LAUNCH_WORKER", section.get("launch_worker", False)))
    if legacy_reason:
        enabled = False
        launch_worker = False
    py = str(env.get("IMAGE2_PYTHON") or section.get("python_executable") or sys.executable)
    opencli_timeout = int(env.get("IMAGE2_OPENCLI_TIMEOUT") or section.get("opencli_timeout_seconds") or 240)
    subprocess_timeout = int(env.get("IMAGE2_SUBPROCESS_TIMEOUT") or section.get("subprocess_timeout_seconds") or 360)
    split_image_lookback = int(env.get("IMAGE2_SPLIT_IMAGE_LOOKBACK_SECONDS") or section.get("split_image_lookback_seconds") or 120)
    marketing_hub_root = _as_path(legacy_root_value, Path(".")) if (legacy_requested and allow_legacy) else None
    return Image2IngressSettings(
        enabled=enabled,
        db_path=db_path,
        runtime_root=runtime_root,
        python_executable=py,
        launch_worker=launch_worker,
        log_dir=log_dir,
        profile_home=profile,
        opencli_timeout_seconds=opencli_timeout,
        subprocess_timeout_seconds=subprocess_timeout,
        split_image_lookback_seconds=split_image_lookback,
        marketing_hub_root=marketing_hub_root,
        legacy_disabled_reason=legacy_reason,
    )


def _message_type_value(message_type: Any) -> str:
    return str(getattr(message_type, "value", message_type) or "").lower()




def _text_appears_to_wait_for_followup_image(text: str) -> bool:
    raw = str(text or "")
    if not raw.strip():
        return False
    if any(word in raw for word in ("这张图", "这张图片", "这张", "这个图片", "原图", "参考图", "把这个", "美化一下这张", "重新美化", "这个海报")):
        return True
    if any(word in raw for word in ("发图", "发个图", "图片我马上发", "等我发图")):
        return True
    return False


def should_handle_feishu_visual_request(*, platform: Any, text: str, message_type: Any = "text") -> bool:
    platform_value = str(getattr(platform, "value", platform) or "").lower()
    if platform_value != "feishu":
        return False
    raw = text or ""
    normalized = raw.lower()
    if _EXPLICIT_PREFIX_RE.search(raw):
        return True
    if _message_type_value(message_type) not in {"text", "command", "photo", "document"}:
        return False
    has_visual = any(keyword in normalized or keyword in raw for keyword in _VISUAL_KEYWORDS)
    if not has_visual:
        return False
    # Avoid hijacking obvious 店长/经营数据 questions unless the user explicitly
    # requested Image 2 with /image2 or /海报.
    if any(keyword in raw for keyword in _NON_VISUAL_DATA_KEYWORDS) and not any(k in raw for k in ("海报", "设计稿", "生图", "出图", "图片")):
        return False
    return True


def _event_thread_identity(event: Any) -> tuple[str, Optional[str], Optional[str]]:
    source = getattr(event, "source", None)
    chat_id = str(getattr(source, "chat_id", "") or getattr(event, "chat_id", "") or "")
    thread_id = (
        getattr(source, "thread_id", None)
        or getattr(event, "thread_id", None)
        or getattr(source, "root_id", None)
        or getattr(event, "root_id", None)
    )
    root_id = getattr(source, "root_id", None) or getattr(event, "root_id", None) or thread_id
    return chat_id, str(root_id or "") or None, str(thread_id or "") or None


def _db_has_table(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table_name,)).fetchone()
    return row is not None


_TERMINAL_IMAGE2_STATUSES = ('readback_verified', 'failed_final', 'needs_login')


def _dedupe_identities(*values: Optional[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def find_existing_image2_thread(
    settings: Image2IngressSettings,
    *,
    chat_id: str,
    root_id: Optional[str],
    thread_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    if not chat_id or (not root_id and not thread_id):
        return None
    db_path = Path(settings.db_path)
    if not db_path.is_file():
        return None
    identities = _dedupe_identities(thread_id, root_id)
    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            if _db_has_table(conn, "image2_jobs"):
                for identity in identities:
                    row = conn.execute(
                        """
                        SELECT task_id, feishu_message_id, chat_id, root_id, thread_id, status
                        FROM image2_jobs
                        WHERE chat_id = ?
                          AND status NOT IN ('readback_verified', 'failed_final', 'needs_login')
                          AND (
                            thread_id = ? OR root_id = ? OR feishu_message_id = ? OR task_id = ?
                          )
                        ORDER BY updated_at DESC
                        LIMIT 1
                        """,
                        (chat_id, identity, identity, identity, identity),
                    ).fetchone()
                    if row is not None:
                        data = dict(row)
                        canonical_root = str(data.get("root_id") or data.get("thread_id") or identity or "")
                        canonical_thread = str(data.get("thread_id") or data.get("root_id") or identity or "")
                        data["root_id"] = canonical_root
                        data["thread_id"] = canonical_thread
                        return data
            if _db_has_table(conn, "image2_generation_sessions"):
                for identity in identities:
                    row = conn.execute(
                        """
                        SELECT design_session_id, chat_id, root_id, thread_id, latest_task_id
                        FROM image2_generation_sessions
                        WHERE chat_id = ?
                          AND (
                            thread_id = ? OR root_id = ? OR latest_task_id = ? OR design_session_id = ?
                          )
                        ORDER BY updated_at DESC
                        LIMIT 1
                        """,
                        (chat_id, identity, identity, identity, identity),
                    ).fetchone()
                    if row is not None:
                        data = dict(row)
                        canonical_root = str(data.get("root_id") or data.get("thread_id") or identity or "")
                        canonical_thread = str(data.get("thread_id") or data.get("root_id") or identity or "")
                        data["root_id"] = canonical_root
                        data["thread_id"] = canonical_thread
                        return data
    except Exception as exc:
        logger.warning("[Image2Ingress] Could not inspect Image2 thread registry for continuation routing: %s", exc)
    return None


def has_existing_image2_thread(settings: Image2IngressSettings, *, chat_id: str, root_id: Optional[str], thread_id: Optional[str]) -> bool:
    return find_existing_image2_thread(settings, chat_id=chat_id, root_id=root_id, thread_id=thread_id) is not None


def canonical_image2_thread_identity(event: Any, *, settings: Image2IngressSettings) -> tuple[str, Optional[str], Optional[str]]:
    chat_id, root_id, thread_id = _event_thread_identity(event)
    match = find_existing_image2_thread(settings, chat_id=chat_id, root_id=root_id, thread_id=thread_id)
    if match is None:
        return chat_id, root_id, thread_id
    return chat_id, str(match.get("root_id") or root_id or thread_id or "") or None, str(match.get("thread_id") or thread_id or root_id or "") or None


def should_handle_feishu_image2_continuation_request(event: Any, *, settings: Image2IngressSettings) -> bool:
    source = getattr(event, "source", None)
    platform_value = str(getattr(getattr(source, "platform", None), "value", getattr(source, "platform", "")) or "").lower()
    if platform_value != "feishu":
        return False
    if _message_type_value(getattr(event, "message_type", "text")) not in {"text", "command", "photo", "document"}:
        return False
    text = str(getattr(event, "text", "") or "").strip()
    if not text and not list(getattr(event, "media_urls", None) or []):
        return False
    if any(keyword in text for keyword in _NON_VISUAL_DATA_KEYWORDS):
        return False
    chat_id, root_id, thread_id = _event_thread_identity(event)
    return has_existing_image2_thread(settings, chat_id=chat_id, root_id=root_id, thread_id=thread_id)


def _object_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _safe_path_component(value: Any, fallback: str = "message") -> str:
    text = str(value or fallback).strip() or fallback
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:120]


def _mime_for_source(parent: Any) -> str:
    for key in ("mime_type", "mime", "content_type"):
        value = _object_get(parent, key)
        if value:
            return str(value)
    filename = str(_object_get(parent, "filename", "") or _object_get(parent, "name", "") or "")
    return mimetypes.guess_type(filename)[0] or "image/jpeg"


def _source_filename_for_mime(mime_type: str) -> str:
    if "png" in mime_type:
        return "source.png"
    if "webp" in mime_type:
        return "source.webp"
    return "source.jpg"


def _quoted_parent_message(event: Any) -> Optional[Any]:
    for key in ("quoted_message", "reply_message", "parent_message", "quoted_parent", "reply_to", "parent"):
        value = _object_get(event, key)
        if value:
            return value
    raw = _object_get(event, "raw_event") or _object_get(event, "raw")
    if isinstance(raw, Mapping):
        for key in ("quoted_message", "reply_message", "parent_message", "quote", "reply_to"):
            value = raw.get(key)
            if value:
                return value
    return None


def _has_parent_image(parent: Any) -> bool:
    if not parent:
        return False
    if any(_object_get(parent, key) for key in ("image_key", "file_key", "media_url", "url", "path", "local_path", "image_bytes", "content_bytes", "bytes")):
        mime = _mime_for_source(parent).lower()
        filename = str(_object_get(parent, "filename", "") or _object_get(parent, "name", "") or "")
        return mime.startswith("image/") or Path(filename).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".heic", ".heif"} or bool(_object_get(parent, "image_key"))
    return False


def _copy_parent_image_without_client(parent: Any, destination: Path) -> Optional[Dict[str, Any]]:
    for key in ("local_path", "path", "file_path"):
        value = _object_get(parent, key)
        if value and Path(str(value)).expanduser().is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(str(value)).expanduser(), destination)
            return {"path": str(destination), "mime_type": _mime_for_source(parent)}
    for key in ("image_bytes", "content_bytes", "bytes"):
        value = _object_get(parent, key)
        if isinstance(value, bytes):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(value)
            return {"path": str(destination), "mime_type": _mime_for_source(parent)}
    return None


def _message_content_dict(message: Any) -> Dict[str, Any]:
    body = _object_get(message, "body") or {}
    content = _object_get(body, "content")
    if isinstance(content, Mapping):
        return dict(content)
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            return dict(parsed) if isinstance(parsed, Mapping) else {}
        except Exception:
            return {}
    return {}


def _message_image_key(message: Any) -> str:
    content = _message_content_dict(message)
    return str(content.get("image_key") or _object_get(message, "image_key") or "")


def _message_time_ms(message: Any) -> Optional[int]:
    value = _object_get(message, "create_time") or _object_get(message, "created_at") or _object_get(message, "timestamp")
    try:
        return int(str(value))
    except Exception:
        return None


def _sender_key(message: Any) -> str:
    sender = _object_get(message, "sender") or {}
    sender_id = _object_get(sender, "sender_id") or {}
    for key in ("open_id", "union_id", "user_id"):
        value = _object_get(sender_id, key)
        if value:
            return f"{key}:{value}"
    value = _object_get(sender, "sender_type")
    return f"sender_type:{value}" if value else ""


def _sender_type(message: Any) -> str:
    sender = _object_get(message, "sender") or {}
    return str(_object_get(sender, "sender_type") or "")


def _message_text(message: Any) -> str:
    content = _message_content_dict(message)
    for key in ("text", "content"):
        value = content.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    body = _object_get(message, "body") or {}
    value = _object_get(body, "text") or _object_get(message, "text")
    return str(value or "").strip()


def select_recent_previous_visual_text_message(
    messages: Sequence[Mapping[str, Any]],
    *,
    current_message_id: str,
    lookback_seconds: int = 120,
) -> Optional[Mapping[str, Any]]:
    current_index = None
    for idx, message in enumerate(messages):
        if str(_object_get(message, "message_id") or "") == str(current_message_id or ""):
            current_index = idx
            break
    if current_index is None:
        return None
    current = messages[current_index]
    current_time = _message_time_ms(current)
    current_sender = _sender_key(current)
    for message in messages[current_index + 1 :]:
        candidate_time = _message_time_ms(message)
        if current_time is not None and candidate_time is not None:
            delta_ms = current_time - candidate_time
            if delta_ms < 0:
                continue
            if delta_ms > lookback_seconds * 1000:
                break
        if str(_object_get(message, "msg_type") or "").lower() not in {"text", "post"}:
            continue
        text = _message_text(message)
        if not should_handle_feishu_visual_request(platform="feishu", text=text, message_type="text"):
            continue
        if not _text_appears_to_wait_for_followup_image(text):
            continue
        candidate_sender = _sender_key(message)
        if current_sender and candidate_sender and current_sender != candidate_sender:
            continue
        if _sender_type(message) and _sender_type(message) != "user":
            continue
        return message
    return None


def resolve_recent_previous_feishu_text(event: Any, *, settings: Image2IngressSettings) -> str:
    source = getattr(event, "source", None)
    chat_id = str(getattr(source, "chat_id", "") or "")
    message_id = str(getattr(event, "message_id", "") or "")
    if not chat_id or not message_id:
        return ""
    try:
        env = _feishu_env(settings)
        token = _feishu_tenant_token(env)
        if not token:
            return ""
        messages = _list_recent_feishu_messages(chat_id=chat_id, env=env, token=token)
        previous = select_recent_previous_visual_text_message(
            messages,
            current_message_id=message_id,
            lookback_seconds=settings.split_image_lookback_seconds,
        )
        return _message_text(previous) if previous else ""
    except Exception as exc:
        logger.warning("[Image2Ingress] Could not resolve recent previous Feishu text for split text/image request: %s", exc)
        return ""


def select_recent_previous_image_message(
    messages: Sequence[Mapping[str, Any]],
    *,
    current_message_id: str,
    lookback_seconds: int = 120,
) -> Optional[Mapping[str, Any]]:
    """Pick the nearest previous user image for split Feishu image+text sends.

    Feishu clients often send an uploaded image and the edit instruction as two
    adjacent top-level messages.  The list API returns newest-first; this helper
    finds the current text message, then scans older messages for a same-sender
    image within a tight lookback window.
    """
    current_index = None
    for idx, message in enumerate(messages):
        if str(_object_get(message, "message_id") or "") == str(current_message_id or ""):
            current_index = idx
            break
    if current_index is None:
        return None

    current = messages[current_index]
    current_time = _message_time_ms(current)
    current_sender = _sender_key(current)
    for message in messages[current_index + 1 :]:
        candidate_time = _message_time_ms(message)
        if current_time is not None and candidate_time is not None:
            delta_ms = current_time - candidate_time
            if delta_ms < 0:
                continue
            if delta_ms > lookback_seconds * 1000:
                break
        if str(_object_get(message, "msg_type") or "").lower() != "image":
            continue
        if not _message_image_key(message):
            continue
        candidate_sender = _sender_key(message)
        if current_sender and candidate_sender and current_sender != candidate_sender:
            continue
        if _sender_type(message) and _sender_type(message) != "user":
            continue
        return message
    return None


_SPLIT_SOURCE_TEXT_MARKERS = (
    "这个图片",
    "这张图片",
    "这张图",
    "图片中",
    "图中",
    "上图",
    "刚才发的图",
    "刚发的图",
    "原图",
    "补总标题",
    "补标题",
    "补卖点",
    "修改意见",
)


def _looks_like_split_source_image_edit(text: str) -> bool:
    raw = text or ""
    if not raw.strip():
        return False
    has_marker = any(marker in raw for marker in _SPLIT_SOURCE_TEXT_MARKERS)
    has_edit_verb = any(verb in raw for verb in ("修改", "增加", "补", "替换", "调整", "加入", "加上"))
    return has_marker and has_edit_verb


def _feishu_env(settings: Image2IngressSettings) -> Dict[str, str]:
    env = os.environ.copy()
    if settings.profile_home:
        env.update(_load_dotenv(settings.profile_home / ".env"))
    env.update({k: v for k, v in _load_dotenv(Path.home() / ".hermes" / ".env").items() if k not in env})
    return env


def _feishu_api_base(env: Mapping[str, str]) -> str:
    base = str(env.get("FEISHU_DOMAIN") or env.get("LARK_DOMAIN") or "").strip().rstrip("/")
    return base if base.startswith("http") else "https://open.feishu.cn"


def _feishu_tenant_token(env: Mapping[str, str]) -> Optional[str]:
    app_id = env.get("FEISHU_APP_ID") or env.get("LARK_APP_ID")
    app_secret = env.get("FEISHU_APP_SECRET") or env.get("LARK_APP_SECRET")
    if not app_id or not app_secret:
        return None
    req = urllib.request.Request(
        _feishu_api_base(env) + "/open-apis/auth/v3/tenant_access_token/internal",
        data=json.dumps({"app_id": app_id, "app_secret": app_secret}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if data.get("code") != 0:
        logger.warning("[Image2Ingress] Feishu token request failed code=%s", data.get("code"))
        return None
    return str(data.get("tenant_access_token") or "") or None


def _list_recent_feishu_messages(*, chat_id: str, env: Mapping[str, str], token: str, page_size: int = 30) -> list[Mapping[str, Any]]:
    params = urllib.parse.urlencode(
        {
            "container_id_type": "chat",
            "container_id": chat_id,
            "page_size": page_size,
            "sort_type": "ByCreateTimeDesc",
        }
    )
    req = urllib.request.Request(
        _feishu_api_base(env) + "/open-apis/im/v1/messages?" + params,
        headers={"Authorization": f"Bearer {token}"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if data.get("code") != 0:
        logger.warning("[Image2Ingress] Feishu recent-message list failed code=%s", data.get("code"))
        return []
    items = (data.get("data") or {}).get("items") or []
    return [item for item in items if isinstance(item, Mapping)]


def _fetch_feishu_message_by_id(*, message_id: str, env: Mapping[str, str], token: str) -> Optional[Mapping[str, Any]]:
    if not message_id:
        return None
    req = urllib.request.Request(
        _feishu_api_base(env) + "/open-apis/im/v1/messages/" + urllib.parse.quote(str(message_id), safe=""),
        headers={"Authorization": f"Bearer {token}"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if data.get("code") != 0:
        logger.warning("[Image2Ingress] Feishu message get failed code=%s message_id=%s", data.get("code"), message_id)
        return None
    payload = data.get("data") or {}
    item = payload.get("item")
    if isinstance(item, Mapping):
        return item
    items = payload.get("items")
    if isinstance(items, list) and items and isinstance(items[0], Mapping):
        return items[0]
    if isinstance(payload, Mapping) and payload.get("message_id"):
        return payload
    return None


def _event_related_feishu_message_ids(event: Any, *, settings: Image2IngressSettings) -> list[str]:
    source = getattr(event, "source", None)
    chat_id, root_id, thread_id = canonical_image2_thread_identity(event, settings=settings)
    del chat_id
    candidates = _dedupe_identities(
        root_id,
        getattr(source, "root_id", None),
        getattr(event, "root_id", None),
        getattr(source, "parent_id", None),
        getattr(event, "parent_id", None),
        thread_id,
        getattr(source, "thread_id", None),
        getattr(event, "thread_id", None),
    )
    return [candidate for candidate in candidates if candidate.startswith("om_")]


def resolve_feishu_thread_image(event: Any, destination: Path, *, settings: Image2IngressSettings) -> Optional[Mapping[str, Any]]:
    """Download a Feishu root/parent image when reply events omit quoted parent payloads.

    Real Feishu replies can carry only IDs: a user may reply to an image message,
    and a later reply may target the text request message while the canonical
    Image2 thread root is still the original image.  This resolver follows the
    canonical root/parent message IDs, fetches message metadata, and downloads
    the first image resource it finds.
    """
    message_ids = _event_related_feishu_message_ids(event, settings=settings)
    if not message_ids:
        return None
    try:
        env = _feishu_env(settings)
        token = _feishu_tenant_token(env)
        if not token:
            return None
        seen: set[str] = set()
        queue = list(message_ids)
        while queue and len(seen) < 8:
            message_id = queue.pop(0)
            if message_id in seen:
                continue
            seen.add(message_id)
            message = _fetch_feishu_message_by_id(message_id=message_id, env=env, token=token)
            if not isinstance(message, Mapping):
                continue
            image_key = _message_image_key(message)
            if image_key:
                destination.parent.mkdir(parents=True, exist_ok=True)
                downloaded = _download_feishu_image_key(
                    image_key=image_key,
                    message_id=str(_object_get(message, "message_id") or message_id),
                    destination=destination,
                    env=env,
                    token=token,
                )
                if not downloaded:
                    return None
                record = dict(downloaded)
                record.setdefault("source", "feishu_thread_root_image")
                record.setdefault("parent_message_id", str(_object_get(message, "message_id") or message_id))
                record.setdefault("image_key", image_key)
                return record
            for key in ("root_id", "parent_id", "thread_id"):
                related = str(_object_get(message, key) or "")
                if related.startswith("om_") and related not in seen:
                    queue.append(related)
    except Exception as exc:
        logger.warning("[Image2Ingress] Could not resolve Feishu thread/root image for Image2 source: %s", exc)
    return None


def _download_feishu_image_key(
    *,
    image_key: str,
    destination: Path,
    env: Mapping[str, str],
    token: str,
    message_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not image_key:
        return None

    def _request(url: str) -> tuple[str, bytes]:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"}, method="GET")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.headers.get("Content-Type") or "image/jpeg", resp.read()

    base = _feishu_api_base(env)
    encoded_key = urllib.parse.quote(image_key, safe="")
    urls = []
    if message_id:
        urls.append(
            base
            + "/open-apis/im/v1/messages/"
            + urllib.parse.quote(str(message_id), safe="")
            + "/resources/"
            + encoded_key
            + "?"
            + urllib.parse.urlencode({"type": "image"})
        )
    urls.append(base + "/open-apis/im/v1/images/" + encoded_key)

    last_error: Optional[Exception] = None
    for url in urls:
        try:
            mime_type, data = _request(url)
            if not data:
                return None
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            return {"path": str(destination), "mime_type": mime_type}
        except urllib.error.HTTPError as exc:
            last_error = exc
            if not message_id:
                raise
            continue
    if last_error is not None:
        raise last_error
    return None


def download_feishu_parent_image(parent: Any, destination: Path, *, settings: Image2IngressSettings) -> Optional[Mapping[str, Any]]:
    image_key = str(_object_get(parent, "image_key") or "")
    if not image_key:
        return None
    env = _feishu_env(settings)
    token = _feishu_tenant_token(env)
    if not token:
        return None
    return _download_feishu_image_key(
        image_key=image_key,
        message_id=str(_object_get(parent, "message_id") or "") or None,
        destination=destination,
        env=env,
        token=token,
    )


def resolve_recent_previous_feishu_image(event: Any, destination: Path, *, settings: Image2IngressSettings) -> Optional[Mapping[str, Any]]:
    if not _looks_like_split_source_image_edit(str(getattr(event, "text", "") or "")):
        return None
    source = getattr(event, "source", None)
    chat_id = str(getattr(source, "chat_id", "") or "")
    message_id = str(getattr(event, "message_id", "") or "")
    if not chat_id or not message_id:
        return None
    try:
        env = _feishu_env(settings)
        token = _feishu_tenant_token(env)
        if not token:
            return None
        messages = _list_recent_feishu_messages(chat_id=chat_id, env=env, token=token)
        previous = select_recent_previous_image_message(
            messages,
            current_message_id=message_id,
            lookback_seconds=settings.split_image_lookback_seconds,
        )
        if not previous:
            return None
        image_key = _message_image_key(previous)
        downloaded = _download_feishu_image_key(
            image_key=image_key,
            message_id=str(_object_get(previous, "message_id") or "") or None,
            destination=destination,
            env=env,
            token=token,
        )
        if not downloaded:
            return None
        record = dict(downloaded)
        record.setdefault("source", "feishu_recent_previous_image")
        record.setdefault("parent_message_id", str(_object_get(previous, "message_id") or ""))
        record.setdefault("image_key", image_key)
        return record
    except Exception as exc:
        logger.warning("[Image2Ingress] Could not resolve recent previous Feishu image for split image/text request: %s", exc)
        return None


def _sha256_local_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mime_for_path(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def _normalise_source_file_record(item: Any, *, default_source: str = "feishu_direct_media") -> Optional[Dict[str, Any]]:
    """Convert direct gateway media records into safe local source manifests.

    Direct Feishu images can arrive as plain local path strings.  Persisting the
    raw string makes later prompt/source detection and candidate SHA gates miss
    the source image.  Signed/remote URLs are deliberately redacted instead of
    being written to the job manifest.
    """
    record: Dict[str, Any]
    if isinstance(item, Mapping):
        record = dict(item)
    elif isinstance(item, (str, os.PathLike)):
        text = str(item).strip()
        if not text:
            return None
        if "://" in text:
            parsed = urllib.parse.urlparse(text)
            name = Path(parsed.path).name
            return {"source": default_source, "remote_url_redacted": True, "name": name or "remote-image"}
        record = {"path": text}
    else:
        return None

    record.setdefault("source", default_source)
    path_value = record.get("path") or record.get("file") or record.get("file_path") or record.get("local_path") or record.get("abs_path")
    if path_value:
        path = Path(str(path_value)).expanduser()
        record["path"] = str(path)
        record.setdefault("mime_type", _mime_for_path(path))
        if path.is_file():
            record.setdefault("sha256", _sha256_local_file(path))
    return record


def _normalise_source_file_records(items: Iterable[Any], *, default_source: str = "feishu_direct_media") -> list[Dict[str, Any]]:
    records: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        record = _normalise_source_file_record(item, default_source=default_source)
        if not record:
            continue
        identity = json.dumps(
            {
                "path": record.get("path"),
                "image_key": record.get("image_key"),
                "file_key": record.get("file_key"),
                "name": record.get("name"),
                "source": record.get("source"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if identity in seen:
            continue
        seen.add(identity)
        records.append(record)
    return records


def collect_image2_source_files(
    event: Any,
    *,
    settings: Image2IngressSettings,
    image_downloader: Optional[Callable[[Any, Path], Optional[Mapping[str, Any]]]] = None,
    recent_image_resolver: Optional[Callable[[Any, Path], Optional[Mapping[str, Any]]]] = None,
    thread_image_resolver: Optional[Callable[[Any, Path], Optional[Mapping[str, Any]]]] = None,
) -> list[Any]:
    """Collect direct, quoted, reply-root, and split-message Feishu image sources for Image2.

    This intentionally persists only local file paths/metadata. It must not log
    tenant tokens, request headers, or signed Feishu download URLs.
    """
    source_files: list[Dict[str, Any]] = _normalise_source_file_records(getattr(event, "media_urls", None) or [])
    parent = _quoted_parent_message(event)
    if _has_parent_image(parent):
        mime_type = _mime_for_source(parent)
        message_id = _safe_path_component(getattr(event, "message_id", None), "message")
        dest = Path(settings.runtime_root) / "_feishu_ingress" / message_id / _source_filename_for_mime(mime_type)
        downloaded: Optional[Mapping[str, Any]] = None
        dest.parent.mkdir(parents=True, exist_ok=True)
        if image_downloader is not None:
            result = image_downloader(parent, dest)
            downloaded = result if isinstance(result, Mapping) else None
        else:
            downloaded = _copy_parent_image_without_client(parent, dest)
            if downloaded is None:
                downloaded = download_feishu_parent_image(parent, dest, settings=settings)
        if downloaded is not None or dest.is_file():
            record: Dict[str, Any] = dict(downloaded or {})
            record["path"] = str(Path(str(record.get("path") or dest)).expanduser())
            record.setdefault("mime_type", mime_type)
            record.setdefault("source", "feishu_quoted_parent")
            record.setdefault("parent_message_id", str(_object_get(parent, "message_id", "") or getattr(getattr(event, "source", None), "thread_id", "") or ""))
            for key in ("image_key", "file_key"):
                value = _object_get(parent, key)
                if value:
                    record[key] = str(value)
            source_files.insert(0, record)
            return source_files

    if source_files:
        return source_files

    message_id = _safe_path_component(getattr(event, "message_id", None), "message")
    dest = Path(settings.runtime_root) / "_feishu_ingress" / message_id / "source.jpg"
    dest.parent.mkdir(parents=True, exist_ok=True)

    root_resolver = thread_image_resolver or (lambda current_event, destination: resolve_feishu_thread_image(current_event, destination, settings=settings))
    thread_image = root_resolver(event, dest)
    if isinstance(thread_image, Mapping) or dest.is_file():
        record = dict(thread_image or {})
        record["path"] = str(Path(str(record.get("path") or dest)).expanduser())
        record.setdefault("mime_type", "image/jpeg")
        record.setdefault("source", "feishu_thread_root_image")
        source_files.insert(0, record)
        return source_files

    resolver = recent_image_resolver or (lambda current_event, destination: resolve_recent_previous_feishu_image(current_event, destination, settings=settings))
    recent = resolver(event, dest)
    if isinstance(recent, Mapping) or dest.is_file():
        record = dict(recent or {})
        record["path"] = str(Path(str(record.get("path") or dest)).expanduser())
        record.setdefault("mime_type", "image/jpeg")
        record.setdefault("source", "feishu_recent_previous_image")
        source_files.insert(0, record)
    return source_files


def build_feishu_message_payload(
    event: Any,
    *,
    settings: Optional[Image2IngressSettings] = None,
    image_downloader: Optional[Callable[[Any, Path], Optional[Mapping[str, Any]]]] = None,
    recent_image_resolver: Optional[Callable[[Any, Path], Optional[Mapping[str, Any]]]] = None,
    thread_image_resolver: Optional[Callable[[Any, Path], Optional[Mapping[str, Any]]]] = None,
    previous_text_resolver: Optional[Callable[[Any], str]] = None,
) -> Dict[str, Any]:
    source = getattr(event, "source", None)
    platform = getattr(getattr(source, "platform", None), "value", getattr(source, "platform", "feishu"))
    if settings is not None:
        chat_id, root_id, thread_id = canonical_image2_thread_identity(event, settings=settings)
    else:
        chat_id, root_id, thread_id = _event_thread_identity(event)
    event_text = str(getattr(event, "text", "") or "")
    if settings is not None and not event_text.strip():
        resolver = previous_text_resolver or (lambda current_event: resolve_recent_previous_feishu_text(current_event, settings=settings))
        event_text = str(resolver(event) or "")
    return {
        "source_platform": str(platform or "feishu"),
        "feishu_message_id": str(getattr(event, "message_id", "") or ""),
        "chat_id": chat_id,
        "root_id": root_id or "",
        "thread_id": thread_id or "",
        "text": event_text,
        "source_files": collect_image2_source_files(
            event,
            settings=settings,
            image_downloader=image_downloader,
            recent_image_resolver=recent_image_resolver,
            thread_image_resolver=thread_image_resolver,
        ) if settings is not None else _normalise_source_file_records(getattr(event, "media_urls", None) or []),
    }


def find_latest_verified_image2_preview(
    settings: Image2IngressSettings,
    *,
    chat_id: str,
    root_id: str | None,
    thread_id: str | None,
) -> Optional[Dict[str, Any]]:
    """Return the latest exactly verified non-print preview in the same Feishu thread."""
    db_path = Path(settings.db_path)
    ids = _dedupe_identities(root_id, thread_id)
    if not db_path.exists() or not str(chat_id or "").strip() or not ids:
        return None
    where = ["chat_id = ?", "status = ?"]
    params: list[Any] = [str(chat_id), "readback_verified"]
    placeholders = ",".join("?" for _ in ids)
    where.append(f"(root_id IN ({placeholders}) OR thread_id IN ({placeholders}))")
    params.extend(ids)
    params.extend(ids)
    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM image2_jobs WHERE " + " AND ".join(where) + " ORDER BY updated_at DESC, created_at DESC LIMIT 20",
                params,
            ).fetchall()
    except sqlite3.Error:
        return None
    for row in rows:
        item = dict(row)
        try:
            payload = json.loads(str(item.get("payload_json") or "{}"))
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, Mapping) and payload.get("print_request"):
            continue
        job_dir = Path(str(item.get("job_dir") or ""))
        result_path = job_dir / "worker_result.json"
        if not result_path.is_file():
            continue
        try:
            worker_result = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        contract = worker_result.get("delivery_contract") or {}
        if contract.get("status") != "ready_to_send":
            continue
        image_path = Path(str(contract.get("image_path") or ""))
        image_sha256 = str(contract.get("image_sha256") or "")
        if not image_path.is_file() or not image_sha256:
            continue
        try:
            if _sha256_local_file(image_path) != image_sha256:
                continue
        except OSError:
            continue
        readback = worker_result.get("delivery_readback") or {}
        if readback.get("verified") is not True or readback.get("readback_msg_type") != "image" or not str(readback.get("message_id") or ""):
            continue
        return {
            "task_id": str(item.get("task_id") or ""),
            "job_dir": str(job_dir),
            "approved_image_path": str(image_path),
            "approved_image_sha256": image_sha256,
            "feishu_image_message_id": str(readback.get("message_id") or ""),
        }
    return None


def enqueue_feishu_job(
    settings: Image2IngressSettings,
    message_payload: Mapping[str, Any],
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> Dict[str, Any]:
    del runner  # Kept for source compatibility; Hermes-owned enqueue is in-process.
    store = Image2JobStore(db_path=Path(settings.db_path), runtime_root=Path(settings.runtime_root))
    return store.enqueue_feishu(message_payload)


def _load_dotenv(path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    if not path.exists():
        return result
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            result[key] = value
    return result


def _worker_env(settings: Image2IngressSettings) -> Dict[str, str]:
    env = os.environ.copy()
    if settings.profile_home:
        env.update(_load_dotenv(settings.profile_home / ".env"))
    # Some installs keep the Feishu credentials one level up.
    env.update({k: v for k, v in _load_dotenv(Path.home() / ".hermes" / ".env").items() if k not in env})
    env["PATH"] = f"{Path.home() / '.local/bin'}:{env.get('PATH', '')}"
    project_root = Path(__file__).resolve().parents[1]
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{project_root}:{existing_pythonpath}" if existing_pythonpath else str(project_root)
    env["IMAGE2_OPENCLI_TIMEOUT"] = str(settings.opencli_timeout_seconds)
    env["IMAGE2_SUBPROCESS_TIMEOUT"] = str(settings.subprocess_timeout_seconds)
    if settings.launch_worker:
        env.setdefault("IMAGE2_WORKER_LIVE_ENABLED", "1")
        env.setdefault("OPENCLI_CHROME_CDP_GUIDANCE", "0")
        env.setdefault("IMAGE2_REVIEWER_PROVIDER", "heuristic-fast-preview")
    return env


def launch_image2_worker(
    settings: Image2IngressSettings,
    *,
    task_id: str,
    popen: Callable[..., Any] = subprocess.Popen,
) -> Dict[str, Any]:
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = settings.log_dir / f"{task_id}.stdout.log"
    stderr_path = settings.log_dir / f"{task_id}.stderr.log"
    project_root = Path(__file__).resolve().parents[1]
    cmd = [
        settings.python_executable,
        "-m",
        "gateway.image2_worker",
        "--db",
        str(settings.db_path),
        "--runtime-root",
        str(settings.runtime_root),
        "--worker-id",
        f"feishu-live-{task_id}",
        "--task-id",
        str(task_id),
    ]
    stdout_fh = stdout_path.open("ab")
    stderr_fh = stderr_path.open("ab")
    try:
        proc = popen(
            cmd,
            cwd=str(project_root),
            env=_worker_env(settings),
            stdout=stdout_fh,
            stderr=stderr_fh,
            start_new_session=True,
        )
    except Exception:
        stdout_fh.close()
        stderr_fh.close()
        raise
    finally:
        # The child process inherited the descriptors during Popen.  Close the
        # parent's copies so repeated launches do not leak file handles.
        stdout_fh.close()
        stderr_fh.close()
    return {
        "pid": getattr(proc, "pid", None),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }


def ack_text_for_job(job: Mapping[str, Any], *, launch_worker: bool = True) -> str:
    task_id = str(job.get("task_id") or "unknown")
    if not launch_worker:
        return (
            f"✅ 收到视觉任务，已进入 Image2 规则验证队列 `{task_id}`。"
            "当前 worker 未开启，不会自动生图或发图。"
        )
    return f"✅ 收到视觉任务，已入队 `{task_id}`。我会走 Image 2 完整海报流程，过审核后直接发图。"


def handle_image2_feishu_ingress_event(
    event: Any,
    *,
    settings: Optional[Image2IngressSettings] = None,
    settings_loader: Callable[[], Image2IngressSettings] = load_image2_ingress_settings,
    enqueue_func: Callable[[Image2IngressSettings, Mapping[str, Any]], Mapping[str, Any]] = enqueue_feishu_job,
    launch_func: Callable[[Image2IngressSettings], Mapping[str, Any]] = None,  # type: ignore[assignment]
) -> Optional[str]:
    """Return an immediate ack for Feishu visual requests, or None to fall through.

    The heavy browser generation remains out-of-band.  Returning a string lets
    the existing gateway adapter send the ack through the normal Feishu reply
    path, while a detached worker handles ChatGPT Image 2 and native image
    delivery.
    """
    loaded = settings if settings is not None else settings_loader()
    if not loaded.enabled:
        return None
    source = getattr(event, "source", None)
    raw_text = str(getattr(event, "text", "") or "")
    chat_id, root_id, thread_id = canonical_image2_thread_identity(event, settings=loaded)
    if should_handle_print_request(raw_text):
        spec = parse_print_spec(raw_text)
        if spec.get("status") == "need_clarification":
            return str(spec.get("message") or "要出最终印刷稿，需要先补尺寸。")
        if spec.get("status") == "ok":
            approved = find_latest_verified_image2_preview(loaded, chat_id=chat_id, root_id=root_id, thread_id=thread_id)
            if not approved:
                return "⚠️ 还没有找到同一飞书话题里已发出并通过回读的预览图。请先定好设计稿，再回复：定稿，出印刷版，尺寸 例如 100×150cm。"
            payload = {
                "source_platform": "feishu",
                "feishu_message_id": str(getattr(event, "message_id", "") or ""),
                "chat_id": chat_id,
                "root_id": root_id or "",
                "thread_id": thread_id or root_id or "",
                "text": raw_text,
                "source_files": [],
                "print_request": {
                    "approved_task_id": approved["task_id"],
                    "approved_image_path": approved["approved_image_path"],
                    "approved_image_sha256": approved.get("approved_image_sha256", ""),
                    "feishu_image_message_id": approved.get("feishu_image_message_id", ""),
                    "spec": spec,
                },
            }
            try:
                job = dict(enqueue_func(loaded, payload))
            except Exception as exc:
                logger.warning("[Image2Ingress] Failed to enqueue Feishu print request: %s", exc.__class__.__name__)
                return "⚠️ 印刷定稿任务入队失败：已停止普通聊天兜底，请稍后重试或联系维护。"
            task_id = str(job.get("task_id") or "")
            if loaded.launch_worker and task_id and not job.get("already_existed"):
                try:
                    launcher = launch_func or (lambda s, *, task_id: launch_image2_worker(s, task_id=task_id))
                    launcher(loaded, task_id=task_id)
                except Exception:
                    logger.exception("[Image2Ingress] Enqueued print %s but failed to launch Image2 worker", task_id)
            return f"✅ 已进入印刷定稿队列 `{task_id}`。我会按 {spec['width_mm']}×{spec['height_mm']}mm / {spec['dpi']}DPI 生成扁平单层 PSD 和 PDF proof，并回传飞书文件。"

    is_visual_request = should_handle_feishu_visual_request(
        platform=getattr(source, "platform", None),
        text=raw_text,
        message_type=getattr(event, "message_type", "text"),
    )

    try:
        payload = build_feishu_message_payload(event, settings=loaded)
    except Exception as exc:
        logger.warning(
            "[Image2Ingress] Failed to collect Feishu visual source files before enqueue: %s",
            exc.__class__.__name__,
        )
        return "⚠️ 视觉任务来源图片读取失败：已停止普通聊天兜底，请重新发送图片或引用后再试。"

    inferred_text = str(payload.get("text") or "")
    has_source = bool(payload.get("source_files"))
    if is_visual_request and not has_source and _text_appears_to_wait_for_followup_image(inferred_text):
        return "收到，这条 Image2 需求我先等图片；请把源图直接发出来，我会把上一条文字和图片合成同一个任务。"

    if not is_visual_request and not should_handle_feishu_image2_continuation_request(event, settings=loaded):
        if not (has_source and should_handle_feishu_visual_request(platform=getattr(source, "platform", None), text=inferred_text, message_type="text")):
            return None

    try:
        job = dict(enqueue_func(loaded, payload))
    except Exception as exc:
        logger.warning("[Image2Ingress] Failed to enqueue Feishu visual request: %s", exc.__class__.__name__)
        return "⚠️ 视觉任务入队失败：已停止普通聊天兜底，请稍后重试或联系维护。"

    task_id = str(job.get("task_id") or "")
    if loaded.launch_worker and task_id and not job.get("already_existed"):
        try:
            launcher = launch_func or (lambda s, *, task_id: launch_image2_worker(s, task_id=task_id))
            launcher(loaded, task_id=task_id)
        except Exception:
            logger.exception("[Image2Ingress] Enqueued %s but failed to launch Image 2 worker", task_id)
            # Keep the ack positive: the durable queue is the important safety
            # net, and a supervisor/manual worker can still pick the job up.
    return ack_text_for_job(job, launch_worker=loaded.launch_worker)

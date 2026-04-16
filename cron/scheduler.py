"""
Cron job scheduler - executes due jobs.

Provides tick() which checks for due jobs and runs them. The gateway
calls this every 60 seconds from a background thread.

Uses a file-based lock (~/.hermes/cron/.tick.lock) so only one tick
runs at a time if multiple processes overlap.
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import subprocess
import sys
from datetime import timedelta

# fcntl is Unix-only; on Windows use msvcrt for file locking
try:
    import fcntl
except ImportError:
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        msvcrt = None
from pathlib import Path
from typing import Any, Optional

# Add parent directory to path for imports BEFORE repo-level imports.
# Without this, standalone invocations (e.g. after `hermes update` reloads
# the module) fail with ModuleNotFoundError for hermes_time et al.
sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_constants import get_hermes_home
from hermes_cli.config import load_config
from hermes_time import now as _hermes_now
from gateway.runtime_canary import (
    DEFAULT_ALERT_THROTTLE_SECONDS,
    DEFAULT_BACKGROUND_STUCK_SECONDS,
    DEFAULT_GATEWAY_STALE_SECONDS,
    DEFAULT_PROVIDER_FAILURE_THRESHOLD,
    DEFAULT_QQ_STALE_SECONDS,
    DEFAULT_SESSION_STUCK_SECONDS,
    run_runtime_canary,
)
from gateway.status import read_runtime_status
from gateway.qq_group_archive import (
    QqGroupArchiveStore,
    format_group_report_for_delivery,
    run_due_qq_group_rollups,
)
from gateway.qq_group_policies import get_group_policy
from gateway.weixin_group_archive import (
    WeixinGroupArchiveStore,
    format_group_report_for_delivery as format_weixin_group_report_for_delivery,
    run_due_weixin_group_rollups,
)
from gateway.weixin_group_policies import get_group_policy as get_weixin_group_policy
from gateway.qq_intel_assignments import list_active_daily_report_workers_for_group
from gateway.qq_intel_assignments import get_intel_worker, reconcile_intel_workers, update_intel_worker

logger = logging.getLogger(__name__)
_QQ_GROUP_REPORT_RETRY_LOOKBACK_DAYS = 7
_WEIXIN_GROUP_REPORT_RETRY_LOOKBACK_DAYS = 7

# Valid delivery platforms — used to validate user-supplied platform names
# in cron delivery targets, preventing env var enumeration via crafted names.
_KNOWN_DELIVERY_PLATFORMS = frozenset({
    "telegram", "discord", "slack", "whatsapp", "signal", "qq_napcat",
    "matrix", "mattermost", "homeassistant", "dingtalk", "feishu",
    "wecom", "sms", "email", "webhook",
})

from cron.jobs import (
    advance_next_run,
    get_due_jobs,
    load_runtime_canary_state,
    mark_job_run,
    save_job_output,
    save_runtime_canary_state,
)

# Sentinel: when a cron agent has nothing new to report, it can start its
# response with this marker to suppress delivery.  Output is still saved
# locally for audit.
SILENT_MARKER = "[SILENT]"

# Resolve Hermes home directory (respects HERMES_HOME override)
_hermes_home = get_hermes_home()


def _get_tick_lock_file() -> Path:
    """Return the tick lock file for the current Hermes home directory.

    ``HERMES_HOME`` can differ per process or per test, so lock paths must be
    resolved at call time rather than frozen at module import time.
    """
    return get_hermes_home() / "cron" / ".tick.lock"


def _resolve_origin(job: dict) -> Optional[dict]:
    """Extract origin info from a job, preserving any extra routing metadata."""
    origin = job.get("origin")
    if not origin:
        return None
    platform = origin.get("platform")
    chat_id = origin.get("chat_id")
    if platform and chat_id:
        return origin
    return None


def _normalize_origin_chat_id(origin: dict) -> str:
    """Convert persisted origin metadata into a concrete delivery target."""
    platform = str(origin.get("platform") or "").lower()
    chat_id = str(origin.get("chat_id"))
    if platform != "qq_napcat":
        return chat_id
    if chat_id.startswith(("group:", "dm:")):
        return chat_id

    chat_type = str(origin.get("chat_type") or "").lower()
    if chat_type == "group":
        return f"group:{chat_id}"
    if chat_type in {"dm", "private"}:
        return f"dm:{chat_id}"
    return chat_id


def _qq_napcat_delivery_error(chat_id: str, *, source: str) -> str:
    """Describe a QQ delivery target that is missing its chat type prefix."""
    sample = str(chat_id or "").strip()
    sample_id = sample.lstrip("-") if sample.lstrip("-").isdigit() else "<id>"
    if source == "home channel":
        return (
            "qq_napcat home channel must include a chat type prefix. "
            f"Use 'group:{sample_id}' or 'dm:{sample_id}'."
        )
    return (
        "qq_napcat deliver target must include a chat type prefix. "
        f"Use 'qq_napcat:group:{sample_id}' or 'qq_napcat:dm:{sample_id}'."
    )


def _is_valid_qq_napcat_delivery_chat_id(chat_id: str) -> bool:
    """QQ auto-delivery targets must be explicit because there is no live session state here."""
    return str(chat_id or "").startswith(("group:", "dm:"))


def _resolve_delivery_target(job: dict) -> Optional[dict]:
    """Resolve the concrete auto-delivery target for a cron job, if any."""
    deliver = job.get("deliver", "local")
    origin = _resolve_origin(job)

    if deliver == "local":
        return None

    if deliver == "origin":
        if origin:
            target = {
                "platform": origin["platform"],
                "chat_id": _normalize_origin_chat_id(origin),
                "thread_id": origin.get("thread_id"),
            }
            if target["platform"].lower() == "qq_napcat" and not _is_valid_qq_napcat_delivery_chat_id(target["chat_id"]):
                return None
            return target
        # Origin missing (e.g. job created via API/script) — try each
        # platform's home channel as a fallback instead of silently dropping.
        for platform_name in ("matrix", "telegram", "discord", "slack", "qq_napcat"):
            chat_id = os.getenv(f"{platform_name.upper()}_HOME_CHANNEL", "")
            if chat_id:
                if platform_name == "qq_napcat" and not _is_valid_qq_napcat_delivery_chat_id(chat_id):
                    return None
                logger.info(
                    "Job '%s' has deliver=origin but no origin; falling back to %s home channel",
                    job.get("name", job.get("id", "?")),
                    platform_name,
                )
                return {
                    "platform": platform_name,
                    "chat_id": chat_id,
                    "thread_id": None,
                }
        return None

    if ":" in deliver:
        platform_name, rest = deliver.split(":", 1)
        platform_key = platform_name.lower()

        from tools.send_message_tool import _parse_target_ref

        parsed_chat_id, parsed_thread_id, is_explicit = _parse_target_ref(platform_key, rest)
        if is_explicit:
            chat_id, thread_id = parsed_chat_id, parsed_thread_id
        else:
            chat_id, thread_id = rest, None

        # Resolve human-friendly labels like "Alice (dm)" to real IDs.
        try:
            from gateway.channel_directory import resolve_channel_name
            resolved = resolve_channel_name(platform_key, chat_id)
            if resolved:
                parsed_chat_id, parsed_thread_id, resolved_is_explicit = _parse_target_ref(platform_key, resolved)
                if resolved_is_explicit:
                    chat_id, thread_id = parsed_chat_id, parsed_thread_id
                else:
                    chat_id = resolved
        except Exception:
            pass

        if platform_key == "qq_napcat" and not _is_valid_qq_napcat_delivery_chat_id(chat_id):
            return None

        return {
            "platform": platform_name,
            "chat_id": chat_id,
            "thread_id": thread_id,
        }

    platform_name = deliver
    if origin and origin.get("platform") == platform_name:
        target = {
            "platform": platform_name,
            "chat_id": _normalize_origin_chat_id(origin),
            "thread_id": origin.get("thread_id"),
        }
        if platform_name.lower() == "qq_napcat" and not _is_valid_qq_napcat_delivery_chat_id(target["chat_id"]):
            return None
        return target

    if platform_name.lower() not in _KNOWN_DELIVERY_PLATFORMS:
        return None
    chat_id = os.getenv(f"{platform_name.upper()}_HOME_CHANNEL", "")
    if not chat_id:
        return None
    if platform_name.lower() == "qq_napcat" and not _is_valid_qq_napcat_delivery_chat_id(chat_id):
        return None

    return {
        "platform": platform_name,
        "chat_id": chat_id,
        "thread_id": None,
    }


# Media extension sets — keep in sync with gateway/platforms/base.py:_process_message_background
_AUDIO_EXTS = frozenset({'.ogg', '.opus', '.mp3', '.wav', '.m4a'})
_VIDEO_EXTS = frozenset({'.mp4', '.mov', '.avi', '.mkv', '.webm', '.3gp'})
_IMAGE_EXTS = frozenset({'.jpg', '.jpeg', '.png', '.webp', '.gif'})


def _send_media_via_adapter(adapter, chat_id: str, media_files: list, metadata: dict | None, loop, job: dict) -> None:
    """Send extracted MEDIA files as native platform attachments via a live adapter.

    Routes each file to the appropriate adapter method (send_voice, send_image_file,
    send_video, send_document) based on file extension — mirroring the routing logic
    in ``BasePlatformAdapter._process_message_background``.
    """
    from pathlib import Path

    for media_path, _is_voice in media_files:
        try:
            ext = Path(media_path).suffix.lower()
            if ext in _AUDIO_EXTS:
                coro = adapter.send_voice(chat_id=chat_id, audio_path=media_path, metadata=metadata)
            elif ext in _VIDEO_EXTS:
                coro = adapter.send_video(chat_id=chat_id, video_path=media_path, metadata=metadata)
            elif ext in _IMAGE_EXTS:
                coro = adapter.send_image_file(chat_id=chat_id, image_path=media_path, metadata=metadata)
            else:
                coro = adapter.send_document(chat_id=chat_id, file_path=media_path, metadata=metadata)

            future = asyncio.run_coroutine_threadsafe(coro, loop)
            result = future.result(timeout=30)
            if result and not getattr(result, "success", True):
                logger.warning(
                    "Job '%s': media send failed for %s: %s",
                    job.get("id", "?"), media_path, getattr(result, "error", "unknown"),
                )
        except Exception as e:
            logger.warning("Job '%s': failed to send media %s: %s", job.get("id", "?"), media_path, e)


def _deliver_result(job: dict, content: str, adapters=None, loop=None) -> Optional[str]:
    """
    Deliver job output to the configured target (origin chat, specific platform, etc.).

    When ``adapters`` and ``loop`` are provided (gateway is running), tries to
    use the live adapter first — this supports E2EE rooms (e.g. Matrix) where
    the standalone HTTP path cannot encrypt.  Falls back to standalone send if
    the adapter path fails or is unavailable.

    Returns None on success, or an error string on failure.
    """
    target = _resolve_delivery_target(job)
    if not target:
        if job.get("deliver", "local") != "local":
            deliver = str(job.get("deliver", "local") or "")
            if deliver.startswith("qq_napcat:"):
                msg = _qq_napcat_delivery_error(deliver.split(":", 1)[1], source="deliver")
            elif deliver == "qq_napcat":
                msg = _qq_napcat_delivery_error(
                    os.getenv("QQ_NAPCAT_HOME_CHANNEL", ""),
                    source="home channel",
                )
            else:
                msg = f"no delivery target resolved for deliver={deliver}"
            logger.warning("Job '%s': %s", job["id"], msg)
            return msg
        return None  # local-only jobs don't deliver — not a failure

    platform_name = target["platform"]
    chat_id = target["chat_id"]
    thread_id = target.get("thread_id")

    from tools.send_message_tool import _send_to_platform
    from gateway.config import load_gateway_config, Platform

    platform_map = {
        "telegram": Platform.TELEGRAM,
        "discord": Platform.DISCORD,
        "slack": Platform.SLACK,
        "whatsapp": Platform.WHATSAPP,
        "signal": Platform.SIGNAL,
        "qq_napcat": Platform.QQ_NAPCAT,
        "matrix": Platform.MATRIX,
        "mattermost": Platform.MATTERMOST,
        "homeassistant": Platform.HOMEASSISTANT,
        "dingtalk": Platform.DINGTALK,
        "feishu": Platform.FEISHU,
        "wecom": Platform.WECOM,
        "email": Platform.EMAIL,
        "sms": Platform.SMS,
    }
    platform = platform_map.get(platform_name.lower())
    if not platform:
        msg = f"unknown platform '{platform_name}'"
        logger.warning("Job '%s': %s", job["id"], msg)
        return msg

    try:
        config = load_gateway_config()
    except Exception as e:
        msg = f"failed to load gateway config: {e}"
        logger.error("Job '%s': %s", job["id"], msg)
        return msg

    pconfig = config.platforms.get(platform)
    if not pconfig or not pconfig.enabled:
        msg = f"platform '{platform_name}' not configured/enabled"
        logger.warning("Job '%s': %s", job["id"], msg)
        return msg

    # Optionally wrap the content with a header/footer so the user knows this
    # is a cron delivery.  Wrapping is on by default; set cron.wrap_response: false
    # in config.yaml for clean output.
    wrap_response = True
    try:
        user_cfg = load_config()
        wrap_response = user_cfg.get("cron", {}).get("wrap_response", True)
    except Exception:
        pass

    if wrap_response:
        task_name = job.get("name", job["id"])
        delivery_content = (
            f"Cronjob Response: {task_name}\n"
            f"-------------\n\n"
            f"{content}\n\n"
            f"Note: The agent cannot see this message, and therefore cannot respond to it."
        )
    else:
        delivery_content = content

    # Extract MEDIA: tags so attachments are forwarded as files, not raw text
    from gateway.platforms.base import BasePlatformAdapter
    media_files, cleaned_delivery_content = BasePlatformAdapter.extract_media(delivery_content)

    # Prefer the live adapter when the gateway is running — this supports E2EE
    # rooms (e.g. Matrix) where the standalone HTTP path cannot encrypt.
    runtime_adapter = (adapters or {}).get(platform)
    if runtime_adapter is not None and loop is not None and getattr(loop, "is_running", lambda: False)():
        send_metadata = {"thread_id": thread_id} if thread_id else None
        try:
            # Send cleaned text (MEDIA tags stripped) — not the raw content
            text_to_send = cleaned_delivery_content.strip()
            adapter_ok = True
            if text_to_send:
                future = asyncio.run_coroutine_threadsafe(
                    runtime_adapter.send(chat_id, text_to_send, metadata=send_metadata),
                    loop,
                )
                send_result = future.result(timeout=60)
                if send_result and not getattr(send_result, "success", True):
                    err = getattr(send_result, "error", "unknown")
                    logger.warning(
                        "Job '%s': live adapter send to %s:%s failed (%s), falling back to standalone",
                        job["id"], platform_name, chat_id, err,
                    )
                    adapter_ok = False  # fall through to standalone path

            # Send extracted media files as native attachments via the live adapter
            if adapter_ok and media_files:
                _send_media_via_adapter(runtime_adapter, chat_id, media_files, send_metadata, loop, job)

            if adapter_ok:
                logger.info("Job '%s': delivered to %s:%s via live adapter", job["id"], platform_name, chat_id)
                return None
        except Exception as e:
            logger.warning(
                "Job '%s': live adapter delivery to %s:%s failed (%s), falling back to standalone",
                job["id"], platform_name, chat_id, e,
            )

    # Standalone path: run the async send in a fresh event loop (safe from any thread)
    coro = _send_to_platform(platform, pconfig, chat_id, cleaned_delivery_content, thread_id=thread_id, media_files=media_files)
    try:
        result = asyncio.run(coro)
    except RuntimeError:
        # asyncio.run() checks for a running loop before awaiting the coroutine;
        # when it raises, the original coro was never started — close it to
        # prevent "coroutine was never awaited" RuntimeWarning, then retry in a
        # fresh thread that has no running loop.
        coro.close()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, _send_to_platform(platform, pconfig, chat_id, cleaned_delivery_content, thread_id=thread_id, media_files=media_files))
            result = future.result(timeout=30)
    except Exception as e:
        msg = f"delivery to {platform_name}:{chat_id} failed: {e}"
        logger.error("Job '%s': %s", job["id"], msg)
        return msg

    if result and result.get("error"):
        msg = f"delivery error: {result['error']}"
        logger.error("Job '%s': %s", job["id"], msg)
        return msg

    logger.info("Job '%s': delivered to %s:%s", job["id"], platform_name, chat_id)
    return None


def _deliver_qq_group_daily_reports(reports: list[dict], adapters=None, loop=None) -> list[dict]:
    """Deliver newly rolled up QQ group reports to configured report targets."""
    store = QqGroupArchiveStore()
    outcomes: list[dict] = []
    for report in reports or []:
        group_id = str(report.get("group_id") or "").strip()
        report_date = str(report.get("report_date") or "").strip()
        if not group_id or not report_date:
            continue

        policy = get_group_policy(group_id)
        target = str(policy.get("daily_report_target") or "").strip()
        if not target:
            continue
        delivery_key = f"policy:{target}"
        if store.has_successful_report_delivery(
            group_id=group_id,
            report_date=report_date,
            delivery_key=delivery_key,
        ):
            continue

        content = format_group_report_for_delivery(
            report,
            group_name=policy.get("group_name"),
        )
        job = {
            "id": f"qq-group-daily-report:{group_id}:{report_date}",
            "name": f"qq-group-daily-report:{group_id}",
            "deliver": target,
        }
        error = _deliver_result(job, content, adapters=adapters, loop=loop)
        store.record_report_delivery(
            group_id=group_id,
            report_date=report_date,
            delivery_key=delivery_key,
            target=target,
            error=error,
        )
        outcomes.append(
            {
                "group_id": group_id,
                "report_date": report_date,
                "target": target,
                "error": error,
            }
        )
    return outcomes


def _deliver_qq_intel_worker_reports(reports: list[dict], adapters=None, loop=None) -> list[dict]:
    """Deliver rolled-up QQ reports to active intel workers assigned to that group."""
    store = QqGroupArchiveStore()
    outcomes: list[dict] = []
    for report in reports or []:
        group_id = str(report.get("group_id") or "").strip()
        report_date = str(report.get("report_date") or "").strip()
        if not group_id or not report_date:
            continue
        for worker in list_active_daily_report_workers_for_group(group_id):
            target = str(worker.get("daily_report_target") or "").strip()
            if not target:
                continue
            worker_name = str(worker.get("worker_name") or "").strip() or "unknown"
            delivery_key = f"worker:{worker_name}:{target}"
            if store.has_successful_report_delivery(
                group_id=group_id,
                report_date=report_date,
                delivery_key=delivery_key,
            ):
                continue
            content = _format_intel_delivery_content(worker, report)
            job = {
                "id": f"qq-intel-report:{group_id}:{report_date}:{worker_name}",
                "name": f"qq-intel-report:{worker_name}",
                "deliver": target,
            }
            error = _deliver_result(job, content, adapters=adapters, loop=loop)
            store.record_report_delivery(
                group_id=group_id,
                report_date=report_date,
                delivery_key=delivery_key,
                target=target,
                error=error,
            )
            if error is None:
                update_intel_worker(
                    worker_name,
                    last_report_at=_hermes_now().isoformat(),
                    updated_by="scheduler",
                )
            outcomes.append(
                {
                    "worker_name": worker_name,
                    "group_id": group_id,
                    "report_date": report_date,
                    "target": target,
                    "error": error,
                }
            )
    return outcomes


def _deliver_weixin_group_daily_reports(reports: list[dict], adapters=None, loop=None) -> list[dict]:
    """Deliver newly rolled up Weixin group reports to configured report targets."""
    store = WeixinGroupArchiveStore()
    outcomes: list[dict] = []
    for report in reports or []:
        chat_id = str(report.get("chat_id") or "").strip()
        report_date = str(report.get("report_date") or "").strip()
        if not chat_id or not report_date:
            continue

        policy = get_weixin_group_policy(chat_id)
        target = str(policy.get("daily_report_target") or "").strip()
        if not target:
            continue
        delivery_key = f"policy:{target}"
        delivery_state = store.get_report_delivery(
            chat_id=chat_id,
            report_date=report_date,
            delivery_key=delivery_key,
        )
        if delivery_state and str(delivery_state.get("delivered_at") or "").strip():
            continue

        content = format_weixin_group_report_for_delivery(
            report,
            group_name=policy.get("group_name"),
        )
        job = {
            "id": f"weixin-group-daily-report:{chat_id}:{report_date}",
            "name": f"weixin-group-daily-report:{chat_id}",
            "deliver": target,
        }
        error = _deliver_result(job, content, adapters=adapters, loop=loop)
        store.record_report_delivery(
            chat_id=chat_id,
            report_date=report_date,
            delivery_key=delivery_key,
            target=target,
            error=error,
        )
        outcomes.append(
            {
                "chat_id": chat_id,
                "report_date": report_date,
                "target": target,
                "error": error,
            }
        )
    return outcomes


def _collect_qq_reports_for_delivery_retry(
    reports: list[dict],
    *,
    store: QqGroupArchiveStore | None = None,
    now=None,
) -> list[dict]:
    archive_store = store or QqGroupArchiveStore()
    merged: dict[tuple[str, str], dict] = {}
    for report in reports or []:
        group_id = str(report.get("group_id") or "").strip()
        report_date = str(report.get("report_date") or "").strip()
        if group_id and report_date:
            merged[(group_id, report_date)] = report

    current_time = now or _hermes_now()
    cutoff_date = (current_time.date() - timedelta(days=_QQ_GROUP_REPORT_RETRY_LOOKBACK_DAYS)).isoformat()
    for report in archive_store.list_reports(limit=512):
        group_id = str(report.get("group_id") or "").strip()
        report_date = str(report.get("report_date") or "").strip()
        if not group_id or not report_date or report_date < cutoff_date:
            continue
        merged.setdefault((group_id, report_date), report)
    return list(merged.values())


def _collect_weixin_reports_for_delivery_retry(
    reports: list[dict],
    *,
    store: WeixinGroupArchiveStore | None = None,
    now=None,
) -> list[dict]:
    archive_store = store or WeixinGroupArchiveStore()
    merged: dict[tuple[str, str], dict] = {}
    for report in reports or []:
        chat_id = str(report.get("chat_id") or "").strip()
        report_date = str(report.get("report_date") or "").strip()
        if chat_id and report_date:
            merged[(chat_id, report_date)] = report

    current_time = now or _hermes_now()
    cutoff_date = (current_time.date() - timedelta(days=_WEIXIN_GROUP_REPORT_RETRY_LOOKBACK_DAYS)).isoformat()
    for report in archive_store.list_reports(limit=512):
        chat_id = str(report.get("chat_id") or "").strip()
        report_date = str(report.get("report_date") or "").strip()
        if not chat_id or not report_date or report_date < cutoff_date:
            continue
        merged.setdefault((chat_id, report_date), report)
    return list(merged.values())


def _format_intel_delivery_content(worker: dict, report: dict) -> str:
    title = f"情报员 {worker.get('worker_name', '未知')} 日报"
    base = format_group_report_for_delivery(
        report,
        group_name=worker.get("target_group_name"),
    )
    objective = str(worker.get("objective") or "").strip()
    if objective:
        return f"{title}\n任务：{objective}\n{base}"
    return f"{title}\n{base}"


def _run_async_safely(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result(timeout=30)


def _qq_napcat_runtime_unavailable() -> bool:
    try:
        from gateway.status import read_runtime_status

        status = read_runtime_status() or {}
        platforms = status.get("platforms") or {}
        qq_state = platforms.get("qq_napcat") if isinstance(platforms, dict) else None
        if not isinstance(qq_state, dict):
            return False
        code = str(qq_state.get("error_code") or "").strip().lower()
        return code in {
            "qq_napcat_runtime_missing",
            "qq_napcat_local_service_offline",
        }
    except Exception:
        return False


async def _fetch_qq_joined_groups_async() -> list[dict]:
    from gateway.config import Platform, load_gateway_config
    from tools.send_message_tool import _qq_napcat_call

    config = load_gateway_config()
    pconfig = config.platforms.get(Platform.QQ_NAPCAT)
    if not pconfig or not pconfig.enabled:
        return []
    if _qq_napcat_runtime_unavailable():
        return []
    data, error = await _qq_napcat_call(pconfig.extra, "get_group_list", {})
    if error:
        raise ValueError(str(error.get("error") or "Failed to fetch QQ joined groups"))
    groups = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        group_id = str(item.get("group_id") or item.get("groupCode") or "").strip()
        if not group_id:
            continue
        groups.append(
            {
                "group_id": group_id,
                "group_name": str(item.get("group_name") or item.get("groupName") or group_id).strip(),
            }
        )
    return groups


def _format_intel_status_change_message(worker: dict, change: dict) -> str:
    worker_name = str(worker.get("worker_name") or change.get("worker_name") or "未知").strip()
    target_name = str(worker.get("target_group_name") or change.get("group_name") or "目标群").strip()
    old_status = str(change.get("from_status") or "unknown")
    new_status = str(change.get("to_status") or "unknown")
    lines = [
        f"情报员 {worker_name} 状态更新",
        f"目标群：{target_name}",
        f"状态：{old_status} -> {new_status}",
    ]
    last_error = str(worker.get("last_error") or "").strip()
    if last_error:
        lines.append(f"备注：{last_error}")
    return "\n".join(lines)


def _reconcile_qq_intel_workers(adapters=None, loop=None) -> dict[str, Any]:
    try:
        joined_groups = _run_async_safely(_fetch_qq_joined_groups_async())
    except Exception as exc:
        return {
            "success": False,
            "changed": 0,
            "changes": [],
            "notifications": [],
            "error": str(exc),
        }

    result = reconcile_intel_workers(joined_groups, updated_by="scheduler")
    notifications: list[dict] = []
    for change in result.get("changes") or []:
        worker = get_intel_worker(str(change.get("worker_name") or ""))
        if not worker:
            continue
        target = str(worker.get("notify_target") or "").strip()
        if not target:
            continue
        error = _deliver_result(
            {
                "id": f"qq-intel-status:{worker['worker_name']}",
                "name": f"qq-intel-status:{worker['worker_name']}",
                "deliver": target,
            },
            _format_intel_status_change_message(worker, change),
            adapters=adapters,
            loop=loop,
        )
        notifications.append(
            {
                "worker_name": worker["worker_name"],
                "target": target,
                "error": error,
            }
        )

    return {
        "success": True,
        "changed": int(result.get("changed") or 0),
        "changes": result.get("changes") or [],
        "notifications": notifications,
    }


_SCRIPT_TIMEOUT = 120  # seconds


def _load_runtime_canary_settings() -> dict[str, Any]:
    try:
        config = load_config()
    except Exception:
        config = {}

    scopes = []
    if isinstance(config, dict):
        scopes.append(config.get("runtime_canary") or {})
        scopes.append((config.get("cron") or {}).get("runtime_canary") or {})
        scopes.append((config.get("gateway") or {}).get("runtime_canary") or {})

    merged: dict[str, Any] = {}
    for scope in scopes:
        if isinstance(scope, dict):
            merged.update({k: v for k, v in scope.items() if v is not None})

    env_target = os.getenv("HERMES_RUNTIME_CANARY_ALERT_TARGET") or os.getenv("HERMES_CANARY_ALERT_TARGET")
    if env_target:
        merged["alert_target"] = env_target.strip()

    return {
        "alert_target": str(merged.get("alert_target") or "").strip() or None,
        "throttle_seconds": int(merged.get("throttle_seconds") or DEFAULT_ALERT_THROTTLE_SECONDS),
        "gateway_stale_seconds": int(merged.get("gateway_stale_seconds") or DEFAULT_GATEWAY_STALE_SECONDS),
        "qq_stale_seconds": int(merged.get("qq_stale_seconds") or DEFAULT_QQ_STALE_SECONDS),
        "session_stuck_seconds": int(merged.get("session_stuck_seconds") or DEFAULT_SESSION_STUCK_SECONDS),
        "background_stuck_seconds": int(merged.get("background_stuck_seconds") or DEFAULT_BACKGROUND_STUCK_SECONDS),
        "provider_failure_threshold": int(
            merged.get("provider_failure_threshold") or DEFAULT_PROVIDER_FAILURE_THRESHOLD
        ),
    }


def _load_runtime_canary_target() -> str | None:
    return _load_runtime_canary_settings().get("alert_target")


def run_runtime_canary_tick(adapters=None, loop=None) -> dict[str, Any]:
    """Run the persisted runtime canary and optionally alert an operator target."""
    settings = _load_runtime_canary_settings()
    result = run_runtime_canary(
        runtime_status=read_runtime_status(),
        alert_state=load_runtime_canary_state(),
        alert_target=settings.get("alert_target"),
        now=_hermes_now(),
        throttle_seconds=int(settings["throttle_seconds"]),
        gateway_stale_seconds=int(settings["gateway_stale_seconds"]),
        qq_stale_seconds=int(settings["qq_stale_seconds"]),
        session_stuck_seconds=int(settings["session_stuck_seconds"]),
        background_stuck_seconds=int(settings["background_stuck_seconds"]),
        provider_failure_threshold=int(settings["provider_failure_threshold"]),
    )
    save_runtime_canary_state(result.get("alert_state") or {"last_alerts": {}})

    if result.get("should_alert") and result.get("alert_target") and result.get("alert_text"):
        job = {
            "id": "runtime-canary",
            "name": "runtime-canary",
            "deliver": result["alert_target"],
        }
        delivery_error = _deliver_result(job, result["alert_text"], adapters=adapters, loop=loop)
        if delivery_error:
            logger.warning("Runtime canary delivery failed: %s", delivery_error)
            result["delivery_error"] = delivery_error

    return result


def _run_job_script(script_path: str) -> tuple[bool, str]:
    """Execute a cron job's data-collection script and capture its output.

    Scripts must reside within HERMES_HOME/scripts/.  Both relative and
    absolute paths are resolved and validated against this directory to
    prevent arbitrary script execution via path traversal or absolute
    path injection.

    Args:
        script_path: Path to a Python script.  Relative paths are resolved
            against HERMES_HOME/scripts/.  Absolute and ~-prefixed paths
            are also validated to ensure they stay within the scripts dir.

    Returns:
        (success, output) — on failure *output* contains the error message so the
        LLM can report the problem to the user.
    """
    from hermes_constants import get_hermes_home

    scripts_dir = get_hermes_home() / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir_resolved = scripts_dir.resolve()

    raw = Path(script_path).expanduser()
    if raw.is_absolute():
        path = raw.resolve()
    else:
        path = (scripts_dir / raw).resolve()

    # Guard against path traversal, absolute path injection, and symlink
    # escape — scripts MUST reside within HERMES_HOME/scripts/.
    try:
        path.relative_to(scripts_dir_resolved)
    except ValueError:
        return False, (
            f"Blocked: script path resolves outside the scripts directory "
            f"({scripts_dir_resolved}): {script_path!r}"
        )

    if not path.exists():
        return False, f"Script not found: {path}"
    if not path.is_file():
        return False, f"Script path is not a file: {path}"

    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=_SCRIPT_TIMEOUT,
            cwd=str(path.parent),
        )
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()

        if result.returncode != 0:
            parts = [f"Script exited with code {result.returncode}"]
            if stderr:
                parts.append(f"stderr:\n{stderr}")
            if stdout:
                parts.append(f"stdout:\n{stdout}")
            return False, "\n".join(parts)

        # Redact any secrets that may appear in script output before
        # they are injected into the LLM prompt context.
        try:
            from agent.redact import redact_sensitive_text
            stdout = redact_sensitive_text(stdout)
        except Exception:
            pass
        return True, stdout

    except subprocess.TimeoutExpired:
        return False, f"Script timed out after {_SCRIPT_TIMEOUT}s: {path}"
    except Exception as exc:
        return False, f"Script execution failed: {exc}"


def _build_job_prompt(job: dict) -> str:
    """Build the effective prompt for a cron job, optionally loading one or more skills first."""
    prompt = job.get("prompt", "")
    skills = job.get("skills")

    # Run data-collection script if configured, inject output as context.
    script_path = job.get("script")
    if script_path:
        success, script_output = _run_job_script(script_path)
        if success:
            if script_output:
                prompt = (
                    "## Script Output\n"
                    "The following data was collected by a pre-run script. "
                    "Use it as context for your analysis.\n\n"
                    f"```\n{script_output}\n```\n\n"
                    f"{prompt}"
                )
            else:
                prompt = (
                    "[Script ran successfully but produced no output.]\n\n"
                    f"{prompt}"
                )
        else:
            prompt = (
                "## Script Error\n"
                "The data-collection script failed. Report this to the user.\n\n"
                f"```\n{script_output}\n```\n\n"
                f"{prompt}"
            )

    # Always prepend cron execution guidance so the agent knows how
    # delivery works and can suppress delivery when appropriate.
    cron_hint = (
        "[SYSTEM: You are running as a scheduled cron job. "
        "DELIVERY: Your final response will be automatically delivered "
        "to the user — do NOT use send_message or try to deliver "
        "the output yourself. Just produce your report/output as your "
        "final response and the system handles the rest. "
        "SILENT: If there is genuinely nothing new to report, respond "
        "with exactly \"[SILENT]\" (nothing else) to suppress delivery. "
        "Never combine [SILENT] with content — either report your "
        "findings normally, or say [SILENT] and nothing more.]\n\n"
    )
    prompt = cron_hint + prompt
    if skills is None:
        legacy = job.get("skill")
        skills = [legacy] if legacy else []

    skill_names = [str(name).strip() for name in skills if str(name).strip()]
    if not skill_names:
        return prompt

    from tools.skills_tool import skill_view

    parts = []
    skipped: list[str] = []
    for skill_name in skill_names:
        loaded = json.loads(skill_view(skill_name))
        if not loaded.get("success"):
            error = loaded.get("error") or f"Failed to load skill '{skill_name}'"
            logger.warning("Cron job '%s': skill not found, skipping — %s", job.get("name", job.get("id")), error)
            skipped.append(skill_name)
            continue

        content = str(loaded.get("content") or "").strip()
        if parts:
            parts.append("")
        parts.extend(
            [
                f'[SYSTEM: The user has invoked the "{skill_name}" skill, indicating they want you to follow its instructions. The full skill content is loaded below.]',
                "",
                content,
            ]
        )

    if skipped:
        notice = (
            f"[SYSTEM: The following skill(s) were listed for this job but could not be found "
            f"and were skipped: {', '.join(skipped)}. "
            f"Start your response with a brief notice so the user is aware, e.g.: "
            f"'⚠️ Skill(s) not found and skipped: {', '.join(skipped)}']"
        )
        parts.insert(0, notice)

    if prompt:
        parts.extend(["", f"The user has provided the following instruction alongside the skill invocation: {prompt}"])
    return "\n".join(parts)


def run_job(job: dict) -> tuple[bool, str, str, Optional[str]]:
    """
    Execute a single cron job.
    
    Returns:
        Tuple of (success, full_output_doc, final_response, error_message)
    """
    from run_agent import AIAgent
    
    # Initialize SQLite session store so cron job messages are persisted
    # and discoverable via session_search (same pattern as gateway/run.py).
    _session_db = None
    try:
        from hermes_state import SessionDB
        _session_db = SessionDB()
    except Exception as e:
        logger.debug("Job '%s': SQLite session store not available: %s", job.get("id", "?"), e)
    
    job_id = job["id"]
    job_name = job["name"]
    prompt = _build_job_prompt(job)
    origin = _resolve_origin(job)
    _cron_session_id = f"cron_{job_id}_{_hermes_now().strftime('%Y%m%d_%H%M%S')}"

    logger.info("Running job '%s' (ID: %s)", job_name, job_id)
    logger.info("Prompt: %s", prompt[:100])

    try:
        # Inject origin context so the agent's send_message tool knows the chat.
        # Must be INSIDE the try block so the finally cleanup always runs.
        if origin:
            os.environ["HERMES_SESSION_PLATFORM"] = origin["platform"]
            os.environ["HERMES_SESSION_CHAT_ID"] = str(origin["chat_id"])
            if origin.get("chat_name"):
                os.environ["HERMES_SESSION_CHAT_NAME"] = origin["chat_name"]
            if origin.get("chat_type"):
                os.environ["HERMES_SESSION_CHAT_TYPE"] = str(origin["chat_type"])
            if origin.get("thread_id") is not None:
                os.environ["HERMES_SESSION_THREAD_ID"] = str(origin["thread_id"])
        # Re-read .env and config.yaml fresh every run so provider/key
        # changes take effect without a gateway restart.
        from dotenv import load_dotenv
        try:
            load_dotenv(str(_hermes_home / ".env"), override=True, encoding="utf-8")
        except UnicodeDecodeError:
            load_dotenv(str(_hermes_home / ".env"), override=True, encoding="latin-1")

        delivery_target = _resolve_delivery_target(job)
        if delivery_target:
            os.environ["HERMES_CRON_AUTO_DELIVER_PLATFORM"] = delivery_target["platform"]
            os.environ["HERMES_CRON_AUTO_DELIVER_CHAT_ID"] = str(delivery_target["chat_id"])
            if delivery_target.get("thread_id") is not None:
                os.environ["HERMES_CRON_AUTO_DELIVER_THREAD_ID"] = str(delivery_target["thread_id"])

        model = job.get("model") or os.getenv("HERMES_MODEL") or ""

        # Load config.yaml for model, reasoning, prefill, toolsets, provider routing
        _cfg = {}
        try:
            import yaml
            _cfg_path = str(_hermes_home / "config.yaml")
            if os.path.exists(_cfg_path):
                with open(_cfg_path) as _f:
                    _cfg = yaml.safe_load(_f) or {}
                _model_cfg = _cfg.get("model", {})
                if not job.get("model"):
                    if isinstance(_model_cfg, str):
                        model = _model_cfg
                    elif isinstance(_model_cfg, dict):
                        model = _model_cfg.get("default", model)
        except Exception as e:
            logger.warning("Job '%s': failed to load config.yaml, using defaults: %s", job_id, e)

        # Reasoning config from config.yaml
        from hermes_constants import parse_reasoning_effort
        effort = str(_cfg.get("agent", {}).get("reasoning_effort", "")).strip()
        reasoning_config = parse_reasoning_effort(effort)

        # Prefill messages from env or config.yaml
        prefill_messages = None
        prefill_file = os.getenv("HERMES_PREFILL_MESSAGES_FILE", "") or _cfg.get("prefill_messages_file", "")
        if prefill_file:
            import json as _json
            pfpath = Path(prefill_file).expanduser()
            if not pfpath.is_absolute():
                pfpath = _hermes_home / pfpath
            if pfpath.exists():
                try:
                    with open(pfpath, "r", encoding="utf-8") as _pf:
                        prefill_messages = _json.load(_pf)
                    if not isinstance(prefill_messages, list):
                        prefill_messages = None
                except Exception as e:
                    logger.warning("Job '%s': failed to parse prefill messages file '%s': %s", job_id, pfpath, e)
                    prefill_messages = None

        # Max iterations
        max_iterations = _cfg.get("agent", {}).get("max_turns") or _cfg.get("max_turns") or 90

        # Optional provider fallback chain (same config contract as CLI/gateway)
        fallback_model = _cfg.get("fallback_providers") or _cfg.get("fallback_model") or None

        # Provider routing
        pr = _cfg.get("provider_routing", {})
        smart_routing = _cfg.get("smart_model_routing", {}) or {}

        from hermes_cli.runtime_provider import (
            resolve_runtime_provider,
            format_runtime_provider_error,
        )
        try:
            runtime_kwargs = {
                "requested": job.get("provider") or os.getenv("HERMES_INFERENCE_PROVIDER"),
            }
            if job.get("base_url"):
                runtime_kwargs["explicit_base_url"] = job.get("base_url")
            runtime = resolve_runtime_provider(**runtime_kwargs)
        except Exception as exc:
            message = format_runtime_provider_error(exc)
            raise RuntimeError(message) from exc

        from agent.smart_model_routing import resolve_turn_route
        turn_route = resolve_turn_route(
            prompt,
            smart_routing,
            {
                "model": model,
                "api_key": runtime.get("api_key"),
                "base_url": runtime.get("base_url"),
                "provider": runtime.get("provider"),
                "api_mode": runtime.get("api_mode"),
                "command": runtime.get("command"),
                "args": list(runtime.get("args") or []),
            },
        )

        agent = AIAgent(
            model=turn_route["model"],
            api_key=turn_route["runtime"].get("api_key"),
            base_url=turn_route["runtime"].get("base_url"),
            provider=turn_route["runtime"].get("provider"),
            api_mode=turn_route["runtime"].get("api_mode"),
            acp_command=turn_route["runtime"].get("command"),
            acp_args=turn_route["runtime"].get("args"),
            max_iterations=max_iterations,
            reasoning_config=reasoning_config,
            prefill_messages=prefill_messages,
            providers_allowed=pr.get("only"),
            providers_ignored=pr.get("ignore"),
            providers_order=pr.get("order"),
            provider_sort=pr.get("sort"),
            fallback_model=fallback_model,
            disabled_toolsets=["cronjob", "messaging", "clarify"],
            quiet_mode=True,
            skip_memory=True,  # Cron system prompts would corrupt user representations
            platform="cron",
            session_id=_cron_session_id,
            session_db=_session_db,
        )
        
        # Run the agent with an *inactivity*-based timeout: the job can run
        # for hours if it's actively calling tools / receiving stream tokens,
        # but a hung API call or stuck tool with no activity for the configured
        # duration is caught and killed.  Default 600s (10 min inactivity);
        # override via HERMES_CRON_TIMEOUT env var.  0 = unlimited.
        #
        # Uses the agent's built-in activity tracker (updated by
        # _touch_activity() on every tool call, API call, and stream delta).
        _cron_timeout = float(os.getenv("HERMES_CRON_TIMEOUT", 600))
        _cron_inactivity_limit = _cron_timeout if _cron_timeout > 0 else None
        _POLL_INTERVAL = 5.0
        _cron_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        _cron_future = _cron_pool.submit(agent.run_conversation, prompt)
        _inactivity_timeout = False
        try:
            if _cron_inactivity_limit is None:
                # Unlimited — just wait for the result.
                result = _cron_future.result()
            else:
                result = None
                while True:
                    done, _ = concurrent.futures.wait(
                        {_cron_future}, timeout=_POLL_INTERVAL,
                    )
                    if done:
                        result = _cron_future.result()
                        break
                    # Agent still running — check inactivity.
                    _idle_secs = 0.0
                    if hasattr(agent, "get_activity_summary"):
                        try:
                            _act = agent.get_activity_summary()
                            _idle_secs = _act.get("seconds_since_activity", 0.0)
                        except Exception:
                            pass
                    if _idle_secs >= _cron_inactivity_limit:
                        _inactivity_timeout = True
                        break
        except Exception:
            _cron_pool.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            _cron_pool.shutdown(wait=False)

        if _inactivity_timeout:
            # Build diagnostic summary from the agent's activity tracker.
            _activity = {}
            if hasattr(agent, "get_activity_summary"):
                try:
                    _activity = agent.get_activity_summary()
                except Exception:
                    pass
            _last_desc = _activity.get("last_activity_desc", "unknown")
            _secs_ago = _activity.get("seconds_since_activity", 0)
            _cur_tool = _activity.get("current_tool")
            _iter_n = _activity.get("api_call_count", 0)
            _iter_max = _activity.get("max_iterations", 0)

            logger.error(
                "Job '%s' idle for %.0fs (inactivity limit %.0fs) "
                "| last_activity=%s | iteration=%s/%s | tool=%s",
                job_name, _secs_ago, _cron_inactivity_limit,
                _last_desc, _iter_n, _iter_max,
                _cur_tool or "none",
            )
            if hasattr(agent, "interrupt"):
                agent.interrupt("Cron job timed out (inactivity)")
            raise TimeoutError(
                f"Cron job '{job_name}' idle for "
                f"{int(_secs_ago)}s (limit {int(_cron_inactivity_limit)}s) "
                f"— last activity: {_last_desc}"
            )

        final_response = result.get("final_response", "") or ""
        # Use a separate variable for log display; keep final_response clean
        # for delivery logic (empty response = no delivery).
        logged_response = final_response if final_response else "(No response generated)"
        
        output = f"""# Cron Job: {job_name}

**Job ID:** {job_id}
**Run Time:** {_hermes_now().strftime('%Y-%m-%d %H:%M:%S')}
**Schedule:** {job.get('schedule_display', 'N/A')}

## Prompt

{prompt}

## Response

{logged_response}
"""
        
        logger.info("Job '%s' completed successfully", job_name)
        return True, output, final_response, None
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.exception("Job '%s' failed: %s", job_name, error_msg)
        
        output = f"""# Cron Job: {job_name} (FAILED)

**Job ID:** {job_id}
**Run Time:** {_hermes_now().strftime('%Y-%m-%d %H:%M:%S')}
**Schedule:** {job.get('schedule_display', 'N/A')}

## Prompt

{prompt}

## Error

```
{error_msg}
```
"""
        return False, output, "", error_msg

    finally:
        # Clean up injected env vars so they don't leak to other jobs
        for key in (
            "HERMES_SESSION_PLATFORM",
            "HERMES_SESSION_CHAT_ID",
            "HERMES_SESSION_CHAT_NAME",
            "HERMES_SESSION_CHAT_TYPE",
            "HERMES_SESSION_THREAD_ID",
            "HERMES_CRON_AUTO_DELIVER_PLATFORM",
            "HERMES_CRON_AUTO_DELIVER_CHAT_ID",
            "HERMES_CRON_AUTO_DELIVER_THREAD_ID",
        ):
            os.environ.pop(key, None)
        if _session_db:
            try:
                _session_db.end_session(_cron_session_id, "cron_complete")
            except (Exception, KeyboardInterrupt) as e:
                logger.debug("Job '%s': failed to end session: %s", job_id, e)
            try:
                _session_db.close()
            except (Exception, KeyboardInterrupt) as e:
                logger.debug("Job '%s': failed to close SQLite session store: %s", job_id, e)


def tick(verbose: bool = True, adapters=None, loop=None) -> int:
    """
    Check and run all due jobs.
    
    Uses a file lock so only one tick runs at a time, even if the gateway's
    in-process ticker and a standalone daemon or manual tick overlap.
    
    Args:
        verbose: Whether to print status messages
        adapters: Optional dict mapping Platform → live adapter (from gateway)
        loop: Optional asyncio event loop (from gateway) for live adapter sends
    
    Returns:
        Number of jobs executed (0 if another tick is already running)
    """
    lock_file_path = _get_tick_lock_file()
    lock_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Cross-platform file locking: fcntl on Unix, msvcrt on Windows
    lock_fd = None
    try:
        lock_fd = open(lock_file_path, "w")
        if fcntl:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        elif msvcrt:
            msvcrt.locking(lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
    except (OSError, IOError):
        logger.debug("Tick skipped — another instance holds the lock")
        if lock_fd is not None:
            lock_fd.close()
        return 0

    try:
        try:
            canary_result = run_runtime_canary_tick(adapters=adapters, loop=loop)
            if canary_result.get("should_alert"):
                logger.warning("Runtime canary alert sent to %s", canary_result.get("alert_target"))
            elif canary_result.get("throttled"):
                logger.info("Runtime canary alert throttled")

            reconcile_result = _reconcile_qq_intel_workers(adapters=adapters, loop=loop)
            if reconcile_result.get("changed"):
                logger.info(
                    "QQ intel reconcile: changes=%d notifications=%d",
                    int(reconcile_result.get("changed") or 0),
                    len(reconcile_result.get("notifications") or []),
                )
            elif reconcile_result.get("error"):
                logger.warning("QQ intel reconcile failed: %s", reconcile_result.get("error"))

            rollup_result = run_due_qq_group_rollups()
            rollup_failures = rollup_result.get("failures") or []
            if rollup_result.get("rolled_up_count") or rollup_failures:
                logger.info(
                    "QQ group rollups: reports=%d purged=%d failures=%d",
                    int(rollup_result.get("rolled_up_count") or 0),
                    int(rollup_result.get("purged_raw_messages") or 0),
                    len(rollup_failures),
                )
            delivery_reports = _collect_qq_reports_for_delivery_retry(
                list(rollup_result.get("reports") or [])
            )
            for delivery in _deliver_qq_group_daily_reports(
                delivery_reports,
                adapters=adapters,
                loop=loop,
            ):
                if delivery.get("error"):
                    logger.warning(
                        "QQ group daily report delivery failed for %s/%s -> %s: %s",
                        delivery.get("group_id"),
                        delivery.get("report_date"),
                        delivery.get("target"),
                        delivery.get("error"),
                    )
            for delivery in _deliver_qq_intel_worker_reports(
                delivery_reports,
                adapters=adapters,
                loop=loop,
            ):
                if delivery.get("error"):
                    logger.warning(
                        "QQ intel report delivery failed for %s/%s -> %s: %s",
                        delivery.get("worker_name"),
                        delivery.get("report_date"),
                        delivery.get("target"),
                        delivery.get("error"),
                    )

            weixin_rollup_result = run_due_weixin_group_rollups()
            weixin_rollup_failures = weixin_rollup_result.get("failures") or []
            if weixin_rollup_result.get("rolled_up_count") or weixin_rollup_failures:
                logger.info(
                    "Weixin group rollups: reports=%d purged=%d failures=%d",
                    int(weixin_rollup_result.get("rolled_up_count") or 0),
                    int(weixin_rollup_result.get("purged_raw_messages") or 0),
                    len(weixin_rollup_failures),
                )
            weixin_delivery_reports = _collect_weixin_reports_for_delivery_retry(
                list(weixin_rollup_result.get("reports") or [])
            )
            for delivery in _deliver_weixin_group_daily_reports(
                weixin_delivery_reports,
                adapters=adapters,
                loop=loop,
            ):
                if delivery.get("error"):
                    logger.warning(
                        "Weixin group daily report delivery failed for %s/%s -> %s: %s",
                        delivery.get("chat_id"),
                        delivery.get("report_date"),
                        delivery.get("target"),
                        delivery.get("error"),
                    )
        except Exception as exc:
            logger.warning("QQ group rollup maintenance failed: %s", exc)

        due_jobs = get_due_jobs()

        if verbose and not due_jobs:
            logger.info("%s - No jobs due", _hermes_now().strftime('%H:%M:%S'))
            return 0

        if verbose:
            logger.info("%s - %s job(s) due", _hermes_now().strftime('%H:%M:%S'), len(due_jobs))

        executed = 0
        for job in due_jobs:
            try:
                # For recurring jobs (cron/interval), advance next_run_at to the
                # next future occurrence BEFORE execution.  This way, if the
                # process crashes mid-run, the job won't re-fire on restart.
                # One-shot jobs are left alone so they can retry on restart.
                advance_next_run(job["id"])

                success, output, final_response, error = run_job(job)

                output_file = save_job_output(job["id"], output)
                if verbose:
                    logger.info("Output saved to: %s", output_file)

                # Deliver the final response to the origin/target chat.
                # If the agent responded with [SILENT], skip delivery (but
                # output is already saved above).  Failed jobs always deliver.
                deliver_content = final_response if success else f"⚠️ Cron job '{job.get('name', job['id'])}' failed:\n{error}"
                should_deliver = bool(deliver_content)
                if should_deliver and success and SILENT_MARKER in deliver_content.strip().upper():
                    logger.info("Job '%s': agent returned %s — skipping delivery", job["id"], SILENT_MARKER)
                    should_deliver = False

                delivery_error = None
                if should_deliver:
                    try:
                        delivery_error = _deliver_result(job, deliver_content, adapters=adapters, loop=loop)
                    except Exception as de:
                        delivery_error = str(de)
                        logger.error("Delivery failed for job %s: %s", job["id"], de)

                mark_job_run(job["id"], success, error, delivery_error=delivery_error)
                executed += 1

            except Exception as e:
                logger.error("Error processing job %s: %s", job['id'], e)
                mark_job_run(job["id"], False, str(e))

        return executed
    finally:
        if fcntl:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        elif msvcrt:
            try:
                msvcrt.locking(lock_fd.fileno(), msvcrt.LK_UNLCK, 1)
            except (OSError, IOError):
                pass
        lock_fd.close()


if __name__ == "__main__":
    tick(verbose=True)

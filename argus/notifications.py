"""
ARGUS notification delivery — multi-platform via raw HTTP APIs.

Discovers configured platforms from ~/.hermes/.env and sends alerts
to all of them. No gateway dependency — works from any process context.

Supported platforms:
  Telegram, Discord, Slack, Signal, Matrix, WhatsApp,
  Email (SMTP), Webhook, Mattermost, DingTalk, Feishu, Wecom

Each platform sender returns (delivered: bool, error: Optional[str]).
All senders are fire-and-forget with 10s timeout.

Gateway Integration (optional):
  If gateway is running, notifications can be routed through it
  for consistent formatting and delivery routing.
"""

import json
import logging
import os
import smtplib
import sqlite3
import urllib.error
import urllib.request
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("argus.notifications")

# =============================================================================
# Credential loading
# =============================================================================

_dotenv_loaded = False


def _ensure_dotenv():
    """Load .env once per process."""
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True
    try:
        from hermes_cli.env_loader import load_hermes_dotenv

        load_hermes_dotenv()
    except (ImportError, TypeError):
        pass
    except Exception as e:
        logger.debug("dotenv load: %s", e)


def _env(key: str) -> Optional[str]:
    """Get env var, loading .env first if needed."""
    _ensure_dotenv()
    val = os.environ.get(key)
    return val.strip() if val else None


# =============================================================================
# Message formatting
# =============================================================================


def format_notification_message(
    session_id: str,
    session: Dict,
    notification_type: str,
    message: str,
) -> str:
    """Build a structured notification message string."""
    return (
        "Agent Watcher Alert\n\n"
        "Session: %s\n"
        "Type: %s\n"
        "Task: %s\n"
        "Action: %s\n"
        "Reason: %s\n"
        "Time: %s"
        % (
            session_id,
            session.get("session_type", "unknown"),
            session.get("task_description", "Unknown"),
            notification_type.upper(),
            message,
            datetime.now().isoformat(),
        )
    )


def _html_message(plain: str) -> str:
    """Convert plain text alert to basic HTML (newlines → <br>)."""
    return plain.replace("\n", "<br>").replace("&", "&amp;")


# =============================================================================
# HTTP helper
# =============================================================================


def _http_post(
    url: str,
    payload: dict,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 10,
    method: str = "POST",
) -> Tuple[bool, Optional[str]]:
    """POST JSON to a URL. Returns (success, error_or_none)."""
    try:
        data = json.dumps(payload).encode("utf-8")
        hdrs = {"Content-Type": "application/json"}
        if headers:
            hdrs.update(headers)

        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            # Some APIs return JSON with an ok/success field
            try:
                result = json.loads(body)
                if "ok" in result:
                    return bool(result["ok"]), None
                if "success" in result:
                    return bool(result["success"]), None
            except json.JSONDecodeError:
                pass
            return resp.status < 400, None

    except urllib.error.HTTPError as e:
        return False, "HTTP %s: %s" % (e.code, e.reason)
    except urllib.error.URLError as e:
        return False, "Connection error: %s" % e.reason
    except Exception as e:
        return False, str(e)


# =============================================================================
# Platform senders
# Each returns (delivered: bool, error: Optional[str])
# =============================================================================


def send_telegram(message: str) -> Tuple[bool, Optional[str]]:
    """Send via Telegram Bot API."""
    token = _env("TELEGRAM_BOT_TOKEN")
    chat_id = _env("TELEGRAM_HOME_CHANNEL")

    if not token or not chat_id:
        return False, "TELEGRAM_BOT_TOKEN/HOME_CHANNEL not configured"

    url = "https://api.telegram.org/bot%s/sendMessage" % token
    ok, err = _http_post(
        url,
        {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML",
        },
    )
    if ok:
        logger.info("Telegram: sent")
    return ok, err


def send_discord(message: str) -> Tuple[bool, Optional[str]]:
    """Send via Discord Bot API."""
    token = _env("DISCORD_BOT_TOKEN")
    channel_id = _env("DISCORD_HOME_CHANNEL")

    if not token or not channel_id:
        return False, "DISCORD_BOT_TOKEN/HOME_CHANNEL not configured"

    url = "https://discord.com/api/v10/channels/%s/messages" % channel_id
    ok, err = _http_post(
        url,
        {"content": message},
        {
            "Authorization": "Bot %s" % token,
        },
    )
    if ok:
        logger.info("Discord: sent")
    return ok, err


def send_slack(message: str) -> Tuple[bool, Optional[str]]:
    """Send via Slack Web API."""
    token = _env("SLACK_BOT_TOKEN")
    channel = _env("SLACK_HOME_CHANNEL")

    if not token or not channel:
        return False, "SLACK_BOT_TOKEN/HOME_CHANNEL not configured"

    url = "https://slack.com/api/chat.postMessage"
    ok, err = _http_post(
        url,
        {"channel": channel, "text": message},
        {
            "Authorization": "Bearer %s" % token,
        },
    )
    if ok:
        logger.info("Slack: sent")
    return ok, err


def send_signal(message: str) -> Tuple[bool, Optional[str]]:
    """Send via Signal CLI HTTP daemon."""
    account = _env("SIGNAL_ACCOUNT")
    recipient = _env("SIGNAL_HOME_CHANNEL")
    daemon_url = _env("SIGNAL_HTTP_URL") or "http://localhost:8080"

    if not account or not recipient:
        return False, "SIGNAL_ACCOUNT/HOME_CHANNEL not configured"

    url = "%s/api/v2/send" % daemon_url.rstrip("/")
    recipients = [recipient] if not recipient.startswith("group:") else []
    group_id = recipient[6:] if recipient.startswith("group:") else None

    payload: Dict = {"message": message, "number": account}
    if group_id:
        payload["groupId"] = group_id
    else:
        payload["recipients"] = recipients

    ok, err = _http_post(url, payload)
    if ok:
        logger.info("Signal: sent")
    return ok, err


def send_matrix(message: str) -> Tuple[bool, Optional[str]]:
    """Send via Matrix Client-Server API."""
    homeserver = _env("MATRIX_HOMESERVER")
    token = _env("MATRIX_ACCESS_TOKEN")
    room_id = _env("MATRIX_HOME_ROOM")

    if not homeserver or not token or not room_id:
        return False, "MATRIX_HOMESERVER/ACCESS_TOKEN/HOME_ROOM not configured"

    # Use a txn_id for idempotency
    import uuid

    txn_id = uuid.uuid4().hex[:12]
    url = "%s/_matrix/client/v3/rooms/%s/send/m.room.message/%s" % (
        homeserver.rstrip("/"),
        room_id,
        txn_id,
    )
    ok, err = _http_post(
        url,
        {
            "msgtype": "m.text",
            "body": message,
        },
        {
            "Authorization": "Bearer %s" % token,
        },
        method="PUT",
    )
    if ok:
        logger.info("Matrix: sent")
    return ok, err


def send_whatsapp(message: str) -> Tuple[bool, Optional[str]]:
    """Send via WhatsApp bridge HTTP API."""
    bridge_port = _env("WHATSAPP_BRIDGE_PORT") or "3000"
    chat_id = _env("WHATSAPP_HOME_CHAT")

    if not chat_id:
        return False, "WHATSAPP_HOME_CHAT not configured"

    url = "http://127.0.0.1:%s/send" % bridge_port
    ok, err = _http_post(url, {"chatId": chat_id, "message": message})
    if ok:
        logger.info("WhatsApp: sent")
    return ok, err


def send_email(message: str) -> Tuple[bool, Optional[str]]:
    """Send via SMTP."""
    host = _env("EMAIL_SMTP_HOST")
    port = int(_env("EMAIL_SMTP_PORT") or "587")
    user = _env("EMAIL_ADDRESS")
    password = _env("EMAIL_PASSWORD")
    to_addr = _env("EMAIL_ADDRESS")

    if not host or not user or not password:
        return False, "EMAIL_SMTP_HOST/ADDRESS/PASSWORD not configured"

    try:
        msg = MIMEMultipart()
        msg["From"] = user
        msg["To"] = to_addr
        msg["Subject"] = "ARGUS Alert"
        msg.attach(MIMEText(message, "plain"))

        with smtplib.SMTP(host, port, timeout=10) as srv:
            srv.starttls()
            srv.login(user, password)
            srv.send_message(msg)

        logger.info("Email: sent to %s", to_addr)
        return True, None
    except Exception as e:
        return False, "SMTP error: %s" % e


def send_webhook(message: str) -> Tuple[bool, Optional[str]]:
    """Send to a generic webhook URL."""
    url = _env("ARGUS_WEBHOOK_URL")
    secret = _env("ARGUS_WEBHOOK_SECRET") or _env("WEBHOOK_SECRET")

    if not url:
        return False, "ARGUS_WEBHOOK_URL not configured"

    payload = {"text": message, "source": "argus"}
    headers = {}
    if secret:
        headers["X-Webhook-Secret"] = secret

    ok, err = _http_post(url, payload, headers)
    if ok:
        logger.info("Webhook: sent")
    return ok, err


def send_mattermost(message: str) -> Tuple[bool, Optional[str]]:
    """Send via Mattermost API."""
    base_url = _env("MATTERMOST_URL")
    token = _env("MATTERMOST_TOKEN")
    channel_id = _env("MATTERMOST_HOME_CHANNEL")

    if not base_url or not token or not channel_id:
        return False, "MATTERMOST_URL/TOKEN/HOME_CHANNEL not configured"

    url = "%s/api/v4/posts" % base_url.rstrip("/")
    ok, err = _http_post(
        url,
        {"channel_id": channel_id, "message": message},
        {
            "Authorization": "Bearer %s" % token,
        },
    )
    if ok:
        logger.info("Mattermost: sent")
    return ok, err


def send_dingtalk(message: str) -> Tuple[bool, Optional[str]]:
    """Send via DingTalk webhook bot."""
    webhook_url = _env("DINGTALK_WEBHOOK_URL")

    if not webhook_url:
        return False, "DINGTALK_WEBHOOK_URL not configured"

    ok, err = _http_post(webhook_url, {"msgtype": "text", "text": {"content": message}})
    if ok:
        logger.info("DingTalk: sent")
    return ok, err


def send_feishu(message: str) -> Tuple[bool, Optional[str]]:
    """Send via Feishu/Lark webhook bot."""
    webhook_url = _env("FEISHU_WEBHOOK_URL")

    if not webhook_url:
        return False, "FEISHU_WEBHOOK_URL not configured"

    ok, err = _http_post(
        webhook_url, {"msg_type": "text", "content": {"text": message}}
    )
    if ok:
        logger.info("Feishu: sent")
    return ok, err


def send_wecom(message: str) -> Tuple[bool, Optional[str]]:
    """Send via WeCom/WeChat Work webhook bot."""
    webhook_url = _env("WECOM_WEBHOOK_URL")

    if not webhook_url:
        return False, "WECOM_WEBHOOK_URL not configured"

    ok, err = _http_post(webhook_url, {"msgtype": "text", "text": {"content": message}})
    if ok:
        logger.info("WeCom: sent")
    return ok, err


# =============================================================================
# Platform registry
# =============================================================================

# All platform senders: (name, function, credential env var to check)
PLATFORM_SENDERS: List[Tuple[str, Callable, str]] = [
    ("telegram", send_telegram, "TELEGRAM_BOT_TOKEN"),
    ("discord", send_discord, "DISCORD_BOT_TOKEN"),
    ("slack", send_slack, "SLACK_BOT_TOKEN"),
    ("signal", send_signal, "SIGNAL_ACCOUNT"),
    ("matrix", send_matrix, "MATRIX_ACCESS_TOKEN"),
    ("whatsapp", send_whatsapp, "WHATSAPP_HOME_CHAT"),
    ("email", send_email, "EMAIL_SMTP_HOST"),
    ("webhook", send_webhook, "ARGUS_WEBHOOK_URL"),
    ("mattermost", send_mattermost, "MATTERMOST_TOKEN"),
    ("dingtalk", send_dingtalk, "DINGTALK_WEBHOOK_URL"),
    ("feishu", send_feishu, "FEISHU_WEBHOOK_URL"),
    ("wecom", send_wecom, "WECOM_WEBHOOK_URL"),
]


def discover_platforms() -> List[Tuple[str, Callable]]:
    """Return list of (name, sender_fn) for platforms with credentials configured."""
    _ensure_dotenv()
    active = []
    for name, sender_fn, env_key in PLATFORM_SENDERS:
        # Check primary credential + alert target
        if os.environ.get(env_key):
            active.append((name, sender_fn))
    return active


# =============================================================================
# Main notification dispatch
# =============================================================================


def send_via_gateway(message: str) -> Tuple[bool, Optional[str]]:
    """Send notification through Hermes gateway if running.
    
    Returns (delivered, error). Falls back to direct send if gateway unavailable.
    """
    try:
        # Try to import gateway delivery router
        from gateway.delivery import DeliveryRouter
        
        router = DeliveryRouter()
        # Route to home channel or admin alert channel
        result = router.route_alert(message)
        return result.get("delivered", False), result.get("error")
    except ImportError:
        logger.debug("gateway.delivery unavailable — using direct send")
        return False, "Gateway not available"
    except Exception as e:
        logger.warning("Gateway send failed: %s", e)
        return False, str(e)


def send_to_all_platforms(message: str, prefer_gateway: bool = False) -> Dict[str, Tuple[bool, Optional[str]]]:
    """Send to all configured platforms in parallel (best-effort).
    
    Args:
        message: Message to send
        prefer_gateway: If True, try gateway first before direct platform sends
    """
    results: Dict[str, Tuple[bool, Optional[str]]] = {}
    
    # Try gateway first if preferred
    if prefer_gateway:
        delivered, error = send_via_gateway(message)
        if delivered:
            results["gateway"] = (True, None)
            return results
        else:
            results["gateway"] = (False, error)
    
    # Direct platform sends
    for name, sender_fn, _ in discover_platforms():
        try:
            delivered, error = sender_fn(message)
            results[name] = (delivered, error)
            if error:
                logger.warning("%s: delivery failed — %s", name, error)
        except Exception as e:
            results[name] = (False, str(e))
            logger.error("%s: sender exception — %s", name, e, exc_info=True)
    
    return results


def send_notification(
    cursor: sqlite3.Cursor,
    conn: sqlite3.Connection,
    session_id: str,
    notification_type: str,
    message: str,
    prefer_gateway: bool = False,
) -> None:
    """Send notification to all configured platforms + record in DB.

    1. Looks up session metadata for message formatting
    2. Sends to all configured platforms (Telegram, Discord, Slack, etc.)
       or through gateway if prefer_gateway=True and gateway available
    3. Records in notifications table with delivery status per platform
    """
    # Look up session for message formatting
    cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    session = (
        dict(row) if row else {"session_type": "unknown", "task_description": "Unknown"}
    )

    full_message = format_notification_message(
        session_id, session, notification_type, message
    )

    # Fan out to all configured platforms (or gateway)
    results = send_to_all_platforms(full_message, prefer_gateway=prefer_gateway)

    # Determine overall delivery status
    any_delivered = any(ok for ok, _ in results.values())
    errors = {name: err for name, (_, err) in results.items() if err}

    # Build combined error string for DB
    if errors:
        delivery_error = "; ".join("%s: %s" % (n, e) for n, e in errors.items())
    else:
        delivery_error = None

    if not results:
        delivery_error = "No platforms configured"

    # Record in database
    try:
        cursor.execute(
            """
            INSERT INTO notifications (session_id, notification_type, message, delivered, delivery_error)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                session_id,
                notification_type,
                full_message,
                any_delivered,
                delivery_error,
            ),
        )
        conn.commit()

        if any_delivered:
            platforms_sent = [n for n, (ok, _) in results.items() if ok]
            logger.info(
                "Notification sent to %s for session %s",
                ", ".join(platforms_sent),
                session_id,
            )
        else:
            logger.warning(
                "No platform delivered notification for session %s: %s",
                session_id,
                delivery_error,
            )

    except Exception as e:
        logger.error("Error recording notification: %s", e, exc_info=True)

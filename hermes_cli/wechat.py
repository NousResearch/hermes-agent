"""CLI helpers for WeChat binding flow."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from gateway.status import read_runtime_status
from gateway.platforms.wechat import WeChatSessionState, WeChatSessionStore
from hermes_constants import display_hermes_home

ALLOWED_ILINK_HOSTS = {"localhost", "127.0.0.1", "ilinkai.weixin.qq.com"}


def wechat_command(args) -> None:
    action = getattr(args, "wechat_action", None)
    if action == "bind":
        _cmd_bind(args)
        return
    if action == "status":
        _cmd_status(args)
        return
    print("Usage: hermes wechat <bind|status>")


def _cmd_bind(args) -> None:
    base_url = _resolve_base_url(getattr(args, "base_url", ""))
    if not base_url:
        print("Error: missing WeChat iLink URL.")
        print(
            "Set WECHAT_ILINK_URL or configure `wechat.ilink_url` in "
            f"{display_hermes_home()}/config.yaml, or pass --base-url."
        )
        return

    timeout = int(getattr(args, "timeout", 180) or 180)
    if timeout <= 0:
        print("Error: --timeout must be > 0.")
        return

    ok, qr_resp = _request_bind_qrcode(base_url)
    if not ok:
        print("Failed to request WeChat binding QR code.")
        print(f"Reason: {qr_resp}")
        return

    qrcode = str(qr_resp.get("qrcode") or "").strip() if isinstance(qr_resp, dict) else ""
    if not qrcode:
        print("QR response missing `qrcode` field.")
        return
    qr_content = _extract_qr_content(qr_resp if isinstance(qr_resp, dict) else {})

    print()
    print("WeChat QR code received.")
    print("Scan this QR code with WeChat:")
    _print_qr(qr_content or qrcode)
    if qr_content:
        print("QR URL:")
        print(qr_content)
    print()
    print(f"Polling QR status for up to {timeout}s ...")

    ok, bind_status = _poll_bind_status(base_url=base_url, qrcode=qrcode, timeout=timeout, poll_interval_seconds=2)
    if not ok:
        print(str(bind_status))
        print("Run `hermes wechat bind` again to request a new QR code.")
        return

    token = _extract_token(bind_status)
    if not token:
        print("Bind confirmed, but no bearer token was found in status response.")
        return

    callback_base = _extract_base_url(bind_status)
    resolved_base = _resolve_base_url(callback_base or base_url)
    if not resolved_base:
        print("Bind response did not include a valid iLink base URL.")
        return

    ok, validation = _validate_session_config(resolved_base, token)
    if not ok:
        print("Bind confirmed, but session validation failed.")
        print(f"Reason: {validation}")
        return

    session_store = WeChatSessionStore()
    # Fresh bind should not inherit stale runtime fields (for example cursor) from prior sessions.
    session_store.save(
        WeChatSessionState(
            bearer_token=token,
            base_url=resolved_base,
            context_tokens=_extract_context_tokens(bind_status),
        )
    )

    print("WeChat binding completed.")
    print(f"Validated via: {resolved_base}/ilink/bot/getconfig")
    print(f"Session stored at: {session_store.path}")
    print("You can now start gateway: hermes gateway run")


def _cmd_status(args) -> None:
    base_url = _resolve_base_url(getattr(args, "base_url", ""))
    session_store = WeChatSessionStore()
    session_state = session_store.load()
    runtime = read_runtime_status() or {}
    platform_state = ((runtime.get("platforms") or {}).get("wechat") or {})

    token_present = bool(str(os.getenv("WECHAT_ILINK_TOKEN", "")).strip() or session_state.bearer_token)
    session_file_present = session_store.path.exists()

    print("WeChat Status")
    print(f"  Base URL: {'configured' if base_url else 'missing'}")
    print(f"  Auth: {'configured' if token_present else 'missing'}")
    print(f"  Session file: {'present' if session_file_present else 'missing'} ({session_store.path})")

    if session_file_present:
        try:
            mode = oct(session_store.path.stat().st_mode & 0o777)
        except OSError:
            mode = "unknown"
        print(f"  Session permissions: {mode}")

    gateway_state = runtime.get("gateway_state")
    if gateway_state:
        print(f"  Gateway: {gateway_state}")
    else:
        print("  Gateway: no runtime status")

    if platform_state:
        state = platform_state.get("state", "unknown")
        print(f"  Adapter state: {state}")
        error_code = platform_state.get("error_code")
        error_message = platform_state.get("error_message")
        if error_code or error_message:
            print(f"  Adapter error: {error_code or 'unknown'}")
            if error_message:
                print(f"  Detail: {error_message}")
    else:
        print("  Adapter state: not reported")


def _resolve_base_url(explicit: str) -> str:
    value = str(explicit or "").strip()
    if not value:
        value = str(os.getenv("WECHAT_ILINK_URL", "")).strip()
    if not value:
        try:
            from hermes_cli.config import load_config

            cfg = load_config()
            wechat_cfg = cfg.get("wechat", {}) if isinstance(cfg, dict) else {}
            if isinstance(wechat_cfg, dict):
                value = str(wechat_cfg.get("ilink_url", "")).strip()
            if not value:
                platforms = cfg.get("platforms", {}) if isinstance(cfg, dict) else {}
                p_wechat = platforms.get("wechat", {}) if isinstance(platforms, dict) else {}
                if isinstance(p_wechat, dict):
                    extra = p_wechat.get("extra", {})
                    if isinstance(extra, dict):
                        value = str(extra.get("ilink_url", "")).strip()
        except Exception:
            value = value.strip()

    value = value.rstrip("/")
    if not value:
        return ""
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https"):
        return ""
    host = str(parsed.hostname or "").strip().lower()
    if host not in ALLOWED_ILINK_HOSTS:
        return ""
    if host not in {"localhost", "127.0.0.1"} and parsed.scheme != "https":
        return ""
    return value


def _http_get_json(url: str, *, headers: Dict[str, str], timeout: int = 15) -> tuple[bool, Dict[str, Any] | str]:
    req = Request(url, headers=headers, method="GET")
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="ignore").strip()
    except HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="ignore")
        except Exception:
            detail = ""
        return False, f"HTTP {exc.code}: {detail or exc.reason}"
    except URLError as exc:
        return False, str(exc.reason or exc)
    except Exception as exc:
        return False, str(exc)

    if not raw:
        return False, "empty response"
    try:
        payload = json.loads(raw)
    except Exception:
        return False, "invalid JSON response"
    if not isinstance(payload, dict):
        return False, "invalid response payload"
    return True, payload


def _request_bind_qrcode(base_url: str) -> tuple[bool, Dict[str, Any] | str]:
    url = f"{base_url.rstrip('/')}/ilink/bot/get_bot_qrcode?bot_type=3"
    ok, payload = _http_get_json(
        url,
        headers={
            "Accept": "application/json",
            "iLink-App-Id": "openclaw-weixin",
        },
    )
    if not ok:
        return False, payload
    assert isinstance(payload, dict)
    if not _is_ok(payload):
        return False, _errmsg(payload)
    return True, payload


def _poll_bind_status(
    *,
    base_url: str,
    qrcode: str,
    timeout: int,
    poll_interval_seconds: int = 2,
) -> tuple[bool, Dict[str, Any] | str]:
    deadline = time.time() + timeout
    last_status: Optional[str] = None
    encoded_qrcode = quote(qrcode, safe="")
    while time.time() < deadline:
        remaining = max(int(deadline - time.time()), 1)
        request_timeout = max(remaining + 10, poll_interval_seconds + 5)
        url = f"{base_url.rstrip('/')}/ilink/bot/get_qrcode_status?qrcode={encoded_qrcode}"
        ok, payload = _http_get_json(
            url,
            headers={
                "Accept": "application/json",
                "iLink-App-Id": "openclaw-weixin",
            },
            timeout=request_timeout,
        )
        if not ok:
            if _is_timeout_error(str(payload)):
                if last_status != "wait":
                    print("Status: waiting for scan")
                    last_status = "wait"
                time.sleep(max(poll_interval_seconds, 1))
                continue
            return False, payload
        assert isinstance(payload, dict)
        if not _is_ok(payload):
            return False, _errmsg(payload)

        status = str(payload.get("status") or "").strip().lower()
        if status != last_status and status:
            if status in ("wait", "waiting"):
                print("Status: waiting for scan")
            elif status in ("scaned", "scanned"):
                print("Status: scanned, waiting for confirmation")
            elif status == "confirmed":
                print("Status: confirmed")
            elif status == "expired":
                print("Status: expired")
            else:
                print(f"Status: {status}")
            last_status = status

        if status == "confirmed":
            return True, payload
        if status == "expired":
            return False, "QR code expired."

        time.sleep(max(poll_interval_seconds, 1))

    return False, "Timed out waiting for QR confirmation."


def _print_qr(text: str) -> None:
    try:
        import qrcode  # type: ignore[import-not-found]

        qr = qrcode.QRCode(border=1)
        qr.add_data(text)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
    except Exception:
        print(text)


def _deep_find_value(payload: Any, keys: set[str]) -> Optional[str]:
    if isinstance(payload, dict):
        for k, v in payload.items():
            if str(k).lower() in keys and isinstance(v, (str, int, float)):
                text = str(v).strip()
                if text:
                    return text
        for v in payload.values():
            found = _deep_find_value(v, keys)
            if found:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _deep_find_value(item, keys)
            if found:
                return found
    return None


def _extract_token(payload: Dict[str, Any]) -> str:
    for key in ("bot_token", "bearer_token", "access_token", "ilink_token", "token"):
        found = _deep_find_value(payload, {key})
        if found:
            return found
    return ""


def _extract_base_url(payload: Dict[str, Any]) -> str:
    for key in ("base_url", "baseurl", "ilink_url", "ilinkurl", "url"):
        found = _deep_find_value(payload, {key})
        if found:
            return found
    return ""


def _extract_context_tokens(payload: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(payload.get("context_tokens"), dict):
        return payload["context_tokens"]
    if isinstance(payload.get("context"), dict):
        return payload["context"]
    return {}


def _extract_qr_content(payload: Dict[str, Any]) -> str:
    return _deep_find_value(payload, {"qrcode_img_content", "qrcode_url", "qr_url"}) or ""


def _validate_session_config(base_url: str, token: str) -> tuple[bool, str]:
    url = f"{base_url.rstrip('/')}/ilink/bot/getconfig"
    payload = {"base_info": {"channel_version": "2.0.0"}}
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
        "AuthorizationType": "ilink_bot_token",
        "iLink-App-Id": "openclaw-weixin",
        "iLink-App-ClientVersion": "hermes-cli-bind",
        "X-WECHAT-UIN": "aGVybWVz",
        "X-ILink-Token": token,
    }
    req = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(req, timeout=15) as resp:
            data = resp.read().decode("utf-8", errors="ignore").strip()
            parsed = json.loads(data) if data else {}
            if not isinstance(parsed, dict):
                return False, "invalid getconfig response shape"
            errcode = parsed.get("errcode")
            ret = parsed.get("ret")
            if errcode in (None, 0, "0") or ret in (0, "0"):
                return True, "ok"
            errmsg = str(parsed.get("errmsg") or parsed.get("message") or "unknown error")
            return False, errmsg
    except HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="ignore")
        except Exception:
            detail = ""
        return False, f"HTTP {exc.code}: {detail or exc.reason}"
    except URLError as exc:
        return False, str(exc.reason or exc)
    except Exception as exc:
        return False, str(exc)


def _is_ok(payload: Dict[str, Any]) -> bool:
    errcode = payload.get("errcode")
    ret = payload.get("ret")
    return (errcode in (None, 0, "0")) and (ret in (None, 0, "0"))


def _is_timeout_error(message: str) -> bool:
    lowered = str(message or "").strip().lower()
    return "timed out" in lowered or "timeout" in lowered


def _errmsg(payload: Dict[str, Any]) -> str:
    return str(payload.get("errmsg") or payload.get("message") or payload.get("error") or "unknown error")

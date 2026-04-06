from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import mimetypes
import os
import re
import secrets
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlencode, quote

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency check only
    aiohttp = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives import padding as crypto_padding
    CRYPTO_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    CRYPTO_AVAILABLE = False

from .wechat_state import WeChatAccount

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://ilinkai.weixin.qq.com"
DEFAULT_CDN_BASE_URL = "https://novac2c.cdn.weixin.qq.com/c2c"
DEFAULT_BOT_TYPE = "3"
SESSION_EXPIRED_ERRCODE = -14
UPLOAD_MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# AES-128-ECB helpers (mirrors official aes-ecb.ts)
# ---------------------------------------------------------------------------

def _aes_ecb_padded_size(plaintext_size: int) -> int:
    """Compute AES-128-ECB ciphertext size (PKCS7 padding to 16-byte boundary)."""
    return math.ceil((plaintext_size + 1) / 16) * 16


def _encrypt_aes_ecb(plaintext: bytes, key: bytes) -> bytes:
    """Encrypt with AES-128-ECB + PKCS7 padding."""
    if not CRYPTO_AVAILABLE:
        raise RuntimeError("cryptography library not installed; required for AES encryption")
    padder = crypto_padding.PKCS7(128).padder()
    padded = padder.update(plaintext) + padder.finalize()
    cipher = Cipher(algorithms.AES(key), modes.ECB())
    encryptor = cipher.encryptor()
    return encryptor.update(padded) + encryptor.finalize()


def _decrypt_aes_ecb(ciphertext: bytes, key: bytes) -> bytes:
    """Decrypt AES-128-ECB + PKCS7 padding."""
    if not CRYPTO_AVAILABLE:
        raise RuntimeError("cryptography library not installed; required for AES decryption")
    cipher = Cipher(algorithms.AES(key), modes.ECB())
    decryptor = cipher.decryptor()
    padded = decryptor.update(ciphertext) + decryptor.finalize()
    unpadder = crypto_padding.PKCS7(128).unpadder()
    return unpadder.update(padded) + unpadder.finalize()


def _parse_aes_key(aes_key_base64: str) -> bytes:
    """Parse CDNMedia.aes_key (base64) into a raw 16-byte AES key.

    Two encodings are seen in the wild (mirrors pic-decrypt.ts):
      - base64(raw 16 bytes)           -> images
      - base64(hex string of 16 bytes) -> file / voice / video
    """
    decoded = base64.b64decode(aes_key_base64)
    if len(decoded) == 16:
        return decoded
    if len(decoded) == 32:
        try:
            hex_str = decoded.decode("ascii")
            if all(c in "0123456789abcdefABCDEF" for c in hex_str):
                return bytes.fromhex(hex_str)
        except (UnicodeDecodeError, ValueError):
            pass
    raise ValueError(
        f"aes_key must decode to 16 raw bytes or 32-char hex string, got {len(decoded)} bytes"
    )


# ---------------------------------------------------------------------------
# CDN URL helpers (mirrors cdn-url.ts)
# ---------------------------------------------------------------------------

def _build_cdn_upload_url(cdn_base_url: str, upload_param: str, filekey: str) -> str:
    return f"{cdn_base_url}/upload?encrypted_query_param={quote(upload_param)}&filekey={quote(filekey)}"


def _build_cdn_download_url(cdn_base_url: str, encrypted_query_param: str) -> str:
    return f"{cdn_base_url}/download?encrypted_query_param={quote(encrypted_query_param)}"


class WeChatSessionExpiredError(RuntimeError):
    pass


class WeChatRateLimitError(RuntimeError):
    pass


class OfficialWeChatTransport:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 30.0,
                 cdn_base_url: str = DEFAULT_CDN_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.cdn_base_url = cdn_base_url.rstrip("/")
        self.timeout = timeout
        self._active_logins: Dict[str, Dict[str, Any]] = {}
        self._app_id = os.getenv("WECHAT_ILINK_APP_ID", "bot")
        self._client_version = os.getenv("WECHAT_ILINK_CLIENT_VERSION", "131072")
        self._channel_version = os.getenv("WECHAT_CHANNEL_VERSION", "0.1.0")
        self._session: Optional[aiohttp.ClientSession] = None

    def _build_base_info(self) -> Dict[str, Any]:
        return {"channel_version": self._channel_version}

    def _generate_client_id(self) -> str:
        return f"hermes-wechat-{uuid.uuid4().hex}"

    def _build_text_item(self, text: str) -> Dict[str, Any]:
        return {"type": 1, "text_item": {"text": self.markdown_to_plain_text(text)}}

    def markdown_to_plain_text(self, text: str) -> str:
        result = text or ""
        result = re.sub(r"```[^\n]*\n?([\s\S]*?)```", lambda m: m.group(1).strip(), result)
        result = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", result)
        result = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", result)
        result = re.sub(r"^\|[\s:|-]+\|$", "", result, flags=re.MULTILINE)
        result = re.sub(
            r"^\|(.+)\|$",
            lambda m: "  ".join(cell.strip() for cell in m.group(1).split("|")),
            result,
            flags=re.MULTILINE,
        )
        result = re.sub(r"^#{1,6}\s*", "", result, flags=re.MULTILINE)
        result = re.sub(r"\*\*(.*?)\*\*", r"\1", result)
        result = re.sub(r"__(.*?)__", r"\1", result)
        result = re.sub(r"(?<!\*)\*(?!\*)(.*?) (?<!\*)\*(?!\*)", r"\1", result)
        result = re.sub(r"(?<!_)_(?!_)(.*?) (?<!_)_(?!_)", r"\1", result)
        return result.strip()

    def _build_media_ref(self, uploaded: Dict[str, Any]) -> Dict[str, Any]:
        aeskey = uploaded.get("aeskey") or uploaded.get("aes_key") or b""
        if isinstance(aeskey, str):
            aeskey_bytes = aeskey.encode("utf-8")
        else:
            aeskey_bytes = aeskey or b""
        return {
            "encrypt_query_param": uploaded.get("downloadEncryptedQueryParam") or uploaded.get("encrypt_query_param") or "",
            "aes_key": base64.b64encode(aeskey_bytes).decode("ascii") if aeskey_bytes else "",
            "encrypt_type": 1,
        }

    def _build_image_item(self, uploaded: Dict[str, Any]) -> Dict[str, Any]:
        return {"type": 2, "image_item": {"media": self._build_media_ref(uploaded), "mid_size": uploaded.get("fileSizeCiphertext") or uploaded.get("mid_size") or 0}}

    def _build_video_item(self, uploaded: Dict[str, Any]) -> Dict[str, Any]:
        return {"type": 5, "video_item": {"media": self._build_media_ref(uploaded), "video_size": uploaded.get("fileSizeCiphertext") or uploaded.get("video_size") or 0}}

    def _build_file_item(self, uploaded: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        return {
            "type": 4,
            "file_item": {
                "media": self._build_media_ref(uploaded),
                "file_name": os.path.basename(file_path),
                "len": str(uploaded.get("fileSize") or uploaded.get("filesize") or 0),
            },
        }

    def _build_voice_item(self, uploaded: Dict[str, Any]) -> Dict[str, Any]:
        return {"type": 3, "voice_item": {"media": self._build_media_ref(uploaded)}}

    def _build_message_envelope(
        self,
        *,
        to_user_id: str,
        context_token: Optional[str],
        item: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "msg": {
                "from_user_id": "",
                "to_user_id": to_user_id,
                "client_id": self._generate_client_id(),
                "message_type": 2,
                "message_state": 2,
                "item_list": [item],
                "context_token": context_token,
            },
            "base_info": self._build_base_info(),
        }

    async def _send_item_sequence(
        self,
        *,
        account: WeChatAccount,
        to_user_id: str,
        context_token: Optional[str],
        text: str,
        media_item: Dict[str, Any],
    ) -> Dict[str, Any]:
        if text:
            await self._api_post(
                "ilink/bot/sendmessage",
                json_body=self._build_message_envelope(
                    to_user_id=to_user_id,
                    context_token=context_token,
                    item=self._build_text_item(text),
                ),
                token=account.token,
                base_url=account.base_url,
            )
        return await self._api_post(
            "ilink/bot/sendmessage",
            json_body=self._build_message_envelope(
                to_user_id=to_user_id,
                context_token=context_token,
                item=media_item,
            ),
            token=account.token,
            base_url=account.base_url,
        )

    def _random_wechat_uin(self) -> str:
        value = secrets.randbelow(2**32)
        return base64.b64encode(str(value).encode("utf-8")).decode("ascii")

    def _build_common_headers(self) -> Dict[str, str]:
        return {
            "iLink-App-Id": self._app_id,
            "iLink-App-ClientVersion": self._client_version,
        }

    def _build_headers(self, token: Optional[str] = None) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "AuthorizationType": "ilink_bot_token",
            "X-WECHAT-UIN": self._random_wechat_uin(),
            **self._build_common_headers(),
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _raise_if_protocol_error(self, body: Dict[str, Any], *, endpoint: str) -> None:
        errcode = body.get("errcode", body.get("ret"))
        if errcode in (None, 0, "0"):
            return
        errmsg = body.get("errmsg") or body.get("message") or body.get("msg") or "unknown error"
        raise RuntimeError(f"wechat {endpoint} failed: errcode={errcode} errmsg={errmsg}")

    # ------------------------------------------------------------------
    # Shared session lifecycle
    # ------------------------------------------------------------------

    async def _get_session(self) -> "aiohttp.ClientSession":
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Low-level HTTP helpers
    # ------------------------------------------------------------------

    async def _api_get(self, endpoint: str, *, params: Optional[Dict[str, Any]] = None, base_url: Optional[str] = None) -> Dict[str, Any]:
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp not installed")
        root = (base_url or self.base_url).rstrip("/")
        query = f"?{urlencode(params or {})}" if params else ""
        url = f"{root}/{endpoint}{query}"
        session = await self._get_session()
        async with session.get(url, headers=self._build_common_headers()) as response:
            body = await response.json(content_type=None)
            if response.status in (429, 503):
                raise WeChatRateLimitError(f"wechat GET {endpoint} rate limited: HTTP {response.status}")
            if response.status >= 400:
                raise RuntimeError(f"wechat GET {endpoint} failed: HTTP {response.status} {body}")
            return body

    async def _api_post(self, endpoint: str, *, json_body: Dict[str, Any], token: Optional[str] = None, base_url: Optional[str] = None) -> Dict[str, Any]:
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp not installed")
        root = (base_url or self.base_url).rstrip("/")
        url = f"{root}/{endpoint}"
        headers = self._build_headers(token=token)
        session = await self._get_session()
        async with session.post(url, data=json.dumps(json_body), headers=headers) as response:
            body = await response.json(content_type=None)
            if response.status in (429, 503):
                raise WeChatRateLimitError(f"wechat POST {endpoint} rate limited: HTTP {response.status}")
            if response.status >= 400:
                raise RuntimeError(f"wechat POST {endpoint} failed: HTTP {response.status} {body}")
            return body

    async def _raw_http_get(self, *, url: str) -> bytes:
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp not installed")
        session = await self._get_session()
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.read()

    async def _raw_http_post(self, *, url: str, headers: Dict[str, str], data: bytes) -> Dict[str, Any]:
        """POST binary data to url. Returns dict with response headers of interest."""
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp not installed")
        session = await self._get_session()
        async with session.post(url, data=data, headers=headers) as response:
            if 400 <= response.status < 500:
                err_msg = response.headers.get("x-error-message") or (await response.text())
                raise RuntimeError(f"CDN upload client error {response.status}: {err_msg}")
            if response.status != 200:
                err_msg = response.headers.get("x-error-message") or f"status {response.status}"
                raise RuntimeError(f"CDN upload server error: {err_msg}")
            download_param = response.headers.get("x-encrypted-param") or ""
            return {
                "ok": True,
                "download_param": download_param,
                "url": str(response.url),
                "size": len(data),
            }

    # ------------------------------------------------------------------
    # CDN upload pipeline (mirrors cdn-upload.ts + cdn-url.ts + upload.ts)
    # ------------------------------------------------------------------

    async def _cdn_upload(self, *, upload_url: str, plaintext: bytes, aes_key: bytes,
                          label: str = "cdn_upload") -> Dict[str, Any]:
        """Encrypt plaintext with AES-128-ECB and upload to CDN.
        Returns dict with 'download_param' from CDN response header.
        """
        ciphertext = _encrypt_aes_ecb(plaintext, aes_key)
        headers = {"Content-Type": "application/octet-stream"}
        logger.debug(f"{label}: CDN POST url={upload_url} ciphertext_size={len(ciphertext)}")

        last_error: Optional[Exception] = None
        result: Optional[Dict[str, Any]] = None

        for attempt in range(1, UPLOAD_MAX_RETRIES + 1):
            try:
                result = await self._raw_http_post(url=upload_url, headers=headers, data=ciphertext)
                if not result.get("download_param"):
                    raise RuntimeError("CDN upload response missing x-encrypted-param header")
                logger.debug(f"{label}: CDN upload success attempt={attempt}")
                break
            except RuntimeError as err:
                last_error = err
                if "client error" in str(err):
                    raise
                if attempt < UPLOAD_MAX_RETRIES:
                    logger.warning(f"{label}: attempt {attempt} failed, retrying... err={err}")
                else:
                    logger.error(f"{label}: all {UPLOAD_MAX_RETRIES} attempts failed err={err}")

        if result is None or not result.get("download_param"):
            raise last_error or RuntimeError(f"CDN upload failed after {UPLOAD_MAX_RETRIES} attempts")

        return result

    def _build_upload_request(self, *, file_path: str, to_user_id: str, media_type: int) -> Dict[str, Any]:
        """Build getUploadUrl request payload (mirrors upload.ts uploadMediaToCdn)."""
        path = Path(file_path)
        data = path.read_bytes()
        rawsize = len(data)
        rawfilemd5 = hashlib.md5(data).hexdigest()
        filesize = _aes_ecb_padded_size(rawsize)
        filekey = secrets.token_hex(16)  # random 32-char hex like official
        aes_key = secrets.token_bytes(16)  # random 16-byte AES key
        return {
            "filekey": filekey,
            "media_type": media_type,
            "to_user_id": to_user_id,
            "rawsize": rawsize,
            "rawfilemd5": rawfilemd5,
            "filesize": filesize,
            "no_need_thumb": True,
            "aeskey": aes_key.hex(),
            "_aes_key_raw": aes_key,  # internal: raw bytes for encryption
            "_plaintext": data,  # internal: file bytes for CDN upload
        }

    def _extract_download_url(self, media: Dict[str, Any]) -> str:
        full_url = str(media.get("full_url") or "").strip()
        if full_url:
            return full_url
        query = str(media.get("encrypt_query_param") or "").strip()
        if not query:
            return ""
        if query.startswith("http://") or query.startswith("https://"):
            return query
        return _build_cdn_download_url(self.cdn_base_url, query)

    async def _maybe_decrypt_media(self, data: bytes, media: Dict[str, Any]) -> bytes:
        """Decrypt AES-128-ECB media if aes_key is present (mirrors pic-decrypt.ts)."""
        aes_key_b64 = str(media.get("aes_key") or "").strip()
        if not aes_key_b64:
            return data
        try:
            key = _parse_aes_key(aes_key_b64)
            return _decrypt_aes_ecb(data, key)
        except Exception as e:
            logger.warning(f"media AES decryption failed, returning raw bytes: {e}")
            return data

    async def fetch_media_bytes(self, media: Dict[str, Any]) -> bytes:
        url = self._extract_download_url(media)
        if not url:
            raise ValueError("media download url is empty")
        data = await self._raw_http_get(url=url)
        return await self._maybe_decrypt_media(data, media)

    async def start_login(self, account_id: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
        session_key = account_id or str(uuid.uuid4())
        if not force and session_key in self._active_logins:
            current = self._active_logins[session_key]
            return {"session_key": session_key, "qrcode_url": current.get("qrcode_url", ""), "message": current.get("message", "二维码已就绪，请使用微信扫描。")}
        body = await self._api_get("ilink/bot/get_bot_qrcode", params={"bot_type": DEFAULT_BOT_TYPE})
        self._raise_if_protocol_error(body, endpoint="ilink/bot/get_bot_qrcode")
        result = {"session_key": session_key, "qrcode": body.get("qrcode"), "qrcode_url": body.get("qrcode_img_content") or "", "message": "使用微信扫描以下二维码，以完成连接。", "base_url": self.base_url}
        self._active_logins[session_key] = result
        return result

    async def wait_login(self, session_key: str, timeout_ms: int = 480_000) -> Dict[str, Any]:
        active = self._active_logins.get(session_key)
        if not active or not active.get("qrcode"):
            return {"connected": False, "message": f"login session {session_key} not found"}
        current_base_url = str(active.get("base_url") or self.base_url)
        body = await self._api_get("ilink/bot/get_qrcode_status", params={"qrcode": active["qrcode"]}, base_url=current_base_url)
        self._raise_if_protocol_error(body, endpoint="ilink/bot/get_qrcode_status")
        status = str(body.get("status") or "wait")
        if status == "confirmed":
            result = {"connected": True, "account_id": body.get("ilink_bot_id"), "bot_token": body.get("bot_token"), "base_url": body.get("baseurl") or current_base_url, "user_id": body.get("ilink_user_id"), "message": "✅ 与微信连接成功！"}
            self._active_logins.pop(session_key, None)
            return result
        if status == "expired":
            qr = await self._api_get("ilink/bot/get_bot_qrcode", params={"bot_type": DEFAULT_BOT_TYPE})
            self._raise_if_protocol_error(qr, endpoint="ilink/bot/get_bot_qrcode")
            active["qrcode"] = qr.get("qrcode")
            active["qrcode_url"] = qr.get("qrcode_img_content") or active.get("qrcode_url", "")
            active["message"] = "二维码已刷新，请重新扫码。"
        elif status == "scaned_but_redirect":
            redirect_host = str(body.get("redirect_host") or "").strip()
            if redirect_host:
                active["base_url"] = f"https://{redirect_host}"
        return {"connected": False, "message": status}

    async def get_updates(
        self,
        *,
        account: WeChatAccount,
        cursor: Optional[str] = None,
        longpolling_timeout_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "get_updates_buf": cursor or "",
            "base_info": self._build_base_info(),
        }
        if longpolling_timeout_ms and int(longpolling_timeout_ms) > 0:
            body["longpolling_timeout_ms"] = int(longpolling_timeout_ms)
        try:
            result = await self._api_post(
                "ilink/bot/getupdates",
                json_body=body,
                token=account.token,
                base_url=account.base_url,
            )
        except TimeoutError:
            return {"msgs": [], "get_updates_buf": cursor}
        errcode = result.get("errcode", result.get("ret"))
        if errcode == SESSION_EXPIRED_ERRCODE:
            raise WeChatSessionExpiredError(f"session expired for account {account.account_id}")
        self._raise_if_protocol_error(result, endpoint="ilink/bot/getupdates")
        return result

    async def get_config(self, *, account: WeChatAccount, ilink_user_id: str, context_token: Optional[str] = None) -> Dict[str, Any]:
        body = await self._api_post("ilink/bot/getconfig", json_body={"ilink_user_id": ilink_user_id, "context_token": context_token, "base_info": self._build_base_info()}, token=account.token, base_url=account.base_url)
        self._raise_if_protocol_error(body, endpoint="ilink/bot/getconfig")
        return body

    async def send_typing(self, *, account: WeChatAccount, ilink_user_id: str, typing_ticket: str, status: int) -> Dict[str, Any]:
        body = await self._api_post("ilink/bot/sendtyping", json_body={"ilink_user_id": ilink_user_id, "typing_ticket": typing_ticket, "status": status, "base_info": self._build_base_info()}, token=account.token, base_url=account.base_url)
        self._raise_if_protocol_error(body, endpoint="ilink/bot/sendtyping")
        return body

    async def get_upload_url(self, *, account: WeChatAccount, upload_request: Dict[str, Any]) -> Dict[str, Any]:
        """Call getuploadurl with the prepared upload request (mirrors upload.ts)."""
        api_body = {k: v for k, v in upload_request.items() if not k.startswith("_")}
        api_body["base_info"] = self._build_base_info()
        body = await self._api_post("ilink/bot/getuploadurl", json_body=api_body,
                                    token=account.token, base_url=account.base_url)
        self._raise_if_protocol_error(body, endpoint="ilink/bot/getuploadurl")
        return body

    async def send_text(self, *, account: WeChatAccount, to_user_id: str, text: str, context_token: Optional[str] = None) -> Dict[str, Any]:
        body = await self._api_post(
            "ilink/bot/sendmessage",
            json_body=self._build_message_envelope(to_user_id=to_user_id, context_token=context_token, item=self._build_text_item(text)),
            token=account.token,
            base_url=account.base_url,
        )
        self._raise_if_protocol_error(body, endpoint="ilink/bot/sendmessage")
        return {"message_id": str(body.get("message_id") or body.get("msgid") or body.get("ret") or "ok"), "raw": body}

    async def send_media_file(self, *, account: WeChatAccount, to_user_id: str, file_path: str, text: str, context_token: Optional[str] = None) -> Dict[str, Any]:
        mime, _ = mimetypes.guess_type(file_path)
        mime = mime or "application/octet-stream"
        if mime.startswith("image/"):
            uploaded = await self._upload_media(account=account, to_user_id=to_user_id, file_path=file_path, media_type=1)
            return await self._send_image_message(account=account, to_user_id=to_user_id, uploaded=uploaded, text=text, context_token=context_token)
        if mime.startswith("video/"):
            uploaded = await self._upload_media(account=account, to_user_id=to_user_id, file_path=file_path, media_type=2)
            return await self._send_video_message(account=account, to_user_id=to_user_id, uploaded=uploaded, text=text, context_token=context_token)
        if mime.startswith("audio/") or file_path.lower().endswith((".amr", ".silk", ".ogg", ".mp3", ".wav")):
            uploaded = await self._upload_media(account=account, to_user_id=to_user_id, file_path=file_path, media_type=4)
            return await self._send_voice_message(account=account, to_user_id=to_user_id, uploaded=uploaded, text=text, context_token=context_token)
        uploaded = await self._upload_media(account=account, to_user_id=to_user_id, file_path=file_path, media_type=3)
        return await self._send_file_message(account=account, to_user_id=to_user_id, uploaded=uploaded, file_path=file_path, text=text, context_token=context_token)

    async def _upload_media(self, *, account: WeChatAccount, to_user_id: str, file_path: str, media_type: int) -> Dict[str, Any]:
        """Common upload pipeline: read file -> hash -> gen aeskey -> getUploadUrl -> CDN upload.
        Mirrors upload.ts uploadMediaToCdn().
        Returns UploadedFileInfo-like dict.
        """
        request = self._build_upload_request(file_path=file_path, to_user_id=to_user_id, media_type=media_type)
        aes_key_raw = request["_aes_key_raw"]
        plaintext = request["_plaintext"]

        upload_resp = await self.get_upload_url(account=account, upload_request=request)

        # Determine CDN upload URL (prefer upload_full_url, fallback to building from upload_param)
        upload_full_url = str(upload_resp.get("upload_full_url") or "").strip()
        upload_param = str(upload_resp.get("upload_param") or "").strip()
        if upload_full_url:
            cdn_url = upload_full_url
        elif upload_param:
            cdn_url = _build_cdn_upload_url(self.cdn_base_url, upload_param, request["filekey"])
        else:
            raise RuntimeError("getUploadUrl returned no upload URL (need upload_full_url or upload_param)")

        cdn_result = await self._cdn_upload(
            upload_url=cdn_url,
            plaintext=plaintext,
            aes_key=aes_key_raw,
            label=f"upload_media[type={media_type}]",
        )

        return {
            "filekey": request["filekey"],
            "downloadEncryptedQueryParam": cdn_result["download_param"],
            "aeskey": request["aeskey"],  # hex-encoded for base64 conversion in _build_media_ref
            "fileSize": request["rawsize"],
            "fileSizeCiphertext": request["filesize"],
        }

    # Keep individual upload helpers as thin wrappers for backward compat in tests
    async def _upload_image(self, *, account: WeChatAccount, to_user_id: str, file_path: str) -> Dict[str, Any]:
        return await self._upload_media(account=account, to_user_id=to_user_id, file_path=file_path, media_type=1)

    async def _upload_file(self, *, account: WeChatAccount, to_user_id: str, file_path: str) -> Dict[str, Any]:
        return await self._upload_media(account=account, to_user_id=to_user_id, file_path=file_path, media_type=3)

    async def _upload_video(self, *, account: WeChatAccount, to_user_id: str, file_path: str) -> Dict[str, Any]:
        return await self._upload_media(account=account, to_user_id=to_user_id, file_path=file_path, media_type=2)

    async def _upload_voice(self, *, account: WeChatAccount, to_user_id: str, file_path: str) -> Dict[str, Any]:
        return await self._upload_media(account=account, to_user_id=to_user_id, file_path=file_path, media_type=4)

    async def _send_image_message(self, *, account: WeChatAccount, to_user_id: str, uploaded: Dict[str, Any], text: str, context_token: Optional[str] = None) -> Dict[str, Any]:
        return await self._send_item_sequence(account=account, to_user_id=to_user_id, context_token=context_token, text=text, media_item=self._build_image_item(uploaded))

    async def _send_file_message(self, *, account: WeChatAccount, to_user_id: str, uploaded: Dict[str, Any], file_path: str, text: str, context_token: Optional[str] = None) -> Dict[str, Any]:
        return await self._send_item_sequence(account=account, to_user_id=to_user_id, context_token=context_token, text=text, media_item=self._build_file_item(uploaded, file_path))

    async def _send_video_message(self, *, account: WeChatAccount, to_user_id: str, uploaded: Dict[str, Any], text: str, context_token: Optional[str] = None) -> Dict[str, Any]:
        return await self._send_item_sequence(account=account, to_user_id=to_user_id, context_token=context_token, text=text, media_item=self._build_video_item(uploaded))

    async def _send_voice_message(self, *, account: WeChatAccount, to_user_id: str, uploaded: Dict[str, Any], text: str, context_token: Optional[str] = None) -> Dict[str, Any]:
        return await self._send_item_sequence(account=account, to_user_id=to_user_id, context_token=context_token, text=text, media_item=self._build_voice_item(uploaded))

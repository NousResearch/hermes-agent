"""
yuanbao_media.py — 元宝平台媒体处理：COS 上传、URL 下载（SSRF 防护）、TIM 媒体消息体。
移植自 TypeScript 版 media.ts（yuanbao-openclaw-plugin），用 httpx 替代 cos-nodejs-sdk-v5。

COS 上传流程：genUploadInfo 获取临时凭证 → 临时凭证 HMAC-SHA1 签名 Authorization 头 → HTTP PUT。
TIM 消息体：build_image_msg_body() → TIMImageElem，build_file_msg_body() → TIMFileElem
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import struct
import time
import urllib.parse
from typing import Optional, Any

import httpx

logger = logging.getLogger(__name__)

UPLOAD_INFO_PATH = "/api/resource/genUploadInfo"
DEFAULT_API_DOMAIN = "yuanbao.tencent.com"
DEFAULT_MAX_SIZE_MB = 50

# MIME → image_format 数字（TIM 协议字段）
_MIME_TO_IMAGE_FORMAT: dict[str, int] = {
    "image/jpeg": 1, "image/jpg": 1, "image/gif": 2, "image/png": 3, "image/bmp": 4,
    "image/webp": 255, "image/heic": 255, "image/tiff": 255,
}

_EXT_TO_MIME: dict[str, str] = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp",
    ".bmp": "image/bmp", ".heic": "image/heic", ".tiff": "image/tiff", ".ico": "image/x-icon",
    ".pdf": "application/pdf",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".txt": "text/plain", ".zip": "application/zip", ".tar": "application/x-tar", ".gz": "application/gzip",
    ".mp3": "audio/mpeg", ".mp4": "video/mp4", ".wav": "audio/wav", ".ogg": "audio/ogg", ".webm": "video/webm",
}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".heic", ".tiff", ".ico"}


def _ext(filename: str) -> str:
    return os.path.splitext(filename)[-1].lower()


def guess_mime_type(filename: str) -> str:
    return _EXT_TO_MIME.get(_ext(filename), "application/octet-stream")


def is_image(filename: str, mime_type: str = "") -> bool:
    """MIME 前缀或扩展名判断是否为图片。"""
    return mime_type.startswith("image/") or _ext(filename) in _IMAGE_EXTS


def get_image_format(mime_type: str) -> int:
    """TIM 图片格式编号（未知 → 255）。"""
    return _MIME_TO_IMAGE_FORMAT.get(mime_type.lower(), 255)


def md5_hex(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def generate_file_id() -> str:
    """随机文件 ID（32 位 hex）。"""
    return secrets.token_hex(16)


# ============ 图片尺寸解析（纯 Python，无需 Pillow） ============

def _wh(w: int, h: int) -> dict[str, int]:
    return {"width": w, "height": h}


def _parse_png_size(buf: bytes) -> Optional[dict[str, int]]:
    if len(buf) < 24 or buf[:4] != b"\x89PNG":
        return None
    return _wh(*struct.unpack(">II", buf[16:24]))


def _parse_jpeg_size(buf: bytes) -> Optional[dict[str, int]]:
    if len(buf) < 4 or buf[:2] != b"\xff\xd8":
        return None
    i = 2
    while i < len(buf) - 9:
        if buf[i] != 0xFF:
            i += 1
        elif buf[i + 1] in {0xC0, 0xC2}:  # SOF0 / SOF2: height then width
            h, w = struct.unpack(">HH", buf[i + 5: i + 9])
            return _wh(w, h)
        elif i + 3 < len(buf):
            i += 2 + struct.unpack(">H", buf[i + 2: i + 4])[0]
        else:
            break
    return None


def _parse_gif_size(buf: bytes) -> Optional[dict[str, int]]:
    if len(buf) < 10 or buf[:6] not in (b"GIF87a", b"GIF89a"):
        return None
    return _wh(*struct.unpack("<HH", buf[6:10]))


def _parse_webp_size(buf: bytes) -> Optional[dict[str, int]]:
    if len(buf) < 16 or buf[:4] != b"RIFF" or buf[8:12] != b"WEBP":
        return None
    chunk = buf[12:16]
    if chunk == b"VP8 " and len(buf) >= 30 and buf[23:26] == b"\x9d\x01\x2a":
        w, h = struct.unpack("<HH", buf[26:30])
        return _wh(w & 0x3FFF, h & 0x3FFF)
    if chunk == b"VP8L" and len(buf) >= 25 and buf[20] == 0x2F:
        bits = struct.unpack("<I", buf[21:25])[0]
        return _wh((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1)
    if chunk == b"VP8X" and len(buf) >= 30:
        return _wh(int.from_bytes(buf[24:27], "little") + 1, int.from_bytes(buf[27:30], "little") + 1)
    return None


def parse_image_size(data: bytes) -> Optional[dict[str, int]]:
    """解析图片宽高（JPEG/PNG/GIF/WebP），返回 {"width", "height"} 或 None。"""
    return _parse_png_size(data) or _parse_jpeg_size(data) or _parse_gif_size(data) or _parse_webp_size(data)


# ============ URL 下载 ============

async def download_url(url: str, max_size_mb: int = DEFAULT_MAX_SIZE_MB) -> tuple[bytes, str]:
    """下载 URL 内容，返回 (bytes, content_type)。

    Raises:
        ValueError: 内容超过 max_size_mb，或 URL / 重定向目标被 SSRF 防护拦截
        httpx.HTTPError: 网络/HTTP 错误
    """
    # SSRF protection: yuanbao downloads model-supplied and inbound URLs server-side.
    # Reject private/internal targets up front, and re-validate every redirect hop so a
    # public URL can't 302 to http://169.254.169.254/.
    from tools.url_safety import create_ssrf_safe_async_client, is_safe_url

    if not is_safe_url(url):
        raise ValueError(f"Blocked unsafe URL (SSRF protection): {url}")

    async def _redirect_guard(response: httpx.Response) -> None:
        if response.is_redirect and response.next_request:
            redirect_url = str(response.next_request.url)
            if not is_safe_url(redirect_url):
                raise ValueError(f"Blocked redirect to private/internal address: {redirect_url}")

    max_bytes = max_size_mb * 1024 * 1024
    async with create_ssrf_safe_async_client(
        timeout=30.0, follow_redirects=True, event_hooks={"response": [_redirect_guard]},
    ) as client:
        try:  # HEAD 预检大小；部分服务器不支持 HEAD，忽略
            head = await client.head(url)
            content_length = int(head.headers.get("content-length", 0) or 0)
            if content_length > 0 and content_length > max_bytes:
                raise ValueError(f"文件过大: {content_length / 1024 / 1024:.1f} MB > {max_size_mb} MB")
        except httpx.HTTPStatusError:
            pass
        async with client.stream("GET", url) as resp:  # 流式 GET，防止超限
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "").split(";")[0].strip()
            chunks: list[bytes] = []
            downloaded = 0
            async for chunk in resp.aiter_bytes(65536):
                downloaded += len(chunk)
                if downloaded > max_bytes:
                    raise ValueError(f"文件过大: 已超过 {max_size_mb} MB 限制")
                chunks.append(chunk)
        return b"".join(chunks), content_type


# ============ COS 鉴权（HMAC-SHA1） ============

def _hmac_sha1_hex(key: str, msg: str) -> str:
    return hmac.new(key.encode("utf-8"), msg.encode("utf-8"), hashlib.sha1).hexdigest()


def _sorted_kv(d: dict[str, str]) -> list[tuple[str, str]]:
    """签名用 (小写 key, URL-encoded value) 按字典序排列"""
    return sorted((k.lower(), urllib.parse.quote(str(v), safe="")) for k, v in d.items())


def _cos_sign(
    method: str, path: str, params: dict[str, str], headers: dict[str, str], secret_id: str, secret_key: str,
    start_time: Optional[int] = None, expire_seconds: int = 3600,
) -> str:
    """COS Authorization 头（q-sign-algorithm=sha1；https://cloud.tencent.com/document/product/436/7778）。

    method 小写、path 已 URL-encode；params / headers 的 key 会被小写；secret_* 为临时 tmpSecretId/Key；
    start_time 默认 now。
    """
    start = start_time or int(time.time())
    q_sign_time = f"{start};{start + expire_seconds}"
    sign_key = _hmac_sha1_hex(secret_key, q_sign_time)  # SignKey = HMAC-SHA1(SecretKey, q-sign-time)
    sorted_params = _sorted_kv(params)
    sorted_headers = _sorted_kv(headers)
    kv = lambda pairs: "&".join(f"{k}={v}" for k, v in pairs)  # noqa: E731
    http_string = "\n".join([method.lower(), path, kv(sorted_params), kv(sorted_headers), ""])
    string_to_sign = "\n".join(["sha1", q_sign_time, hashlib.sha1(http_string.encode("utf-8")).hexdigest(), ""])
    return (
        f"q-sign-algorithm=sha1&q-ak={secret_id}&q-sign-time={q_sign_time}&q-key-time={q_sign_time}"
        f"&q-header-list={';'.join(k for k, _ in sorted_headers)}"
        f"&q-url-param-list={';'.join(k for k, _ in sorted_params)}"
        f"&q-signature={_hmac_sha1_hex(sign_key, string_to_sign)}"
    )


# ============ 主要公开 API ============

async def get_cos_credentials(
    app_key: str, api_domain: str, token: str, filename: str = "file", file_id: Optional[str] = None,
    bot_id: str = "", route_env: str = "",
) -> dict:
    """调用 genUploadInfo 获取 COS 临时密钥及上传配置。

    X-ID 头取 bot_id（优先）或 app_key；api_domain 如 https://bot.yuanbao.tencent.com；file_id 不传则自动生成。
    返回接口 data dict：bucketName, region, location (COS key), encryptTmpSecretId, encryptTmpSecretKey,
    encryptToken, startTime, expiredTime, resourceUrl, resourceID(可选)。
    Raises RuntimeError: 接口返回非 0 code 或 bucketName/location 缺失
    """
    headers = {"Content-Type": "application/json", "X-Token": token, "X-ID": bot_id or app_key, "X-Source": "web"}
    if route_env:
        headers["X-Route-Env"] = route_env
    body = {"fileName": filename, "fileId": file_id if file_id is not None else generate_file_id(), "docFrom": "localDoc", "docOpenId": ""}
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(f"{api_domain.rstrip('/')}{UPLOAD_INFO_PATH}", json=body, headers=headers)
        resp.raise_for_status()
        result: dict[str, Any] = resp.json()
    code = result.get("code")
    if code != 0 and code is not None:
        raise RuntimeError(f"genUploadInfo 失败: code={code}, msg={result.get('msg', '')}")
    data = result.get("data") or result
    missing = [f for f in ["bucketName", "location"] if not data.get(f)]
    if missing:
        raise RuntimeError(f"genUploadInfo 返回字段不完整: 缺少字段 {missing}")
    return data


async def upload_to_cos(
    file_bytes: bytes, filename: str, content_type: str, credentials: dict, bucket: str, region: str,
) -> dict:
    """用临时凭证（get_cos_credentials() 返回的 dict）HMAC-SHA1 签名，httpx PUT 上传到 COS（走全球加速域名）。

    Returns {url, uuid (内容 MD5), size, width?, height? (仅图片)}
    Raises httpx.HTTPStatusError（COS 非 2xx）/ RuntimeError（credentials 字段缺失）
    """
    secret_id, secret_key, session_token, cos_key = (
        credentials.get(k, "") for k in ("encryptTmpSecretId", "encryptTmpSecretKey", "encryptToken", "location")
    )
    start_time: Optional[int] = credentials.get("startTime")
    expired_time: Optional[int] = credentials.get("expiredTime")
    if not secret_id or not secret_key or not cos_key:
        raise RuntimeError(
            f"COS credentials 不完整: secretId={bool(secret_id)}, secretKey={bool(secret_key)}, location={bool(cos_key)}"
        )
    cos_host = f"{bucket}.cos.accelerate.myqcloud.com"
    encoded_key = urllib.parse.quote(cos_key, safe="/").lstrip("/")
    cos_url = f"https://{cos_host}/{encoded_key}"
    if not content_type or content_type == "application/octet-stream":
        content_type = guess_mime_type(filename) if is_image(filename) else "application/octet-stream"
    file_size = len(file_bytes)
    now = int(time.time())
    authorization = _cos_sign(
        method="put", path=f"/{encoded_key}", params={},
        headers={"host": cos_host, "content-type": content_type, "x-cos-security-token": session_token},
        secret_id=secret_id, secret_key=secret_key,
        start_time=start_time if start_time else now,
        expire_seconds=(expired_time - now) if expired_time and expired_time > now else 3600,
    )
    put_headers = {"Authorization": authorization, "Content-Type": content_type, "x-cos-security-token": session_token}
    logger.info("COS PUT: bucket=%s region=%s key=%s size=%d mime=%s", bucket, region, cos_key, file_size, content_type)
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.put(cos_url, content=file_bytes, headers=put_headers)
        resp.raise_for_status()
    result: dict[str, Any] = {"url": credentials.get("resourceUrl", "") or cos_url, "uuid": md5_hex(file_bytes), "size": file_size}
    if content_type.startswith("image/"):
        result.update(parse_image_size(file_bytes) or {})
    logger.info("COS 上传成功: url=%s size=%d", result["url"], file_size)
    return result


# ============ TIM 媒体消息构建（https://cloud.tencent.com/document/product/269/2720） ============

def build_image_msg_body(
    url: str, uuid: Optional[str] = None, filename: Optional[str] = None, size: int = 0, width: int = 0,
    height: int = 0, mime_type: str = "",
) -> list[dict]:
    """TIMImageElem 消息体（可直接放入 msg_body）。uuid 缺省依次退到 filename / URL basename / "image"。"""
    return [{
        "msg_type": "TIMImageElem",
        "msg_content": {
            "uuid": uuid or filename or _basename_from_url(url) or "image",
            "image_format": get_image_format(mime_type) if mime_type else 255,
            "image_info_array": [{"type": 1, "size": size, "width": width, "height": height, "url": url}],  # type 1 = 原图
        },
    }]


def build_file_msg_body(url: str, filename: str, uuid: Optional[str] = None, size: int = 0) -> list[dict]:
    """TIMFileElem 消息体（可直接放入 msg_body）。uuid 缺省使用 filename。"""
    return [{
        "msg_type": "TIMFileElem",
        "msg_content": {"uuid": uuid or filename, "file_name": filename, "file_size": size, "url": url},
    }]


def _basename_from_url(url: str) -> str:
    try:
        return os.path.basename(urllib.parse.urlparse(url).path)
    except Exception:
        return ""

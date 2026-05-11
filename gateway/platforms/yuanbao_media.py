"""
yuanbao_media.py — 元宝平台媒体处理模块

提供 COS 上传、文件下载、TIM 媒体消息构建等功能。
移植自 TypeScript 版 media.ts（yuanbao-openclaw-plugin），
使用 httpx 替代 cos-nodejs-sdk-v5，避免引入额外 SDK 依赖。

COS 上传流程：
  1. 调用 genUploadInfo 获取临时凭证（tmpSecretId/tmpSecretKey/sessionToken）
  2. 用临时凭证通过 HMAC-SHA1 签名构建 Authorization 头
  3. HTTP PUT 上传到 COS

TIM 消息体构建：
  - buildImageMsgBody() → TIMImageElem
  - buildFileMsgBody()  → TIMFileElem
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

# ============ 常量 ============

UPLOAD_INFO_PATH = "/api/resource/genUploadInfo"
DEFAULT_API_DOMAIN = "yuanbao.tencent.com"
DEFAULT_MAX_SIZE_MB = 50

# COS 加速域名后缀（优先使用全球加速）
COS_USE_ACCELERATE = True

# ============ 类型映射 ============

# MIME → image_format 数字（TIM 协议字段）
_MIME_TO_IMAGE_FORMAT: dict[str, int] = {
    "image/jpeg": 1,
    "image/jpg": 1,
    "image/gif": 2,
    "image/png": 3,
    "image/bmp": 4,
    "image/webp": 255,
    "image/heic": 255,
    "image/tiff": 255,
}

# 文件扩展名 → MIME
_EXT_TO_MIME: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".heic": "image/heic",
    ".tiff": "image/tiff",
    ".ico": "image/x-icon",
    ".pdf": "application/pdf",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".txt": "text/plain",
    ".zip": "application/zip",
    ".tar": "application/x-tar",
    ".gz": "application/gzip",
    ".mp3": "audio/mpeg",
    ".mp4": "video/mp4",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".webm": "video/webm",
}


# ============ 工具函数 ============

def guess_mime_type(filename: str) -> str:
    """根据文件扩展名猜测 MIME 类型。"""
    ext = os.path.splitext(filename)[-1].lower()
    return _EXT_TO_MIME.get(ext, "application/octet-stream")


def is_image(filename: str, mime_type: str = "") -> bool:
    """判断是否为图片类型。"""
    if mime_type.startswith("image/"):
        return True
    ext = os.path.splitext(filename)[-1].lower()
    return ext in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".heic", ".tiff", ".ico"}


def get_image_format(mime_type: str) -> int:
    """获取 TIM 图片格式编号。"""
    return _MIME_TO_IMAGE_FORMAT.get(mime_type.lower(), 255)


def md5_hex(data: bytes) -> str:
    """计算 MD5 十六进制摘要。"""
    return hashlib.md5(data).hexdigest()


def generate_file_id() -> str:
    """生成随机文件 ID（32 位 hex）。"""
    return secrets.token_hex(16)



# ============ 图片尺寸解析（纯 Python，无需 Pillow） ============

def parse_image_size(data: bytes) -> Optional[dict[str, int]]:
    """
    解析图片宽高（支持 JPEG/PNG/GIF/WebP），无需第三方依赖。
    返回 {"width": w, "height": h} 或 None（无法识别）。
    """
    return (
        _parse_png_size(data)
        or _parse_jpeg_size(data)
        or _parse_gif_size(data)
        or _parse_webp_size(data)
    )


def _parse_png_size(buf: bytes) -> Optional[dict[str, int]]:
    if len(buf) < 24:
        return None
    if buf[:4] != b"\x89PNG":
        return None
    w = struct.unpack(">I", buf[16:20])[0]
    h = struct.unpack(">I", buf[20:24])[0]
    return {"width": w, "height": h}


def _parse_jpeg_size(buf: bytes) -> Optional[dict[str, int]]:
    if len(buf) < 4 or buf[0] != 0xFF or buf[1] != 0xD8:
        return None
    i = 2
    while i < len(buf) - 9:
        if buf[i] != 0xFF:
            i += 1
            continue
        marker = buf[i + 1]
        if marker in (0xC0, 0xC2):
            h = struct.unpack(">H", buf[i + 5: i + 7])[0]
            w = struct.unpack(">H", buf[i + 7: i + 9])[0]
            return {"width": w, "height": h}
        if i + 3 < len(buf):
            i += 2 + struct.unpack(">H", buf[i + 2: i + 4])[0]
        else:
            break
    return None


def _parse_gif_size(buf: bytes) -> Optional[dict[str, int]]:
    if len(buf) < 10:
        return None
    sig = buf[:6].decode("ascii", errors="replace")
    if sig not in ("GIF87a", "GIF89a"):
        return None
    w = struct.unpack("<H", buf[6:8])[0]
    h = struct.unpack("<H", buf[8:10])[0]
    return {"width": w, "height": h}


def _parse_webp_size(buf: bytes) -> Optional[dict[str, int]]:
    if len(buf) < 16:
        return None
    if buf[:4] != b"RIFF" or buf[8:12] != b"WEBP":
        return None
    chunk = buf[12:16].decode("ascii", errors="replace")
    if chunk == "VP8 ":
        if len(buf) >= 30 and buf[23] == 0x9D and buf[24] == 0x01 and buf[25] == 0x2A:
            w = struct.unpack("<H", buf[26:28])[0] & 0x3FFF
            h = struct.unpack("<H", buf[28:30])[0] & 0x3FFF
            return {"width": w, "height": h}
    elif chunk == "VP8L":
        if len(buf) >= 25 and buf[20] == 0x2F:
            bits = struct.unpack("<I", buf[21:25])[0]
            w = (bits & 0x3FFF) + 1
            h = ((bits >> 14) & 0x3FFF) + 1
            return {"width": w, "height": h}
    elif chunk == "VP8X":
        if len(buf) >= 30:
            w = (buf[24] | (buf[25] << 8) | (buf[26] << 16)) + 1
            h = (buf[27] | (buf[28] << 8) | (buf[29] << 16)) + 1
            return {"width": w, "height": h}
    return None


# ============ URL 下载 ============

async def download_url(
    url: str,
    max_size_mb: int = DEFAULT_MAX_SIZE_MB,
) -> tuple[bytes, str]:
    """
    下载 URL 内容，返回 (bytes, content_type)。

    Args:
        url:          HTTP(S) URL
        max_size_mb:  最大允许大小（MB），超过则抛出异常

    Returns:
        (data_bytes, content_type_string)

    Raises:
        ValueError:  内容超过大小限制
        httpx.HTTPError: 网络/HTTP 错误
    """
    max_bytes = max_size_mb * 1024 * 1024
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        # 先 HEAD 检查大小
        try:
            head = await client.head(url)
            content_length = int(head.headers.get("content-length", 0) or 0)
            if content_length > 0 and content_length > max_bytes:
                raise ValueError(
                    f"文件过大: {content_length / 1024 / 1024:.1f} MB > {max_size_mb} MB"
                )
        except httpx.HTTPStatusError:
            pass  # 部分服务器不支持 HEAD，忽略

        # GET 下载（流式读取，防止超限）
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "").split(";")[0].strip()

            chunks: list[bytes] = []
            downloaded = 0
            async for chunk in resp.aiter_bytes(65536):
                downloaded += len(chunk)
                if downloaded > max_bytes:
                    raise ValueError(
                        f"文件过大: 已超过 {max_size_mb} MB 限制"
                    )
                chunks.append(chunk)

        data = b"".join(chunks)
        return data, content_type


# ============ COS 鉴权（HMAC-SHA1） ============

def _cos_sign(
    method: str,
    path: str,
    params: dict[str, str],
    headers: dict[str, str],
    secret_id: str,
    secret_key: str,
    start_time: Optional[int] = None,
    expire_seconds: int = 3600,
) -> str:
    """
    构建 COS 请求签名（q-sign-algorithm=sha1 方案）。
    参考：https://cloud.tencent.com/document/product/436/7778

    Args:
        method:         HTTP 方法（小写，如 "put"）
        path:           URL 路径（URL encode 后的小写）
        params:         URL 查询参数 dict（用于签名）
        headers:        参与签名的请求头 dict（key 需小写）
        secret_id:      临时 SecretId（tmpSecretId）
        secret_key:     临时 SecretKey（tmpSecretKey）
        start_time:     签名起始 Unix 时间戳（默认 now）
        expire_seconds: 签名有效期（秒，默认 3600）

    Returns:
        Authorization header 值（完整字符串）
    """
    now = int(time.time())
    q_sign_time = f"{start_time or now};{(start_time or now) + expire_seconds}"

    # Step 1: SignKey = HMAC-SHA1(SecretKey, q-sign-time)
    sign_key = hmac.new(
        secret_key.encode("utf-8"),
        q_sign_time.encode("utf-8"),
        hashlib.sha1,
    ).hexdigest()

    # Step 2: HttpString
    # 参数和头部需按字典序排列，key 小写
    sorted_params = sorted((k.lower(), urllib.parse.quote(str(v), safe="") ) for k, v in params.items())
    sorted_headers = sorted((k.lower(), urllib.parse.quote(str(v), safe="") ) for k, v in headers.items())

    url_param_list = ";".join(k for k, _ in sorted_params)
    url_params = "&".join(f"{k}={v}" for k, v in sorted_params)
    header_list = ";".join(k for k, _ in sorted_headers)
    header_str = "&".join(f"{k}={v}" for k, v in sorted_headers)

    http_string = "\n".join([
        method.lower(),
        path,
        url_params,
        header_str,
        "",
    ])

    # Step 3: StringToSign = sha1 hash of HttpString
    sha1_of_http = hashlib.sha1(http_string.encode("utf-8")).hexdigest()
    string_to_sign = "\n".join([
        "sha1",
        q_sign_time,
        sha1_of_http,
        "",
    ])

    # Step 4: Signature = HMAC-SHA1(SignKey, StringToSign)
    signature = hmac.new(
        sign_key.encode("utf-8"),
        string_to_sign.encode("utf-8"),
        hashlib.sha1,
    ).hexdigest()

    return (
        f"q-sign-algorithm=sha1"
        f"&q-ak={secret_id}"
        f"&q-sign-time={q_sign_time}"
        f"&q-key-time={q_sign_time}"
        f"&q-header-list={header_list}"
        f"&q-url-param-list={url_param_list}"
        f"&q-signature={signature}"
    )


# ============ 主要公开 API ============

async def get_cos_credentials(
    app_key: str,
    api_domain: str,
    token: str,
    filename: str = "file",
    file_id: Optional[str] = None,
    bot_id: str = "",
    route_env: str = "",
) -> dict:
    """
    调用 genUploadInfo 接口获取 COS 临时密钥及上传配置。

    Args:
        app_key:        应用 Key（用于 X-ID 头）
        api_domain:     API 域名（如 https://bot.yuanbao.tencent.com）
        token:          当前有效的签票 token（X-Token 头）
        filename:       待上传的文件名（含扩展名）
        file_id:        客户端生成的唯一文件 ID（不传则自动生成）
        bot_id:         Bot 账号 ID（用于 X-ID 头）

    Returns:
        COS 上传配置 dict，包含以下字段：
            bucketName         (str)  — COS Bucket 名称
            region             (str)  — COS 地域
            location           (str)  — 上传 Key（对象路径）
            encryptTmpSecretId (str)  — 临时 SecretId
            encryptTmpSecretKey(str)  — 临时 SecretKey
            encryptToken       (str)  — SessionToken
            startTime          (int)  — 凭证起始时间戳（Unix）
            expiredTime        (int)  — 凭证过期时间戳（Unix）
            resourceUrl        (str)  — 上传后的公网访问 URL
            resourceID         (str)  — 资源 ID（可选）

    Raises:
        RuntimeError: 接口返回非 0 code 或字段缺失
    """
    if file_id is None:
        file_id = generate_file_id()

    upload_url = f"{api_domain.rstrip('/')}{UPLOAD_INFO_PATH}"

    headers = {
        "Content-Type": "application/json",
        "X-Token": token,
        "X-ID": bot_id or app_key,
        "X-Source": "web",
    }
    if route_env:
        headers["X-Route-Env"] = route_env
    body = {
        "fileName": filename,
        "fileId": file_id,
        "docFrom": "localDoc",
        "docOpenId": "",
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(upload_url, json=body, headers=headers)
        resp.raise_for_status()
        result: dict[str, Any] = resp.json()

    code = result.get("code")
    if code != 0 and code is not None:
        raise RuntimeError(
            f"genUploadInfo 失败: code={code}, msg={result.get('msg', '')}"
        )

    data = result.get("data") or result
    required_fields = ["bucketName", "location"]
    missing = [f for f in required_fields if not data.get(f)]
    if missing:
        raise RuntimeError(
            f"genUploadInfo 返回字段不完整: 缺少字段 {missing}"
        )

    return data


async def upload_to_cos(
    file_bytes: bytes,
    filename: str,
    content_type: str,
    credentials: dict,
    bucket: str,
    region: str,
) -> dict:
    """
    通过 httpx PUT 请求将文件上传到 COS。
    使用临时凭证（tmpSecretId/tmpSecretKey/sessionToken）构建 HMAC-SHA1 签名。

    Args:
        file_bytes:   文件二进制内容
        filename:     文件名（用于辅助计算 MIME、UUID）
        content_type: MIME 类型（如 "image/jpeg"）
        credentials:  get_cos_credentials() 返回的 dict，包含：
                        encryptTmpSecretId  → tmpSecretId
                        encryptTmpSecretKey → tmpSecretKey
                        encryptToken        → sessionToken
                        location            → COS key（对象路径）
                        resourceUrl         → 上传后公网 URL
                        startTime           → 凭证起始时间（Unix）
                        expiredTime         → 凭证过期时间（Unix）
        bucket:       COS Bucket 名称（如 chatbot-1234567890）
        region:       COS 地域（如 ap-guangzhou）

    Returns:
        上传结果 dict，包含：
            url       (str)           — COS 公网访问 URL
            uuid      (str)           — 文件内容 MD5
            size      (int)           — 文件大小（字节）
            width     (int, optional) — 图片宽度（仅图片）
            height    (int, optional) — 图片高度（仅图片）

    Raises:
        httpx.HTTPStatusError: COS 返回非 2xx 状态
        RuntimeError:          credentials 字段缺失
    """
    secret_id: str = credentials.get("encryptTmpSecretId", "")
    secret_key: str = credentials.get("encryptTmpSecretKey", "")
    session_token: str = credentials.get("encryptToken", "")
    cos_key: str = credentials.get("location", "")
    resource_url: str = credentials.get("resourceUrl", "")
    start_time: Optional[int] = credentials.get("startTime")
    expired_time: Optional[int] = credentials.get("expiredTime")

    if not secret_id or not secret_key or not cos_key:
        raise RuntimeError(
            f"COS credentials 不完整: secretId={bool(secret_id)}, "
            f"secretKey={bool(secret_key)}, location={bool(cos_key)}"
        )

    # 构建 COS 上传 URL（优先使用全球加速域名）
    if COS_USE_ACCELERATE:
        cos_host = f"{bucket}.cos.accelerate.myqcloud.com"
    else:
        cos_host = f"{bucket}.cos.{region}.myqcloud.com"

    # URL encode cos_key（保留 /）
    encoded_key = urllib.parse.quote(cos_key, safe="/")
    cos_url = f"https://{cos_host}/{encoded_key.lstrip('/')}"

    # 确定 Content-Type
    if not content_type or content_type == "application/octet-stream":
        if is_image(filename):
            content_type = guess_mime_type(filename)
        else:
            content_type = "application/octet-stream"

    # 计算文件 MD5 + size
    file_uuid = md5_hex(file_bytes)
    file_size = len(file_bytes)

    # 参与签名的请求头
    sign_headers = {
        "host": cos_host,
        "content-type": content_type,
        "x-cos-security-token": session_token,
    }

    # 计算签名有效期
    now = int(time.time())
    sign_start = start_time if start_time else now
    sign_expire = (expired_time - now) if expired_time and expired_time > now else 3600

    authorization = _cos_sign(
        method="put",
        path=f"/{encoded_key.lstrip('/')}",
        params={},
        headers=sign_headers,
        secret_id=secret_id,
        secret_key=secret_key,
        start_time=sign_start,
        expire_seconds=sign_expire,
    )

    put_headers = {
        "Authorization": authorization,
        "Content-Type": content_type,
        "x-cos-security-token": session_token,
    }

    logger.info(
        "COS PUT: bucket=%s region=%s key=%s size=%d mime=%s",
        bucket, region, cos_key, file_size, content_type,
    )

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.put(
            cos_url,
            content=file_bytes,
            headers=put_headers,
        )
        resp.raise_for_status()

    # 解析图片尺寸（仅图片类型）
    result: dict[str, Any] = {
        "url": resource_url or cos_url,
        "uuid": file_uuid,
        "size": file_size,
    }

    if content_type.startswith("image/"):
        size_info = parse_image_size(file_bytes)
        if size_info:
            result["width"] = size_info["width"]
            result["height"] = size_info["height"]

    logger.info(
        "COS 上传成功: url=%s size=%d",
        result["url"], file_size,
    )
    return result


# ============ TIM 媒体消息构建 ============

def build_image_msg_body(
    url: str,
    uuid: Optional[str] = None,
    filename: Optional[str] = None,
    size: int = 0,
    width: int = 0,
    height: int = 0,
    mime_type: str = "",
) -> list[dict]:
    """
    构建腾讯 IM TIMImageElem 消息体。
    参考：https://cloud.tencent.com/document/product/269/2720

    Args:
        url:       图片公网访问 URL（COS resourceUrl）
        uuid:      文件 UUID（MD5 或其他唯一标识）
        filename:  文件名（uuid 为空时作为备用）
        size:      文件大小（字节）
        width:     图片宽度（像素）
        height:    图片高度（像素）
        mime_type: MIME 类型（用于确定 image_format）

    Returns:
        TIMImageElem 消息体列表（适合直接放入 msg_body）
    """
    _uuid = uuid or filename or _basename_from_url(url) or "image"
    image_format = get_image_format(mime_type) if mime_type else 255

    return [
        {
            "msg_type": "TIMImageElem",
            "msg_content": {
                "uuid": _uuid,
                "image_format": image_format,
                "image_info_array": [
                    {
                        "type": 1,       # 1 = 原图
                        "size": size,
                        "width": width,
                        "height": height,
                        "url": url,
                    }
                ],
            },
        }
    ]


def build_file_msg_body(
    url: str,
    filename: str,
    uuid: Optional[str] = None,
    size: int = 0,
) -> list[dict]:
    """
    构建腾讯 IM TIMFileElem 消息体。
    参考：https://cloud.tencent.com/document/product/269/2720

    Args:
        url:      文件公网访问 URL（COS resourceUrl）
        filename: 文件名（含扩展名）
        uuid:     文件 UUID（MD5 或其他唯一标识，不传则使用 filename）
        size:     文件大小（字节）

    Returns:
        TIMFileElem 消息体列表（适合直接放入 msg_body）
    """
    _uuid = uuid or filename

    return [
        {
            "msg_type": "TIMFileElem",
            "msg_content": {
                "uuid": _uuid,
                "file_name": filename,
                "file_size": size,
                "url": url,
            },
        }
    ]


# ============ 内部工具 ============

def _basename_from_url(url: str) -> str:
    """从 URL 提取文件名。"""
    try:
        parsed = urllib.parse.urlparse(url)
        return os.path.basename(parsed.path)
    except Exception:
        return ""

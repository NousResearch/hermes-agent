"""Response-level checks for claims that need delivery evidence.

The helpers here target a narrow failure mode: a model says a file or
attachment was sent/generated, but emits no MEDIA directive, path, link, or
inline payload the gateway can deliver.
"""

from __future__ import annotations

import re


_DELIVERABLE_EXTS = (
    "doc",
    "docx",
    "pdf",
    "ppt",
    "pptx",
    "xls",
    "xlsx",
    "csv",
    "htm",
    "html",
    "txt",
    "md",
    "zip",
    "rar",
    "7z",
    "png",
    "jpg",
    "jpeg",
    "gif",
    "webp",
    "mp3",
    "wav",
    "m4a",
    "mp4",
    "mov",
)

_EXT_ALT = "|".join(sorted(_DELIVERABLE_EXTS, key=len, reverse=True))

_MEDIA_TAG_RE = re.compile(
    r"MEDIA:\s*[`\"']?(?:~/|/|[A-Za-z]:[/\\])\S+\.(?:" + _EXT_ALT + r")\b",
    re.IGNORECASE,
)
_LOCAL_FILE_PATH_RE = re.compile(
    r"(?<![/:\w.])(?:~/|/|[A-Za-z]:[/\\])\S+\.(?:" + _EXT_ALT + r")\b",
    re.IGNORECASE,
)
_RELATIVE_FILE_PATH_RE = re.compile(
    r"(?<![\w:/\\.-])(?:\.{1,2}[/\\])?"
    r"(?:[\w\u4e00-\u9fff（）()【】《》_.-]+[/\\])+\S+\.(?:"
    + _EXT_ALT
    + r")\b",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_EMAIL_ADDRESS_RE = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)
_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\([^)]+\)", re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```[\s\S]+```")

_NEGATION_RE = re.compile(
    r"(?:"
    r"\b(?:cannot|can't|couldn't|unable|not able|failed|did not|haven't|hasn't)\b"
    r"|(?:没有|还没|尚未|未能|无法|不能|失败)"
    r")",
    re.IGNORECASE,
)

_CLAIM_RE = re.compile(
    r"(?:"
    r"\b(?:file|attachment|document|docx|word|download|attached|uploaded|sent)\b"
    r"|(?:文件|附件|文档|报告|下载|上传|发送|发给你|查收)"
    r")",
    re.IGNORECASE,
)

_COMPLETION_CLAIM_RE = re.compile(
    r"(?:"
    r"\b(?:attached|uploaded|sent|downloadable|here(?:'s| is)? the file|file is below|attachment is below)\b"
    r"|(?:文件如下|附件如下|已附上|已上传|已发送|已下载|下载给你|下载好了|文件已生成|文件已创建|发给你|请查收|供下载)"
    r")",
    re.IGNORECASE,
)

_DELIVERY_PROMISE_RE = re.compile(
    r"(?:"
    r"\b(?:"
    r"I(?:'ll| will)?\s+(?:re)?send"
    r"|(?:re)?sending"
    r"|will\s+(?:be\s+)?(?:attach|upload|send)"
    r"|should\s+(?:appear|show up)\s+as\s+(?:an?\s+)?attachment"
    r"|as\s+(?:an?\s+)?attachment"
    r")\b"
    r"|(?:"
    r"我再发一次"
    r"|重新发(?:一次)?"
    r"|重发(?:一次)?"
    r"|再发(?:一次)?"
    r"|以附件形式"
    r"|作为附件"
    r"|应该会.{0,12}附件"
    r"|会.{0,12}附件"
    r")"
    r")",
    re.IGNORECASE,
)

_EMPTY_AFTER_MARKER_RE = re.compile(
    r"(?:"
    r"(?:文件|附件|文档|word\s*文件|\bfile\b|\battachment\b)"
    r".{0,24}?"
    r"(?:如下|在下方|below|attached)"
    r")\s*[:：,.。!！-]*\s*$",
    re.IGNORECASE,
)

_EMAIL_DELIVERY_RE = re.compile(
    r"(?:\b(?:email|mail|emailed)\b|(?:邮件|邮箱|发送到|发到))",
    re.IGNORECASE,
)


def has_file_delivery_payload(text: str) -> bool:
    """Return True when text carries evidence of a deliverable artifact."""
    if not text:
        return False
    body = str(text)
    return bool(
        _MEDIA_TAG_RE.search(body)
        or _LOCAL_FILE_PATH_RE.search(body)
        or _URL_RE.search(body)
        or _MARKDOWN_LINK_RE.search(body)
        or _CODE_FENCE_RE.search(body)
    )


def looks_like_unbacked_file_delivery_claim(text: str) -> bool:
    """Detect short file/attachment delivery claims with no payload evidence."""
    if not isinstance(text, str):
        return False
    body = " ".join(text.strip().split())
    if not body:
        return False
    if has_file_delivery_payload(body):
        return False
    if _NEGATION_RE.search(body):
        return False
    if _EMAIL_ADDRESS_RE.search(body) and _EMAIL_DELIVERY_RE.search(body):
        return False
    if len(body) > 220:
        return False
    if not _CLAIM_RE.search(body):
        return False
    if _RELATIVE_FILE_PATH_RE.search(body):
        return True
    if _EMPTY_AFTER_MARKER_RE.search(body):
        return True
    if _DELIVERY_PROMISE_RE.search(body):
        return True
    return bool(_COMPLETION_CLAIM_RE.search(body) and len(body) <= 160)

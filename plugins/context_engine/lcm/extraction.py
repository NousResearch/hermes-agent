"""Pre-compaction extraction — extract decisions and commitments before summarization.

Best-effort: failures never block compaction. Extracted content is written to
daily note files so key decisions survive even if the DAG summary loses nuance.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .model_routing import apply_lcm_model_route

logger = logging.getLogger(__name__)

_MEDIA_DATA_URI_RE = re.compile(
    r"data:(?:image|audio|video)/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\s]{16,}",
    re.IGNORECASE,
)
_MEDIA_ATTACHMENT_MARKER = "[Media attachment]"
_MEDIA_ATTACHMENT_SUFFIX = "[with media attachment]"
_TEXT_BLOCK_TYPES = {"text", "input_text", "output_text"}
_MEDIA_BLOCK_HINTS = ("image", "audio", "video")
_STRUCTURED_METADATA_KEYS = ("file_id", "filename", "name", "mime_type", "url", "file_url", "id")


EXTRACTION_PROMPT = """Extract decisions, commitments, outcomes, and rules from this conversation segment.

Format as a flat list of bullet points. Each bullet should be self-contained and understandable
without the surrounding conversation. Include:
- Decisions made (what was chosen, and why if stated)
- Commitments (who will do what)
- Outcomes (what happened as a result of an action)
- Rules or constraints discovered

Skip: greetings, meta-discussion, reasoning that led nowhere, repeated information.
If there is nothing worth extracting, respond with exactly: NOTHING_TO_EXTRACT

CONTENT:
{text}"""


def _call_extraction_llm(prompt: str, model: str = "",
                          timeout: float | None = None) -> Optional[str]:
    """Call the Hermes auxiliary LLM for extraction."""
    try:
        from agent.auxiliary_client import call_llm
        call_kwargs = {
            "task": "extraction",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 2000,
        }
        apply_lcm_model_route(call_kwargs, model)
        if timeout is not None:
            call_kwargs["timeout"] = timeout
        response = call_llm(**call_kwargs)
        content = response.choices[0].message.content
        if not isinstance(content, str):
            content = str(content) if content else ""
        from .escalation import _strip_reasoning_blocks
        return _strip_reasoning_blocks(content).strip()
    except Exception as e:
        logger.debug("Extraction LLM call failed: %s", e)
        return None


def _sanitize_string_media(text: str) -> str:
    if not text:
        return ""
    if not _MEDIA_DATA_URI_RE.search(text):
        return text

    without_media = _MEDIA_DATA_URI_RE.sub("", text)
    without_media = without_media.strip()
    without_media = re.sub(r"\n{3,}", "\n\n", without_media)

    if not without_media:
        return _MEDIA_ATTACHMENT_MARKER
    if _MEDIA_ATTACHMENT_SUFFIX in without_media:
        return without_media
    return f"{without_media}\n{_MEDIA_ATTACHMENT_SUFFIX}"


def _looks_like_media_block(block_type: str, block: Dict[str, Any]) -> bool:
    if any(hint in block_type for hint in _MEDIA_BLOCK_HINTS):
        return True
    return any(key in block for key in ("image_url", "input_image", "output_image", "audio_url", "video_url"))


def _extract_structured_metadata(block: Dict[str, Any]) -> str:
    parts: List[str] = []
    block_type = str(block.get("type", "")).strip()
    if block_type:
        parts.append(f"type={block_type}")

    for key in _STRUCTURED_METADATA_KEYS:
        value = block.get(key)
        if isinstance(value, dict):
            for nested_key in _STRUCTURED_METADATA_KEYS:
                nested_value = value.get(nested_key)
                if isinstance(nested_value, (str, int, float)) and nested_value:
                    parts.append(f"{nested_key}={nested_value}")
                    break
            continue
        if isinstance(value, (str, int, float)) and value:
            parts.append(f"{key}={value}")

    if not parts:
        return "[Structured content]"
    return "[Structured content: " + ", ".join(dict.fromkeys(parts)) + "]"


def _sanitize_content_block(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return _sanitize_string_media(content)
    if isinstance(content, list):
        parts: List[str] = []
        media_seen = False
        for block in content:
            block_text = _sanitize_content_block(block)
            if not block_text:
                continue
            if block_text == _MEDIA_ATTACHMENT_MARKER:
                media_seen = True
                continue
            if block_text.endswith(_MEDIA_ATTACHMENT_SUFFIX):
                media_seen = True
                block_text = block_text[: -len(_MEDIA_ATTACHMENT_SUFFIX)].rstrip()
                if not block_text:
                    continue
            parts.append(block_text)
        combined = "\n".join(part for part in parts if part).strip()
        combined = re.sub(r"\n{3,}", "\n\n", combined)
        if media_seen and combined:
            return f"{combined}\n{_MEDIA_ATTACHMENT_SUFFIX}"
        if media_seen:
            return _MEDIA_ATTACHMENT_MARKER
        return combined
    if isinstance(content, dict):
        block_type = str(content.get("type", "")).lower()
        if block_type in _TEXT_BLOCK_TYPES:
            text_value = content.get("text")
            if isinstance(text_value, dict):
                text_value = text_value.get("value", "")
            if not text_value:
                text_value = content.get("content", "")
            return _sanitize_content_block(text_value)
        if _looks_like_media_block(block_type, content):
            return _MEDIA_ATTACHMENT_MARKER
        for key in ("text", "content"):
            if key in content:
                return _sanitize_content_block(content.get(key))
        return _extract_structured_metadata(content)
    return str(content)


def _sanitize_json_like(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            _sanitize_string_media(key) if isinstance(key, str) else key: _sanitize_json_like(val)
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_json_like(item) for item in value]
    if isinstance(value, str):
        return _sanitize_string_media(value)
    return value


def sanitize_pre_compaction_content(text: Any) -> str:
    """Replace inline media/base64 payloads with compact attachment markers."""
    return _sanitize_content_block(text)


def sanitize_pre_compaction_tool_arguments(arguments: Any) -> str:
    """Clean tool-call argument payloads while preserving JSON-like structure when possible."""
    if arguments is None:
        return ""
    if isinstance(arguments, (dict, list)):
        return json.dumps(_sanitize_json_like(arguments), ensure_ascii=False)
    if not isinstance(arguments, str):
        return sanitize_pre_compaction_content(arguments)
    try:
        parsed = json.loads(arguments)
    except Exception:
        return sanitize_pre_compaction_content(arguments)
    return json.dumps(_sanitize_json_like(parsed), ensure_ascii=False)


def extract_before_compaction(
    serialized_messages: str,
    output_path: str,
    session_id: str = "",
    model: str = "",
    timeout: float | None = None,
) -> bool:
    """Extract decisions from messages about to be compacted and write to a daily file.

    Returns True if extraction succeeded, False otherwise.
    Never raises — failures are logged and swallowed.
    """
    try:
        prompt = EXTRACTION_PROMPT.format(text=serialized_messages)
        result = _call_extraction_llm(prompt, model=model, timeout=timeout)

        if not result or result.strip() == "NOTHING_TO_EXTRACT":
            logger.debug("Pre-compaction extraction: nothing to extract")
            return True

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        date_str = datetime.now().strftime("%Y-%m-%d")
        file_path = output_dir / f"{date_str}.md"

        header = f"\n\n## Extraction — {datetime.now().strftime('%H:%M')}"
        if session_id:
            header += f" ({session_id})"
        header += "\n\n"

        with open(file_path, "a", encoding="utf-8") as f:
            f.write(header)
            f.write(result)
            f.write("\n")

        logger.info("Pre-compaction extraction written to %s", file_path)
        return True

    except Exception as e:
        logger.warning("Pre-compaction extraction failed (non-blocking): %s", e)
        return False

"""Upload a Hermes session transcript to Hugging Face as an agent trace, re-emitted in the **Claude Code
JSONL** shape the HF Agent Trace Viewer auto-detects (https://huggingface.co/docs/hub/agent-traces).
Deterministic, zero LLM turns. Private by default: traces can carry prompts, tool output, local paths and
secrets, so the dataset is created private and every text body passes the secret redactor (``force=True``)
unless ``redact=False``. :func:`upload_session_trace` never raises (returns a user-facing status string);
programmatic callers use :func:`build_trace_jsonl` + :func:`_do_upload`."""

from __future__ import annotations

import json
import logging
import os
import uuid
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME = "hermes-traces"
_HERMES_VERSION = "hermes-agent"
_REDACTION_BLOCKED_MESSAGE = (
    "Trace upload blocked: secret redaction failed, so the transcript may "
    "still contain credentials or other sensitive data. Fix the redactor or "
    "rerun with --no-redact only after manually reviewing the transcript."
)
_NO_TOKEN_MESSAGE = (
    "Can't upload — no Hugging Face token is available. To set it up:\n"
    "\n"
    "1. Create a token with WRITE access at https://huggingface.co/settings/tokens\n"
    "   (New token -> type \"Write\" -> copy it).\n"
    "2. Add it to your environment as HF_TOKEN (e.g. in ~/.hermes/.env):\n"
    "     HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx\n"
    "3. Run /upload-trace again (or `hermes trace upload`)."
)
_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")


class TraceRedactionError(RuntimeError):
    """Raised when a trace cannot be safely redacted before upload."""


# --- Conversion: Hermes OpenAI-format messages -> Claude Code JSONL ---

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _redact(text: Any, enabled: bool) -> Any:
    """Redact a string body when enabled (``force=True``: an upload scrubs even if log redaction is off)."""
    if not enabled or not isinstance(text, str) or not text:
        return text
    try:
        from agent.redact import redact_sensitive_text
        return redact_sensitive_text(text, force=True)
    except Exception as exc:
        logger.warning("Trace upload redaction failed; refusing upload", exc_info=True)
        raise TraceRedactionError(_REDACTION_BLOCKED_MESSAGE) from exc


def _text_block(text: Any, redact: bool) -> Dict[str, Any]:
    return {"type": "text", "text": _redact(text, redact)}


def _part_to_block(part: Any, redact: bool) -> Dict[str, Any]:
    if not isinstance(part, dict):
        return _text_block(str(part), redact)
    if part.get("type") == "text":
        return _text_block(part.get("text", ""), redact)
    if part.get("type") in ("image_url", "image"):
        return {"type": "text", "text": "[image omitted]"}  # the viewer renders text turns; no base64
    return _text_block(json.dumps(part), redact)


def _content_to_blocks(content: Any, redact: bool) -> List[Dict[str, Any]]:
    """Normalize a message ``content`` field into Anthropic content blocks."""
    if isinstance(content, list):
        return [_part_to_block(part, redact) for part in content]
    return [] if content is None else [_text_block(content if isinstance(content, str) else json.dumps(content), redact)]


def _parse_tool_args(raw_args: Any) -> Dict[str, Any]:
    if not isinstance(raw_args, str):
        return raw_args if isinstance(raw_args, dict) else {}
    try:
        return json.loads(raw_args) if raw_args.strip() else {}
    except (json.JSONDecodeError, ValueError):
        return {"_raw": raw_args}


def _tool_calls_to_blocks(tool_calls: Any, redact: bool) -> List[Dict[str, Any]]:
    """Convert OpenAI tool_calls into Anthropic ``tool_use`` content blocks."""
    blocks: List[Dict[str, Any]] = []
    for tc in tool_calls if isinstance(tool_calls, list) else ():
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        parsed = _parse_tool_args(fn.get("arguments"))
        if redact:
            try:
                parsed = json.loads(_redact(json.dumps(parsed), redact))
            except (json.JSONDecodeError, ValueError):
                logger.warning("Trace upload redacted tool arguments are not valid JSON; refusing upload")
                raise TraceRedactionError(_REDACTION_BLOCKED_MESSAGE)
        blocks.append({"type": "tool_use", "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:16]}",
                       "name": fn.get("name") or tc.get("name") or "tool", "input": parsed})
    return blocks


def _git_branch(cwd: str) -> str:
    if not cwd:
        return ""
    try:
        import subprocess
        r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3, cwd=cwd,
                           stdin=subprocess.DEVNULL)
    except Exception:
        return ""
    return r.stdout.strip() if r.returncode == 0 else ""


def _assistant_message(msg: Dict[str, Any], model: str, redact: bool) -> Dict[str, Any]:
    blocks = _content_to_blocks(msg.get("content"), redact) + _tool_calls_to_blocks(msg.get("tool_calls"), redact)
    return {"role": "assistant", "model": model or "unknown", "content": blocks or [{"type": "text", "text": ""}]}


def _tool_result_message(msg: Dict[str, Any], model: str, redact: bool) -> Dict[str, Any]:
    content = msg.get("content")
    return {"role": "user", "content": [{
        "type": "tool_result", "tool_use_id": msg.get("tool_call_id") or msg.get("tool_name") or "tool",
        "content": _redact(content if isinstance(content, str) else json.dumps(content), redact),
    }]}


def _user_message(msg: Dict[str, Any], model: str, redact: bool) -> Dict[str, Any]:
    content = msg.get("content")
    return {"role": "user", "content": _redact(content, redact) if isinstance(content, str) else _content_to_blocks(content, redact)}


# role -> (Claude Code line type, message builder). Unknown roles render as user.
_ROLE_RENDERERS: Dict[Any, Tuple[str, Any]] = {"assistant": ("assistant", _assistant_message), "tool": ("user", _tool_result_message)}


def build_trace_jsonl(messages: List[Dict[str, Any]], *, session_id: str, model: str = "", cwd: str = "", redact: bool = True) -> str:
    """One JSONL line per non-system message: ``user``/``tool`` -> type user (tool results ride on user turns as
    ``tool_result`` keyed by ``tool_call_id``), ``assistant`` -> text + ``tool_use`` blocks; turns link via ``parentUuid``."""
    lines: List[str] = []
    parent: Optional[str] = None
    base_ts = _now_iso()
    git_branch = _git_branch(cwd)
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            continue
        turn_uuid = str(uuid.uuid4())
        line_type, render = _ROLE_RENDERERS.get(role, ("user", _user_message))
        entry = {  # key order is the wire order
            "parentUuid": parent, "isSidechain": False, "userType": "external", "cwd": cwd or os.getcwd(),
            "sessionId": session_id, "version": _HERMES_VERSION, "gitBranch": git_branch, "uuid": turn_uuid,
            "timestamp": base_ts, "type": line_type, "message": render(msg, model, redact),
        }
        lines.append(json.dumps(entry, ensure_ascii=False))
        parent = turn_uuid
    return "\n".join(lines) + ("\n" if lines else "")


# --- Upload ---

def _resolve_hf_token() -> Optional[str]:
    """Return the user's Hugging Face token from the usual env vars."""
    return next((val for var in _TOKEN_ENV_VARS if (val := (os.getenv(var) or "").strip())), None)


def _do_upload(jsonl: str, *, token: str, session_id: str, dataset_name: str = DEFAULT_DATASET_NAME, private: bool = True) -> str:
    """Create the dataset (idempotent) and push the trace file; user-facing status string, never raises."""
    with suppress(Exception):  # lazy-install unavailable/declined — the import below surfaces the hint
        from tools import lazy_deps
        lazy_deps.ensure("tool.trace_upload", prompt=False)
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return "Hugging Face upload needs the `huggingface_hub` package (`pip install huggingface_hub`)."
    api = HfApi(token=token)
    try:
        who = api.whoami()
    except Exception as e:
        logger.warning("HF whoami failed: %s", e)
        return "Your Hugging Face token was rejected (whoami failed). Make sure it has WRITE access and isn't expired."
    user = who.get("name") if isinstance(who, dict) else None
    if not user:
        return "Could not resolve your Hugging Face username from the token."
    repo_id = f"{user}/{dataset_name}"
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    except Exception as e:
        logger.warning("HF create_repo failed for %s: %s", repo_id, e)
        return f"Could not create/access dataset {repo_id}: {e}"
    path_in_repo = f"sessions/{session_id}.jsonl"
    try:
        api.upload_file(path_or_fileobj=jsonl.encode("utf-8"), path_in_repo=path_in_repo, repo_id=repo_id,
                        repo_type="dataset", commit_message=f"add session trace {session_id}")
    except Exception as e:
        logger.warning("HF upload_file failed for %s: %s", repo_id, e)
        return f"Upload to Hugging Face failed: {e}"
    return (f"Uploaded -> https://huggingface.co/datasets/{repo_id}/blob/main/{path_in_repo}\n"
            f"View in the trace viewer: https://huggingface.co/datasets/{repo_id}")


def load_session_messages(session_id: str, db_path=None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """``(messages, meta)`` from SQLite; ``meta`` is ``{}`` when the session row is missing (a live, untitled
    session may still have messages)."""
    from hermes_state import SessionDB
    db = SessionDB(db_path=db_path) if db_path else SessionDB()
    try:
        resolved = db.resolve_session_id(session_id) or session_id
        meta = db.get_session(resolved) or {}
        return db.get_messages_as_conversation(resolved), meta
    finally:
        try:
            db.close()
        except Exception:
            logger.debug("Failed to close trace-upload SessionDB", exc_info=True)


def upload_session_trace(
    session_id: str, *, model: str = "", cwd: str = "", redact: bool = True, private: bool = True,
    dataset_name: str = DEFAULT_DATASET_NAME, db_path=None, token: Optional[str] = None,
) -> str:
    """CLI/gateway entry point: load, convert, upload to ``{user}/hermes-traces``. Status string, never raises."""
    if not session_id:
        return "No active session to upload."
    token = token or _resolve_hf_token()
    if not token:
        return _NO_TOKEN_MESSAGE
    try:
        messages, meta = load_session_messages(session_id, db_path=db_path)
    except Exception as e:
        logger.warning("Failed to load session %s for trace upload: %s", session_id, e)
        return f"Could not load session {session_id}: {e}"
    if not messages:
        return "No transcript to upload for this session yet."
    try:
        jsonl = build_trace_jsonl(messages, session_id=session_id, model=model or meta.get("model") or "", cwd=cwd, redact=redact)
    except TraceRedactionError:
        return _REDACTION_BLOCKED_MESSAGE
    if not jsonl.strip():
        return "No transcript content to upload for this session."
    return _do_upload(jsonl, token=token, session_id=session_id, dataset_name=dataset_name, private=private)

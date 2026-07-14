"""Upload a Hermes session transcript to Hugging Face as an agent trace.

Hermes stores sessions in its own SQLite store (``hermes_state.SessionDB``),
so we reconstruct the conversation and emit it in the **Claude Code JSONL**
shape — one of the three formats the Hugging Face Agent Trace Viewer
auto-detects (Claude Code / Codex / Pi). No dataset-side preprocessing is
needed; the Hub tags the dataset ``agent-traces`` and opens it in the viewer.

Docs: https://huggingface.co/docs/hub/agent-traces

Design notes
------------
* **Zero LLM turn.** This is a deterministic export — it never spends a
  model call. The ``hermes trace upload`` subcommand calls
  :func:`upload_session_trace` directly.
* **Private by default.** Traces can contain prompts, tool output, local
  paths, and secrets. The dataset is created private and every text body
  is passed through Hermes' secret redactor (``force=True``) unless the
  caller explicitly opts out with ``redact=False``.
* **Never raises.** Returns a user-facing status string so command
  handlers can echo it straight back to the user. Programmatic callers
  that need the URL can use :func:`build_trace_jsonl` + :func:`_do_upload`
  directly.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
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


class TraceRedactionError(RuntimeError):
    """Raised when a trace cannot be safely redacted before upload."""


# ---------------------------------------------------------------------------
# Conversion: Hermes OpenAI-format messages -> Claude Code JSONL
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _redact(text: Any, enabled: bool) -> Any:
    """Redact secrets from a string body when redaction is enabled.

    Non-strings pass through untouched. Uses Hermes' shared redactor with
    ``force=True`` so an upload always scrubs known secret shapes even if
    the user disabled log redaction globally.
    """
    if not enabled or not isinstance(text, str) or not text:
        return text
    try:
        from agent.redact import redact_sensitive_text
        return redact_sensitive_text(text, force=True)
    except Exception as exc:
        logger.warning("Trace upload redaction failed; refusing upload", exc_info=True)
        raise TraceRedactionError(_REDACTION_BLOCKED_MESSAGE) from exc


def _content_to_blocks(content: Any, redact: bool) -> List[Dict[str, Any]]:
    """Normalize a message ``content`` field into Anthropic content blocks."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": _redact(content, redact)}]
    if isinstance(content, list):
        blocks: List[Dict[str, Any]] = []
        for part in content:
            if isinstance(part, dict):
                ptype = part.get("type")
                if ptype == "text":
                    blocks.append({"type": "text", "text": _redact(part.get("text", ""), redact)})
                elif ptype in ("image_url", "image"):
                    # Keep a placeholder; the viewer renders text turns and we
                    # don't want to inline base64 blobs into a trace.
                    blocks.append({"type": "text", "text": "[image omitted]"})
                else:
                    blocks.append({"type": "text", "text": _redact(json.dumps(part), redact)})
            else:
                blocks.append({"type": "text", "text": _redact(str(part), redact)})
        return blocks
    return [{"type": "text", "text": _redact(json.dumps(content), redact)}]


def _tool_calls_to_blocks(tool_calls: Any, redact: bool) -> List[Dict[str, Any]]:
    """Convert OpenAI tool_calls into Anthropic ``tool_use`` content blocks."""
    blocks: List[Dict[str, Any]] = []
    if not isinstance(tool_calls, list):
        return blocks
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        name = fn.get("name") or tc.get("name") or "tool"
        raw_args = fn.get("arguments")
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args) if raw_args.strip() else {}
            except (json.JSONDecodeError, ValueError):
                parsed = {"_raw": raw_args}
        elif isinstance(raw_args, dict):
            parsed = raw_args
        else:
            parsed = {}
        if redact:
            try:
                parsed = json.loads(_redact(json.dumps(parsed), redact))
            except (json.JSONDecodeError, ValueError) as _b904_exc:
                logger.warning("Trace upload redacted tool arguments are not valid JSON; refusing upload")
                raise TraceRedactionError(_REDACTION_BLOCKED_MESSAGE) from _b904_exc
        blocks.append({
            "type": "tool_use",
            "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:16]}",
            "name": name,
            "input": parsed,
        })
    return blocks


def build_trace_jsonl(
    messages: List[Dict[str, Any]],
    *,
    session_id: str,
    model: str = "",
    cwd: str = "",
    redact: bool = True,
) -> str:
    """Render Hermes conversation messages as Claude Code JSONL text.

    Each non-system message becomes one JSONL line in the Claude Code
    transcript shape the HF Agent Trace Viewer auto-detects:

    * ``user`` / ``tool`` -> ``{"type": "user", "message": {...}}``
    * ``assistant``       -> ``{"type": "assistant", "message": {...}}``
      with ``content`` blocks (text + ``tool_use``).

    Tool results are emitted as user turns carrying a ``tool_result``
    block keyed by ``tool_call_id`` — the same way Claude Code records
    them. Turns are linked via ``uuid`` / ``parentUuid``.
    """
    lines: List[str] = []
    parent: Optional[str] = None
    base_ts = _now_iso()
    git_branch = ""
    try:
        import subprocess
        if cwd:
            r = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, timeout=3, cwd=cwd,
            )
            if r.returncode == 0:
                git_branch = r.stdout.strip()
    except Exception:
        git_branch = ""

    def _common(turn_uuid: str) -> Dict[str, Any]:
        return {
            "parentUuid": parent,
            "isSidechain": False,
            "userType": "external",
            "cwd": cwd or os.getcwd(),
            "sessionId": session_id,
            "version": _HERMES_VERSION,
            "gitBranch": git_branch,
            "uuid": turn_uuid,
            "timestamp": base_ts,
        }

    for msg in messages:
        role = msg.get("role")
        if role == "system":
            continue
        turn_uuid = str(uuid.uuid4())

        if role == "assistant":
            blocks = _content_to_blocks(msg.get("content"), redact)
            blocks.extend(_tool_calls_to_blocks(msg.get("tool_calls"), redact))
            if not blocks:
                blocks = [{"type": "text", "text": ""}]
            entry = _common(turn_uuid)
            entry["type"] = "assistant"
            entry["message"] = {
                "role": "assistant",
                "model": model or "unknown",
                "content": blocks,
            }
            lines.append(json.dumps(entry, ensure_ascii=False))
            parent = turn_uuid
            continue

        if role == "tool":
            tool_use_id = msg.get("tool_call_id") or msg.get("tool_name") or "tool"
            result_content = _redact(
                msg.get("content") if isinstance(msg.get("content"), str)
                else json.dumps(msg.get("content")),
                redact,
            )
            entry = _common(turn_uuid)
            entry["type"] = "user"
            entry["message"] = {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result_content,
                }],
            }
            lines.append(json.dumps(entry, ensure_ascii=False))
            parent = turn_uuid
            continue

        # Default: user (and any unknown role) -> user turn.
        content = msg.get("content")
        if isinstance(content, str):
            message_content: Any = _redact(content, redact)
        else:
            message_content = _content_to_blocks(content, redact)
        entry = _common(turn_uuid)
        entry["type"] = "user"
        entry["message"] = {"role": "user", "content": message_content}
        lines.append(json.dumps(entry, ensure_ascii=False))
        parent = turn_uuid

    return "\n".join(lines) + ("\n" if lines else "")


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def _resolve_hf_token() -> Optional[str]:
    """Return the user's Hugging Face token from the usual env vars."""
    for var in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        val = os.getenv(var)
        if val and val.strip():
            return val.strip()
    return None


_NO_TOKEN_MESSAGE = (
    "Can't upload — no Hugging Face token is available. To set it up:\n"
    "\n"
    "1. Create a token with WRITE access at https://huggingface.co/settings/tokens\n"
    "   (New token -> type \"Write\" -> copy it).\n"
    "2. Add it to your environment as HF_TOKEN (e.g. in ~/.hermes/.env):\n"
    "     HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx\n"
    "3. Run /upload-trace again (or `hermes trace upload`)."
)


def _do_upload(
    jsonl: str,
    *,
    token: str,
    session_id: str,
    dataset_name: str = DEFAULT_DATASET_NAME,
    private: bool = True,
) -> str:
    """Create (idempotently) the private dataset and push the trace file.

    Returns a user-facing status string. Never raises.
    """
    try:
        from tools import lazy_deps
        lazy_deps.ensure("tool.trace_upload", prompt=False)
    except Exception:
        # lazy-install unavailable/declined — fall through to the import,
        # which surfaces the install hint below if the package is missing.
        pass
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return ("Hugging Face upload needs the `huggingface_hub` package "
                "(`pip install huggingface_hub`).")

    api = HfApi(token=token)
    try:
        who = api.whoami()
        user = who.get("name") if isinstance(who, dict) else None
    except Exception as e:
        logger.warning("HF whoami failed: %s", e)
        return ("Your Hugging Face token was rejected (whoami failed). "
                "Make sure it has WRITE access and isn't expired.")
    if not user:
        return "Could not resolve your Hugging Face username from the token."

    repo_id = f"{user}/{dataset_name}"
    try:
        api.create_repo(
            repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True,
        )
    except Exception as e:
        logger.warning("HF create_repo failed for %s: %s", repo_id, e)
        return f"Could not create/access dataset {repo_id}: {e}"

    path_in_repo = f"sessions/{session_id}.jsonl"
    try:
        api.upload_file(
            path_or_fileobj=jsonl.encode("utf-8"),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"add session trace {session_id}",
        )
    except Exception as e:
        logger.warning("HF upload_file failed for %s: %s", repo_id, e)
        return f"Upload to Hugging Face failed: {e}"

    return (f"Uploaded -> https://huggingface.co/datasets/{repo_id}/blob/main/{path_in_repo}\n"
            f"View in the trace viewer: https://huggingface.co/datasets/{repo_id}")


def load_session_messages(
    session_id: str, db_path=None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load a session's conversation + metadata from the SQLite store.

    Returns ``(messages, meta)``. ``meta`` is ``{}`` when the session row is
    missing (messages may still be present for a live, untitled session).
    """
    from hermes_state import SessionDB
    db = SessionDB(db_path=db_path) if db_path else SessionDB()
    resolved = db.resolve_session_id(session_id) or session_id
    meta = db.get_session(resolved) or {}
    messages = db.get_messages_as_conversation(resolved)
    return messages, meta


def upload_session_trace(
    session_id: str,
    *,
    model: str = "",
    cwd: str = "",
    redact: bool = True,
    private: bool = True,
    dataset_name: str = DEFAULT_DATASET_NAME,
    db_path=None,
    token: Optional[str] = None,
) -> str:
    """Top-level entry point used by the CLI/gateway/subcommand.

    Loads the session, converts it to Claude Code JSONL, and uploads it to
    the user's private ``{user}/hermes-traces`` dataset. Returns a
    user-facing status string and never raises.
    """
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

    resolved_model = model or meta.get("model") or ""
    try:
        jsonl = build_trace_jsonl(
            messages,
            session_id=session_id,
            model=resolved_model,
            cwd=cwd,
            redact=redact,
        )
    except TraceRedactionError:
        return _REDACTION_BLOCKED_MESSAGE
    if not jsonl.strip():
        return "No transcript content to upload for this session."

    return _do_upload(
        jsonl,
        token=token,
        session_id=session_id,
        dataset_name=dataset_name,
        private=private,
    )

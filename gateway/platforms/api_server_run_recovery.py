"""Conservative recovery of cooperatively interrupted, run-owned file work.

This is not a generic tool-effect ledger. A checkpoint is sealed only AFTER the
synchronous worker returned, with a complete durable transcript. Only sequential
native file writes with a verifiable receipt (and known no-effect tools) qualify.
Hard crashes, partial batches, unverified writes and other effects stay blocked.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from agent.tool_result_classification import tool_may_have_side_effect
from gateway.platforms.api_server_run_recovery_files import inspect_file
from tools.file_operations_common import _normalize_line_endings, _UTF8_BOM

POLICY = "verified_file_steps"


class RecoveryBlocked(ValueError):
    """The evidence is insufficient; never infer that a failed check is retryable."""


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def request_binding(body: dict, gateway_session_key: str | None) -> str:
    # Retain every execution/identity option, not secrets or input text. A new
    # authenticated instruction is allowed, but the original execution policy is not changed.
    return digest({"body": {k: v for k, v in body.items() if k not in {"input", "checkpoint_id"}},
                   "gateway_session_key": gateway_session_key})


def validate_request(body: dict, *, durable: bool, idempotency_key: str) -> None:
    if body.get("recovery_policy") != POLICY:
        raise RecoveryBlocked("Unsupported recovery policy")
    if not durable or not idempotency_key:
        raise RecoveryBlocked("Recovery requires durable admission and an Idempotency-Key")
    # Caller-authored transcripts must never become apparent handler receipts.
    if not isinstance(body.get("input"), str) or any(
        body.get(k) for k in ("conversation_history", "session_id", "previous_response_id", "hosted_room_dispatch")
    ):
        raise RecoveryBlocked("Recovery requires a new run-owned session and string input")


def _object(text: Any) -> dict:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise RecoveryBlocked("Duplicate JSON key in recovery evidence")
            result[key] = value
        return result

    if not isinstance(text, str):
        raise RecoveryBlocked("Recovery evidence is not an original JSON string")
    try:
        result = json.loads(text, object_pairs_hook=unique_pairs)
    except (TypeError, ValueError) as exc:
        raise RecoveryBlocked("Incomplete or non-JSON recovery evidence") from exc
    if not isinstance(result, dict):
        raise RecoveryBlocked("Recovery evidence is not an object")
    return result


def _write_receipt(call: dict, result: dict) -> dict:
    args = _object(call.get("function", {}).get("arguments"))
    content, path = args.get("content"), args.get("path")
    target = result.get("resolved_path")
    size = result.get("bytes_written")
    if (not isinstance(content, str) or not isinstance(path, str) or not isinstance(target, str)
            or result.get("error") or result.get("verified") is not True
            or type(size) is not int or size < 0 or result.get("files_modified") != [target]):
        raise RecoveryBlocked("Missing verified native write receipt")
    if not Path(path).is_absolute() or Path(path) != Path(target):
        raise RecoveryBlocked("The receipt does not bind the original absolute target")
    # The native writer preserves existing CRLF/BOM. Reproduce its transformations
    # rather than incorrectly hashing the raw argument. Length must identify ONE
    # possible byte sequence; e.g. BOM+LF and CRLF can have equal lengths. Reject
    # that ambiguity instead of accepting whichever candidate happens to exist.
    variants = {content, _normalize_line_endings(content, "\r\n")}
    variants |= {_UTF8_BOM + text for text in tuple(variants) if not text.startswith(_UTF8_BOM)}
    candidates = {text.encode("utf-8", "surrogateescape") for text in variants}
    candidates = {data for data in candidates if len(data) == size}
    if len(candidates) != 1:
        raise RecoveryBlocked("The verified write's effective bytes are ambiguous")
    expected = hashlib.sha256(candidates.pop()).hexdigest()
    return {"path": target, "bytes_written": size, "sha256": expected}


def _check_files(receipts: list[dict]) -> list[dict]:
    snapshots = []
    try:
        for receipt in receipts:
            snapshot = inspect_file(receipt["path"], receipt["bytes_written"])
            if snapshot["sha256"] != receipt["sha256"]:
                raise RecoveryBlocked("The verified file changed after its checkpoint")
            snapshots.append({**receipt, "snapshot": snapshot})
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RecoveryBlocked("Cannot validate the checkpoint's local files") from exc
    return snapshots


def checkpoint(messages: list[dict], *, run_id: str, session_id: str, binding: str,
               runtime_sha256: str) -> dict:
    pending = None
    seen = set()
    files = {}
    for message in messages:
        if not isinstance(message, dict) or message.get("effect_disposition") == "unknown":
            raise RecoveryBlocked("Unknown tool effect in the checkpoint")
        calls = message.get("tool_calls")
        if calls:
            if pending is not None or message.get("role") != "assistant" or len(calls) != 1:
                raise RecoveryBlocked("Parallel or incomplete tool batch")
            pending = calls[0]
            call_id = pending.get("id")
            if not isinstance(call_id, str) or not call_id or call_id in seen:
                raise RecoveryBlocked("Ambiguous tool-call identity")
            seen.add(call_id)
        elif message.get("role") == "tool":
            if pending is None or message.get("tool_call_id") != pending.get("id"):
                raise RecoveryBlocked("Unpaired tool receipt")
            name = pending.get("function", {}).get("name")
            if message.get("tool_name") not in (None, name):
                raise RecoveryBlocked("Tool-call and receipt names disagree")
            if name == "write_file":
                receipt = _write_receipt(pending, _object(message.get("content")))
                files[receipt["path"]] = receipt
            elif not isinstance(name, str) or tool_may_have_side_effect(name):
                raise RecoveryBlocked("No recovery verifier for this effectful tool")
            pending = None
        elif pending is not None:
            raise RecoveryBlocked("Unsettled tool call")
    if pending is not None or not files:
        raise RecoveryBlocked("No complete verified file step")
    receipts = _check_files(list(files.values()))
    proof = {"policy": POLICY, "run_id": run_id, "session_id": session_id,
             "request_binding": binding, "runtime_sha256": runtime_sha256,
             "transcript_sha256": digest(messages), "files": receipts}
    return {**proof, "checkpoint_id": digest(proof), "disposition": "safe_to_continue"}


def verify_checkpoint(proof: dict, messages: list[dict], *, binding: str) -> None:
    if proof.get("disposition") != "safe_to_continue" or proof.get("request_binding") != binding:
        raise RecoveryBlocked("No compatible, unclaimed recovery checkpoint")
    fresh = checkpoint(messages, run_id=proof["run_id"], session_id=proof["session_id"],
                       binding=binding, runtime_sha256=proof["runtime_sha256"])
    if fresh != proof:
        raise RecoveryBlocked("The checkpoint transcript or receipts changed")


def runtime_binding(agent) -> str:
    """Fingerprint the resolved runtime, not the client's alias or default."""
    return digest({name: getattr(agent, name, None) for name in (
        "model", "provider", "api_mode", "base_url", "tools", "max_iterations")})


def seal_after_worker(run, agent, result: dict) -> dict:
    """Prepare in the real tool scope; publish only AFTER full executor return."""
    blocked = {"disposition": "blocked", "reason": "checkpoint_not_proven"}
    if result.get("interrupted") is not True:
        return blocked
    try:
        from tools.environments.local import LocalEnvironment
        from tools.terminal_tool_lifecycle import get_active_env
        # A host Path must not accidentally stand in for a container/SSH path.
        if not isinstance(get_active_env(run.session_id), LocalEnvironment):
            raise RecoveryBlocked("Only local native file effects can be verified")
        db = agent._session_db
        if db is None or str(db.db_path) == ":memory:" or agent.session_id != run.session_id:
            raise RecoveryBlocked("The original durable session is unavailable or rotated")
        if getattr(agent, "_persist_disabled", False):
            raise RecoveryBlocked("Transcript persistence is disabled")
        messages = db.get_messages_as_conversation(run.session_id)
        if messages != db.get_messages_as_conversation(run.session_id, include_inactive=True):
            raise RecoveryBlocked("Compacted or rewound history cannot prove all prior effects")
        return checkpoint(messages, run_id=run.run_id, session_id=run.session_id,
                          binding=run.recovery_binding, runtime_sha256=runtime_binding(agent))
    except (RecoveryBlocked, AttributeError, TypeError, ValueError, OSError) as exc:
        return {**blocked, "reason": str(exc)}


def prepare_continuation(run, agent) -> None:
    """Recheck after queuing; durably seed a NEW run-owned session before inference.

    Failures after admission leave the source claimed. An uncertain setup crash
    must not enable an automatic second successor.
    """
    from tools.environments.local import LocalEnvironment
    from tools.file_tools import _get_file_ops

    proof, db = run.continuation_proof, agent._session_db
    if not db or proof.get("runtime_sha256") != runtime_binding(agent):
        raise RecoveryBlocked("The resolved runtime changed since the source checkpoint")
    if not isinstance(_get_file_ops(run.session_id).env, LocalEnvironment):
        raise RecoveryBlocked("The continuation backend is not local")
    original = db.get_messages_as_conversation(proof["session_id"])
    verify_checkpoint(proof, original, binding=run.recovery_binding)
    if original != run.conversation_history:
        raise RecoveryBlocked("The source transcript changed while continuation was queued")
    if db.get_messages_as_conversation(run.session_id):
        raise RecoveryBlocked("The successor session is not empty")
    db.create_session(run.session_id, source="api_server", parent_session_id=proof["session_id"],
                      model_config={"_branched_from": proof["session_id"]})
    db.append_messages_batch(run.session_id, [
        {k: v for k, v in message.items() if not k.startswith("_")} for message in original])
    copied = db.get_messages_as_conversation(run.session_id)
    if digest(copied) != proof["transcript_sha256"]:
        raise RecoveryBlocked("The successor transcript copy was not verified")
    run.conversation_history = copied

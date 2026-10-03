"""Parser fixtures are not execution proof; see the native-writer HTTP tests."""

import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from gateway.platforms.api_server_run_recovery import RecoveryBlocked, checkpoint, verify_checkpoint


def transcript(path, content="settled\n"):
    path.write_text(content, encoding="utf-8", newline="")
    return [
        {"role": "user", "content": "Write the file, then continue."},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "step-one", "type": "function", "function": {
                "name": "write_file", "arguments": json.dumps({"path": str(path), "content": content})}}]},
        {"role": "tool", "tool_call_id": "step-one", "tool_name": "write_file",
         "content": json.dumps({"verified": True, "bytes_written": path.stat().st_size,
                                "resolved_path": str(path), "files_modified": [str(path)]})},
    ]


def seal(messages):
    return checkpoint(messages, run_id="source", session_id="session", binding="request", runtime_sha256="runtime")


@pytest.mark.parametrize("change", [
    "missing-result", "unverified", "unknown-effect", "duplicate-id", "wrong-id", "parallel",
    "unknown-tool", "patch", "non-json", "duplicate-json-key", "different-target", "name-mismatch",
])
def test_rejects_ambiguous_or_unsupported_evidence(tmp_path, change):
    messages = transcript(tmp_path / "output.txt")
    result = json.loads(messages[-1]["content"])
    call = messages[1]["tool_calls"][0]
    if change == "missing-result":
        messages.pop()
    elif change == "unverified":
        result.pop("verified")
        messages[-1]["content"] = json.dumps(result)
    elif change == "unknown-effect":
        messages[-1]["effect_disposition"] = "unknown"
    elif change == "duplicate-id":
        messages.extend(copy.deepcopy(messages[1:]))
    elif change == "wrong-id":
        messages[-1]["tool_call_id"] = "other"
    elif change == "parallel":
        messages[1]["tool_calls"].append(copy.deepcopy(call))
    elif change in {"unknown-tool", "patch"}:
        call["function"]["name"] = "mcp_mutation" if change == "unknown-tool" else "patch"
    elif change == "non-json":
        messages[-1]["content"] = "[tool output omitted]"
    elif change == "duplicate-json-key":
        messages[-1]["content"] = '{"verified":false,"verified":true}'
    elif change == "different-target":
        result["resolved_path"] = str(tmp_path / "other.txt")
        messages[-1]["content"] = json.dumps(result)
    elif change == "name-mismatch":
        messages[-1]["tool_name"] = "terminal"
    with pytest.raises(RecoveryBlocked):
        seal(messages)


def test_equal_length_bom_crlf_candidates_are_not_proof(tmp_path):
    content = "a\nb\nc\n"
    path = tmp_path / "ambiguous.txt"
    messages = transcript(path, content)
    # BOM+LF and no-BOM+CRLF have equal byte lengths but different bytes.
    path.write_bytes(content.replace("\n", "\r\n").encode())
    result = json.loads(messages[-1]["content"])
    result["bytes_written"] = path.stat().st_size
    messages[-1]["content"] = json.dumps(result)
    with pytest.raises(RecoveryBlocked, match="ambiguous"):
        seal(messages)


@pytest.mark.parametrize("change", ["content", "identity"])
def test_receipt_and_transcript_are_revalidated(tmp_path, change):
    path = tmp_path / "output.txt"
    messages = transcript(path)
    proof = seal(messages)
    verify_checkpoint(proof, messages, binding="request")
    changed = copy.deepcopy(messages)
    changed[0]["content"] = "a different original task"
    with pytest.raises(RecoveryBlocked):
        verify_checkpoint(proof, changed, binding="request")
    if change == "content":
        path.write_text("external change", encoding="utf-8")
    else:
        original_identity = path.stat().st_ino
        replacement = tmp_path / "replacement.txt"
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
        assert path.stat().st_ino != original_identity
    with pytest.raises(RecoveryBlocked):
        verify_checkpoint(proof, messages, binding="request")


def test_source_claim_is_atomic_across_connections_and_different_keys(tmp_path):
    path = tmp_path / "runs.db"
    stores = [RunIdempotencyStore(str(path)), RunIdempotencyStore(str(path))]
    source_status = {"run_id": "source", "status": "cancelled", "recovery": {
        "disposition": "safe_to_continue", "checkpoint_id": "checkpoint"}}
    stores[0].reserve("scope", "original", "original-body", "source", source_status)
    barrier = threading.Barrier(2)

    def attempt(index):
        barrier.wait(timeout=5)
        return stores[index].reserve("scope", f"key-{index}", f"body-{index}", f"successor-{index}",
            {"run_id": f"successor-{index}", "status": "queued"},
            source_run_id="source", checkpoint_id="checkpoint")

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(attempt, range(2)))
        assert sorted(outcome[0] for outcome in outcomes) == ["created", "unrecoverable"]
        winner = next(index for index, outcome in enumerate(outcomes) if outcome[0] == "created")
        source_record = stores[0].status_for_run("scope", "source")
        assert source_record is not None
        source = source_record["status"]
        assert source["recovery"]["successor_run_id"] == f"successor-{winner}"
        replay = stores[1].reserve("scope", f"key-{winner}", f"body-{winner}", "unused",
            {"status": "queued"}, source_run_id="source", checkpoint_id="checkpoint")
        assert replay[0] == "reused"
        assert stores[0].status_for_run("scope", "unused") is None
        assert stores[0].status_for_run("other-principal", "source") is None
    finally:
        for store in stores:
            store.close()

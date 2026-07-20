from __future__ import annotations

import errno
import importlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
PLUGIN_DIR = REPO_ROOT / "plugins" / "truth-ledger"


def _load_truth_plugin_init():
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.truth_ledger",
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec is not None
    assert spec.loader is not None
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hermes_plugins.truth_ledger"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_truth_plugin_schemas_module():
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.truth_ledger.schemas",
        PLUGIN_DIR / "schemas.py",
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hermes_plugins.truth_ledger.schemas"] = mod
    spec.loader.exec_module(mod)
    return mod


class _FakeCtx:
    def __init__(self, profile_name: str = "default") -> None:
        self.profile_name = profile_name
        self.hooks: list[str] = []

    def register_hook(self, name: str, callback):
        self.hooks.append(name)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"completed": False, "turn_exit_reason": "max_iterations_reached(3/3)"},
        {"failed": True},
        {"interrupted": True},
        {"kanban_task_id": "t_worker"},
        {"is_subagent": True, "delegate_depth": 1},
    ],
)
def test_post_llm_call_rejects_ineligible_contexts(kwargs, tmp_path):
    plugin = _load_truth_plugin_init()
    plugin.register(_FakeCtx())

    payload = {
        "session_id": "sess-1",
        "turn_id": "turn-1",
        "task_id": "task-1",
        "platform": "cli",
        "completed": True,
        "failed": False,
        "interrupted": False,
        "turn_exit_reason": "text_response(finish_reason=stop)",
        "delegate_depth": 0,
        "is_subagent": False,
        "kanban_task_id": None,
        "speaker_id": None,
        "conversation_id": None,
        "thread_id": None,
        "user_message": "remember this",
        "assistant_response": "ok",
        "conversation_history": [{"role": "user", "content": "do not persist"}],
    }
    payload.update(kwargs)

    plugin.on_post_llm_call(**payload)

    pending = tmp_path / ".hermes" / "truth-ledger" / "spool" / "pending"
    assert list(pending.glob("*.json")) == []


def test_post_llm_call_enqueues_once_and_never_persists_conversation_history(tmp_path):
    plugin = _load_truth_plugin_init()
    plugin.register(_FakeCtx(profile_name="automation-operator"))
    schemas_mod = _load_truth_plugin_schemas_module()

    payload = {
        "session_id": "sess-eligible",
        "turn_id": "turn-eligible",
        "task_id": "task-eligible",
        "platform": "cli",
        "completed": True,
        "failed": False,
        "interrupted": False,
        "turn_exit_reason": "text_response(finish_reason=stop)",
        "delegate_depth": 0,
        "is_subagent": False,
        "kanban_task_id": None,
        "speaker_id": None,
        "conversation_id": None,
        "thread_id": None,
        "user_message": "remember api key sk-live-1234567890ABCDEFGHIJ and keep replies concise",
        "assistant_response": "Got it, I noted sk-live-1234567890ABCDEFGHIJ and will keep replies concise",
        "conversation_history": [{"role": "assistant", "content": "raw transcript"}],
    }

    captured: list[dict[str, Any]] = []

    def _capture_envelope(envelope):
        captured.append(envelope)
        return {"ok": True}

    plugin._SEEN_ENVELOPES.clear()
    plugin._enqueue_source_envelope = _capture_envelope

    plugin.on_post_llm_call(**payload)
    plugin.on_post_llm_call(**payload)

    assert len(captured) == 1
    envelope = captured[0]
    schemas_mod.validate_document("source-envelope.v1", envelope)

    assert envelope["profile"] == "automation-operator"
    assert envelope["session_id"] == "sess-eligible"
    assert envelope["turn_id"] == "turn-eligible"
    assert envelope["input"]["user_message"]
    assert envelope["output"]["assistant_response"]
    assert len(envelope["input"]["user_message"]) <= 65536
    assert len(envelope["output"]["assistant_response"]) <= 65536
    assert "conversation_history" not in json.dumps(envelope)
    assert "sk-live-1234567890ABCDEFGHIJ" not in json.dumps(envelope)
    assert "keep replies concise" in envelope["input"]["user_message"]

    extractor_mod = importlib.import_module("hermes_plugins.truth_ledger.extractor")
    blocks = extractor_mod._build_input_blocks(envelope)
    assert len(blocks) == 1
    block_payload = json.loads(blocks[0].text)
    assert block_payload["input"]["user_message"]
    assert block_payload["output"]["assistant_response"]
    assert "sk-live-1234567890ABCDEFGHIJ" not in blocks[0].text
    assert "keep replies concise" in block_payload["input"]["user_message"]


def test_post_llm_call_real_enqueue_persists_source_envelope_contract(tmp_path):
    plugin = _load_truth_plugin_init()
    plugin.register(_FakeCtx(profile_name="automation-operator"))
    schemas_mod = _load_truth_plugin_schemas_module()

    plugin._SEEN_ENVELOPES.clear()
    plugin.on_post_llm_call(
        session_id="sess-real-enqueue",
        turn_id="turn-real-enqueue",
        task_id="task-real-enqueue",
        platform="cli",
        completed=True,
        failed=False,
        interrupted=False,
        turn_exit_reason="text_response(finish_reason=stop)",
        delegate_depth=0,
        is_subagent=False,
        kanban_task_id=None,
        speaker_id=None,
        conversation_id=None,
        thread_id=None,
        user_message="remember api key sk-live-1234567890ABCDEFGHIJ and keep replies concise",
        assistant_response="Acknowledged. I will keep replies concise.",
    )

    pending_dir = tmp_path / ".hermes" / "truth-ledger" / "spool" / "pending"
    pending_files = list(pending_dir.glob("*.json"))
    assert len(pending_files) == 1

    stored_record = json.loads(pending_files[0].read_text(encoding="utf-8"))
    schemas_mod.validate_document("spool-record.v1", stored_record)
    payload_path = Path(stored_record["payload_path"])
    assert payload_path.exists()

    stored_envelope = json.loads(payload_path.read_text(encoding="utf-8"))
    schemas_mod.validate_document("source-envelope.v1", stored_envelope)
    assert "attempt_count" not in stored_envelope
    assert "first_seen_at" not in stored_envelope
    assert "next_retry_at" not in stored_envelope


def test_on_session_start_recovers_stale_processing(tmp_path):
    plugin = _load_truth_plugin_init()
    plugin.register(_FakeCtx())

    root = tmp_path / ".hermes" / "truth-ledger"
    processing = root / "spool" / "processing"
    processing.mkdir(parents=True, exist_ok=True)
    src = processing / "stale.json"
    src.write_text('{"session_id":"s","turn_id":"t"}', encoding="utf-8")
    stale_ts = 0
    os.utime(src, (stale_ts, stale_ts))

    plugin.on_session_start(session_id="sess")

    pending = root / "spool" / "pending"
    dead_letter = root / "spool" / "dead-letter"
    assert src.exists() is False
    assert len(list(pending.glob("*.json"))) == 0
    dead_files = list(dead_letter.glob("*.json"))
    assert len(dead_files) == 1
    record = json.loads(dead_files[0].read_text(encoding="utf-8"))
    assert record["flow"]["dead_letter_reason"] == "invalid_spool_record"


def test_register_declares_required_hooks_and_manifest_lists_them():
    plugin = _load_truth_plugin_init()
    ctx = _FakeCtx()
    plugin.register(ctx)
    assert "on_session_start" in ctx.hooks
    assert "post_llm_call" in ctx.hooks

    manifest_path = PLUGIN_DIR / "plugin.yaml"
    assert manifest_path.exists()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["name"] == "truth-ledger"
    assert "on_session_start" in manifest.get("hooks", [])
    assert "post_llm_call" in manifest.get("hooks", [])


def test_post_llm_call_is_fail_open_when_spool_fails(monkeypatch):
    plugin = _load_truth_plugin_init()
    plugin.register(_FakeCtx())

    def _boom(*_args, **_kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(plugin, "_enqueue_source_envelope", _boom)

    plugin.on_post_llm_call(
        session_id="s",
        turn_id="t",
        task_id="task",
        platform="cli",
        completed=True,
        failed=False,
        interrupted=False,
        turn_exit_reason="text_response(finish_reason=stop)",
        delegate_depth=0,
        is_subagent=False,
        kanban_task_id=None,
        speaker_id=None,
        conversation_id=None,
        thread_id=None,
        user_message="u",
        assistant_response="a",
    )


def test_post_llm_call_fail_open_on_enospc_at_payload_fsync_cleans_tmp_and_retries(monkeypatch):
    plugin = _load_truth_plugin_init()
    plugin.register(_FakeCtx(profile_name="automation-operator"))
    plugin._SEEN_ENVELOPES.clear()

    spool_mod = importlib.import_module("hermes_plugins.truth_ledger.spool")
    real_fsync = spool_mod.os.fsync
    state = {"raised": False}

    def _raise_enospc_once(fd):
        if not state["raised"]:
            state["raised"] = True
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_fsync(fd)

    monkeypatch.setattr(spool_mod.os, "fsync", _raise_enospc_once)

    payload = dict(
        session_id="sess-enospc",
        turn_id="turn-enospc",
        task_id="task-enospc",
        platform="cli",
        completed=True,
        failed=False,
        interrupted=False,
        turn_exit_reason="text_response(finish_reason=stop)",
        delegate_depth=0,
        is_subagent=False,
        kanban_task_id=None,
        speaker_id=None,
        conversation_id=None,
        thread_id=None,
        user_message="u",
        assistant_response="a",
    )

    plugin.on_post_llm_call(**payload)

    spool = plugin.TruthSpool(plugin._truth_root())
    assert list(spool.pending_dir.glob("*.json")) == []
    assert list(spool.payloads_dir.glob(".tmp-*.json")) == []
    assert ("automation-operator", "sess-enospc", "turn-enospc") not in plugin._SEEN_ENVELOPES

    plugin.on_post_llm_call(**payload)
    pending = list(spool.pending_dir.glob("*.json"))
    assert len(pending) == 1


def test_post_llm_call_is_fail_open_on_permission_denied_at_pending_atomic_replace(monkeypatch):
    plugin = _load_truth_plugin_init()
    plugin.register(_FakeCtx(profile_name="automation-operator"))
    plugin._SEEN_ENVELOPES.clear()

    spool_mod = importlib.import_module("hermes_plugins.truth_ledger.spool")
    real_replace = spool_mod.os.replace

    def _deny_pending_replace(src, dst):
        if Path(dst).parent.name == "pending":
            raise PermissionError("permission denied")
        return real_replace(src, dst)

    monkeypatch.setattr(spool_mod.os, "replace", _deny_pending_replace)

    plugin.on_post_llm_call(
        session_id="sess-perm",
        turn_id="turn-perm",
        task_id="task-perm",
        platform="cli",
        completed=True,
        failed=False,
        interrupted=False,
        turn_exit_reason="text_response(finish_reason=stop)",
        delegate_depth=0,
        is_subagent=False,
        kanban_task_id=None,
        speaker_id=None,
        conversation_id=None,
        thread_id=None,
        user_message="u",
        assistant_response="a",
    )

    spool = plugin.TruthSpool(plugin._truth_root())
    assert list(spool.pending_dir.glob("*.json")) == []
    assert ("automation-operator", "sess-perm", "turn-perm") not in plugin._SEEN_ENVELOPES


def test_post_llm_call_recovers_from_interruption_between_tmp_write_and_pending_replace(monkeypatch):
    plugin = _load_truth_plugin_init()
    plugin.register(_FakeCtx(profile_name="automation-operator"))
    plugin._SEEN_ENVELOPES.clear()

    spool_mod = importlib.import_module("hermes_plugins.truth_ledger.spool")
    real_replace = spool_mod.os.replace
    interrupted = {"done": False}

    def _interrupt_once(src, dst):
        if Path(dst).parent.name == "pending" and not interrupted["done"]:
            interrupted["done"] = True
            raise InterruptedError("simulated kill window before pending replace")
        return real_replace(src, dst)

    monkeypatch.setattr(spool_mod.os, "replace", _interrupt_once)

    payload = dict(
        session_id="sess-int",
        turn_id="turn-int",
        task_id="task-int",
        platform="cli",
        completed=True,
        failed=False,
        interrupted=False,
        turn_exit_reason="text_response(finish_reason=stop)",
        delegate_depth=0,
        is_subagent=False,
        kanban_task_id=None,
        speaker_id=None,
        conversation_id=None,
        thread_id=None,
        user_message="u",
        assistant_response="a",
    )

    plugin.on_post_llm_call(**payload)
    plugin.on_post_llm_call(**payload)

    spool = plugin.TruthSpool(plugin._truth_root())
    pending = list(spool.pending_dir.glob("*.json"))
    assert len(pending) == 1
    assert ("automation-operator", "sess-int", "turn-int") in plugin._SEEN_ENVELOPES


def test_post_llm_call_bounds_seen_dedupe_state(monkeypatch):
    plugin = _load_truth_plugin_init()
    plugin.register(_FakeCtx(profile_name="automation-operator"))

    monkeypatch.setattr(plugin, "_enqueue_source_envelope", lambda envelope: {"ok": True})
    plugin._SEEN_ENVELOPES.clear()

    for i in range(3000):
        plugin.on_post_llm_call(
            session_id="sess-bounded",
            turn_id=f"turn-{i}",
            task_id=f"task-{i}",
            platform="cli",
            completed=True,
            failed=False,
            interrupted=False,
            turn_exit_reason="text_response(finish_reason=stop)",
            delegate_depth=0,
            is_subagent=False,
            kanban_task_id=None,
            speaker_id="user-1",
            conversation_id="conv-1",
            thread_id="thread-1",
            user_message="remember this",
            assistant_response="ok",
        )

    assert len(plugin._SEEN_ENVELOPES) <= 1024


def test_realistic_synthetic_credential_never_leaks_across_spool_or_extractor_surfaces(tmp_path):
    plugin = _load_truth_plugin_init()
    plugin.register(_FakeCtx(profile_name="automation-operator"))

    raw_credential = "sk_live_SYNTHETIC_1234567890abcdef"
    safe_phrase = "keep replies concise"
    plugin._SEEN_ENVELOPES.clear()
    plugin.on_post_llm_call(
        session_id="sess-secret",
        turn_id="turn-secret",
        task_id="task-secret",
        platform="cli",
        completed=True,
        failed=False,
        interrupted=False,
        turn_exit_reason="text_response(finish_reason=stop)",
        delegate_depth=0,
        is_subagent=False,
        kanban_task_id=None,
        speaker_id=None,
        conversation_id=None,
        thread_id=None,
        user_message=f"remember {raw_credential} and {safe_phrase}",
        assistant_response=f"Acknowledged. I noted {raw_credential} and will {safe_phrase}.",
    )

    spool = plugin.TruthSpool(plugin._truth_root())
    pending_files = list(spool.pending_dir.glob("*.json"))
    assert len(pending_files) == 1
    pending_record = json.loads(pending_files[0].read_text(encoding="utf-8"))
    pending_text = json.dumps(pending_record)
    assert raw_credential not in pending_text

    pending_payload_text = Path(pending_record["payload_path"]).read_text(encoding="utf-8")
    assert raw_credential not in pending_payload_text
    assert safe_phrase in pending_payload_text

    claim = spool.claim_next(owner="worker-1")
    assert claim is not None
    processing_path = Path(claim["path"])
    processing_record = json.loads(processing_path.read_text(encoding="utf-8"))
    processing_text = json.dumps(processing_record)
    assert raw_credential not in processing_text

    processing_payload_text = Path(processing_record["payload_path"]).read_text(encoding="utf-8")
    assert raw_credential not in processing_payload_text
    assert safe_phrase in processing_payload_text

    extractor_mod = importlib.import_module("hermes_plugins.truth_ledger.extractor")
    blocks = extractor_mod._build_input_blocks(claim["envelope"])
    assert len(blocks) == 1
    assert raw_credential not in blocks[0].text
    assert safe_phrase in blocks[0].text

    retry = spool.retry_processing(processing_path, error_code="TEMP")
    retry_pending_path = Path(retry["path"])
    retry_pending_record = json.loads(retry_pending_path.read_text(encoding="utf-8"))
    retry_pending_text = json.dumps(retry_pending_record)
    assert raw_credential not in retry_pending_text

    retry_payload_text = Path(retry_pending_record["payload_path"]).read_text(encoding="utf-8")
    assert raw_credential not in retry_payload_text
    assert safe_phrase in retry_payload_text

    reclaim = spool.claim_next(owner="worker-2")
    assert reclaim is not None
    dead = spool.dead_letter(Path(reclaim["path"]), reason="permanent")
    dead_record = json.loads(Path(dead["path"]).read_text(encoding="utf-8"))
    dead_text = json.dumps(dead_record)
    assert raw_credential not in dead_text

    dead_payload_path = Path(dead_record["payload_path"])
    assert dead_payload_path.exists() is False

    strict = plugin.TruthSpool(tmp_path / "strict", soft_count=99, hard_count=1)
    env = plugin._build_source_envelope(
        {
            "session_id": "strict-session",
            "turn_id": "strict-turn-1",
            "platform": "cli",
            "conversation_id": None,
            "thread_id": None,
            "speaker_id": None,
            "user_message": f"remember {raw_credential} and {safe_phrase}",
            "assistant_response": "ok",
        },
        profile="automation-operator",
    )
    assert strict.enqueue(env)["ok"] is True
    env2 = dict(env)
    env2["turn_id"] = "strict-turn-2"
    reject = strict.enqueue(env2)
    assert reject["ok"] is False
    errors_text = (strict.errors_dir / "errors.jsonl").read_text(encoding="utf-8")
    assert raw_credential not in errors_text

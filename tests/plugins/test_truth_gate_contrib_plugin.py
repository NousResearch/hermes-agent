import importlib.util
import sys
from pathlib import Path
import tempfile

PLUGIN_DIR = Path(__file__).resolve().parents[2] / "contrib" / "plugins" / "truth_gate"


def load_plugin():
    spec = importlib.util.spec_from_file_location("contrib_truth_gate_plugin_under_test", PLUGIN_DIR / "__init__.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_vendor_hash_mismatch_blocks_transform_output():
    mod = load_plugin()
    setattr(mod, "_stop_mod", None)
    mod.SOURCE_HASHES["truth-stop-gate.py"] = "0" * 64
    try:
        out = mod.transform_llm_output("unsafe original text", session_id="hash-check")
        assert "TRUTH GATE BLOCK" in out
        assert "plugin unavailable" in out
        assert "unsafe original text" not in out
    finally:
        setattr(mod, "_stop_mod", None)
        mod.SOURCE_HASHES["truth-stop-gate.py"] = "cbe01769b2d45c3c31708433f9bf926d10edda3c7d269cfeb627224751d73548"


def test_vendor_hash_mismatch_fails_closed():
    mod = load_plugin()
    setattr(mod, "_stop_mod", None)
    mod.SOURCE_HASHES["truth-stop-gate.py"] = "0" * 64
    try:
        try:
            mod.validate_response("unproven final answer", session_id="hash-check")
        except RuntimeError as exc:
            assert "hash mismatch" in str(exc)
        else:
            raise AssertionError("expected hash mismatch RuntimeError")
    finally:
        setattr(mod, "_stop_mod", None)
        mod.SOURCE_HASHES["truth-stop-gate.py"] = "cbe01769b2d45c3c31708433f9bf926d10edda3c7d269cfeb627224751d73548"


def test_valid_footer_passes_unchanged_without_packet():
    mod = load_plugin()
    text = '''Tool result cited.\n\nTRUTH_PROVEN:\n| ID | Claim | Ledger anchor | Verified |\n|---|---|---|---|\n| P1 | tool_result content proof | contrib/plugins/truth_gate/__init__.py:1 | YES |\n\nTRUTH_PARTIAL:\n| ID | What proven | What not proven | Why partial | What closes it | Objective-critical |\n|---|---|---|---|---|---|\n| PT1 | plugin adapter shape | side-door runtime coverage | side doors intentionally no | future surface tests | NO |\n\nTRUTH_GAP:\n| ID | Gap | Fillable | Missing proof | Next read-test-action | Blocks PASS |\n|---|---|---|---|---|---|\n| G1 | nothing blocking | NO | none | none | NO |\n\nCURRENT_STATE:\n| Item | State | Proof |\n|---|---|---|\n| plugin | under test | tests/plugins/test_truth_gate_contrib_plugin.py |\n\nSTATE_NEXT:\n| State | Next | Owner | Proof |\n|---|---|---|---|\n| idle | await review | assistant | focused test |\n\nMETRICS GATE:\n| Metric | Required | Actual | Pass/Fail |\n|---|---:|---:|---|\n| GAPS_FILLED | 100% | 100% | PASS |\n| DISCOVERY | 100% | 100% | PASS |\n| BUILD_CONFIDENCE | >=95% | 100% | PASS |\n| METRICS_GATE | PASS only if all above pass | PASS | PASS |\n\nBEHAVIOR_FAIL:\n| ID | Failure | Proof | Blocks PASS |\n|---|---|---|---|\n| BF1 | none | no blocking behavior failure | NO |\n'''
    with tempfile.TemporaryDirectory() as td:
        state = Path(td)
        out = mod.transform_llm_output(response_text=text, session_id="valid", state_dir=str(state))
        assert out == text
        assert not any((state / "packets").glob("*.json"))


def test_invalid_answer_blocks_packet_without_retry_flag():
    mod = load_plugin()
    with tempfile.TemporaryDirectory() as td:
        state = Path(td)
        out = mod.transform_llm_output(
            response_text="I fixed it completely.",
            session_id="s1",
            model="m",
            platform="cli",
            state_dir=str(state),
        )
        assert "TRUTH GATE BLOCK" in out
        assert "evidence.schema.canonical-footer.always-required" in out
        assert any((state / "packets").glob("*.json"))
        assert not (state / "rewrite-required-flags" / "s1.json").exists()
        assert not (state / "inactive-correction.flag").exists()
        assert not (state / "inactive-correction-flags" / "s1.flag").exists()
        assert not (state / "inactive-correction-stuck.flag").exists()
        assert "same-session" not in out.lower()


def test_packet_contract_is_block_only():
    mod = load_plugin()
    with tempfile.TemporaryDirectory() as td:
        result = mod.validate_response("unproven final answer", session_id="s2", state_dir=td)
        packet = result.get("packet") or {}
        assert packet["enforcement_mode"] == "block_only"
        assert packet["correction_enabled"] is False
        assert "truth_gate_exact_auto_rewrite_parity" not in packet
        assert "same-session auto-rewrite" not in str(packet).lower()


def test_status_reports_front_door_yes_side_doors_no():
    mod = load_plugin()
    status = mod.get_status()
    assert status["front_door"]["agent_final_response"] == "yes"
    assert status["side_doors"]["raw_tool_stdout"] == "no"
    assert status["side_doors"]["no_agent_cron_stdout"] == "no"
    assert status["side_doors"]["direct_send_message"] == "no"
    assert status["side_doors"]["system_platform_messages"] == "no"
    assert status["trigger_metric"] == "front_door_yes_and_violation_count_gt_0"
    assert status["enforcement_mode"] == "block_only"
    assert status["correction_enabled"] is False

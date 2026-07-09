"""Tests for verify-hook.py."""
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(r"C:\Users\bbask\AppData\Local\hermes\hooks\verify-hook.py")


def run_hook(payload: dict) -> dict:
    """Run the hook with the given payload, return parsed JSON response."""
    r = subprocess.run(
        [sys.executable, str(SCRIPT)],
        input=json.dumps(payload),
        capture_output=True, text=True, timeout=10
    )
    return json.loads(r.stdout)


# --- contains_claim ---

def test_contains_claim_tests_pass():
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("vh", SCRIPT)
    vh = module_from_spec(spec)
    spec.loader.exec_module(vh)
    assert vh.contains_claim("All tests pass") is True


def test_contains_claim_all_checks_passed():
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("vh", SCRIPT)
    vh = module_from_spec(spec)
    spec.loader.exec_module(vh)
    assert vh.contains_claim("All checks passed") is True


def test_contains_claim_is_done():
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("vh", SCRIPT)
    vh = module_from_spec(spec)
    spec.loader.exec_module(vh)
    assert vh.contains_claim("Build is done") is True


def test_contains_claim_pr_opened():
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("vh", SCRIPT)
    vh = module_from_spec(spec)
    spec.loader.exec_module(vh)
    assert vh.contains_claim("PR opened at github.com/...") is True


def test_contains_claim_no_match():
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("vh", SCRIPT)
    vh = module_from_spec(spec)
    spec.loader.exec_module(vh)
    assert vh.contains_claim("Let me think about this") is False


# --- contains_verification ---

def test_contains_verification_marker():
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("vh", SCRIPT)
    vh = module_from_spec(spec)
    spec.loader.exec_module(vh)
    text = "Tests pass.\nVerified: pytest, output 85 passed"
    assert vh.contains_verification(text) is True


def test_contains_verification_exit_code():
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("vh", SCRIPT)
    vh = module_from_spec(spec)
    spec.loader.exec_module(vh)
    text = "Tests pass with exit code 0"
    assert vh.contains_verification(text) is True


def test_contains_verification_no_match():
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("vh", SCRIPT)
    vh = module_from_spec(spec)
    spec.loader.exec_module(vh)
    text = "Tests pass without any proof"
    assert vh.contains_verification(text) is False


# --- main() flow ---

def test_no_payload_returns_continue():
    r = run_hook({})
    assert r["action"] == "continue"


def test_empty_text_returns_continue():
    r = run_hook({"text": ""})
    assert r["action"] == "continue"


def test_short_text_skips_check():
    """Text < 80 chars skips the check entirely."""
    r = run_hook({"text": "Tests pass."})
    assert r["action"] == "continue"
    assert "message" not in r


def test_unverified_claim_triggers_followup():
    text = "All 22 tests pass and the install is healthy. No issues found anywhere in the codebase right now."
    r = run_hook({"text": text})
    assert r["action"] == "continue"
    assert "message" in r
    assert "verification" in r["message"].lower()


def test_verified_claim_passes():
    text = """All tests pass.
Verified: pytest tests/, output 85 passed in 2.5s
Done."""
    r = run_hook({"text": text})
    assert r["action"] == "continue"
    assert "message" not in r


def test_pr_opened_without_verification_blocked():
    text = "PR opened at github.com/foo/bar/pull/1 and ready for review by the team. The build succeeded and CI passed cleanly."
    r = run_hook({"text": text})
    assert "message" in r


def test_no_claim_no_action():
    text = "Let me think about how to approach this problem. There are several factors to consider."
    r = run_hook({"text": text})
    assert r["action"] == "continue"
    assert "message" not in r


def test_claim_with_verified_by_passes():
    text = "Tests pass. Verified by running pytest locally."
    r = run_hook({"text": text})
    assert r["action"] == "continue"
    assert "message" not in r


def test_claim_with_curl_output_passes():
    text = "Gateway healthy: curl http://127.0.0.1:9720/ returned 200"
    r = run_hook({"text": text})
    assert r["action"] == "continue"
    assert "message" not in r


def test_malformed_payload_doesnt_crash():
    """If payload can't be parsed, default to continue (don't block)."""
    r = subprocess.run(
        [sys.executable, str(SCRIPT)],
        input="not json at all",
        capture_output=True, text=True, timeout=10
    )
    parsed = json.loads(r.stdout)
    assert parsed["action"] == "continue"
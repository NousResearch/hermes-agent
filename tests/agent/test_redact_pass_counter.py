"""``PASS=174 FAIL=0`` test summaries stay readable; credential shapes mask.

A bare ``PASS`` key is a password env var for _ENV_ASSIGN_RE, so test-runner
summaries egressed as ``PASS=*** FAIL=0`` and reviewers read the masked count
as a hidden result. Only the counter shape is exempt (see
``agent.redact._is_test_pass_counter``); every credential-bearing near-miss
must still mask.
"""

import json

import pytest

from agent.redact import redact_sensitive_text


@pytest.fixture(autouse=True)
def _ensure_redaction_enabled(monkeypatch):
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


@pytest.mark.parametrize("text", [
    "test-fleet-land.sh: PASS=174 FAIL=0",
    "PASS=174 FAIL=0",
    "PASS=3 FAIL=1 SKIP=2",
    "FAIL=0 PASS=174",
    "summary PASS=12, FAIL=0",
    "PASS=5 FAILED=0",
    "PASS=9 ERRORS=0",
    "line one\nsuite: PASS=8 FAIL=0\nline three",
])
def test_counter_preserved(text):
    assert redact_sensitive_text(text) == text


@pytest.mark.parametrize("text,secret", [
    ("PASS=174", "174"),                      # lone numeric PASS may be a PIN
    ("export PASS=174", "174"),
    ("PASS=hunter2 FAIL=0", "hunter2"),
    ("PASS=174abc FAIL=0", "174abc"),
    ("PASS=174!@#$ FAIL=0", "174!@#$"),
    ("PASS='174' FAIL=0", "174"),
    ("PASS=1234567890 FAIL=0", "1234567890"),
    ("DB_PASS=174 FAIL=0", "174"),
    ("MYSQL_PASS=174 FAIL=0", "174"),
    ("PASSWORD=174 FAIL=0", "174"),
    ("PASSWD=174 FAIL=0", "174"),
    ("PASS=174 FAIL=zero", "174"),
    ("PASS=174 xFAIL=0", "174"),
    ("PASS=174 FAIL_COUNT=0", "174"),
    ("PASS=174 FAILURE_MODE=x", "174"),
    ("PASS=174\nFAIL=0", "174"),              # sibling on another line
])
def test_credential_shapes_still_masked(text, secret):
    out = redact_sensitive_text(text)
    assert "***" in out, out
    assert secret not in out.replace("FAIL=0", ""), out


def test_only_counter_survives_next_to_real_password():
    out = redact_sensitive_text("PASS=hunter2 PASS=174 FAIL=0")
    assert "hunter2" not in out
    assert out.endswith("PASS=174 FAIL=0"), out


@pytest.mark.parametrize("value", [
    "test-fleet-land.sh: PASS=174 FAIL=0",
    "suite: FAIL=0 PASS=174",               # counter flush against the closing quote
])
def test_counter_survives_kanban_metadata_json_egress(value):
    # kanban_tools: redact_sensitive_text(json.dumps(metadata), force=True)
    out = redact_sensitive_text(json.dumps({"tests": value}), force=True)
    assert json.loads(out) == {"tests": value}

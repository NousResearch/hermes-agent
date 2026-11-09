"""Telegram recognition starts at a credential boundary, not inside an identifier.

Every value here is synthetic and assembled at runtime; none is a live credential.
"""
import json
import sys

import pytest

from agent.redact import redact_for_egress, redact_sensitive_text, redact_terminal_output


def _redact(payload, mode):
    if mode == "terminal":
        return redact_terminal_output(payload, "python synthetic.py", force=True)
    if mode == "egress":
        return redact_for_egress(payload)
    return redact_sensitive_text(payload, force=True, code_file=mode == "code")


@pytest.mark.parametrize("mode", ["normal", "code", "terminal", "egress"])
@pytest.mark.parametrize("bot_prefix", ["", "bot"])
@pytest.mark.parametrize("left", [
    r"\n", r"\t", r"\r", r"\b", r"\f", r"\v", r"\x1b", r"\u0000",
    r"\u4ee4\u724c", r"\u0020", r"\U0001f600", "%0A", "%0a", "%09", "%20", "%3A",
    "https://example.invalid/?message=%0A",
])
def test_encoded_boundary_masks_token_without_rewriting_representation(left, bot_prefix, mode):
    token = bot_prefix + "987654321:" + "Qa_7-" * 7
    neighbor = "ghp_" + "y" * 36
    payload = left + token + " " + neighbor
    result = _redact(payload, mode)
    assert result.startswith(left + bot_prefix + "987654321:***")
    assert "Qa_7-" not in result
    assert neighbor not in result


@pytest.mark.parametrize("mode", ["normal", "terminal", "egress"])
@pytest.mark.parametrize("bot_prefix", ["", "bot"])
@pytest.mark.parametrize("message", ["\n", "\t", "凭据"])
def test_json_serialized_message_masks_token(message, bot_prefix, mode):
    token = bot_prefix + "987654321:" + "Qa_7-" * 7
    payload = json.dumps({"message": message + token})
    result = _redact(payload, mode)
    assert json.loads(result)["message"] == message + bot_prefix + "987654321:***"


@pytest.mark.parametrize("left", [r"\u0061", r"\u005f", r"\u0031", r"\u0661", r"\x41", "%41", "%5F", "%31"])
@pytest.mark.parametrize("bot_prefix", ["", "bot"])
def test_encoded_identifier_character_is_not_a_credential_boundary(left, bot_prefix):
    payload = left + bot_prefix + "987654321:" + "q-" + "z" * 32
    assert redact_sensitive_text(payload, force=True) == payload


@pytest.mark.parametrize("left", [r"\n", r"\u0020", "%0A"])
def test_encoded_boundary_long_digit_run_completes_fast(left):
    import time

    payload = "dump: " + left + "0" * 100_000 + "\n"
    start = time.perf_counter()
    assert redact_sensitive_text(payload, force=True) == payload
    assert time.perf_counter() - start < 2.0


def test_repr_message_masks_token_without_rewriting_representation():
    token = "987654321:" + "Qa_7-" * 7
    payload = repr({"message": "\n" + token})
    assert _redact(payload, "terminal") == repr({"message": "\n987654321:***"})


@pytest.mark.parametrize("prefix", ["a", "Z", "_", "job-a", "a0"])
@pytest.mark.parametrize("run_prefix", ["q-", "rv-", "prior-"])
def test_embedded_digit_suffix_preserves_business_identity(prefix, run_prefix):
    identity = "ITEM-000042:job-" + prefix + "12345678:" + run_prefix + "a" * 32
    payload = json.dumps({"research_pending": [{"key": identity}]})
    assert redact_sensitive_text(payload, force=True) == payload


@pytest.mark.parametrize("left", ["", " ", "\n", "\t", "\x00", "\x1b", "'", '"', "(", "=", "/", ":", "令牌：", "令牌", "https://api.telegram.org/"])
@pytest.mark.parametrize("bot_prefix", ["", "bot"])
def test_token_boundary_still_masks_token_and_neighbor(left, bot_prefix):
    token = bot_prefix + "123456789:" + "Ab_9-" * 7
    neighbor = "ghp_" + "x" * 36
    payload = left + token + " " + neighbor
    result = redact_sensitive_text(payload, force=True)
    assert result.startswith(left + bot_prefix + "123456789:***")
    assert "Ab_9-" not in result
    assert neighbor not in result


def test_cron_stdout_preserves_identity_and_masks_adjacent_token(tmp_path, monkeypatch):
    from cron.monitor import check_monitor

    home = tmp_path / "home"
    scripts = home / "scripts"
    scripts.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)
    identity = "ITEM-000042:job-a12345678:q-" + "a" * 32
    token = "123456789:" + "Ab_9-" * 7
    script = scripts / "synthetic_monitor.py"
    # A real child emits JSON; resolution, capture, redaction and context stay real.
    script.write_text("import json\nprint(json.dumps(" + repr({"key": identity, "credential": token, "message": "\n" + token}) + "))\n")
    snapshots = []
    # Only replace persistence: no cron job store or production monitor is touched.
    monkeypatch.setattr("cron.monitor._persist_monitor_state", lambda *args: snapshots.append(args))
    outcome = check_monitor({"id": "synthetic-monitor", "monitor_script": script.name, "interpreter": sys.executable})
    assert outcome.ok, outcome.error
    assert outcome.changed
    result = json.loads(snapshots[0][2])
    assert result["key"] == identity
    # The JSON key-name pass may mask "credential" first; either way the token body is gone.
    assert token not in result["credential"]
    assert result["message"] == "\n123456789:***"
    assert outcome.context_block is not None
    assert "### Current output" in outcome.context_block
    assert identity in outcome.context_block
    assert token not in outcome.context_block


@pytest.mark.parametrize("mode", ["normal", "code", "terminal", "egress"])
@pytest.mark.parametrize("color", ["\x1b[32m", "\x1b[1;32m", "\x1b[0m"])
def test_token_right_after_an_ansi_color_code_is_masked(color, mode):
    token = "987654321:" + "Qa_7-" * 7
    for payload in (color + token + "\x1b[0m", json.dumps(color + token)):
        result = _redact(payload, mode)
        assert "Qa_7-" not in result
        assert "987654321:***" in result

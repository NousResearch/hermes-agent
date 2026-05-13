"""Tests for provider/tool boundary error shaping (content filter, redaction)."""

import importlib
import json

from agent import provider_error_mapping as pem


def test_classify_openai_content_policy_body():
    class _E(Exception):
        pass

    exc = _E("wrap")
    exc.body = {
        "error": {
            "message": "oh no",
            "type": "invalid_request_error",
            "code": "content_policy_violation",
        }
    }
    assert pem.classify_provider_exception(exc) == "content_filter"


def test_safe_payload_omits_verbatim_on_filter():
    class _E(Exception):
        pass

    exc = _E("SHOULD_NOT_LEAK_USER_PROMPT")
    exc.body = {
        "error": {"code": "content_policy_violation", "message": "verbose provider text"}
    }
    p = pem.safe_tool_error_payload(exc)
    assert p.get("error_code") == "content_filter"
    assert "SHOULD_NOT_LEAK" not in p["error"]
    assert "details omitted" in p["error"].lower()


def test_format_tool_boundary_error_is_json_with_keys():
    err = RuntimeError("plain failure")
    raw = pem.format_tool_boundary_error("web_search", err)
    d = json.loads(raw)
    assert "error" in d
    assert "web_search" in d["error"]


def test_save_trajectory_redacts_sk_like_prefix(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_REDACT_SECRETS", "true")
    import agent.redact as redact_mod

    importlib.reload(redact_mod)

    from agent.trajectory import save_trajectory

    long_key = "sk-" + ("a" * 40)
    traj = [
        {"role": "user", "content": f"tell me about {long_key}"},
    ]
    fn = tmp_path / "sample.jsonl"
    save_trajectory(traj, model="gpt-test", completed=True, filename=str(fn))
    line = fn.read_text(encoding="utf-8").strip()
    parsed = json.loads(line)
    blob = json.dumps(parsed, ensure_ascii=False)
    assert long_key not in blob
    assert "***" in blob or "..." in blob

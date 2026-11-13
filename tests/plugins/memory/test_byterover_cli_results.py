"""ByteRover's JSON result, not only its exit code, determines success."""

import json
import logging
from types import SimpleNamespace

import pytest

import plugins.memory.byterover as brv


@pytest.mark.parametrize("payload", [
    {"success": False, "data": {"message": "Query argument is required.", "status": "error"}},
    {"success": False, "data": {"event": "error", "message": "Provider unavailable"}},
    {"data": {"event": "error", "message": "Provider unavailable"}},
])
def test_zero_exit_json_error_reaches_tools_and_skips_prefetch(monkeypatch, tmp_path, payload):
    calls = []
    output = json.dumps({"success": True, "data": {"event": "started"}}) + "\n" + json.dumps(payload)

    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(brv, "_resolve_brv_path", lambda: "/fake/brv")
    monkeypatch.setattr(brv, "_brv_child_env", lambda _: {})
    monkeypatch.setattr(brv.subprocess, "run", run)
    provider = brv.ByteRoverMemoryProvider({})
    provider.initialize("test", hermes_home=tmp_path)
    for tool, args in [("brv_query", {"query": "How does authentication work?"}),
                       ("brv_curate", {"content": "A substantive memory"}),
                       ("brv_status", {})]:
        result = json.loads(provider.handle_tool_call(tool, args))
        assert "error" in result
        assert payload["data"]["message"] in result["error"]
    assert provider.prefetch("How does authentication work?") == ""
    assert calls[0][1:4] == ["query", "--format", "json"]
    assert calls[1][1:4] == ["curate", "--format", "json"]
    assert "--" in calls[0]


@pytest.mark.parametrize("returncode,stdout,stderr,expected", [
    (0, "ordinary legacy text", "", {"success": True, "output": "ordinary legacy text"}),
    (0, '[1, 2]\nnull\n{"data": null}', "", {"success": True, "output": '[1, 2]\nnull\n{"data": null}'}),
    (0, '{"success": true, "data": {"event": "completed", "result": "Useful remembered context."}}', "",
     {"success": True, "output": '{"success": true, "data": {"event": "completed", "result": "Useful remembered context."}}'}),
    (1, "failure output", "failure stderr", {"success": False, "error": "failure stderr"}),
])
def test_success_and_nonzero_exit_contract(monkeypatch, tmp_path, returncode, stdout, stderr, expected):
    monkeypatch.setattr(brv, "_resolve_brv_path", lambda: "/fake/brv")
    monkeypatch.setattr(brv, "_brv_child_env", lambda _: {})
    monkeypatch.setattr(brv.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=returncode, stdout=stdout, stderr=stderr))
    assert brv._run_brv(["status"], cwd=str(tmp_path)) == expected


@pytest.mark.parametrize("text", ["", "Useful remembered context, including literal success: false in quoted notes.", "x" * 9000])
def test_json_query_completion_and_failed_background_curate(monkeypatch, tmp_path, caplog, text):
    output = json.dumps({"success": True, "data": {"event": "started"}}) + "\n" + json.dumps(
        {"success": True, "data": {"event": "completed", "result": text}})
    monkeypatch.setattr(brv, "_run_brv", lambda *a, **kw: {"success": True, "output": output})
    provider = brv.ByteRoverMemoryProvider({})
    provider.initialize("test", hermes_home=tmp_path)
    expected = text[:8000] + "\n\n[... truncated]" if len(text) > 8000 else text or "No relevant memories found."
    assert json.loads(provider.handle_tool_call("brv_query", {"query": "A useful question"})) == {"result": expected}
    assert provider.prefetch("A useful question") == ("## ByteRover Context\n" + text if text else "")
    monkeypatch.setattr(brv, "_run_brv", lambda *a, **kw: {"success": False, "error": "Provider unavailable"})
    with caplog.at_level(logging.DEBUG, logger=brv.__name__):
        thread = provider._curate_in_background("A memory", name="test-brv", what="test", on_done="Saved memory")
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert "Saved memory" not in caplog.text
    assert "failed" in caplog.text

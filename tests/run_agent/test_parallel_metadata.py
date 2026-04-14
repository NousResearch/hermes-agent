from types import SimpleNamespace

import pytest

import run_agent


def _mock_tool_call(name: str, arguments: str = "{}", call_id: str = "tc_1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _patch_metadata_lookup(monkeypatch, mapping):
    monkeypatch.setattr(
        run_agent.registry,
        "get_metadata",
        lambda name: mapping.get(name, {}),
    )


def test_metadata_true_allows_parallel_candidate_outside_legacy_set(monkeypatch):
    _patch_metadata_lookup(
        monkeypatch,
        {"meta_safe": {"parallel_safe_default": True}},
    )
    tc1 = _mock_tool_call(name="meta_safe", arguments='{"q":"one"}', call_id="c1")
    tc2 = _mock_tool_call(name="meta_safe", arguments='{"q":"two"}', call_id="c2")

    assert run_agent._should_parallelize_tool_batch([tc1, tc2]) is True


def test_metadata_true_does_not_bypass_path_conflict_runtime_guard(tmp_path, monkeypatch):
    _patch_metadata_lookup(
        monkeypatch,
        {"write_file": {"parallel_safe_default": True}},
    )
    monkeypatch.chdir(tmp_path)
    tc1 = _mock_tool_call(
        name="write_file",
        arguments='{"path":"notes.txt","content":"one"}',
        call_id="c1",
    )
    tc2 = _mock_tool_call(
        name="write_file",
        arguments=f'{{"path":"{tmp_path / "notes.txt"}","content":"two"}}',
        call_id="c2",
    )

    assert run_agent._should_parallelize_tool_batch([tc1, tc2]) is False


def test_metadata_false_blocks_legacy_parallel_safe_tool(monkeypatch):
    assert "web_search" in run_agent._PARALLEL_SAFE_TOOLS
    _patch_metadata_lookup(
        monkeypatch,
        {"web_search": {"parallel_safe_default": False}},
    )
    tc1 = _mock_tool_call(name="web_search", arguments='{"q":"one"}', call_id="c1")
    tc2 = _mock_tool_call(name="web_search", arguments='{"q":"two"}', call_id="c2")

    assert run_agent._should_parallelize_tool_batch([tc1, tc2]) is False


@pytest.mark.parametrize(
    "metadata",
    [{}, {"parallel_safe_default": None}],
    ids=["missing", "none"],
)
def test_missing_parallel_metadata_falls_back_to_legacy_allowlist(metadata, monkeypatch):
    assert "web_search" in run_agent._PARALLEL_SAFE_TOOLS
    _patch_metadata_lookup(
        monkeypatch,
        {"web_search": metadata},
    )
    tc1 = _mock_tool_call(name="web_search", arguments='{"q":"one"}', call_id="c1")
    tc2 = _mock_tool_call(name="web_search", arguments='{"q":"two"}', call_id="c2")

    assert run_agent._should_parallelize_tool_batch([tc1, tc2]) is True


def test_tool_not_in_candidate_set_cannot_parallelize(monkeypatch):
    _patch_metadata_lookup(monkeypatch, {})
    tc1 = _mock_tool_call(name="terminal", arguments='{"command":"pwd"}', call_id="c1")
    tc2 = _mock_tool_call(name="terminal", arguments='{"command":"ls"}', call_id="c2")

    assert run_agent._should_parallelize_tool_batch([tc1, tc2]) is False

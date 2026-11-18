"""Reading a prune archive does not create another copy of its rendered output."""

import json

import pytest

from tests.agent.test_proactive_prune_archive import _archive_path, _compressor, _round


def _readback_messages(arguments):
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "read"}]
    messages += _round(0, "1|original output\n" + "x" * 12_000)
    messages[2]["tool_calls"][0]["function"]["arguments"] = arguments
    for i in range(1, 8):
        messages += _round(i, "ok")
    return messages + [{"role": "user", "content": "continue"}]


def _own_archive(home):
    archive = home / "cache" / "pruned" / "x.json"
    archive.parent.mkdir(parents=True)
    archive.write_text('{"output": "original"}')
    return archive


@pytest.mark.parametrize("path_style", ["absolute", "tilde"])
def test_archive_readback_reuses_the_original_pointer(tmp_path, monkeypatch, path_style):
    """A read-back of a file in this home's cache/pruned points at that file again: no second copy."""
    home = tmp_path / "home" / "hermes"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    archive = _own_archive(home)
    path = str(archive) if path_style == "absolute" else "~/hermes/cache/pruned/x.json"
    messages = _readback_messages(json.dumps({"path": path}))
    result, count = _compressor().prune_tool_results_only(messages, current_tokens=100_000)
    assert count == 1
    assert _archive_path(result[3]["content"]) == archive
    assert list(archive.parent.iterdir()) == [archive]
    assert archive.read_text() == '{"output": "original"}'


@pytest.mark.parametrize("case", ["lookalike_elsewhere", "missing_in_archive_dir", "relative", "nul"])
def test_a_path_that_is_not_an_existing_archive_is_archived_normally(tmp_path, monkeypatch, case):
    """Only an existing file in this home's cache/pruned counts; anything else keeps its own archive."""
    home = tmp_path / "home" / "hermes"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    if case == "lookalike_elsewhere":
        other = tmp_path / "work" / "cache" / "pruned" / "x.json"
        other.parent.mkdir(parents=True)
        other.write_text("not an archive")
        path = str(other)
    elif case == "missing_in_archive_dir":
        path = str(home / "cache" / "pruned" / "gone.json")
    elif case == "nul":
        path = str(home / "cache" / "pruned" / "x\x00.json")
    else:
        path = "cache/pruned/x.json"
    messages = _readback_messages(json.dumps({"path": path}))
    result, count = _compressor().prune_tool_results_only(messages, current_tokens=100_000)
    assert count == 1
    archives = list((home / "cache" / "pruned").iterdir())
    assert len(archives) == 1
    assert archives[0].read_text() == messages[3]["content"]
    assert _archive_path(result[3]["content"]) == archives[0]

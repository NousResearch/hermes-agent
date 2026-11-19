"""A sandboxed terminal backend must keep a pruned result whole.

The prune has no environment to sync an archive into and ``cache/pruned`` is not mounted, so on
those backends a stub would name a host-only file the model cannot open. These tests use the real
path translation (``to_agent_visible_cache_path`` is not mocked), so they see what the active
backend actually gets.
"""

import pytest

from tools.tool_result_storage import write_pruned_original

_BODY = "original body " * 200


@pytest.mark.parametrize("backend", ["docker", "modal", "ssh", "daytona", "vercel_sandbox"])
def test_sandboxed_backend_gets_no_archive_and_no_stub_path(tmp_path, monkeypatch, backend):
    """On a sandboxed backend the writer returns None, so the caller keeps the result whole."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_ENV", backend)

    assert write_pruned_original(_BODY, "s1", "read_file", "call_1") is None
    pruned = tmp_path / "cache" / "pruned"
    assert not pruned.exists() or not any(pruned.iterdir())


def test_local_backend_still_archives(tmp_path, monkeypatch):
    """The local backend reads host paths, so the original is archived and the stub can name it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_ENV", "local")

    archived = write_pruned_original(_BODY, "s1", "read_file", "call_1")

    assert archived is not None
    assert any((tmp_path / "cache" / "pruned").iterdir())

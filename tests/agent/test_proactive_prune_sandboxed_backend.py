"""A pruned result's stub must name a file the model's own ``read_file`` can open.

``cache/pruned`` is published like ``cache/spillover``: docker bind-mounts it, modal/ssh/daytona/vercel_sandbox
file-sync it. On those backends the archive is written host-side and the stub names the sandbox path; on a
backend that does not publish the folder (a plugin, an unknown name, a hosted scope) the result stays whole.
These tests use the real path translation (``to_agent_visible_cache_path`` is not mocked).
"""

import pytest

from tools.credential_files import get_cache_directory_mounts
from tools.tool_result_storage import archive_compaction_drops, write_pruned_original

_BODY = "original body " * 200


@pytest.mark.parametrize(("backend", "visible_root"), [
    ("docker", "/root/.hermes/cache/pruned/"), ("modal", "/root/.hermes/cache/pruned/"),
    ("ssh", "~/.hermes/cache/pruned/"), ("daytona", "~/.hermes/cache/pruned/"),
    ("vercel_sandbox", "~/.hermes/cache/pruned/")])
def test_publishing_backend_archives_and_names_the_sandbox_path(tmp_path, monkeypatch, backend, visible_root):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_ENV", backend)

    archived = write_pruned_original(_BODY, "s1", "read_file", "call_1")

    assert archived is not None
    host_path, stub_path = archived
    assert open(host_path, encoding="utf-8").read() == _BODY
    assert stub_path.startswith(visible_root) and stub_path.endswith(host_path.rsplit("/", 1)[1])
    assert str(tmp_path) not in stub_path  # never the host-only path


def test_the_folder_is_published_with_the_other_cache_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    mounts = {m["container_path"]: m["host_path"] for m in get_cache_directory_mounts()}
    assert mounts["/root/.hermes/cache/pruned"] == str(tmp_path / "cache" / "pruned")


@pytest.mark.parametrize("backend", ["singularity-plugin", "some_future_backend"])
def test_backend_that_does_not_publish_the_folder_keeps_results_whole(tmp_path, monkeypatch, backend):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_ENV", backend)

    assert write_pruned_original(_BODY, "s1", "read_file", "call_1") is None
    original = [{"role": "tool", "content": _BODY}]
    replacement = [{"role": "tool", "content": "shorter"}]
    assert archive_compaction_drops(original, replacement, "s1") == (replacement, [], 0)
    pruned = tmp_path / "cache" / "pruned"
    assert not pruned.exists() or not any(pruned.iterdir())


def test_hosted_scope_keeps_results_whole(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setattr("tools.environments.sandbox_scope.hosted_scope", lambda: object())

    assert write_pruned_original(_BODY, "s1", "read_file", "call_1") is None


def test_local_backend_still_archives_with_the_display_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_ENV", "local")

    archived = write_pruned_original(_BODY, "s1", "read_file", "call_1")

    assert archived is not None
    host_path, stub_path = archived
    assert open(host_path, encoding="utf-8").read() == _BODY
    assert stub_path.endswith("/cache/pruned/" + host_path.rsplit("/", 1)[1])


def test_read_file_pushes_a_synced_backend_before_opening_an_archive(tmp_path, monkeypatch):
    """An archive written since the last (rate-limited) sync must reach the sandbox before read_file opens it."""
    import json

    import tools.file_tools as ft

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    archived = write_pruned_original(_BODY, "s1", "read_file", "call_1")
    assert archived is not None
    forced = []

    class Sync:
        def sync(self, *, force=False):
            forced.append(force)

    real_ops = ft._get_file_ops("sync-task")
    real_env = real_ops.env
    real_env._sync_manager = Sync()  # the local env with a sync manager attached, as a synced backend has
    try:
        out = json.loads(ft.read_file_tool(archived[0], task_id="sync-task"))
    finally:
        del real_env._sync_manager

    assert forced == [True]
    assert "original body" in out.get("content", ""), out


def test_sandbox_archive_is_read_at_the_path_the_sandbox_names(tmp_path, monkeypatch):
    """On daytona the stub names ``~/.hermes/...``: the sandbox's ``~``, not the host's. The byte read must get that
    path, or a one-line JSON archive falls through to the ordinary reader and comes back cut at the per-line cap."""
    import base64
    import json

    import tools.file_tools as ft
    from tools.file_operations import ShellFileOperations
    from tools.file_operations_common import ReadResult
    from tools.tool_output_limits import get_max_line_length

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_ENV", "daytona")
    monkeypatch.setenv("HOME", str(tmp_path / "host_home"))  # the host's ~ is not the sandbox's
    body = json.dumps({"output": "x" * (get_max_line_length() * 3)})
    archived = write_pruned_original(body, "s1", "terminal", "call_1")
    assert archived is not None
    host_path, stub_path = archived
    assert stub_path.startswith("~/.hermes/")
    raw = open(host_path, "rb").read()
    requested = []

    class SandboxOps:
        env = object()  # no uses_host_filesystem declaration: not the host filesystem
        _add_line_numbers = ShellFileOperations._add_line_numbers

        def read_file_bytes(self, path, max_bytes=None):
            requested.append(path)
            if path != stub_path:
                return ReadResult(error=f"File not found: {path}", not_found=True)
            return ReadResult(base64_content=base64.b64encode(raw).decode(), file_size=len(raw))

        def read_file(self, path, offset=1, limit=2000):  # the ordinary reader: the line cut at the cap
            return ReadResult(content=self._add_line_numbers(body, offset), total_lines=1, file_size=len(raw))

    monkeypatch.setattr(ft, "_get_file_ops", lambda task_id="default": SandboxOps())

    out = json.loads(ft.read_file_tool(stub_path, task_id="sandbox-task"))

    assert requested == [stub_path], out
    assert out.get("long_lines_wrapped_at"), out
    assert "[truncated]" not in out["content"]

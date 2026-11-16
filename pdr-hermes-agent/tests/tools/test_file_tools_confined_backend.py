"""A backend that confines execution keeps the file tools inside it.

The file tools answer reads, writes, patches and searches on the host when the terminal backend runs
on the host. A backend that subclasses ``LocalEnvironment`` but confines its commands (a bubblewrap
sandbox: bind mounts, read-only paths, its own ``$TMPDIR``) sets ``uses_host_filesystem = False``, and
then every file tool goes through ``env.execute`` and sees what the sandbox sees.

The test backend exposes a real directory under a second path, the way a bind mount does: the view
path exists only for commands it runs, so a tool result that names it came through ``execute``.
"""

import json

import pytest

from tools import terminal_tool
from tools.environments.local import LocalEnvironment
from tools.file_tools import clear_file_ops_cache
from tools.registry import registry

# The host fast path is POSIX-only (``_native_read_enabled``); Windows always takes the shell path.
pytestmark = pytest.mark.platforms("posix")


class _MountedViewEnv(LocalEnvironment):
    """Commands see ``real`` at ``view``; the host sees only ``real``."""

    def __init__(self, view: str, real: str):
        super().__init__(cwd=real, timeout=30, persistent=False)
        self._view, self._real = view, real

    def execute(self, command, cwd="", **kwargs):
        result = super().execute(command.replace(self._view, self._real),
                                 cwd=cwd.replace(self._view, self._real), **kwargs)
        result["output"] = result.get("output", "").replace(self._real, self._view)
        return result


class _ConfinedEnv(_MountedViewEnv):
    uses_host_filesystem = False


def _dispatch(name, task_id, **arguments):
    return json.loads(registry.dispatch(name, arguments, task_id=task_id))


@pytest.fixture
def mounted(tmp_path, monkeypatch, request):
    """``(env, view, real, task_id)`` with the env active for the task; ``a.txt`` holds two lines."""
    monkeypatch.delenv("HERMES_NATIVE_FILE_READ", raising=False)  # the fast path at its default: on
    view, real = tmp_path / "view", tmp_path / "real"
    real.mkdir()
    (real / "a.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    env = request.param(str(view), str(real))
    task_id = f"confined-{request.node.name}"
    monkeypatch.setitem(terminal_tool._active_environments, terminal_tool._resolve_container_task_id(task_id), env)
    yield env, view, real, task_id
    clear_file_ops_cache()
    env.cleanup()


@pytest.mark.parametrize("mounted", [_ConfinedEnv], indirect=True)
def test_confined_backend_runs_every_file_tool_through_execute(mounted):
    _env, view, real, task = mounted

    read = _dispatch("read_file", task, path=str(view / "a.txt"))
    assert "alpha" in read.get("content", ""), read

    written = _dispatch("write_file", task, path=str(view / "b.txt"), content="gamma\n")
    assert "error" not in written, written
    assert (real / "b.txt").read_text(encoding="utf-8") == "gamma\n"

    patched = _dispatch("patch", task, path=str(view / "a.txt"), old_string="beta", new_string="delta")
    assert "error" not in patched, patched
    assert (real / "a.txt").read_text(encoding="utf-8") == "alpha\ndelta\n"

    found = _dispatch("search_files", task, pattern="delta", path=str(view))
    assert [m["path"] for m in found.get("matches", [])] == [str(view / "a.txt")], found

    assert not view.exists(), "a file tool touched the host path instead of the sandbox"


@pytest.mark.parametrize("mounted", [_MountedViewEnv], indirect=True)
def test_host_backend_keeps_the_host_fast_path(mounted):
    _env, view, real, task = mounted

    read = _dispatch("read_file", task, path=str(view / "a.txt"))
    assert "alpha" not in json.dumps(read), read

    written = _dispatch("write_file", task, path=str(view / "b.txt"), content="gamma\n")
    assert "error" not in written, written
    assert (view / "b.txt").read_text(encoding="utf-8") == "gamma\n"
    assert not (real / "b.txt").exists()

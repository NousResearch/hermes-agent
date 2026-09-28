"""The post-swap child must start in the checkout, not the caller's cwd.

A stale setuptools finder cannot import a new top-level package. ``python -m``
only adds the process cwd to ``sys.path``, so a hand-off spawned from outside
the checkout dies before the dependency sync can refresh the map.
"""

import os
import subprocess
import sys
import textwrap
import time

import hermes_cli.update_handoff as handoff


FINDER_SRC = textwrap.dedent(
    """\
    import sys
    from importlib.machinery import PathFinder
    from pathlib import Path

    MAPPING = {"oldpkg": %r}

    class _Finder:
        @classmethod
        def find_spec(cls, fullname, path, target=None):
            root = MAPPING.get(fullname.split(".", 1)[0])
            if root is None:
                return None
            return PathFinder.find_spec(fullname, [str(Path(root).parent)])

    def install():
        if _Finder not in sys.meta_path:
            sys.meta_path.insert(0, _Finder)
    """
)

CHILD_SRC = textwrap.dedent(
    """\
    import importlib.util
    import os
    import pathlib

    spec = importlib.util.spec_from_file_location(
        "stale_finder", os.environ["HANDOFF_FINDER"])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.install()
    import oldpkg
    pathlib.Path(os.environ["HANDOFF_MARKER"]).write_text("sync:" + oldpkg.MARKER)
    """
)


class _Child:
    def wait(self, timeout=None):
        return 0


def _layout(tmp_path):
    checkout = tmp_path / "checkout"
    outside = tmp_path / "outside"
    outside.mkdir()
    oldpkg = checkout / "oldpkg"
    newpkg = checkout / "handoff_newpkg"
    oldpkg.mkdir(parents=True)
    newpkg.mkdir()
    (oldpkg / "__init__.py").write_text(
        "import handoff_newpkg\nMARKER = handoff_newpkg.MARKER\n",
        encoding="utf-8",
    )
    (newpkg / "__init__.py").write_text("MARKER = 'sync'\n", encoding="utf-8")
    finder = tmp_path / "stale_finder.py"
    finder.write_text(FINDER_SRC % str(oldpkg), encoding="utf-8")
    return checkout, outside, finder


def _child_env(finder, marker):
    env = os.environ.copy()
    env["HANDOFF_FINDER"] = str(finder)
    env["HANDOFF_MARKER"] = str(marker)
    env["PYTHONPATH"] = ""
    env.pop("PYTHONSAFEPATH", None)
    return env


def _run(cmd_cwd, env):
    return subprocess.run(
        [sys.executable, "-c", CHILD_SRC],
        cwd=cmd_cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_both_handoff_launches_use_the_checkout_cwd(monkeypatch, tmp_path):
    recorded = []
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    assert handoff._post_swap_cwd() == str(handoff.Path(handoff.__file__).resolve().parent.parent)

    monkeypatch.setattr(handoff, "write_handoff", lambda payload: tmp_path / "handoff.json")
    monkeypatch.setattr(handoff, "post_swap_python", lambda: sys.executable)
    monkeypatch.setattr(handoff, "post_swap_child_env", lambda: {"HERMES_UPDATE_POST_SWAP": "1"})
    monkeypatch.setattr(handoff, "detached_shim_child_env", lambda env: env)
    monkeypatch.setattr(handoff, "_post_swap_cwd", lambda: str(checkout))
    monkeypatch.setattr(
        handoff.subprocess,
        "Popen",
        lambda cmd, **kwargs: recorded.append(kwargs.get("cwd")) or _Child(),
    )

    monkeypatch.setattr(handoff, "_running_from_windows_shim", lambda: False)
    assert handoff.continue_update_in_fresh_interpreter({}, argv_tail=["--yes"]) == 0

    monkeypatch.setattr(handoff, "_running_from_windows_shim", lambda: True)
    assert handoff.continue_update_in_fresh_interpreter({}, argv_tail=["--yes"]) == 0

    assert recorded == [str(checkout), str(checkout)]


def test_outside_cwd_child_reaches_the_new_package(monkeypatch, tmp_path):
    """Parent cwd is outside; the stale map omits the new package; the child still imports it."""
    checkout, outside, finder = _layout(tmp_path)
    marker = tmp_path / "sync.txt"
    env = _child_env(finder, marker)
    monkeypatch.chdir(outside)

    direct = _run(outside, env)
    assert direct.returncode != 0
    assert "handoff_newpkg" in (direct.stderr or "")
    assert not marker.exists()

    inside = _run(checkout, env)
    assert inside.returncode == 0
    assert marker.read_text(encoding="utf-8") == "sync:sync"
    marker.unlink()

    monkeypatch.setattr(handoff, "write_handoff", lambda payload: tmp_path / "handoff.json")
    monkeypatch.setattr(handoff, "post_swap_command", lambda path, tail: [sys.executable, "-c", CHILD_SRC])
    monkeypatch.setattr(handoff, "post_swap_child_env", lambda: env)
    monkeypatch.setattr(handoff, "detached_shim_child_env", lambda child_env: child_env)
    monkeypatch.setattr(handoff, "_post_swap_cwd", lambda: str(checkout))
    monkeypatch.setattr(handoff, "_running_from_windows_shim", lambda: False)
    monkeypatch.setattr(sys, "stdin", open(os.devnull, "r"))

    assert handoff.continue_update_in_fresh_interpreter({}, argv_tail=[]) == 0
    assert marker.read_text(encoding="utf-8") == "sync:sync"
    marker.unlink()

    monkeypatch.setattr(handoff, "_running_from_windows_shim", lambda: True)
    assert handoff.continue_update_in_fresh_interpreter({}, argv_tail=[]) == 0
    for _ in range(50):
        if marker.exists():
            break
        time.sleep(0.1)
    assert marker.read_text(encoding="utf-8") == "sync:sync"

"""A store root spelled in Windows' extended-length form (``\\\\?\\C:\\...``) stays inside PM.

The spelling exists only so PM's own file calls beneath the root pass MAX_PATH. Text that
leaves PM must carry the ordinary spelling: the same folder spelled two ways hashes and
compares differently, and child tools (cmd.exe, gpg, installers) may refuse the prefix.
The roots below are plain text, so both invariants hold on every OS.
"""
import subprocess
from pathlib import Path

from pm import environment as environment_module
from pm.environment import PythonEnvironment
from pm.lock import Facts
from pm.package import compose_env
from pm.runtime import _inputs
from pm.store import Store

VERBATIM = "\\\\?\\"
ROOT = Path(VERBATIM + r"C:\Program Files\WindowsApps\Hermes\agent-payload\tools")


def test_identity_does_not_depend_on_the_root_spelling(tmp_path):
    for name in ("pyproject.toml", "uv.lock"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    plain = Path(r"C:\Program Files\WindowsApps\Hermes\agent-payload\tools\python-3.12\python.exe")

    assert _inputs(tmp_path, Path(VERBATIM + str(plain))) == _inputs(tmp_path, plain)


def test_no_verbatim_spelling_leaves_through_records_environment_or_uv(tmp_path, monkeypatch):
    store = Store(ROOT)
    facts = Facts(tmp_path / "facts.json")
    facts.record("git", "1", "git-1", {"PATH": [str(store.entry("git-1") / "cmd")]}, store.root)
    facts.record("chromium", "1", "chromium-1",
                 {"AGENT_BROWSER_EXECUTABLE_PATH": str(store.entry("chromium-1") / "chrome.exe")}, store.root)
    record = (tmp_path / "facts.json").read_text(encoding="utf-8")

    env = compose_env([facts.env_for("git", store.root), facts.env_for("chromium", store.root)], base={})

    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(environment_module.subprocess, "run", run)
    PythonEnvironment(uv=store.entry("uv-1") / "uv.exe", python=store.entry("python-1") / "python.exe",
                      destination=tmp_path / "venv", cache=tmp_path / "cache",
                      env={}).sync(store.entry("source"))
    (command, kwargs), = calls

    leaving = [record, *env.values(), *command, kwargs["cwd"], *kwargs["env"].values()]
    assert [text for text in leaving if VERBATIM in text] == []
    assert env["AGENT_BROWSER_EXECUTABLE_PATH"].endswith("chrome.exe")

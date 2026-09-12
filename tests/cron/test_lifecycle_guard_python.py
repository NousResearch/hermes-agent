"""Python data is not executable source; literal process descendants still are."""

import json
import shlex
from pathlib import Path

import pytest

import cron.lifecycle_guard as guard


class SyntheticRemote:
    """Serve only bounded reader requests; never execute the inspected commands."""

    def __init__(self):
        self.files = {}
        self.reads = []

    def execute(self, command):
        words = shlex.split(command)
        assert words[:2] == ["head", "-c"] and words[3] == "<"
        limit = int(words[2])
        assert 0 < limit <= guard._MAX_REFERENCED_SCRIPT_BYTES + 1
        path = Path(words[4])
        self.reads.append(path)
        data = self.files.get(path, "").encode("utf-8")[:limit]
        return {"returncode": 0, "output": data.decode("utf-8", errors="replace")}


@pytest.mark.parametrize("backend", ["local", "remote"])
@pytest.mark.parametrize("invocation", [
    "./report",
    "python3 report",
    "python3 --check-hash-based-pycs always report",
    "python3 --check-hash-based-pycs never report",
    "python3 --check-hash-based-pycs default report",
])
@pytest.mark.parametrize("data_size", [32, 2 * guard._MAX_REFERENCED_SCRIPT_BYTES])
def test_python_data_does_not_consume_script_budget(
    tmp_path, monkeypatch, backend, invocation, data_size,
):
    from tools import process_registry
    from tools.terminal_tool_guards import gateway_lifecycle_block

    home = tmp_path / "home"
    scripts = home / "scripts"
    scripts.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: True)
    data = scripts / "report.json"
    data.write_text(json.dumps({"value": "x" * data_size}), encoding="utf-8")
    source = (
        "#!/usr/bin/env python3\nimport json\nfrom pathlib import Path\n"
        f"report = json.loads(Path({str(data)!r}).read_text())\n"
    )
    remote = SyntheticRemote() if backend == "remote" else None
    cwd = tmp_path / "remote" if remote else scripts
    wrapper = cwd / "report"
    if remote:
        remote.files[wrapper] = source
    else:
        wrapper.write_text(source, encoding="utf-8")
        wrapper.chmod(0o755)
    reads = []
    real_read = guard._read_referenced_script

    def read(path, *, max_bytes=None):
        reads.append(path)
        return real_read(path, max_bytes=max_bytes)

    def verdict(command):
        return gateway_lifecycle_block(
            command=command, env=remote, env_type="ssh" if remote else "local",
            cwd=str(cwd), workdir=str(cwd), session_key="python-data",
        )

    monkeypatch.setattr(guard, "_read_referenced_script", read)
    assert verdict("hermes gateway restart") is not None  # supervised guard is active
    assert verdict(invocation) is None
    assert wrapper in reads
    if remote:
        assert wrapper in remote.reads
        assert data not in remote.reads
    # Cron fixture uses its unambiguous Python extension, not shebang dispatch.
    cron_script = scripts / "report.py"
    cron_script.write_text(source, encoding="utf-8")
    guard.check_gateway_lifecycle("Generate report", str(cron_script))
    assert data not in reads

    oversized = source + "#" * (guard._MAX_REFERENCED_SCRIPT_BYTES + 1)
    if remote:
        remote.files[wrapper] = oversized
    else:
        wrapper.write_text(oversized, encoding="utf-8")
    assert verdict(invocation) is not None  # source size still consumes the budget
    cron_script.write_text(oversized, encoding="utf-8")
    with pytest.raises(guard.GatewayLifecycleBlocked):
        guard.check_gateway_lifecycle("Generate report", str(cron_script))


@pytest.mark.parametrize("backend", ["local", "remote"])
@pytest.mark.parametrize("invocation", [
    "python3 wrapper.py",
    "python3 --check-hash-based-pycs always wrapper.py",
    "python3 --check-hash-based-pycs never wrapper.py",
    "python3 --check-hash-based-pycs default wrapper.py",
])
@pytest.mark.parametrize("shell", [False, True])
def test_python_literal_process_descendants_keep_interpretation(
    tmp_path, monkeypatch, backend, invocation, shell,
):
    from tools import process_registry
    from tools.terminal_tool_guards import gateway_lifecycle_block

    home = tmp_path / "home"
    scripts = home / "scripts"
    scripts.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: True)
    remote = SyntheticRemote() if backend == "remote" else None
    cwd = tmp_path / "remote" if remote else scripts
    wrapper = cwd / "wrapper.py"

    def put(name, text):
        path = cwd / name
        if remote:
            remote.files[path] = text
        else:
            path.write_text(text, encoding="utf-8")
            path.chmod(0o755)

    def verdict(command=invocation):
        return gateway_lifecycle_block(
            command=command, env=remote, env_type="ssh" if remote else "local",
            cwd=str(cwd), workdir=str(cwd), session_key="python-descendant",
        )

    # With a shell the space separates argv; without it the whole path is executable.
    args = repr("./child helper.sh") if shell else repr(["./child helper.sh"])
    put("wrapper.py", f"import subprocess\nsubprocess.run({args}, shell={shell!r})\n")
    actual, other = ("child", "child helper.sh") if shell else ("child helper.sh", "child")
    put(actual, "#!/bin/sh\nexit 0\n")
    put(other, "#!/bin/sh\nhermes gateway restart\n")
    assert verdict() is None
    put(actual, "#!/bin/sh\nhermes gateway restart\n")
    put(other, "#!/bin/sh\nexit 0\n")
    blocked = verdict()
    assert blocked is not None
    assert json.loads(blocked)["exit_code"] == 1
    if remote:
        assert cwd / actual in remote.reads
        assert cwd / other not in remote.reads
    else:
        with pytest.raises(guard.GatewayLifecycleBlocked):
            guard.check_gateway_lifecycle("Generate report", str(wrapper))
    put(actual, "# descendant line limit\n" * guard._MAX_LIFECYCLE_SCAN_LINES)
    assert verdict() is not None

    # An inert Python expression is a real shell subshell: Python visits cannot hide it.
    # These fixtures are only classified, never executed (including by the remote adapter).
    put("child helper.sh", "#!/bin/sh\nhermes gateway restart\n")
    mixed = "python3 wrapper.py; sh wrapper.py"
    direct_commands = ("./wrapper.py", "python3 wrapper.py; ./wrapper.py")
    for shebang in ("#!/usr/bin/env python3\n", ""):
        put("wrapper.py", shebang + "('./child helper.sh')\n")
        assert verdict() is None
        assert verdict("sh wrapper.py") is not None
        assert verdict(mixed) is not None
        for command in direct_commands:
            # Without a shebang, Bash falls back to shell even for a .py filename.
            assert (verdict(command) is not None) == (not shebang)
        if not remote:
            # Cron selects Python by .py independently of direct terminal execution.
            guard.check_gateway_lifecycle("Generate report", str(wrapper))
            guard.check_gateway_lifecycle("python3 wrapper.py", str(wrapper))
            with pytest.raises(guard.GatewayLifecycleBlocked):
                guard.check_gateway_lifecycle(mixed, str(wrapper))
            for command in direct_commands:
                if shebang:
                    guard.check_gateway_lifecycle(command, str(wrapper))
                else:
                    with pytest.raises(guard.GatewayLifecycleBlocked):
                        guard.check_gateway_lifecycle(command, str(wrapper))

    # Re-visits share one path allowance, but a new interpretation pays for source again.
    source = "#!/usr/bin/env python3\n# " + "x" * 80 + "\n"
    put("wrapper.py", source)
    same = "python3 wrapper.py; python3 wrapper.py"
    with monkeypatch.context() as limits:
        limits.setattr(guard, "_MAX_LIFECYCLE_SCAN_PATHS", 1)
        assert verdict(mixed) is None
        limits.setattr(guard, "_MAX_LIFECYCLE_SCAN_BYTES", len(same) + len(source))
        assert verdict(same) is None
        assert verdict(mixed) is not None
    if remote:
        with monkeypatch.context() as limits:
            limits.setattr(guard, "_MAX_LIFECYCLE_SCAN_REMOTE_READS", 1)
            assert verdict(same) is None
            assert verdict(mixed) is not None

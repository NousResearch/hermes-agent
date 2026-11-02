"""Captured package-tool output must not depend on the host's ANSI code page."""

import importlib
import locale
import os
from pathlib import Path
import subprocess
import sys

import pytest

import pm
import pm.workspace as ws
from pm.package import InstallError, Runner


@pytest.fixture
def legacy_locale_child(monkeypatch):
    # The canonical runner enables UTF-8 mode. Pin only subprocess's default
    # encoding seam to a legacy locale; keep the real host and pipe I/O.
    monkeypatch.setattr(locale, "getencoding", lambda: "cp1252")
    monkeypatch.setattr(subprocess, "_text_encoding", locale.getencoding)
    real_run = subprocess.run

    def run(*, stdout=b"", stderr=b"", returncode=0, **kwargs):
        script = (
            "import sys; "
            f"sys.stdout.buffer.write({stdout!r}); "
            f"sys.stderr.buffer.write({stderr!r}); "
            f"sys.exit({returncode})"
        )
        return real_run([sys.executable, "-c", script], **kwargs)

    return run


@pytest.mark.parametrize("stage", ["lock", "sync"])
@pytest.mark.parametrize("stream", ["stdout", "stderr"])
@pytest.mark.parametrize("suffix", [b"", b"\xff"], ids=["utf8", "invalid-byte"])
def test_uv_failure_retains_utf8_build_diagnostic(
    tmp_path, monkeypatch, legacy_locale_child, stage, stream, suffix
):
    diagnostic = "🔍 cryptography: OpenSSL headers not found"
    raw = diagnostic.encode("utf-8") + suffix + b"\n"
    expected = diagnostic + ("�" if suffix else "")
    core = tmp_path / "core"
    core.mkdir()
    (core / "pyproject.toml").write_text('[project]\nname="test-core"\nversion="1"\n')
    from pm.environment import PythonEnvironment

    environment = PythonEnvironment(
        uv=Path(sys.executable), python=Path(sys.executable),
        destination=tmp_path / "venv", cache=tmp_path / "cache", env=dict(os.environ),
    )
    completed = []

    def run_uv(cmd, **kwargs):
        # Successful lock output must also be decoded before sync can run.
        output = {stream: raw} if cmd[1] == stage else {"stdout": raw, "stderr": raw}
        result = legacy_locale_child(
            **output, returncode=17 if cmd[1] == stage else 0, **kwargs
        )
        completed.append(result)
        return result

    monkeypatch.setattr("pm.environment.subprocess.run", run_uv)
    with pytest.raises(InstallError) as excinfo:
        ws.lock_and_sync([], [], root=tmp_path / "workspace", source=core,
                         seed_lock=None, environment=environment)

    assert type(excinfo.value) is InstallError  # A build error is not a resolver conflict.
    assert excinfo.value.cause == f"uv {stage} exited 17: {expected}"
    assert getattr(completed[-1], stream) == expected + "\n"


@pytest.mark.parametrize("install_cmd", ["ci", "install"])
@pytest.mark.parametrize("stream", ["stdout", "stderr"])
@pytest.mark.parametrize("returncode", [0, 17])
def test_node_sidecar_retains_output_and_exit_status(
    tmp_path, monkeypatch, legacy_locale_child, install_cmd, stream, returncode
):
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    if install_cmd == "ci":
        (tmp_path / "package-lock.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        importlib.import_module("pm.install"), "lazy_installs_allowed", lambda: True
    )
    diagnostic = "🔍 node-gyp: build toolchain unavailable"
    raw = diagnostic.encode("utf-8") + b"\xff\n"
    completed = []
    npm_dir = tmp_path / "pm-bin"
    npm_dir.mkdir()
    npm = npm_dir / ("npm.cmd" if os.name == "nt" else "npm")
    npm.write_text("process boundary fixture", encoding="utf-8")
    npm.chmod(0o755)
    context = Runner("npm", dict(os.environ, PATH=str(npm_dir)))
    acquisitions = []

    def acquire(name, **kwargs):
        acquisitions.append((name, kwargs))
        return context

    def run_npm(cmd, **kwargs):
        assert cmd == [str(npm), install_cmd, "--no-audit", "--no-fund"]
        assert kwargs["env"] == context.env
        assert kwargs["cwd"] == str(tmp_path)
        # Keep the real Runner and decoding path; only replace npm's process
        # with a Python child that emits controlled bytes and an exit status.
        result = legacy_locale_child(**{stream: raw}, returncode=returncode, **kwargs)
        completed.append(result)
        return result

    monkeypatch.setattr(pm, "ensure", acquire)
    monkeypatch.setattr("pm.package.subprocess.run", run_npm)
    error = ws.install_node_sidecar(tmp_path)

    assert acquisitions == [("npm", {"explicit": False})]
    assert len(completed) == 1
    assert completed[0].returncode == returncode
    expected = diagnostic + "�"
    if returncode:
        assert error == f"npm {install_cmd} exited {returncode}: {expected}"
    else:
        assert error is None
    assert getattr(completed[-1], stream) == expected + "\n"
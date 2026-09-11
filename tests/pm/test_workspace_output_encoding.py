"""Captured package-tool output must not depend on the host's ANSI code page."""

import importlib
import locale
import subprocess
import sys

import pytest

import pm.workspace as ws
from pm.package import InstallError


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
    monkeypatch.setattr(ws, "_generate_pyproject", lambda *a, **k: (tmp_path, False))
    from pathlib import Path

    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (Path(sys.executable), Path(sys.executable)))
    completed = []

    def run_uv(cmd, **kwargs):
        # Successful lock output must also be decoded before sync can run.
        output = {stream: raw} if cmd[1] == stage else {"stdout": raw, "stderr": raw}
        result = legacy_locale_child(
            **output, returncode=17 if cmd[1] == stage else 0, **kwargs
        )
        completed.append(result)
        return result

    monkeypatch.setattr(ws.subprocess, "run", run_uv)
    with pytest.raises(InstallError) as excinfo:
        ws.lock_and_sync([], venv_dir=tmp_path / "venv", root=tmp_path)

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
        importlib.import_module("pm.ensure"), "lazy_installs_allowed", lambda: True
    )
    diagnostic = "🔍 node-gyp: build toolchain unavailable"
    raw = diagnostic.encode("utf-8") + b"\xff\n"
    completed = []

    def run_npm(cmd, **kwargs):
        result = legacy_locale_child(**{stream: raw}, returncode=returncode, **kwargs)
        completed.append(result)
        return result

    error = ws.install_node_sidecar(tmp_path, npm_bin=sys.executable, runner=run_npm)

    expected = diagnostic + "�"
    if returncode:
        assert error == f"npm {install_cmd} exited {returncode}: {expected}"
    else:
        assert error is None
    assert getattr(completed[-1], stream) == expected + "\n"
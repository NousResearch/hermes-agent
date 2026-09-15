"""Hindsight's shipped probe and bridge execute in an isolated side interpreter.

The tiny SDK models only the child protocol, not upstream ML/CPU compatibility.
PM generation acquisition is owned by tests/pm; no upstream downloads occur here.
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import plugins.memory.hindsight.embedded_runtime as rt


@pytest.fixture()
def side_root(tmp_path, monkeypatch):
    root = tmp_path / "hermes-home" / "profiles" / "Hindsight" / "env"
    monkeypatch.setattr(rt, "sideenv_root", lambda: root)
    return root


def test_ensure_sideenv_uses_pm_selected_python(side_root, monkeypatch):
    import pm
    from hermes_constants import venv_python_path

    python = venv_python_path(side_root / "selected")
    seen = {}

    def ensure(name, requirements, **kwargs):
        seen.update(name=name, requirements=requirements, **kwargs)
        return python

    monkeypatch.setattr(pm, "ensure_environment", ensure)
    assert rt.ensure_sideenv() == python
    assert seen == dict(
        name="hindsight", requirements=(f"hindsight-embed=={rt._EMBED_VERSION}",
                                        f"hindsight-api-slim[all]=={rt._API_SLIM_VERSION}"),
        root=side_root, explicit=True, timeout=3600,
    )


def test_sideenv_python_is_pm_passive_selection(side_root, monkeypatch):
    import pm
    from hermes_constants import venv_python_path

    python = venv_python_path(side_root / "selected")
    seen = []
    monkeypatch.setattr(pm, "environment_python",
                        lambda name, **kw: seen.append((name, kw)) or python)
    assert rt.sideenv_python() == python
    assert seen == [("hindsight", {"root": side_root})]
    assert not side_root.exists()


@pytest.fixture
def side_python(side_root, monkeypatch, tmp_path):
    import pm

    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(side_root)],
                   check=True, capture_output=True, timeout=30)
    from hermes_constants import venv_python_path
    python = venv_python_path(side_root)
    site = Path(subprocess.check_output(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        text=True, timeout=10,
    ).strip())
    sdk = site / "hindsight_embed"
    sdk.mkdir()
    (sdk / "__init__.py").touch()
    for module in (sdk / "daemon_embed_manager.py", site / "hindsight_api.py", site / "sentence_transformers.py"):
        module.write_text("", encoding="utf-8")
    (sdk / "daemon_client.py").write_text("""import json, os, sys
from pathlib import Path
stopped = None
def stop_daemon(profile):
    global stopped
    stopped = profile
def ensure_daemon_running(config, profile):
    Path(os.environ['TEST_RECEIPT']).write_text(json.dumps({
        'config': config, 'profile': profile, 'stopped': stopped,
        'prefix': sys.prefix, 'env': dict(os.environ)}), encoding='utf-8')
    print('manager startup noise')
    return True
def get_daemon_url(profile):
    return 'http://127.0.0.1:54321/' + profile
""", encoding="utf-8")
    receipt = tmp_path / "bridge.json"
    monkeypatch.setenv("TEST_RECEIPT", str(receipt))
    monkeypatch.setattr(pm, "environment_python", lambda *a, **kw: python)
    return site, receipt


@pytest.mark.parametrize("grace,override,expected", [
    (60, None, "60.0"), ("45", None, "45.0"), (0, None, "0.0"),
    (None, None, None), ("", None, None), ("invalid", None, None),
    (-5, None, None), (60, "99", "99"),
])
def test_side_interpreter_probe_and_bridge(side_python, side_root, monkeypatch, caplog, grace, override, expected):
    site, receipt = side_python
    markers = ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONEXECUTABLE")
    for key in markers:
        monkeypatch.setenv(key, "/invalid/main-environment")
    monkeypatch.delenv("HINDSIGHT_EMBED_PORT_HEALTH_GRACE_TIMEOUT", raising=False)
    if override:
        monkeypatch.setenv("HINDSIGHT_EMBED_PORT_HEALTH_GRACE_TIMEOUT", override)
    parent = dict(os.environ)
    assert rt.check_local_runtime() == (True, None)
    with caplog.at_level("INFO", logger=rt.__name__):
        assert rt.ensure_daemon_and_url({"profile": "test-profile", "port_health_grace_timeout": grace}, restart=True) == "http://127.0.0.1:54321/test-profile"
    child = json.loads(receipt.read_text(encoding="utf-8"))
    assert child["config"] == {}
    assert child["profile"] == child["stopped"] == "test-profile"
    assert Path(child["prefix"]) == side_root
    assert child["env"].get("HINDSIGHT_EMBED_PORT_HEALTH_GRACE_TIMEOUT") == expected
    assert child["env"]["HINDSIGHT_EMBED_API_VERSION"] == rt._API_SLIM_VERSION
    assert not set(markers).intersection(child["env"])
    assert dict(os.environ) == parent
    assert not {"hindsight_embed", "hindsight_api", "sentence_transformers"}.intersection(sys.modules)
    assert "daemon ready" in caplog.text and "starting side-env daemon manager" in caplog.text
    (site / "sentence_transformers.py").write_text("raise RuntimeError('NumPy SIMD unavailable')", encoding="utf-8")
    ok, reason = rt.check_local_runtime()
    assert not ok and "NumPy SIMD unavailable" in reason


@pytest.mark.parametrize("outcome,message", [
    (SimpleNamespace(returncode=1, stdout="", stderr="illegal instruction"), "illegal instruction"),
    (SimpleNamespace(returncode=0, stdout='__HERMES_HINDSIGHT_BRIDGE__{"ok": false, "error": "daemon refused"}', stderr=""), "daemon refused"),
    (SimpleNamespace(returncode=0, stdout="noise only", stderr=""), "no bridge result"),
    (subprocess.TimeoutExpired("side-python", 5), "timed out"),
    (OSError("cannot execute"), "could not run"),
])
def test_bridge_failure_is_actionable(side_root, monkeypatch, caplog, outcome, message):
    monkeypatch.setattr(rt, "sideenv_python", lambda: side_root / "python")
    def run(*args, **kwargs):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    monkeypatch.setattr(rt, "subprocess", SimpleNamespace(run=run, TimeoutExpired=subprocess.TimeoutExpired))
    with pytest.raises(RuntimeError, match=message):
        rt.ensure_daemon_and_url({})
    if not isinstance(outcome, OSError):
        assert caplog.records


@pytest.mark.parametrize("mode,available", [("cloud", False), ("local_external", False), ("local_embedded", True), ("local_embedded", False)])
def test_provider_unavailable_reason(side_root, monkeypatch, mode, available):
    import plugins.memory.hindsight as hs

    monkeypatch.setattr(hs, "_load_config", lambda: {"mode": mode})
    def probe():
        assert mode == "local_embedded"
        return available, "NumPy SIMD unavailable"
    monkeypatch.setattr(hs, "_check_local_runtime", probe)
    reason = hs.HindsightMemoryProvider().unavailable_reason()
    if mode != "local_embedded" or available:
        assert reason == ""
    else:
        assert reason == reason.strip()
        assert "NumPy SIMD unavailable" in reason and "hermes memory setup" in reason
        assert str(side_root) in reason
        assert all(requirement in reason for requirement in rt._REQUIREMENTS)
        assert "uv pip install --python" not in reason


def test_missing_side_environment_never_starts_daemon(side_root):
    ok, reason = rt.check_local_runtime()
    assert not ok and "hermes memory setup" in reason and str(side_root) in reason
    with pytest.raises(RuntimeError, match="hermes memory setup"):
        rt.ensure_daemon_and_url({})

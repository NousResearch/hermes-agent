"""uv spells ``--system-certs`` as ``UV_SYSTEM_CERTS``; PM's allowlist must forward it.

uv 0.11 deprecates ``--native-tls`` in favour of ``--system-certs``, and PM runs uv with a
temporary ``XDG_CONFIG_HOME``, so a ``system-certs = true`` in the user's ``uv.toml`` is never
read. On a network whose index sits behind a TLS-intercepting proxy every lazy extra install
therefore dies with ``invalid peer certificate: UnknownIssuer`` even though the user configured
uv the recommended way (#152632). ``UV_SYSTEM_CERTS`` is a transport knob, not an index
redirect: it must survive both the ambient-``UV_*`` filter and ``no_config`` isolation, exactly
like ``UV_NATIVE_TLS``.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pytest

from pm.environment import PythonEnvironment, _base_environment


@pytest.fixture
def clean_uv_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(("UV_", "PIP_")):
            monkeypatch.delenv(key)
    monkeypatch.setenv("PIP_CONFIG_FILE", os.devnull)


def test_system_certs_reaches_uv_while_selection_knobs_do_not(clean_uv_env, monkeypatch):
    monkeypatch.setenv("UV_SYSTEM_CERTS", "1")
    monkeypatch.setenv("UV_PYTHON", "/poison/python")
    monkeypatch.setenv("UV_CACHE_DIR", "/poison/cache")

    env = _base_environment()

    assert env["UV_SYSTEM_CERTS"] == "1"
    assert not {"UV_PYTHON", "UV_CACHE_DIR"} & env.keys()


def test_no_config_isolation_keeps_the_transport_knob(tmp_path):
    """The pinned runtime builder strips index redirects, not transport knobs."""
    ambient = dict(os.environ, UV_INDEX_URL="https://fixture-mirror.invalid/simple",
                   UV_SYSTEM_CERTS="1")
    child = ("import os, sys\n"
             "if 'UV_INDEX_URL' in os.environ:\n"
             "    sys.exit('leaked index redirect: UV_INDEX_URL')\n"
             "if os.environ.get('UV_SYSTEM_CERTS') != '1':\n"
             "    sys.exit('transport knob dropped: UV_SYSTEM_CERTS')\n")

    environment = PythonEnvironment(
        uv=Path(sys.executable), python=Path(sys.executable), destination=tmp_path / "venv",
        cache=tmp_path / "cache", env=ambient, output=io.StringIO(), no_config=True)

    result = environment._run(["-c", child], cwd=tmp_path, timeout=60)

    assert result.returncode == 0, f"transport knob dropped by isolation: {result.stderr}"

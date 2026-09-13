"""Embedded-daemon health grace timeout (issue #13125 comment thread).

On resource-contended hosts the embedded Hindsight daemon can exceed a single
2s /health check and get needlessly killed + restarted. Upstream exposes the
grace window via HINDSIGHT_EMBED_PORT_HEALTH_GRACE_TIMEOUT, read at import
time of the process that runs DaemonEmbedManager — which is now the SIDE-env
daemon bridge process, so the value must ride the subprocess environment
built by embedded_runtime (never Hermes' own os.environ).
"""

import plugins.memory.hindsight.embedded_runtime as rt

_ENV = rt._PORT_HEALTH_GRACE_ENV


def test_configured_value_rides_subprocess_env():
    env = rt._daemon_subprocess_env({"port_health_grace_timeout": 60})
    assert float(env[_ENV]) == 60.0


def test_string_value_parsed():
    env = rt._daemon_subprocess_env({"port_health_grace_timeout": "45"})
    assert float(env[_ENV]) == 45.0


def test_blank_and_missing_are_noops():
    env = rt._daemon_subprocess_env({})
    assert _ENV not in env
    assert _ENV not in rt._daemon_subprocess_env({"port_health_grace_timeout": ""})
    assert _ENV not in rt._daemon_subprocess_env({"port_health_grace_timeout": None})


def test_invalid_and_negative_ignored():
    assert _ENV not in rt._daemon_subprocess_env({"port_health_grace_timeout": "not-a-number"})
    assert _ENV not in rt._daemon_subprocess_env({"port_health_grace_timeout": -5})


def test_explicit_env_wins_over_config(monkeypatch):
    monkeypatch.setenv(_ENV, "99")
    env = rt._daemon_subprocess_env({"port_health_grace_timeout": 60})
    # setdefault must not clobber an operator-set env override.
    assert env[_ENV] == "99"


def test_main_env_interpreter_markers_stripped(monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", "/some/main-venv")
    monkeypatch.setenv("PYTHONPATH", "/some/shim")
    env = rt._daemon_subprocess_env({})
    assert "VIRTUAL_ENV" not in env
    assert "PYTHONPATH" not in env

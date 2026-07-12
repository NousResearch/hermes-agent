from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import canonical_writer_gateway_bootstrap as bootstrap


_POLICY = """\
canonical_brain:
  writer_boundary:
    enabled: true
  discord_edge:
    enabled: false
plugins:
  enabled: []
  disabled: []
cron:
  provider: builtin
"""


def _allow_acl_query(monkeypatch):
    monkeypatch.setattr(bootstrap.os, "listxattr", lambda _target: [], raising=False)


def test_strict_managed_policy_loader_accepts_only_exact_stable_file(
    tmp_path,
    monkeypatch,
):
    _allow_acl_query(monkeypatch)
    managed = tmp_path / "hermes"
    managed.mkdir(mode=0o755)
    path = managed / "config.yaml"
    path.write_text(_POLICY, encoding="utf-8")
    path.chmod(0o444)

    loaded = bootstrap.load_strict_managed_writer_only_policy(
        path,
        _expected_uid=os.geteuid(),
        _expected_gid=os.getegid(),
    )

    assert loaded["canonical_brain"]["writer_boundary"] == {"enabled": True}
    assert loaded["canonical_brain"]["discord_edge"] == {"enabled": False}
    assert loaded["plugins"] == {"enabled": [], "disabled": []}
    assert loaded["cron"] == {"provider": "builtin"}


@pytest.mark.parametrize(
    "payload",
    [
        _POLICY.replace("enabled: true", "enabled: true\n    enabled: false"),
        _POLICY + "model:\n  default: forbidden\n",
        _POLICY.replace("enabled: true", "enabled: &boundary true").replace(
            "enabled: false",
            "enabled: *boundary",
        ),
        _POLICY.replace("provider: builtin", "provider: external"),
    ],
)
def test_strict_managed_policy_loader_rejects_ambiguous_or_broad_policy(
    tmp_path,
    monkeypatch,
    payload,
):
    _allow_acl_query(monkeypatch)
    managed = tmp_path / "hermes"
    managed.mkdir(mode=0o755)
    path = managed / "config.yaml"
    path.write_text(payload, encoding="utf-8")
    path.chmod(0o444)

    with pytest.raises(RuntimeError, match="policy"):
        bootstrap.load_strict_managed_writer_only_policy(
            path,
            _expected_uid=os.geteuid(),
            _expected_gid=os.getegid(),
        )


def test_minimal_gateway_binds_entry_identity_and_stops_cleanly(
    tmp_path,
    monkeypatch,
):
    _allow_acl_query(monkeypatch)
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    runtime.chmod(0o700)
    stop = threading.Event()
    startup_origins = []
    startup_notifications = []
    liveness_writes = []
    liveness_notifications = []
    removals = []

    def startup_attestor(*, _module_identity_provider):
        origin, digest = _module_identity_provider()
        startup_origins.append((origin, digest))
        return {"phase": len(startup_origins), "origin": origin, "sha256": digest}

    def startup_notifier(receipt, *, ready):
        startup_notifications.append((receipt["phase"], ready))
        return True

    def liveness_attestor(generation, **kwargs):
        assert kwargs["_persist"] is False
        assert kwargs["_deadline_monotonic_ns"] > 0
        return {"generation": generation}

    def liveness_notifier(startup_sha256, receipt):
        assert len(startup_sha256) == 64
        liveness_notifications.append(receipt["generation"])
        stop.set()
        return True

    isolated = SimpleNamespace(
        enabled=True,
        discord_edge_enabled=False,
        model_tools_enabled=False,
    )
    assert bootstrap.run_writer_only_gateway(
        runtime_dir=runtime,
        liveness_interval_seconds=0.001,
        _policy_loader=lambda: {"trusted": True},
        _hardener=lambda policy: policy == {"trusted": True},
        _config_provider=lambda _policy: isolated,
        _startup_attestor=startup_attestor,
        _startup_notifier=startup_notifier,
        _liveness_attestor=liveness_attestor,
        _liveness_writer=liveness_writes.append,
        _liveness_notifier=liveness_notifier,
        _liveness_remover=lambda: removals.append(True),
        _process_start_provider=lambda pid: pid + 100,
        _stop_event=stop,
        _install_signal_handlers=False,
        _platform="linux",
    )

    expected_origin = str(Path(bootstrap.__file__).resolve(strict=True))
    assert len(startup_origins) == 2
    assert {origin for origin, _digest in startup_origins} == {expected_origin}
    assert all(len(digest) == 64 for _origin, digest in startup_origins)
    assert startup_notifications == [(1, False), (2, True)]
    assert liveness_writes == [{"generation": 1}]
    assert liveness_notifications == [1]
    assert len(removals) == 2
    assert not (runtime / "gateway.pid").exists()
    assert (runtime / "gateway.lock").stat().st_mode & 0o777 == 0o600


def test_minimal_gateway_fails_closed_when_systemd_does_not_accept_readiness(
    tmp_path,
    monkeypatch,
):
    _allow_acl_query(monkeypatch)
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    runtime.chmod(0o700)
    removals = []
    isolated = SimpleNamespace(
        enabled=True,
        discord_edge_enabled=False,
        model_tools_enabled=False,
    )

    with pytest.raises(RuntimeError, match="pre-readiness"):
        bootstrap.run_writer_only_gateway(
            runtime_dir=runtime,
            _policy_loader=lambda: {},
            _hardener=lambda _policy: True,
            _config_provider=lambda _policy: isolated,
            _startup_attestor=lambda **_kwargs: {"receipt": True},
            _startup_notifier=lambda _receipt, **_kwargs: False,
            _liveness_remover=lambda: removals.append(True),
            _process_start_provider=lambda _pid: 123,
            _install_signal_handlers=False,
            _platform="linux",
        )

    assert len(removals) == 2
    assert not (runtime / "gateway.pid").exists()


def test_minimal_gateway_writes_before_liveness_notify_and_invalidates_failure(
    tmp_path,
    monkeypatch,
):
    _allow_acl_query(monkeypatch)
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    runtime.chmod(0o700)
    events = []
    isolated = SimpleNamespace(
        enabled=True,
        discord_edge_enabled=False,
        model_tools_enabled=False,
    )

    with pytest.raises(RuntimeError, match="liveness systemd notify"):
        bootstrap.run_writer_only_gateway(
            runtime_dir=runtime,
            _policy_loader=lambda: {},
            _hardener=lambda _policy: True,
            _config_provider=lambda _policy: isolated,
            _startup_attestor=lambda **_kwargs: {"ready": True},
            _startup_notifier=lambda _receipt, **_kwargs: True,
            _liveness_attestor=lambda generation, **_kwargs: {
                "generation": generation
            },
            _liveness_writer=lambda receipt: events.append(
                ("write", receipt["generation"])
            ),
            _liveness_notifier=lambda _startup, receipt: events.append(
                ("notify", receipt["generation"])
            )
            or False,
            _liveness_remover=lambda: events.append(("remove",)),
            _process_start_provider=lambda _pid: 123,
            _install_signal_handlers=False,
            _platform="linux",
        )

    assert events == [
        ("remove",),
        ("write", 1),
        ("notify", 1),
        ("remove",),
    ]
    assert not (runtime / "gateway.pid").exists()


def test_import_surface_never_loads_general_gateway_or_dynamic_capabilities():
    probe = """
import sys
import gateway.canonical_writer_gateway_bootstrap as entry
forbidden = (
    'agent', 'tools', 'run_agent', 'model_tools', 'cron', 'plugins',
    'providers', 'dotenv', 'gateway.run', 'gateway.platforms',
    'hermes_cli.config', 'hermes_cli.env_loader',
)
loaded = sorted(
    name for name in sys.modules
    if any(name == prefix or name.startswith(prefix + '.') for prefix in forbidden)
)
assert not loaded, loaded
assert entry.__file__.endswith('/gateway/canonical_writer_gateway_bootstrap.py')
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
        env={
            key: value
            for key, value in os.environ.items()
            if key not in {"PYTHONHOME", "PYTHONPATH"}
        },
    )
    assert completed.returncode == 0, completed.stderr

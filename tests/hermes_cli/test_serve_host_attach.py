"""A second `hermes serve` attaches to the host backend instead of binding a second port."""

import json
import os
from types import SimpleNamespace

import pytest

from gateway import host_rendezvous as hr
from hermes_cli.main_dashboard import _attach_to_host_backend


def _args(**over):
    base = dict(host="127.0.0.1", port=9200, no_open=True, isolated=False, open_profile="")
    return SimpleNamespace(**{**base, **over})


@pytest.fixture
def host_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    return tmp_path


def _publish(create_time, pid=None):
    record = hr.HostRecord(
        role=hr.ROLE_SERVE, pid=pid if pid is not None else os.getpid(), create_time=create_time,
        host="127.0.0.1", port=9119, protocol_version=hr.HOST_PROTOCOL_VERSION,
        token_fingerprint="", profiles=("default", "alpha"),
        updated_at="2026-01-01T00:00:00+00:00")
    path = hr.record_path(hr.ROLE_SERVE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record.to_json()), encoding="utf-8")


def test_second_serve_attaches_to_the_live_host_backend(host_dir, capsys):
    """A live record (real pid + its real creation time) ends the second launch at exit 0 —
    it never reaches the bind."""
    _publish(hr.process_create_time())

    with pytest.raises(SystemExit) as exc:
        _attach_to_host_backend(_args(), headless_backend=True)

    assert exc.value.code == 0
    assert "port 9119" in capsys.readouterr().out


def test_stale_record_is_ignored_and_the_launch_proceeds(host_dir):
    """A record whose creation time does not match the live PID is a recycled PID, not a
    backend: the launch must fall through and bind, never attach."""
    _publish(1.0)

    assert _attach_to_host_backend(_args(), headless_backend=True) is None


def test_isolated_never_attaches(host_dir):
    """`--isolated` is load-bearing for Desktop's SSH backend ownership proof."""
    _publish(hr.process_create_time())

    assert _attach_to_host_backend(_args(isolated=True), headless_backend=True) is None

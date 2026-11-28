"""The post-move check compares the gateway's sign-in providers with the old backend's list. When that list
could not be read (``/api/health`` answered, ``/api/auth/providers`` did not), the apply refuses before any
change (exit 3) instead of stopping the old backend and failing its own check afterwards."""

from __future__ import annotations

from hermes_cli import gateway_migrate_backends_apply as engine
from tests.hermes_cli.test_gateway_migrate_backends_apply import FakeHost, _args, _config_files, _nothing_applied


def test_an_unreadable_provider_list_is_refused_before_the_old_backend_stops(home, capsys, monkeypatch):
    host = FakeHost(home)
    real = host.http_status

    def providers_unreadable(dial, port):
        status = real(dial, port)
        return {**status, "auth_providers": None} if status and host.old["active"] else status

    monkeypatch.setattr(host, "http_status", providers_unreadable)
    before = _config_files(home)

    code = engine.cmd_apply(host, _args())

    out = capsys.readouterr().out
    assert code == engine.EXIT_BLOCKED, out
    _nothing_applied(host, home, before)
    assert host.old == {"enabled": True, "active": True}
    assert "/api/auth/providers" in out and "Nothing changed" in out, out


def test_a_readable_provider_list_still_moves(home):
    assert engine.cmd_apply(FakeHost(home), _args()) == engine.EXIT_OK


def test_the_post_move_check_reads_an_unknown_provider_list_as_not_verified(home):
    """A record that holds the unknown list (``providers: null``) is a check that fails with a reason, never a
    TypeError."""
    host = FakeHost(home)
    assert engine.cmd_apply(host, _args()) == engine.EXIT_OK
    manifest = engine.read_manifest(home)
    manifest["auth_before"]["providers"] = None

    assert "unknown" in engine.gateway_serving(host, manifest)

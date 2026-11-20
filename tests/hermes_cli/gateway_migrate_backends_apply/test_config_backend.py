"""The migration needs the file config backend: its backups, rollback and ``--restore`` cover config.yaml only.

With Remote Config selected (the real remote backend against the stub plane) an apply would write the
``webapp.*`` settings to Remote Config while a rollback or ``--restore`` puts back only the local file. The
apply refuses (exit 3) before any change instead, and says why.
"""

from __future__ import annotations

import pytest

from hermes_cli import gateway_migrate_backends_apply as engine
from plugins.config_backends import remote as remote_pkg
from plugins.config_backends.remote import backend as backend_mod
from plugins.config_backends.remote import credentials as cred_mod
from tests.hermes_cli.test_gateway_migrate_backends_apply import FakeHost, _args, _config_files, _nothing_applied
from tests.plugins.config_backends.stub_plane import StubPlane, remote_env


@pytest.fixture
def remote(home, monkeypatch):
    with StubPlane() as plane:
        for key, value in remote_env(plane).items():
            monkeypatch.setenv(key, value)
        monkeypatch.setattr(backend_mod, "BOOT_RETRY_DELAYS", (0.0, 0.0))
        remote_pkg._reset_for_tests()
        cred_mod._IDP_CACHE.clear()
        yield plane
        remote_pkg._reset_for_tests()
        cred_mod._IDP_CACHE.clear()


def _remote_webapp():
    from hermes_cli.config_backend import get_config_backend
    from hermes_cli.config import get_hermes_home
    doc = get_config_backend().read_user_doc(get_hermes_home()) or {}
    return (doc.get("webapp") if isinstance(doc, dict) else None) or {}


def test_a_non_file_config_backend_is_refused_before_any_change(home, remote, capsys):
    host = FakeHost(home)
    before = _config_files(home)

    code = engine.cmd_apply(host, _args())

    out = capsys.readouterr().out
    assert code == engine.EXIT_BLOCKED, out
    _nothing_applied(host, home, before)
    assert host.old == {"enabled": True, "active": True}
    assert "HERMES_CONFIG_BACKEND=remote" in out and "file config backend" in out, out
    assert not _remote_webapp().get("enabled")  # Remote Config was not written either

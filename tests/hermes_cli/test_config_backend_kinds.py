"""``HERMES_CONFIG_BACKEND=<kind>`` resolves to the package ``plugins/config_backends/<kind>/``
through one entry point (``get_backend()``), and everything core needs from a backend is a protocol
method (teknium1 P2-1 on #122509). A backend core has never heard of is served end to end: the
fake package below lives in a temp directory, and nothing in core is patched or edited."""
from __future__ import annotations

import os
import sys
import textwrap

import pytest

import plugins.config_backends
from hermes_cli import config_backend as cb

FAKE = "fakekind_p21"

_FAKE_PACKAGE = '''
import contextlib

from hermes_cli.config_backend import FileBackend

EVENTS = []


class FakeBackend(FileBackend):
    name = "{name}"

    def bootstrap_env_names(self):
        return frozenset({{"FAKE_P21_PLANE"}})

    @contextlib.contextmanager
    def preserve_read_base(self):
        EVENTS.append("enter")
        try:
            yield
        finally:
            EVENTS.append("exit")

    def doctor_warnings(self, home):
        return (("fake warning for " + home.name, "(fake hint)", "fake issue"),)


_INSTANCE = FakeBackend()


def get_backend():
    return _INSTANCE
'''


@pytest.fixture
def fake_kind(tmp_path, monkeypatch):
    """Registers ``plugins/config_backends/<FAKE>/`` from a temp dir (the package path is extended,
    the real import machinery resolves it) and starts a fresh process bootstrap."""
    root = tmp_path / "backends"
    (root / FAKE).mkdir(parents=True)
    (root / FAKE / "__init__.py").write_text(textwrap.dedent(_FAKE_PACKAGE.format(name=FAKE)), encoding="utf-8")
    monkeypatch.setattr(plugins.config_backends, "__path__", [*plugins.config_backends.__path__, str(root)])
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv(cb.BACKEND_ENV, raising=False)
    monkeypatch.delenv("FAKE_P21_PLANE", raising=False)
    monkeypatch.setattr(cb, "_BOOTSTRAPPED", False)
    yield home
    for name in [m for m in sys.modules if m.startswith(f"plugins.config_backends.{FAKE}")]:
        del sys.modules[name]


def test_unknown_kind_is_served_through_protocol_methods_only(fake_kind):
    from hermes_cli.doctor_config import _check_config_backend
    from hermes_cli.doctor_report import Finding

    home = fake_kind
    (home / ".env").write_text(f"HERMES_CONFIG_BACKEND={FAKE}\nFAKE_P21_PLANE=https://plane.test\n", encoding="utf-8")
    (home / "config.yaml").write_text("a: 1\n", encoding="utf-8")

    backend = cb.get_config_backend()
    assert backend.name == FAKE
    fake = sys.modules[f"plugins.config_backends.{FAKE}"]
    assert backend is fake.get_backend()
    # bootstrap_env_names(): published from the launch home's .env before the first read
    assert os.environ["FAKE_P21_PLANE"] == "https://plane.test"
    # preserve_read_base(): keeping_read_base runs the read inside the backend's context
    assert cb.keeping_read_base(lambda: cb.read_config_doc(home / "config.yaml")) == {"a": 1}
    assert fake.EVENTS == ["enter", "exit"]
    # doctor hook
    f = Finding()
    assert _check_config_backend(backend, home, f) is True
    assert f.issues == ["fake issue"]


def test_file_backend_protocol_defaults_are_no_ops(tmp_path):
    fb = cb.FileBackend()
    assert fb.bootstrap_env_names() == frozenset()
    assert fb.doctor_warnings(tmp_path) == ()
    with fb.preserve_read_base():
        pass


@pytest.mark.parametrize("kind", ["../etc", "remote.backend", "remote x", "os;id", "_remote", "9lives",
                                  "x" * 65, "remote/backend", "plugins.config_backends.remote"])
def test_kind_is_validated_before_any_import(kind, monkeypatch):
    def no_import(name, *a, **k):
        raise AssertionError(f"imported {name!r} for an invalid kind")
    monkeypatch.setattr(cb.importlib, "import_module", no_import)
    monkeypatch.setattr(cb, "_BOOTSTRAPPED", True)
    monkeypatch.setenv(cb.BACKEND_ENV, kind)
    with pytest.raises(cb.ConfigBackendUnavailable, match="is not a config backend"):
        cb.get_config_backend()


def test_absent_kind_fails_closed(monkeypatch):
    monkeypatch.setattr(cb, "_BOOTSTRAPPED", True)
    monkeypatch.setenv(cb.BACKEND_ENV, "nosuchbackend")
    with pytest.raises(cb.ConfigBackendUnavailable, match="is not a config backend"):
        cb.get_config_backend()


def test_broken_backend_package_shows_its_own_import_error(fake_kind):
    pkg = plugins.config_backends.__path__[-1]
    with open(f"{pkg}/{FAKE}/__init__.py", "w", encoding="utf-8") as fh:
        fh.write("import hermes_missing_dependency_p21\n")
    with pytest.raises(ModuleNotFoundError, match="hermes_missing_dependency_p21"):
        cb._backend_for(FAKE)

"""Tests for the terminal environment provider registry + ABC.

Mirrors tests/agent/test_image_gen_registry.py in structure. Covers:
- registration / re-registration / type & name validation
- reserved built-in name rejection
- scoped registrations (multiplexed-gateway profiles)
- classification helpers (provider_flag, plugin_strip_env_keys)
- restore_registration semantics (plugin unload)
- fail-soft behavior for raising providers
"""

import pytest

from agent.terminal_env_provider import TerminalEnvironmentProvider
from agent import terminal_env_registry as reg


class _Env:
    def execute(self, command, **kwargs):
        return {"output": "", "exit_code": 0}

    def cleanup(self):
        pass


class _Provider(TerminalEnvironmentProvider):
    name = "testbox"
    display_name = "TestBox"
    is_remote = True
    is_container = True
    session_isolated_when_nonpersistent = True

    @property
    def cache_path_base(self):
        return "~/.hermes"

    @property
    def strip_env_keys(self):
        return frozenset({"TESTBOX_TOKEN", "TESTBOX_SECRET"})

    def is_available(self):
        return True

    def create_environment(self, *, cwd, timeout, task_id="default",
                           image=None, container_config=None, **kwargs):
        return _Env()


@pytest.fixture(autouse=True)
def _clean_registry():
    reg._reset_for_tests()
    yield
    reg._reset_for_tests()


def test_register_and_get():
    p = _Provider()
    reg.register_provider(p)
    assert reg.get_provider("testbox") is p
    assert reg.get_provider("TESTBOX") is p  # case-insensitive lookup
    assert reg.plugin_backend_names() == ["testbox"]


def test_rejects_non_provider():
    with pytest.raises(TypeError):
        reg.register_provider(object())


def test_rejects_empty_name():
    class Bad(_Provider):
        name = "  "

    with pytest.raises(ValueError):
        reg.register_provider(Bad())


@pytest.mark.parametrize("reserved", sorted(reg.BUILTIN_BACKEND_NAMES))
def test_rejects_builtin_names(reserved):
    class Shadow(_Provider):
        name = reserved

    with pytest.raises(ValueError):
        reg.register_provider(Shadow())
    assert reg.get_provider(reserved) is None


def test_reregistration_overwrites():
    p1, p2 = _Provider(), _Provider()
    reg.register_provider(p1)
    reg.register_provider(p2)
    assert reg.get_provider("testbox") is p2


def test_scoped_registration_isolated():
    p_global, p_scoped = _Provider(), _Provider()
    reg.register_provider(p_global)
    reg.register_provider(p_scoped, scope="profile-a")
    assert reg.get_provider("testbox", scope="profile-a") is p_scoped
    # Different scope falls back to global
    assert reg.get_provider("testbox", scope="profile-b") is p_global


def test_provider_flag_reads_attributes():
    reg.register_provider(_Provider())
    assert reg.provider_flag("testbox", "is_remote") is True
    assert reg.provider_flag("testbox", "is_container") is True
    assert reg.provider_flag("testbox", "session_isolated_when_nonpersistent") is True
    assert reg.provider_flag("testbox", "cache_path_base", None) == "~/.hermes"
    assert reg.provider_flag("testbox", "skip_container_guards") is True


def test_provider_flag_unknown_backend_returns_default():
    assert reg.provider_flag("missing", "is_remote", False) is False
    assert reg.provider_flag("missing", "cache_path_base", None) is None


def test_provider_flag_fail_soft_on_raising_property():
    class Broken(_Provider):
        name = "broken"

        @property
        def cache_path_base(self):
            raise RuntimeError("boom")

    reg.register_provider(Broken())
    assert reg.provider_flag("broken", "cache_path_base", None) is None


def test_plugin_strip_env_keys_union():
    class Other(_Provider):
        name = "other"

        @property
        def strip_env_keys(self):
            return frozenset({"OTHER_KEY"})

    reg.register_provider(_Provider())
    reg.register_provider(Other())
    keys = reg.plugin_strip_env_keys()
    assert keys == frozenset({"TESTBOX_TOKEN", "TESTBOX_SECRET", "OTHER_KEY"})


def test_plugin_strip_env_keys_fail_soft():
    class Broken(_Provider):
        name = "broken"

        @property
        def strip_env_keys(self):
            raise RuntimeError("boom")

    reg.register_provider(Broken())
    assert reg.plugin_strip_env_keys() == frozenset()


def test_restore_registration_unregisters():
    p = _Provider()
    reg.register_provider(p)
    assert reg.restore_registration("testbox", p, None) is True
    assert reg.get_provider("testbox") is None


def test_restore_registration_noop_when_replaced():
    p1, p2 = _Provider(), _Provider()
    reg.register_provider(p1)
    reg.register_provider(p2)  # p1 replaced
    assert reg.restore_registration("testbox", p1, None) is False
    assert reg.get_provider("testbox") is p2


def test_restore_registration_restores_previous():
    p1, p2 = _Provider(), _Provider()
    reg.register_provider(p1)
    reg.register_provider(p2)
    assert reg.restore_registration("testbox", p2, p1) is True
    assert reg.get_provider("testbox") is p1


def test_abc_defaults():
    p = _Provider()
    assert p.skip_container_guards is True  # defaults to is_container
    assert "TestBox" in p.env_description
    assert p.probe() == ("ready", "")
    assert p.setup_instructions() == []
    rows = p.doctor_checks()
    assert rows and rows[0][0] is True


def test_probe_needs_setup_when_unavailable():
    class Off(_Provider):
        name = "offbox"

        def is_available(self):
            return False

    status, detail = Off().probe()
    assert status == "needs_setup"
    assert detail


def test_registry_generation_bumps():
    g0 = reg.registry_generation()
    reg.register_provider(_Provider())
    g1 = reg.registry_generation()
    assert g1 != g0

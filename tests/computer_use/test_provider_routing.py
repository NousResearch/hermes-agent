"""Which machine computer_use drives is a provider decision, resolved once.

``computer_use`` used to pick its backend class from
``HERMES_COMPUTER_USE_BACKEND`` with a hardcoded two-name branch, so the only
runtime it could ever drive was cua-driver on the gateway host. The registry
replaces that branch: the host backend is a registered provider like any
other, and a plugin can supply one that owns a container display, a leased
sandbox, or a bridge back to the desktop client.

The load-bearing property is that selection is explicit and loud. Silently
falling back to the host provider means driving the user's own desktop when
they asked for a container, so a name nobody registered raises.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.computer_use_provider import ComputerUseProvider
from agent.computer_use_registry import (
    HOST_PROVIDER_NAME,
    UnknownComputerUseProvider,
    get_provider,
    list_providers,
    register_provider,
    resolve_provider,
    restore_registration,
    snapshot_registration,
)
from tools.computer_use import tool as cu_tool


class FakeProvider(ComputerUseProvider):
    def __init__(self, name="fake", available=True):
        self._name = name
        self.available = available
        self.created = []
        self.cleaned = 0

    @property
    def name(self):
        return self._name

    def is_available(self):
        return self.available

    def create_backend(self, session_id, permission_mode):
        backend = MagicMock()
        self.created.append((session_id, permission_mode))

        return backend

    def emergency_cleanup(self):
        self.cleaned += 1


@pytest.fixture
def clean_provider(monkeypatch):
    """A registered provider that is the configured one, with caches cleared."""
    provider = FakeProvider()
    register_provider(provider)
    monkeypatch.setenv("HERMES_COMPUTER_USE_BACKEND", "fake")
    cu_tool.reset_backend_for_tests()
    yield provider
    cu_tool.reset_backend_for_tests()
    restore_registration("fake", provider, None)


class TestRegistry:
    def test_the_host_backend_is_itself_a_provider(self):
        """One path to a backend, so a plugin provider is exercised by the
        same code the default is rather than by a branch nobody runs."""
        assert get_provider(HOST_PROVIDER_NAME) is not None
        assert HOST_PROVIDER_NAME in {p.name for p in list_providers()}

    @pytest.mark.parametrize("configured", ["", None, "local", "cua", "cua-driver", "builtin", "  CUA  "])
    def test_unset_and_every_legacy_name_still_mean_the_host(self, configured):
        assert resolve_provider(configured).name == HOST_PROVIDER_NAME

    def test_an_unregistered_name_raises_instead_of_driving_this_desktop(self):
        with pytest.raises(UnknownComputerUseProvider) as excinfo:
            resolve_provider("webtop-pool")

        # The message has to carry the fix: this surfaces to the user as the
        # tool's entire output.
        assert "webtop-pool" in str(excinfo.value)
        assert "config.yaml" in str(excinfo.value)

    def test_registering_requires_the_abc(self):
        with pytest.raises(TypeError):
            register_provider(object())

    def test_unloading_a_plugin_puts_back_what_it_replaced(self):
        first, second = FakeProvider("dup"), FakeProvider("dup")
        register_provider(first)
        previous = snapshot_registration("dup")
        register_provider(second)

        restore_registration("dup", second, previous)

        assert get_provider("dup") is first
        restore_registration("dup", first, None)

    def test_a_later_plugin_owns_the_name_and_is_not_rolled_back(self):
        """Restoring on unload must not take a third plugin's provider away."""
        first, second, third = FakeProvider("race"), FakeProvider("race"), FakeProvider("race")
        register_provider(first)
        previous = snapshot_registration("race")
        register_provider(second)
        register_provider(third)

        restore_registration("race", second, previous)

        assert get_provider("race") is third
        restore_registration("race", third, None)


class TestSelection:
    def test_the_configured_provider_builds_the_session_backend(self, clean_provider):
        cu_tool._get_backend(session_id="s1")

        assert [sid for sid, _ in clean_provider.created] == ["s1"]

    def test_the_resolved_permission_mode_reaches_the_provider(self, clean_provider):
        with patch.object(cu_tool, "_cua_permission_mode", return_value="bounded"):
            cu_tool._get_backend(session_id="s1")

        assert clean_provider.created[0][1] == "bounded"

    def test_a_missing_provider_reaches_the_model_as_the_fix(self, monkeypatch):
        monkeypatch.setenv("HERMES_COMPUTER_USE_BACKEND", "webtop-pool")
        cu_tool.reset_backend_for_tests()

        out = json.loads(cu_tool.handle_computer_use({"action": "capture"}, session_id="s1"))

        assert "webtop-pool" in out["error"]
        # The cua-driver install hint would send them after the wrong thing.
        assert "hint" not in out

    def test_config_is_read_once_per_profile_not_once_per_call(self, monkeypatch):
        """This runs on every dispatch and every tool-registration paint."""
        monkeypatch.delenv("HERMES_COMPUTER_USE_BACKEND", raising=False)
        cu_tool.reset_backend_for_tests()

        with patch("hermes_cli.config.load_config", return_value={}) as load:
            cu_tool.active_computer_use_provider()
            cu_tool.active_computer_use_provider()

        assert load.call_count == 1
        cu_tool.reset_backend_for_tests()

    def test_each_profile_resolves_its_own_config(self, monkeypatch, tmp_path):
        """One process serves several profiles under gateway.multiplex_profiles,
        and computer_use.provider is per-profile."""
        provider = FakeProvider("second-profile")
        register_provider(provider)
        monkeypatch.delenv("HERMES_COMPUTER_USE_BACKEND", raising=False)
        cu_tool.reset_backend_for_tests()

        with patch("hermes_cli.config.load_config", return_value={}):
            assert cu_tool.active_computer_use_provider().name == HOST_PROVIDER_NAME

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "other"))

        with patch(
            "hermes_cli.config.load_config",
            return_value={"computer_use": {"provider": "second-profile"}},
        ):
            assert cu_tool.active_computer_use_provider() is provider

        cu_tool.reset_backend_for_tests()
        restore_registration("second-profile", provider, None)

    def test_the_retired_env_var_still_wins_but_says_so(self, clean_provider, caplog):
        cu_tool.reset_backend_for_tests()

        with caplog.at_level("WARNING"):
            assert cu_tool.active_computer_use_provider() is clean_provider

        assert "HERMES_COMPUTER_USE_BACKEND" in caplog.text
        assert "config.yaml" in caplog.text

    def test_config_selects_the_provider_when_no_env_override_is_set(self, monkeypatch):
        provider = FakeProvider("from-config")
        register_provider(provider)
        monkeypatch.delenv("HERMES_COMPUTER_USE_BACKEND", raising=False)
        cu_tool.reset_backend_for_tests()

        with patch(
            "hermes_cli.config.load_config",
            return_value={"computer_use": {"provider": "from-config"}},
        ):
            assert cu_tool.active_computer_use_provider() is provider

        cu_tool.reset_backend_for_tests()
        restore_registration("from-config", provider, None)

    def test_an_unreadable_config_still_gets_the_host(self, monkeypatch):
        monkeypatch.delenv("HERMES_COMPUTER_USE_BACKEND", raising=False)
        cu_tool.reset_backend_for_tests()

        with patch("hermes_cli.config.load_config", side_effect=OSError("boom")):
            assert cu_tool.active_computer_use_provider().name == HOST_PROVIDER_NAME

        cu_tool.reset_backend_for_tests()


class TestLifecycle:
    def test_process_exit_cleans_up_what_the_provider_owns(self, clean_provider):
        """Leases outlive the backend objects that drove them, so the hook runs
        after every backend has been stopped."""
        cu_tool._get_backend(session_id="s1")

        cu_tool._shutdown_backend_atexit()

        assert clean_provider.cleaned == 1

    def test_a_provider_that_was_never_resolved_is_never_woken_at_exit(self):
        cu_tool.reset_backend_for_tests()
        provider = FakeProvider("dormant")
        register_provider(provider)

        cu_tool._shutdown_backend_atexit()

        assert provider.cleaned == 0
        restore_registration("dormant", provider, None)

    def test_a_throwing_cleanup_cannot_escape_atexit(self, clean_provider):
        """An exception out of atexit prints a traceback on every exit."""
        cu_tool._get_backend(session_id="s1")
        clean_provider.emergency_cleanup = MagicMock(side_effect=RuntimeError("nope"))

        cu_tool._shutdown_backend_atexit()

    def test_releasing_a_session_stops_its_backend(self, clean_provider):
        """The provider does not cache, so the existing session release path is
        the whole per-session lifecycle — no second one to keep in sync."""
        backend = cu_tool._get_backend(session_id="s1")

        assert cu_tool.release_computer_use_session("s1") is True
        backend.stop.assert_called_once()


class TestAvailabilityGate:
    def test_a_provider_answers_for_its_own_runtime_not_the_hosts(self, clean_provider):
        """A container pool supplies displays a headless gateway does not have,
        so the host platform gate is not its to fail."""
        with patch.object(cu_tool.sys, "platform", "freebsd13"):
            assert cu_tool.check_computer_use_requirements() is True

    def test_a_throwing_provider_is_an_absent_one(self, clean_provider):
        clean_provider.is_available = MagicMock(side_effect=RuntimeError("boom"))

        assert cu_tool.check_computer_use_requirements() is False

    def test_a_misconfigured_provider_keeps_the_tool_so_it_can_explain(self, monkeypatch):
        """Stripped from the schema, the model has no way to say why it cannot
        help; kept, the dispatcher names the missing provider."""
        monkeypatch.setenv("HERMES_COMPUTER_USE_BACKEND", "webtop-pool")
        cu_tool.reset_backend_for_tests()

        assert cu_tool.check_computer_use_requirements() is True

    def test_the_host_provider_still_gates_on_platform_and_binary(self, monkeypatch):
        monkeypatch.setenv("HERMES_COMPUTER_USE_BACKEND", "local")
        cu_tool.reset_backend_for_tests()

        with patch.object(cu_tool.sys, "platform", "freebsd13"):
            assert cu_tool.check_computer_use_requirements() is False

    def test_a_provider_that_refuses_to_build_reports_the_cause_not_a_timeout(
        self, clean_provider
    ):
        """A provider that knows its runtime is gone says so from
        create_backend, before anything is spawned — the cheap answer, rather
        than a start() timeout reporting the symptom minutes later."""
        clean_provider.create_backend = MagicMock(
            side_effect=RuntimeError("webtop pool is down")
        )

        result = json.loads(cu_tool.handle_computer_use({"action": "screenshot"}))

        assert "webtop pool is down" in result["error"]

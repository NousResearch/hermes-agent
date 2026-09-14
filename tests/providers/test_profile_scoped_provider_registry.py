"""Profile-scoped provider plugin discovery regression."""

from __future__ import annotations

import sys

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _write_provider(home, marker: str) -> None:
    plugin = home / "plugins" / "model-providers" / "same-provider"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        "name: same-provider\nkind: model-provider\nversion: 1.0.0\n",
        encoding="utf-8",
    )
    (plugin / "__init__.py").write_text(
        "from providers import register_provider\n"
        "from providers.base import ProviderProfile\n"
        f"register_provider(ProviderProfile(name='same-provider', description='{marker}', "
        f"env_vars=('KEY_{marker.upper().replace('-', '_')}',), base_url='https://{marker}.example/v1'))\n",
        encoding="utf-8",
    )


def _clear_provider_state() -> None:
    import providers

    providers._REGISTRY.clear()
    providers._ALIASES.clear()
    providers._PROVIDER_LIST_CACHE = None
    providers._discovered = False
    providers._PROFILE_STATES.clear()
    for name in list(sys.modules):
        if name.startswith((
            "_hermes_user_provider_", "_hermes_scoped_provider_",
            "plugins.model_providers.",
        )):
            sys.modules.pop(name, None)


def test_provider_plugin_registry_follows_profile_home(tmp_path):
    import providers
    from agent.secret_scope import set_multiplex_active

    home_a, home_b = tmp_path / "a", tmp_path / "b"
    _write_provider(home_a, "profile-a")
    _write_provider(home_b, "profile-b")
    _clear_provider_state()
    set_multiplex_active(True)

    try:
        token = set_hermes_home_override(str(home_a))
        try:
            first = providers.get_provider_profile("same-provider")
            from hermes_cli.auth import get_provider_config, resolve_provider
            first_config = get_provider_config("same-provider")
            first_route = resolve_provider("same-provider")
        finally:
            reset_hermes_home_override(token)

        token = set_hermes_home_override(str(home_b))
        try:
            second = providers.get_provider_profile("same-provider")
            second_config = get_provider_config("same-provider")
            second_route = resolve_provider("same-provider")
        finally:
            reset_hermes_home_override(token)
    finally:
        set_multiplex_active(False)
        providers._PROFILE_STATES.clear()

    assert first is not None and first.description == "profile-a"
    assert second is not None and second.description == "profile-b"
    assert first is not second
    assert first_config is not None and first_config.inference_base_url == "https://profile-a.example/v1"
    assert second_config is not None and second_config.inference_base_url == "https://profile-b.example/v1"
    assert first_config.api_key_env_vars == ("KEY_PROFILE_A",)
    assert second_config.api_key_env_vars == ("KEY_PROFILE_B",)
    assert first_route == second_route == "same-provider"


def test_import_time_auth_seed_never_freezes_first_profile(tmp_path):
    import importlib
    import providers
    from agent.secret_scope import set_multiplex_active

    home_a, home_b = tmp_path / "seed-a", tmp_path / "seed-b"
    _write_provider(home_a, "profile-a")
    _write_provider(home_b, "profile-b")
    _clear_provider_state()
    set_multiplex_active(True)
    token = set_hermes_home_override(home_a)
    try:
        import hermes_cli.auth as auth
        auth = importlib.reload(auth)
        assert "same-provider" not in auth.PROVIDER_REGISTRY
        assert auth.get_provider_config("same-provider").inference_base_url == "https://profile-a.example/v1"
    finally:
        reset_hermes_home_override(token)

    token = set_hermes_home_override(home_b)
    try:
        assert auth.get_provider_config("same-provider").inference_base_url == "https://profile-b.example/v1"
    finally:
        reset_hermes_home_override(token)
        set_multiplex_active(False)
        _clear_provider_state()
        importlib.reload(auth)
        providers._discover_providers()


def test_profile_provider_status_uses_profile_config(tmp_path, monkeypatch):
    import providers
    from agent.secret_scope import reset_secret_scope, set_multiplex_active, set_secret_scope

    home = tmp_path / "status"
    _write_provider(home, "profile-a")
    _clear_provider_state()
    set_multiplex_active(True)
    home_token = set_hermes_home_override(home)
    secret_token = set_secret_scope({"KEY_PROFILE_A": "fixture-status-key"})
    try:
        from hermes_cli.auth import get_api_key_provider_status
        status = get_api_key_provider_status("same-provider")
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)
        set_multiplex_active(False)
        _clear_provider_state()
    assert status["configured"] is True
    assert status["logged_in"] is True


def test_bare_module_entry_point_replays_profile_for_each_home(tmp_path, monkeypatch):
    import importlib.metadata as md
    import providers
    from agent.secret_scope import set_multiplex_active
    from providers.base import ProviderProfile

    class _Module:
        PROFILE = ProviderProfile(
            name="module-entry", description="module-form",
            env_vars=("MODULE_ENTRY_KEY",), base_url="https://module.example/v1")

    module = _Module()

    class _EP:
        name = "module-entry"
        group = "hermes_agent.plugins"
        def load(self):
            return module

    class _EPs:
        def select(self, group):
            return [_EP()] if group == "hermes_agent.plugins" else []

    import hermes_cli.plugins as hp
    monkeypatch.setattr(md, "entry_points", lambda: _EPs())
    monkeypatch.setattr(hp, "_get_enabled_plugins", lambda: {"module-entry"})
    monkeypatch.setattr(hp, "_get_disabled_plugins", lambda: set())
    _clear_provider_state()
    set_multiplex_active(True)
    homes = [tmp_path / "module-a", tmp_path / "module-b"]
    seen = []
    try:
        for home in homes:
            home.mkdir()
            token = set_hermes_home_override(home)
            try:
                seen.append(providers.get_provider_profile("module-entry"))
            finally:
                reset_hermes_home_override(token)
    finally:
        set_multiplex_active(False)
        _clear_provider_state()
    assert all(profile is not None and profile.description == "module-form" for profile in seen)


def test_entry_point_registration_is_profile_scoped(tmp_path, monkeypatch):
    import importlib.metadata as md
    import providers
    from agent.secret_scope import set_multiplex_active
    from providers.base import ProviderProfile

    class _EP:
        name = "profile-entry"
        group = "hermes_agent.plugins"

        def load(self):
            def register():
                providers.register_provider(
                    ProviderProfile(name="profile-entry", description="entry-point"))
            return register

    class _EPs:
        def select(self, group):
            return [_EP()] if group == "hermes_agent.plugins" else []

    import hermes_cli.plugins as hp
    monkeypatch.setattr(md, "entry_points", lambda: _EPs())
    monkeypatch.setattr(hp, "_get_enabled_plugins", lambda: {"profile-entry"})
    monkeypatch.setattr(hp, "_get_disabled_plugins", lambda: set())

    home = tmp_path / "profile"
    home.mkdir()
    _clear_provider_state()
    set_multiplex_active(True)
    token = set_hermes_home_override(home)
    try:
        assert providers.get_provider_profile("profile-entry").description == "entry-point"
        assert "profile-entry" not in providers._REGISTRY
    finally:
        reset_hermes_home_override(token)
        set_multiplex_active(False)
        _clear_provider_state()


def test_multiplex_registry_fails_closed_without_profile_context():
    import providers
    from agent.secret_scope import set_multiplex_active

    _clear_provider_state()
    set_multiplex_active(True)
    try:
        try:
            providers.get_provider_profile("same-provider")
        except RuntimeError as exc:
            assert "requires a profile-scoped HERMES_HOME override" in str(exc)
        else:
            raise AssertionError("multiplex provider lookup must fail closed without profile context")
    finally:
        set_multiplex_active(False)
        _clear_provider_state()

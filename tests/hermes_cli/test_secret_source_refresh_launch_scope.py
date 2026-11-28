"""A plugin rediscovery inside a launch-profile body must not shrink that body's secret scope.

Under multiplexing the launch profile's bound scope is ``launch_secret_scope``: its ``.env`` over
the env frozen at activation, which is where systemd / ``op run`` / docker ``-e`` credentials and
the YAML-bridged allowlists (``telegram.allow_from`` -> ``TELEGRAM_ALLOWED_USERS``) live. A forced
discovery that re-applies plugin secret sources refreshes the installed scope in place; rebuilt
from the files alone it dropped those env-only values for the rest of the body.
"""
from __future__ import annotations

import os

import pytest

_FIXTURE_PLUGIN = (
    "from agent.secret_sources.base import SECRET_SOURCE_API_VERSION, FetchResult, SecretSource\n\n"
    "class FixtureVault(SecretSource):\n"
    "    name = 'fixturevault'\n"
    "    label = 'Fixture vault'\n"
    "    api_version = SECRET_SOURCE_API_VERSION\n"
    "    shape = 'bulk'\n\n"
    "    def fetch(self, cfg, home_path):\n"
    # SYSTEMD_ONLY_TOKEN is the operator's (set before the source runs): supplied, never written.
    "        return FetchResult(secrets={'VAULT_ONLY_KEY': 'from-plugin', 'SYSTEMD_ONLY_TOKEN': 'from-plugin'})\n\n"
    "def register(ctx):\n"
    "    ctx.register_secret_source(FixtureVault())\n"
)


@pytest.fixture
def launch_home(tmp_path, monkeypatch):
    import agent.secret_sources.registry as reg
    from hermes_cli import env_loader

    home = tmp_path / ".hermes"
    plugin_dir = home / "plugins" / "fixture-secret-source"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: fixture-secret-source\nversion: 0.1.0\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(_FIXTURE_PLUGIN, encoding="utf-8")
    (home / "profiles" / "ops").mkdir(parents=True)
    (home / "config.yaml").write_text(
        "allow_all_users: true\n"
        "telegram:\n  allow_from: ['111']\n"
        "plugins:\n  enabled: [fixture-secret-source]\n"
        "secrets:\n  fixturevault:\n    enabled: true\n",
        encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in ("TELEGRAM_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS", "VAULT_ONLY_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SYSTEMD_ONLY_TOKEN", "env-only")  # e.g. systemd Environment=, no .env line
    reg._reset_registry_for_tests()
    env_loader.reset_secret_source_cache()
    try:
        yield home
    finally:
        os.environ.pop("VAULT_ONLY_KEY", None)
        reg._reset_registry_for_tests()
        env_loader.reset_secret_source_cache()


def test_forced_rediscovery_keeps_launch_env_in_the_bound_launch_scope(launch_home):
    from agent.secret_scope import current_secret_scope, set_multiplex_active
    from gateway.config import load_gateway_config
    from gateway.platforms._shared import platform_gate_env
    from gateway.run import _profile_runtime_scope
    from hermes_cli.plugins import get_plugin_manager
    from tui_gateway.launch_profile_policy import activate_multi_profile_hosting

    load_gateway_config()  # the launch YAML bridge: allow_from -> TELEGRAM_ALLOWED_USERS, allow-all grant
    assert os.environ["TELEGRAM_ALLOWED_USERS"] == "111"
    manager = get_plugin_manager()
    manager.discover_and_load()  # gateway boot discovery
    activate_multi_profile_hosting()
    try:
        with _profile_runtime_scope(launch_home):  # a launch-profile turn
            assert platform_gate_env("TELEGRAM_ALLOWED_USERS") == "111"
            # A tool in the turn force-refreshes discovery (image_gen / TTS / STT "provider missing" retry).
            manager.discover_and_load(force=True)
            scope = current_secret_scope()
            assert scope.get("VAULT_ONLY_KEY") == "from-plugin"
            assert platform_gate_env("GATEWAY_ALLOW_ALL_USERS") == "true"
            # The list that narrows the grant, and the env-only credential, survive the refresh.
            assert platform_gate_env("TELEGRAM_ALLOWED_USERS") == "111"
            assert scope.get("SYSTEMD_ONLY_TOKEN") == "env-only"
        # A secondary's refresh stays files-only: the launch env never reaches its scope.
        from agent.secret_scope import refresh_installed_secret_scope
        with _profile_runtime_scope(launch_home / "profiles" / "ops"):
            assert refresh_installed_secret_scope(launch_home / "profiles" / "ops") is True
            assert "SYSTEMD_ONLY_TOKEN" not in (current_secret_scope() or {})
            assert platform_gate_env("TELEGRAM_ALLOWED_USERS") == ""
    finally:
        set_multiplex_active(False)


def test_a_removed_plugin_sources_boot_write_leaves_the_frozen_launch_env(launch_home):
    """Boot discovery writes the plugin's value into os.environ, activation freezes that env, then
    the plugin is disabled and a forced discovery revokes the write. The launch scope (refreshed in
    the running turn, and every fresh launch bind after it) must lose the value with os.environ,
    while the operator's own value for a name the source also supplied stays."""
    from agent.secret_scope import current_secret_scope, set_multiplex_active
    from gateway.run import _profile_runtime_scope
    from hermes_cli.plugins import get_plugin_manager
    from tui_gateway.launch_profile_policy import activate_multi_profile_hosting, launch_secret_scope

    manager = get_plugin_manager()
    manager.discover_and_load()  # gateway boot discovery: the source writes VAULT_ONLY_KEY
    assert os.environ["VAULT_ONLY_KEY"] == "from-plugin"
    activate_multi_profile_hosting()
    try:
        config = launch_home / "config.yaml"
        config.write_text(config.read_text(encoding="utf-8").replace(
            "enabled: [fixture-secret-source]", "enabled: []"), encoding="utf-8")
        with _profile_runtime_scope(launch_home):
            assert current_secret_scope().get("VAULT_ONLY_KEY") == "from-plugin"
            manager.discover_and_load(force=True)
            assert "VAULT_ONLY_KEY" not in os.environ
            assert "VAULT_ONLY_KEY" not in current_secret_scope()
            assert current_secret_scope().get("SYSTEMD_ONLY_TOKEN") == "env-only"
        fresh = launch_secret_scope(launch_home)
        assert "VAULT_ONLY_KEY" not in fresh
        assert fresh.get("SYSTEMD_ONLY_TOKEN") == "env-only"
    finally:
        set_multiplex_active(False)


def test_a_rotated_then_removed_sources_write_leaves_the_frozen_launch_env(launch_home):
    """An override source rotates its value after activation, then is removed: the frozen launch env
    must lose the pre-rotation value too, not only the source's last write."""
    from agent.secret_scope import current_secret_scope, set_multiplex_active
    from gateway.run import _profile_runtime_scope
    from hermes_cli.plugins import get_plugin_manager
    from tui_gateway.launch_profile_policy import activate_multi_profile_hosting, launch_secret_scope

    plugin = launch_home / "plugins" / "fixture-secret-source" / "__init__.py"
    plugin.write_text(plugin.read_text(encoding="utf-8").replace(
        "'VAULT_ONLY_KEY': 'from-plugin'", "'VAULT_ONLY_KEY': open(f'{home_path}/vault.txt').read()"), encoding="utf-8")
    config = launch_home / "config.yaml"
    config.write_text(config.read_text(encoding="utf-8").replace(
        "enabled: true\n", "enabled: true\n    override_existing: true\n"), encoding="utf-8")
    (launch_home / "vault.txt").write_text("v1", encoding="utf-8")
    manager = get_plugin_manager()
    manager.discover_and_load()  # boot: writes v1, and SYSTEMD_ONLY_TOKEN over the operator's value
    activate_multi_profile_hosting()
    try:
        (launch_home / "vault.txt").write_text("v2", encoding="utf-8")
        with _profile_runtime_scope(launch_home):
            manager.discover_and_load(force=True)
            assert current_secret_scope().get("VAULT_ONLY_KEY") == "v2"
        config.write_text(config.read_text(encoding="utf-8").replace(
            "enabled: [fixture-secret-source]", "enabled: []"), encoding="utf-8")
        with _profile_runtime_scope(launch_home):
            manager.discover_and_load(force=True)
            assert "VAULT_ONLY_KEY" not in current_secret_scope()
        fresh = launch_secret_scope(launch_home)
        assert "VAULT_ONLY_KEY" not in fresh
        assert fresh.get("SYSTEMD_ONLY_TOKEN") == "env-only"
    finally:
        set_multiplex_active(False)


def _rotating_override_source(launch_home, fetch_prelude: str = "") -> None:
    """The fixture source reads VAULT_ONLY_KEY from vault.txt (v1) as an override_existing source."""
    plugin = launch_home / "plugins" / "fixture-secret-source" / "__init__.py"
    plugin.write_text(plugin.read_text(encoding="utf-8").replace(
        "        return FetchResult(", fetch_prelude + "        return FetchResult(").replace(
        "'VAULT_ONLY_KEY': 'from-plugin'", "'VAULT_ONLY_KEY': open(f'{home_path}/vault.txt').read()"),
        encoding="utf-8")
    config = launch_home / "config.yaml"
    config.write_text(config.read_text(encoding="utf-8").replace(
        "enabled: true\n", "enabled: true\n    override_existing: true\n"), encoding="utf-8")
    (launch_home / "vault.txt").write_text("v1", encoding="utf-8")


def test_the_launch_launchers_env_file_template_follows_the_frozen_env_through_a_rotation_and_removal(launch_home):
    """The plugin host's launcher builds the launch scope again at a start, its ``.env`` templates resolved
    against the frozen launch env. A template naming a source's value follows that env as it follows the
    source: the rotated value after a rotation, the template's default after the source is removed, as a
    launch-profile body reads it."""
    from agent.secret_scope import set_multiplex_active
    from gateway.run import _profile_runtime_scope
    from hermes_cli.plugins import get_plugin_manager
    from tui_gateway.launch_profile_policy import (activate_multi_profile_hosting, launch_secret_scope,
                                                   launch_template_env)

    _rotating_override_source(launch_home)
    (launch_home / ".env").write_text("RUNNER_TAG=${VAULT_ONLY_KEY:-none}/tag\n", encoding="utf-8")
    config = launch_home / "config.yaml"
    manager = get_plugin_manager()
    manager.discover_and_load()  # boot: the source writes v1
    activate_multi_profile_hosting()
    try:
        seen = []
        (launch_home / "vault.txt").write_text("v2", encoding="utf-8")
        for change in ("rotate", "remove"):
            if change == "remove":
                config.write_text(config.read_text(encoding="utf-8").replace(
                    "enabled: [fixture-secret-source]", "enabled: []"), encoding="utf-8")
            with _profile_runtime_scope(launch_home):
                manager.discover_and_load(force=True)
            seen.append((launch_secret_scope(launch_home, template_env=launch_template_env())["RUNNER_TAG"],
                         launch_secret_scope(launch_home)["RUNNER_TAG"]))
        assert seen == [("v2/tag", "v2/tag"), ("none/tag", "none/tag")]
    finally:
        set_multiplex_active(False)


def test_the_launch_launchers_env_file_template_keeps_a_sources_value_equal_to_what_the_env_file_published(
        launch_home, monkeypatch):
    """A name a secret source holds in the frozen launch env is the source's value, even after it rotates to the
    value the launch ``.env`` once published for that name: the launcher's ``.env`` templates see it, rather than
    taking it for dotenv output and peeling it back to what the env held before the ``.env`` loaded."""
    from agent.secret_scope import set_multiplex_active
    from gateway.run import _profile_runtime_scope
    from hermes_cli import env_loader
    from hermes_cli.plugins import get_plugin_manager
    from tui_gateway.launch_profile_policy import (activate_multi_profile_hosting, launch_secret_scope,
                                                   launch_template_env)

    monkeypatch.setattr(env_loader, "_DOTENV_PUBLISHED", {})
    monkeypatch.setenv("RUNNER_TAG", "registered-for-restore")
    monkeypatch.delenv("RUNNER_TAG")
    _rotating_override_source(launch_home)
    # The template first, so the file's own later line cannot stand in for the env's value.
    (launch_home / ".env").write_text("RUNNER_TAG=${VAULT_ONLY_KEY:-none}/tag\nVAULT_ONLY_KEY=same\n", encoding="utf-8")
    env_loader.load_hermes_dotenv(hermes_home=launch_home, load_external_secrets=False)  # publishes "same"
    manager = get_plugin_manager()
    manager.discover_and_load()  # boot: the override source writes v1 over it
    activate_multi_profile_hosting()
    try:
        (launch_home / "vault.txt").write_text("same", encoding="utf-8")
        with _profile_runtime_scope(launch_home):
            manager.discover_and_load(force=True)  # the source rotates to the value the .env published
        assert launch_secret_scope(launch_home, template_env=launch_template_env())["RUNNER_TAG"] == "same/tag"
    finally:
        set_multiplex_active(False)


def test_a_concurrent_write_during_the_fetch_never_enters_the_frozen_launch_env(launch_home):
    """Another thread writes the source's name into os.environ while the vault fetch runs (stood in
    for by the fetch itself, once foreign.txt exists); the source re-applies its unchanged value.
    The frozen launch env must keep the value the source published, not the foreign one."""
    from agent.secret_scope import set_multiplex_active
    from gateway.run import _profile_runtime_scope
    from hermes_cli.plugins import get_plugin_manager
    from tui_gateway.launch_profile_policy import activate_multi_profile_hosting, capture_launch_env

    _rotating_override_source(launch_home, (
        "        import os, pathlib\n"
        "        foreign = pathlib.Path(home_path) / 'foreign.txt'\n"
        "        if foreign.exists():\n"
        "            os.environ['VAULT_ONLY_KEY'] = foreign.read_text()\n"))
    manager = get_plugin_manager()
    manager.discover_and_load()  # boot: the source writes v1
    activate_multi_profile_hosting()
    try:
        assert capture_launch_env()["VAULT_ONLY_KEY"] == "v1"
        (launch_home / "foreign.txt").write_text("from-secondary", encoding="utf-8")
        with _profile_runtime_scope(launch_home):
            manager.discover_and_load(force=True)
        assert capture_launch_env()["VAULT_ONLY_KEY"] == "v1"
        # Removing the source takes back only its own value: the foreign one stays in os.environ.
        config = launch_home / "config.yaml"
        config.write_text(config.read_text(encoding="utf-8").replace(
            "enabled: [fixture-secret-source]", "enabled: []"), encoding="utf-8")
        with _profile_runtime_scope(launch_home):
            manager.discover_and_load(force=True)
        assert os.environ["VAULT_ONLY_KEY"] == "from-secondary"
        assert "VAULT_ONLY_KEY" not in capture_launch_env()
    finally:
        set_multiplex_active(False)


def _remove_source_in_a_launch_turn(launch_home, manager) -> None:
    from gateway.run import _profile_runtime_scope

    config = launch_home / "config.yaml"
    config.write_text(config.read_text(encoding="utf-8").replace(
        "enabled: [fixture-secret-source]", "enabled: []"), encoding="utf-8")
    with _profile_runtime_scope(launch_home):
        manager.discover_and_load(force=True)


@pytest.mark.parametrize("foreign_write", ["during_a_fetch", "between_refreshes"])
def test_a_foreign_value_never_enters_the_frozen_launch_env_when_a_later_refresh_precedes_removal(
        launch_home, foreign_write):
    """Another writer puts its value in os.environ after activation (during a fetch, or between two
    refreshes), one more refresh re-applies the source over it, then the source is removed. The
    other writer keeps its value in os.environ, but the frozen launch env only ever loses the
    source's own write: it must never gain the foreign value (the launch env never saw it)."""
    from agent.secret_scope import set_multiplex_active
    from gateway.run import _profile_runtime_scope
    from hermes_cli.plugins import get_plugin_manager
    from tui_gateway.launch_profile_policy import activate_multi_profile_hosting, capture_launch_env

    _rotating_override_source(launch_home, (
        "        import os, pathlib\n"
        "        foreign = pathlib.Path(home_path) / 'foreign.txt'\n"
        "        if foreign.exists():\n"
        "            os.environ['VAULT_ONLY_KEY'] = foreign.read_text()\n"
        "            foreign.unlink()\n"))
    manager = get_plugin_manager()
    manager.discover_and_load()  # boot: the source writes v1
    activate_multi_profile_hosting()
    try:
        if foreign_write == "during_a_fetch":
            (launch_home / "foreign.txt").write_text("from-secondary", encoding="utf-8")
            with _profile_runtime_scope(launch_home):
                manager.discover_and_load(force=True)
        else:
            os.environ["VAULT_ONLY_KEY"] = "from-secondary"
        with _profile_runtime_scope(launch_home):
            manager.discover_and_load(force=True)  # re-applies v1 over the foreign value
        assert capture_launch_env()["VAULT_ONLY_KEY"] == "v1"
        _remove_source_in_a_launch_turn(launch_home, manager)
        assert os.environ["VAULT_ONLY_KEY"] == "from-secondary"
        assert "VAULT_ONLY_KEY" not in capture_launch_env()
    finally:
        set_multiplex_active(False)


def _foreign_fetch_source(launch_home) -> None:
    """The rotating override source; a fetch also writes foreign.txt's value (once) into os.environ,
    standing in for another writer racing the fetch, and fails while fail.txt exists."""
    _rotating_override_source(launch_home, (
        "        import os, pathlib\n"
        "        from agent.secret_sources.base import ErrorKind\n"
        "        foreign = pathlib.Path(home_path) / 'foreign.txt'\n"
        "        if foreign.exists():\n"
        "            os.environ['VAULT_ONLY_KEY'] = foreign.read_text()\n"
        "            foreign.unlink()\n"
        "        if (pathlib.Path(home_path) / 'fail.txt').exists():\n"
        "            return FetchResult().fail('vault down', ErrorKind.NETWORK)\n"))


def _refresh(launch_home, manager) -> None:
    from gateway.run import _profile_runtime_scope

    with _profile_runtime_scope(launch_home):
        manager.discover_and_load(force=True)


@pytest.mark.parametrize("scenario", [
    "foreign_during_fetch_reapply_rotate",   # foreign value in a fetch, re-apply, rotate, remove
    "foreign_between_refreshes_rotate",      # foreign value, then rotation to v2, then remove
    "foreign_then_failed_fetch",             # foreign value, a refresh whose fetch fails, remove
    "rotate_twice",                          # v1 -> v2 -> v3, then remove
])
def test_the_frozen_launch_env_loses_an_override_sources_value_however_os_environ_diverged(
        launch_home, scenario):
    """However os.environ diverged from the source's write after activation, removing the source
    takes its value out of the frozen launch env, and the snapshot never holds a foreign value."""
    from agent.secret_scope import set_multiplex_active
    from hermes_cli.plugins import get_plugin_manager
    from tui_gateway.launch_profile_policy import activate_multi_profile_hosting, capture_launch_env

    _foreign_fetch_source(launch_home)
    manager = get_plugin_manager()
    manager.discover_and_load()  # boot: the source writes v1
    activate_multi_profile_hosting()
    seen = []
    try:
        if scenario == "foreign_during_fetch_reapply_rotate":
            (launch_home / "foreign.txt").write_text("from-secondary", encoding="utf-8")
            _refresh(launch_home, manager)
            seen.append(capture_launch_env().get("VAULT_ONLY_KEY"))
            _refresh(launch_home, manager)  # re-applies v1 over the foreign value
            seen.append(capture_launch_env().get("VAULT_ONLY_KEY"))
            (launch_home / "vault.txt").write_text("v2", encoding="utf-8")
            _refresh(launch_home, manager)
        elif scenario == "foreign_between_refreshes_rotate":
            os.environ["VAULT_ONLY_KEY"] = "from-secondary"
            (launch_home / "vault.txt").write_text("v2", encoding="utf-8")
            _refresh(launch_home, manager)
        elif scenario == "foreign_then_failed_fetch":
            os.environ["VAULT_ONLY_KEY"] = "from-secondary"
            (launch_home / "fail.txt").write_text("", encoding="utf-8")
            _refresh(launch_home, manager)
            assert os.environ["VAULT_ONLY_KEY"] == "from-secondary"
        else:
            for value in ("v2", "v3"):
                (launch_home / "vault.txt").write_text(value, encoding="utf-8")
                _refresh(launch_home, manager)
                seen.append(capture_launch_env().get("VAULT_ONLY_KEY"))
            assert seen == ["v2", "v3"]
        seen.append(capture_launch_env().get("VAULT_ONLY_KEY"))
        assert "from-secondary" not in seen
        _remove_source_in_a_launch_turn(launch_home, manager)
        if scenario != "rotate_twice":
            assert os.environ["VAULT_ONLY_KEY"] == "from-secondary"  # the other writer keeps os.environ
        assert "VAULT_ONLY_KEY" not in capture_launch_env()
    finally:
        set_multiplex_active(False)


@pytest.mark.parametrize("foreign_cleared_and_reapplied", [False, True])
def test_a_displaced_default_sources_boot_write_leaves_the_frozen_launch_env(
        launch_home, foreign_cleared_and_reapplied):
    """A default (non-override) source's boot write is frozen at activation, then another writer
    displaces it in os.environ and a refresh runs. Optionally the foreign value is then cleared and
    one more refresh re-applies the source. Removal must take from-plugin out of the frozen launch
    env either way, and the snapshot never holds the foreign value."""
    from agent.secret_scope import set_multiplex_active
    from hermes_cli.plugins import get_plugin_manager
    from tui_gateway.launch_profile_policy import activate_multi_profile_hosting, capture_launch_env

    manager = get_plugin_manager()
    manager.discover_and_load()  # boot: the source writes from-plugin
    activate_multi_profile_hosting()
    try:
        assert capture_launch_env()["VAULT_ONLY_KEY"] == "from-plugin"
        os.environ["VAULT_ONLY_KEY"] = "from-secondary"
        _refresh(launch_home, manager)
        assert capture_launch_env()["VAULT_ONLY_KEY"] == "from-plugin"
        if foreign_cleared_and_reapplied:
            del os.environ["VAULT_ONLY_KEY"]
            _refresh(launch_home, manager)
            assert os.environ["VAULT_ONLY_KEY"] == "from-plugin"
            assert capture_launch_env()["VAULT_ONLY_KEY"] == "from-plugin"
        _remove_source_in_a_launch_turn(launch_home, manager)
        assert os.environ.get("VAULT_ONLY_KEY") == (None if foreign_cleared_and_reapplied else "from-secondary")
        assert "VAULT_ONLY_KEY" not in capture_launch_env()
    finally:
        set_multiplex_active(False)


def test_removal_gives_the_frozen_launch_env_back_its_own_pre_write_value(launch_home, monkeypatch):
    """The operator had VAULT_ONLY_KEY=operator before the override source wrote v1 at boot. After
    a foreign write, a re-apply and a rotation, removal restores the snapshot's own pre-write value,
    never the foreign one and never the rotated value."""
    from agent.secret_scope import set_multiplex_active
    from hermes_cli.plugins import get_plugin_manager
    from tui_gateway.launch_profile_policy import activate_multi_profile_hosting, capture_launch_env

    monkeypatch.setenv("VAULT_ONLY_KEY", "operator")
    _foreign_fetch_source(launch_home)
    manager = get_plugin_manager()
    manager.discover_and_load()  # boot: v1 over the operator's value
    activate_multi_profile_hosting()
    try:
        assert capture_launch_env()["VAULT_ONLY_KEY"] == "v1"
        os.environ["VAULT_ONLY_KEY"] = "from-secondary"
        _refresh(launch_home, manager)  # re-applies v1 over the foreign value
        (launch_home / "vault.txt").write_text("v2", encoding="utf-8")
        _refresh(launch_home, manager)
        assert capture_launch_env()["VAULT_ONLY_KEY"] == "v2"
        _remove_source_in_a_launch_turn(launch_home, manager)
        assert capture_launch_env()["VAULT_ONLY_KEY"] == "operator"
    finally:
        set_multiplex_active(False)


def test_a_sources_first_write_after_activation_stays_out_of_the_frozen_launch_env(launch_home):
    """A source enabled only after activation writes and then rotates: neither value is the launch
    env's, so the snapshot never gains the name (only a re-application over a pre-activation write
    is followed)."""
    from agent.secret_scope import set_multiplex_active
    from gateway.run import _profile_runtime_scope
    from hermes_cli.plugins import get_plugin_manager
    from tui_gateway.launch_profile_policy import activate_multi_profile_hosting, capture_launch_env

    _rotating_override_source(launch_home)
    config = launch_home / "config.yaml"
    config.write_text(config.read_text(encoding="utf-8").replace(
        "enabled: [fixture-secret-source]", "enabled: []"), encoding="utf-8")
    manager = get_plugin_manager()
    manager.discover_and_load()
    activate_multi_profile_hosting()
    try:
        config.write_text(config.read_text(encoding="utf-8").replace(
            "enabled: []", "enabled: [fixture-secret-source]"), encoding="utf-8")
        for value in ("v1", "v2"):
            (launch_home / "vault.txt").write_text(value, encoding="utf-8")
            with _profile_runtime_scope(launch_home):
                manager.discover_and_load(force=True)
            assert os.environ["VAULT_ONLY_KEY"] == value
            assert "VAULT_ONLY_KEY" not in capture_launch_env()
    finally:
        set_multiplex_active(False)


def test_a_source_enabled_after_activation_never_revokes_an_equal_operator_value(launch_home, monkeypatch):
    """The operator's VAULT_ONLY_KEY=from-plugin is frozen at activation; another writer then sets
    late-foreign in os.environ, and an override source enabled only now writes the same string and
    is removed again. The snapshot holds no write of that source, so it keeps the operator's value:
    matching the string is not ownership."""
    from agent.secret_scope import set_multiplex_active
    from hermes_cli.plugins import get_plugin_manager
    from tui_gateway.launch_profile_policy import activate_multi_profile_hosting, capture_launch_env

    monkeypatch.setenv("VAULT_ONLY_KEY", "from-plugin")
    config = launch_home / "config.yaml"
    config.write_text(config.read_text(encoding="utf-8").replace(
        "enabled: true\n", "enabled: true\n    override_existing: true\n").replace(
        "enabled: [fixture-secret-source]", "enabled: []"), encoding="utf-8")
    manager = get_plugin_manager()
    manager.discover_and_load()
    activate_multi_profile_hosting()
    try:
        os.environ["VAULT_ONLY_KEY"] = "late-foreign"
        config.write_text(config.read_text(encoding="utf-8").replace(
            "enabled: []", "enabled: [fixture-secret-source]"), encoding="utf-8")
        _refresh(launch_home, manager)
        assert os.environ["VAULT_ONLY_KEY"] == "from-plugin"
        _remove_source_in_a_launch_turn(launch_home, manager)
        assert os.environ["VAULT_ONLY_KEY"] == "late-foreign"
        assert capture_launch_env()["VAULT_ONLY_KEY"] == "from-plugin"
    finally:
        set_multiplex_active(False)


def test_an_activation_racing_the_pass_start_never_gives_the_snapshot_a_foreign_prior(launch_home, monkeypatch):
    """Activation lands just as a pass starts, and another writer sets the name before the pass copies
    os.environ. The pass must not record that post-activation value as the snapshot's prior: removing
    the source would otherwise write it into the frozen launch env, which keeps the operator's value."""
    from agent.secret_scope import set_multiplex_active
    from hermes_cli.plugins import get_plugin_manager
    import tui_gateway.launch_profile_policy as policy
    from tui_gateway.launch_profile_policy import activate_multi_profile_hosting, capture_launch_env

    monkeypatch.setenv("VAULT_ONLY_KEY", "from-plugin")
    config = launch_home / "config.yaml"
    config.write_text(config.read_text(encoding="utf-8").replace(
        "enabled: true\n", "enabled: true\n    override_existing: true\n").replace(
        "enabled: [fixture-secret-source]", "enabled: []"), encoding="utf-8")
    manager = get_plugin_manager()
    manager.discover_and_load()
    config.write_text(config.read_text(encoding="utf-8").replace(
        "enabled: []", "enabled: [fixture-secret-source]"), encoding="utf-8")

    real_captured = policy._launch_env_captured
    raced = []

    def captured_then_activate():
        was = real_captured()
        if not raced:
            raced.append(True)
            activate_multi_profile_hosting()
            os.environ["VAULT_ONLY_KEY"] = "F-foreign"
        return was

    monkeypatch.setattr(policy, "_launch_env_captured", captured_then_activate)
    try:
        _refresh(launch_home, manager)
        assert raced
        assert capture_launch_env()["VAULT_ONLY_KEY"] == "from-plugin"
        _remove_source_in_a_launch_turn(launch_home, manager)
        assert capture_launch_env()["VAULT_ONLY_KEY"] == "from-plugin"
    finally:
        set_multiplex_active(False)


def test_an_activation_landing_after_the_publish_still_lets_removal_revoke_the_write(launch_home, monkeypatch):
    """Activation lands mid-pass, after the source's first write is published but before the pass
    records it: the snapshot froze that write, so removing the source must still take it out."""
    from agent.secret_scope import set_multiplex_active
    from hermes_cli import env_loader
    from hermes_cli.plugins import get_plugin_manager
    from tui_gateway.launch_profile_policy import activate_multi_profile_hosting, capture_launch_env

    real_sanitize = env_loader._sanitize_loaded_credentials

    def sanitize_then_activate():
        real_sanitize()
        activate_multi_profile_hosting()

    monkeypatch.setattr(env_loader, "_sanitize_loaded_credentials", sanitize_then_activate)
    manager = get_plugin_manager()
    try:
        manager.discover_and_load()  # boot: the source writes VAULT_ONLY_KEY, activation lands before it is recorded
        assert capture_launch_env()["VAULT_ONLY_KEY"] == "from-plugin"
        monkeypatch.setattr(env_loader, "_sanitize_loaded_credentials", real_sanitize)
        _remove_source_in_a_launch_turn(launch_home, manager)
        assert "VAULT_ONLY_KEY" not in capture_launch_env()
        assert capture_launch_env()["SYSTEMD_ONLY_TOKEN"] == "env-only"
    finally:
        set_multiplex_active(False)

def test_a_source_reapplying_over_a_value_it_did_not_write_leaves_the_frozen_launch_env(launch_home):
    """The source wrote v1 at boot, then the operator's value replaced it before activation: the
    frozen env holds the operator's value, and the source's next write is not over its own earlier
    write, so the snapshot must not follow it."""
    from agent.secret_scope import set_multiplex_active
    from gateway.run import _profile_runtime_scope
    from hermes_cli.plugins import get_plugin_manager
    from tui_gateway.launch_profile_policy import activate_multi_profile_hosting, capture_launch_env

    _rotating_override_source(launch_home)
    manager = get_plugin_manager()
    manager.discover_and_load()  # boot: the source writes v1
    os.environ["VAULT_ONLY_KEY"] = "operator-late"
    activate_multi_profile_hosting()
    try:
        (launch_home / "vault.txt").write_text("v2", encoding="utf-8")
        with _profile_runtime_scope(launch_home):
            manager.discover_and_load(force=True)
        assert os.environ["VAULT_ONLY_KEY"] == "v2"
        assert capture_launch_env()["VAULT_ONLY_KEY"] == "operator-late"
    finally:
        set_multiplex_active(False)


def test_a_rotated_write_the_ascii_sweep_rewrote_is_still_followed(launch_home):
    """A rotation to a value carrying a zero-width space is published, then stripped to ASCII by the
    credential sweep: the snapshot follows the sanitized value, so the later removal still drops it."""
    from agent.secret_scope import set_multiplex_active
    from gateway.run import _profile_runtime_scope
    from hermes_cli.plugins import get_plugin_manager
    from tui_gateway.launch_profile_policy import activate_multi_profile_hosting, capture_launch_env

    _rotating_override_source(launch_home)
    manager = get_plugin_manager()
    manager.discover_and_load()  # boot: the source writes v1
    activate_multi_profile_hosting()
    try:
        (launch_home / "vault.txt").write_text("v2\u200b", encoding="utf-8")
        with _profile_runtime_scope(launch_home):
            manager.discover_and_load(force=True)
        assert os.environ["VAULT_ONLY_KEY"] == "v2"
        assert capture_launch_env()["VAULT_ONLY_KEY"] == "v2"
        config = launch_home / "config.yaml"
        config.write_text(config.read_text(encoding="utf-8").replace(
            "enabled: [fixture-secret-source]", "enabled: []"), encoding="utf-8")
        with _profile_runtime_scope(launch_home):
            manager.discover_and_load(force=True)
        assert "VAULT_ONLY_KEY" not in capture_launch_env()
    finally:
        set_multiplex_active(False)


@pytest.mark.parametrize("rotate", [False, True])
def test_an_activation_landing_after_the_tracking_step_still_lets_removal_revoke_the_write(
        launch_home, monkeypatch, rotate):
    """Activation lands after the pass tracked the source's write (no snapshot yet) but before it
    published its write record: capture must still learn of the write, or removal leaves it frozen."""
    from agent.secret_scope import set_multiplex_active
    from hermes_cli.plugins import get_plugin_manager
    import tui_gateway.launch_profile_policy as policy
    from tui_gateway.launch_profile_policy import activate_multi_profile_hosting, capture_launch_env

    if rotate:
        _rotating_override_source(launch_home)
    manager = get_plugin_manager()
    if rotate:
        manager.discover_and_load()  # boot: the source writes v1, recorded before any activation
        (launch_home / "vault.txt").write_text("v2", encoding="utf-8")
    real_track = policy._track_launch_env_write

    def track_then_activate(*args, **kwargs):
        real_track(*args, **kwargs)
        if args[1] == "VAULT_ONLY_KEY":
            activate_multi_profile_hosting()

    monkeypatch.setattr(policy, "_track_launch_env_write", track_then_activate)
    try:
        if rotate:
            manager.discover_and_load(force=True)
        else:
            manager.discover_and_load()  # boot: the source's first write
        assert capture_launch_env()["VAULT_ONLY_KEY"] == ("v2" if rotate else "from-plugin")
        monkeypatch.setattr(policy, "_track_launch_env_write", real_track)
        _remove_source_in_a_launch_turn(launch_home, manager)
        assert "VAULT_ONLY_KEY" not in capture_launch_env()
    finally:
        set_multiplex_active(False)

def test_a_refresh_for_one_home_never_replaces_another_homes_installed_scope(launch_home):
    """A scope stamped for a secondary (no home override) is not the launch home's to refresh: the
    launch profile's files and frozen env must not replace it."""
    from agent.secret_scope import (current_secret_scope, refresh_installed_secret_scope,
                                    reset_secret_scope, set_multiplex_active, set_secret_scope)
    from tui_gateway.launch_profile_policy import activate_multi_profile_hosting

    (launch_home / ".env").write_text("LAUNCH_FILE_KEY=launch\n", encoding="utf-8")
    ops = launch_home / "profiles" / "ops"
    activate_multi_profile_hosting()
    token = set_secret_scope({"OPS_KEY": "ops"}, profile_home=str(ops))
    try:
        assert refresh_installed_secret_scope(launch_home) is False
        assert dict(current_secret_scope()) == {"OPS_KEY": "ops"}
    finally:
        reset_secret_scope(token)
        set_multiplex_active(False)


@pytest.mark.require_symlinks
def test_an_unstamped_scope_on_a_pinned_host_belongs_to_the_pinned_home(tmp_path, monkeypatch):
    """A host that pins its launch home and mirrors the served profile into HERMES_HOME (Hermes
    WebUI): an unstamped scope is the pinned launch home's, so the mirrored profile's refresh must
    not replace it, and a symlinked spelling of the pinned home still owns it."""
    from agent import secret_scope
    from hermes_constants import pin_process_hermes_home

    launch = tmp_path / "launch"
    ops = launch / "profiles" / "ops"
    ops.mkdir(parents=True)
    (ops / ".env").write_text("OPS_KEY=o\n", encoding="utf-8")
    link = tmp_path / "launch-link"
    link.symlink_to(launch, target_is_directory=True)
    pin_process_hermes_home(launch)
    monkeypatch.setenv("HERMES_HOME", str(ops))  # the mirror
    token = secret_scope.set_secret_scope({"LAUNCH_KEY": "l"})
    try:
        assert secret_scope.refresh_installed_secret_scope(ops) is False
        assert dict(secret_scope.current_secret_scope()) == {"LAUNCH_KEY": "l"}
        assert secret_scope.refresh_installed_secret_scope(link) is True
        assert "OPS_KEY" not in secret_scope.current_secret_scope()
    finally:
        secret_scope.reset_secret_scope(token)
        pin_process_hermes_home(None)

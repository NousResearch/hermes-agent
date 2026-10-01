"""Component-scoped updates: `hermes update` respects `components:` (#123828).

A Desktop-only user sets e.g. ``components: {tui: false, web: false}`` and the
update must skip the ui-tui/web npm install + build entirely. Defaults (no
``components:`` key, malformed values) build everything, as before.

The gate is unioned over every live home sharing the install (#124495): the skipped
artifacts live in the checkout, not in a home, so one profile's ``components:`` must not
narrow a tree its sibling profiles still build from.
"""
from __future__ import annotations

import json

import pytest


def _checkout_with_frontends(tmp_path):
    root = tmp_path / "checkout"
    for workspace in ("ui-tui", "web"):
        directory = root / workspace
        directory.mkdir(parents=True)
        (directory / "package.json").write_text('{"name": "%s"}\n' % workspace, encoding="utf-8")
    return root


@pytest.fixture
def update_spy(monkeypatch, tmp_path_factory):
    """Run build_update_products with every side effect recorded, not executed."""
    from hermes_cli import source_build

    calls = []

    monkeypatch.setattr(source_build, "source_build_env", lambda **kwargs: {"PATH": "spy"})
    monkeypatch.setattr(
        source_build, "prepare_source_dependencies",
        lambda project_root, workspaces, **kwargs: calls.append(("deps", tuple(workspaces))))
    monkeypatch.setattr(
        source_build, "build_source_tui",
        lambda project_root, **kwargs: calls.append(("tui",)))
    monkeypatch.setattr(
        source_build, "build_source_web",
        lambda project_root, **kwargs: calls.append(("web",)))

    import hermes_cli.update_stage as update_stage
    monkeypatch.setattr(update_stage, "publish_stage", lambda *args, **kwargs: None)
    import hermes_cli.main_install_repair as repair
    monkeypatch.setattr(repair, "_warn_configured_features_missing_deps", lambda: None)
    import hermes_cli.memory_provider_migration as migration
    monkeypatch.setattr(migration, "migrate_all_homes", lambda: calls.append(("migrate",)))
    import hermes_cli.main_desktop as desktop
    monkeypatch.setattr(
        desktop, "build_prepared_desktop",
        lambda *args, **kwargs: calls.append(("desktop",)))
    monkeypatch.setattr(
        desktop, "_install_rebuilt_desktop_app", lambda *args, **kwargs: ([], []))

    def configure(*configs):
        """Make *configs* the live homes sharing this install, one config.yaml each.

        ``configure({})`` is the single default home; two configs are two profile homes that
        share one checkout (#124495) — ``_components_gate`` reads the union."""
        import pm.plugins_state as plugins_state

        root = tmp_path_factory.mktemp("homes")
        homes = []
        for index, config in enumerate(configs or [{}]):
            home = root / f"home{index}"
            home.mkdir()
            (home / "config.yaml").write_text(json.dumps(config) + "\n", encoding="utf-8")
            homes.append(home)
        monkeypatch.setattr(plugins_state, "dependency_homes", lambda: list(homes))

    return calls, configure


def _run(root):
    from hermes_cli.source_build import build_update_products
    build_update_products(root, desktop=False)


def test_update_builds_everything_by_default(tmp_path, update_spy):
    calls, configure = update_spy
    configure({})
    _run(_checkout_with_frontends(tmp_path))
    kinds = [call[0] for call in calls]
    assert "tui" in kinds and "web" in kinds
    assert calls[0] == ("deps", ("ui-tui", "web"))


def test_update_skips_disabled_tui_and_web(tmp_path, update_spy):
    calls, configure = update_spy
    configure({"components": {"desktop": True, "tui": False, "web": False}})
    _run(_checkout_with_frontends(tmp_path))
    kinds = [call[0] for call in calls]
    assert "tui" not in kinds and "web" not in kinds
    assert not any(kind == "deps" for kind in kinds)


def test_update_disables_only_tui(tmp_path, update_spy):
    calls, configure = update_spy
    configure({"components": {"tui": False}})
    _run(_checkout_with_frontends(tmp_path))
    kinds = [call[0] for call in calls]
    assert "tui" not in kinds and "web" in kinds
    assert calls[0] == ("deps", ("web",))


def test_update_malformed_components_fails_open(tmp_path, update_spy):
    calls, configure = update_spy
    configure({"components": "nope"})
    _run(_checkout_with_frontends(tmp_path))
    kinds = [call[0] for call in calls]
    assert "tui" in kinds and "web" in kinds


def test_update_builds_when_a_sibling_home_still_wants_the_component(tmp_path, update_spy):
    """The skipped artifacts live in the shared checkout, so one profile cannot drop them."""
    calls, configure = update_spy
    configure({"components": {"tui": False, "web": False}}, {})
    _run(_checkout_with_frontends(tmp_path))
    kinds = [call[0] for call in calls]
    assert "tui" in kinds and "web" in kinds
    assert calls[0] == ("deps", ("ui-tui", "web"))


def test_update_skips_a_component_only_when_every_home_disables_it(tmp_path, update_spy):
    calls, configure = update_spy
    configure(
        {"components": {"tui": False}},
        {"components": {"desktop": True, "tui": False, "web": True}},
    )
    _run(_checkout_with_frontends(tmp_path))
    kinds = [call[0] for call in calls]
    assert "tui" not in kinds and "web" in kinds
    assert calls[0] == ("deps", ("web",))


def test_update_desktop_false_skips_desktop_rebuild(tmp_path, update_spy, monkeypatch):
    from hermes_cli import source_build

    calls, configure = update_spy
    configure({"components": {"desktop": False}})
    root = _checkout_with_frontends(tmp_path)
    (root / "apps/desktop/package.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "apps/desktop/package.json").write_text('{"name": "desktop"}\n', encoding="utf-8")
    source_build.build_update_products(root, desktop=True)
    kinds = [call[0] for call in calls]
    assert "desktop" not in kinds
    assert "tui" in kinds and "web" in kinds

"""Regression for #135217: CLI labels must not reuse a previous release's date."""

import json
from types import SimpleNamespace

import pytest


@pytest.fixture
def stamped_version(tmp_path, monkeypatch):
    from hermes_cli.version_info import _reset_version_info_cache

    root = tmp_path / "install"
    root.mkdir()
    (root / "install-stamp.json").write_text(json.dumps({
        "source": "docker", "distribution": "docker", "updateMechanism": "external",
        "commit": "a" * 40, "baseVersion": "0.21.6", "displayVersion": "0.21.6",
        "commitDate": 1791417600,
    }))
    monkeypatch.setattr("hermes_cli.config.get_project_root", lambda: root)
    monkeypatch.setattr("pm.paths.repo_root", lambda: root)
    monkeypatch.delenv("HERMES_INSTALL_ROOT", raising=False)
    monkeypatch.setattr("hermes_cli.update_channel.resolve_update_channel", lambda *_: "main")
    monkeypatch.setattr("hermes_cli.banner.get_git_banner_state", lambda: {
        "upstream": "aaaaaaaa", "local": "aaaaaaaa", "ahead": 0,
    })
    _reset_version_info_cache()
    yield
    _reset_version_info_cache()


def test_stamped_banner_keeps_version_and_provenance(stamped_version):
    from hermes_cli.banner import format_banner_version_label

    assert format_banner_version_label() == "Hermes Agent v0.21.6 · upstream aaaaaaaa"


@pytest.mark.parametrize("fallback", [False, True])
def test_fast_version_has_no_stale_release_date(stamped_version, monkeypatch, capsys, fallback):
    from hermes_cli import _startup_fast, banner

    if fallback:
        def unavailable():
            raise RuntimeError("banner unavailable")
        monkeypatch.setattr(banner, "format_banner_version_label", unavailable)
    _startup_fast.print_fast_version_info(check_updates=False)
    label = capsys.readouterr().out.splitlines()[0]
    expected = "Hermes Agent v0.21.6"
    if not fallback:
        expected += " · upstream aaaaaaaa"
    assert label == expected


@pytest.mark.parametrize("lang", ["en", "uk", "zh", "zh-hant"])
def test_fast_compact_banner_preserves_localized_version(stamped_version, monkeypatch, lang):
    from agent.i18n import t
    from hermes_cli import cli_render

    monkeypatch.setenv("HERMES_FAST_STARTUP_BANNER", "1")
    monkeypatch.setattr(cli_render.shutil, "get_terminal_size", lambda: SimpleNamespace(columns=90))
    monkeypatch.setattr(cli_render, "t", lambda key, **kwargs: t(key, lang=lang, **kwargs))
    rendered = cli_render._build_compact_banner()
    expected = "Hermes Agent, версія 0.21.6" if lang == "uk" else "Hermes Agent v0.21.6"
    assert expected in rendered
    assert "2026.9.24" not in rendered
    assert "{date}" not in rendered
    assert "()" not in rendered
    assert "（）" not in rendered


def test_fast_compact_banner_accepts_legacy_locale_overlay(stamped_version, tmp_path, monkeypatch):
    from agent.i18n import reset_language_cache, t
    from hermes_cli import cli_render

    home = tmp_path / "profile"
    overlays = home / "locales"
    overlays.mkdir(parents=True)
    (overlays / "en.yaml").write_text(
        'cli:\n  render:\n    banner_version: "Hermes Agent v{version} ({date})"\n'
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_FAST_STARTUP_BANNER", "1")
    monkeypatch.setattr(cli_render.shutil, "get_terminal_size", lambda: SimpleNamespace(columns=90))
    monkeypatch.setattr(cli_render, "t", lambda key, **kwargs: t(key, lang="en", **kwargs))
    reset_language_cache()
    try:
        rendered = cli_render._build_compact_banner()
        assert "Hermes Agent v0.21.6" in rendered
        assert "{version}" not in rendered
        assert "{date}" not in rendered
        assert "2026.9.24" not in rendered
    finally:
        reset_language_cache()

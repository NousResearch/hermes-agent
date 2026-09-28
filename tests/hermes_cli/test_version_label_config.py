"""display.version_label (#123538): /api/health serves the release-only short
form when configured, the historical ``+<distance>`` form otherwise.

The pill's short label is the one place the distance reads as "N commits
behind" on an up-to-date install; the tooltip and the expanded version details
keep the full form, so only the health payload shortens.
"""

from hermes_cli.version_info import VersionInfo, release_only_version

import pytest


def test_release_only_version_strips_the_distance_past_a_release():
    ahead = VersionInfo("0.21.5", "0.21.5+1913.gf83a9e9", 1913, "f" * 40, "main", "git")
    assert release_only_version(ahead.display_version, ahead.base_version) == "0.21.5"
    # A tagless checkout keeps its `git.<sha>` identity — no release to shorten to.
    tagless = VersionInfo("unknown", "git.d0288be5", None, "d" * 40, "main", "git")
    assert release_only_version(tagless.display_version, tagless.base_version) == "git.d0288be5"
    # A stamp that already lost its distance is unchanged.
    on_tag = VersionInfo("0.21.5", "0.21.5", 0, None, None, "git")
    assert release_only_version(on_tag.display_version, on_tag.base_version) == "0.21.5"


def _health_with_config(monkeypatch, tmp_path, config_yaml: str):
    import hermes_cli.web_routers.status as status_router

    info = VersionInfo("0.21.5", "0.21.5+1913.gf83a9e9", 1913, "f" * 40, "main", "git")
    monkeypatch.setattr(status_router, "get_version_info", lambda: info)

    config = tmp_path / "config.yaml"
    config.write_text(config_yaml, encoding="utf-8")
    monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: config)

    return status_router.get_health()


@pytest.mark.asyncio
async def test_health_shortens_display_version_when_release_label(monkeypatch, tmp_path):
    payload = await _health_with_config(monkeypatch, tmp_path, "display:\n  version_label: release\n")

    assert payload["displayVersion"] == "0.21.5"
    assert payload["version"] == "0.21.5"


@pytest.mark.asyncio
async def test_health_keeps_the_distance_form_by_default(monkeypatch, tmp_path):
    payload = await _health_with_config(monkeypatch, tmp_path, "display:\n  version_label: release+distance\n")

    assert payload["displayVersion"] == "0.21.5+1913"


@pytest.mark.asyncio
async def test_health_keeps_the_distance_form_without_the_key(monkeypatch, tmp_path):
    payload = await _health_with_config(monkeypatch, tmp_path, "")

    assert payload["displayVersion"] == "0.21.5+1913"


def test_version_label_default_registered():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["display"]["version_label"] == "release+distance"

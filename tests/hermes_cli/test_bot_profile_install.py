from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from hermes_cli.bot_catalog import BotCatalogEntry, BotCatalogError, resolve_bot_catalog_entry
from hermes_cli.bot_profile_install import BotProfileInstallSpec, install_bot_profile
from hermes_cli.bot_routines import list_bot_routines
from hermes_cli.bot_setup import bot_setup_status, installed_catalog_entry
from hermes_cli.profiles import profiles_to_serve


@pytest.fixture
def profile_homes(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default = tmp_path / ".hermes"
    default.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(default))
    (default / "config.yaml").write_text("model:\n  provider: default-provider\n  default: default/model\n")
    (default / ".env").write_text("SOURCE_API_KEY=wrong\n", encoding="utf-8")

    source = default / "profiles" / "source"
    source.mkdir(parents=True)
    (source / "config.yaml").write_text("model:\n  provider: source-provider\n  default: source/model\n")
    (source / ".env").write_text(
        "SOURCE_API_KEY=right\nTELEGRAM_BOT_TOKEN=do-not-copy\n", encoding="utf-8"
    )
    return default, source


def _spec(**overrides):
    values = {
        "catalog_name": "research-analyst",
        "name": "my-researcher",
        "source_profile": "source",
        "credentials": "copy_api_keys",
    }
    values.update(overrides)
    return BotProfileInstallSpec(**values)


def test_install_publishes_one_complete_profile_from_explicit_source(profile_homes, monkeypatch):
    default, _source = profile_homes
    entry = resolve_bot_catalog_entry("research-analyst", include_live=False)
    observed = []
    def observe_publish(name):
        published = dict(profiles_to_serve(True))[name]
        observed.append((name, sorted(path.name for path in published.iterdir())))

    monkeypatch.setattr("hermes_cli.profiles._notify_multiplexer", observe_publish)

    receipt = install_bot_profile(entry, _spec())

    profile = default / "profiles" / "my-researcher"
    assert receipt.committed is True
    assert receipt.name == "my-researcher"
    assert receipt.catalog_name == "research-analyst"
    assert receipt.source_profile == "source"
    assert receipt.copied_credentials == (".env",)
    assert receipt.setup_state == "needs_setup"
    assert profile == receipt.path
    assert (profile / "SOUL.md").read_text(encoding="utf-8") == entry.profile.soul
    assert "SOURCE_API_KEY=right" in (profile / ".env").read_text(encoding="utf-8")
    assert "TELEGRAM_BOT_TOKEN" not in (profile / ".env").read_text(encoding="utf-8")
    config = yaml.safe_load((profile / "config.yaml").read_text(encoding="utf-8"))
    assert config["model"] == {"provider": "source-provider", "default": "source/model"}
    assert "enabled_toolsets" not in config.get("tools", {})
    assert config["platform_toolsets"]["cli"] == ["hermes-cli", "web"]
    assert config["platform_toolsets"]["desktop"] == ["hermes-desktop", "web"]
    assert config["platform_toolsets"]["cron"] == ["hermes-cron", "web"]
    meta = yaml.safe_load((profile / "profile.yaml").read_text(encoding="utf-8"))
    assert meta["provenance"]["catalog_name"] == "research-analyst"
    assert meta["ui_meta"]["hermes-bots"]["title"] == entry.title
    assert all((profile / "skills" / "/".join(skill.split("/")[1:]) / "SKILL.md").is_file()
               for skill in entry.capabilities.skills)
    assert observed and observed[0][0] == "my-researcher"
    assert {"SOUL.md", "config.yaml", "profile.yaml", "bot-blueprint.yaml", ".env", "skills"}.issubset(observed[0][1])

    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setattr(
        "hermes_cli.bot_catalog.resolve_bot_catalog_entry",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("live catalog was consulted")),
    )
    saved = installed_catalog_entry()
    assert saved == entry
    manifest = yaml.safe_load((profile / "bot-blueprint.yaml").read_text(encoding="utf-8"))
    assert manifest["catalog_entry"]["version"] == entry.version


def test_installed_snapshot_survives_removed_upstream_skills(profile_homes, monkeypatch):
    default, _source = profile_homes
    entry = resolve_bot_catalog_entry("research-analyst", include_live=False)
    install_bot_profile(entry, _spec())
    profile = default / "profiles" / "my-researcher"
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setattr("hermes_cli.bot_catalog.resolve_reviewed_skill_path", lambda _identifier: None)

    saved = installed_catalog_entry()
    status = bot_setup_status(saved, runtime={
        "ok": True,
        "provider": "custom:local",
        "model": "local/model",
        "source": "no-key-required",
    })

    assert saved == entry
    assert status["catalog_name"] == entry.name
    assert [item["id"] for item in list_bot_routines(saved)] == [item.id for item in entry.routines]
    assert all((profile / "skills" / "/".join(skill.split("/")[1:]) / "SKILL.md").is_file()
               for skill in saved.capabilities.skills)


def test_new_install_rejects_structurally_valid_unavailable_dependencies(profile_homes):
    source = resolve_bot_catalog_entry("research-analyst", include_live=False).model_dump(mode="python")
    missing_skill = {**source, "capabilities": {
        **source["capabilities"], "skills": ["official/bots/not-shipped"],
    }}
    missing_toolset = {**source, "setup": {"requirements": [{
        "kind": "toolset",
        "id": "not-shipped",
        "required": True,
        "purpose": "Exercise the reviewed workflow.",
    }]}}

    for name, raw in (("missing-skill", missing_skill), ("missing-toolset", missing_toolset)):
        entry = BotCatalogEntry.model_validate(raw)
        with pytest.raises(BotCatalogError, match="not available|unknown toolset"):
            install_bot_profile(entry, _spec(name=name))


def test_pre_publish_failure_leaves_no_profile_or_staging_residue(profile_homes, monkeypatch):
    default, _source = profile_homes
    entry = resolve_bot_catalog_entry("research-analyst", include_live=False)
    monkeypatch.setattr(
        "hermes_cli.bot_profile_install._copy_reviewed_skills",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("skill copy failed")),
    )

    with pytest.raises(RuntimeError, match="skill copy failed"):
        install_bot_profile(entry, _spec(name="broken"))

    assert not (default / "profiles" / "broken").exists()
    assert not list((default / "profiles").glob(".broken.staging-*"))
    assert "broken" not in {name for name, _path in profiles_to_serve(True)}


def test_missing_explicit_source_is_refused_before_staging(profile_homes):
    default, _source = profile_homes
    entry = resolve_bot_catalog_entry("research-analyst", include_live=False)
    with pytest.raises(FileNotFoundError, match="source profile"):
        install_bot_profile(entry, _spec(name="missing-source", source_profile="does-not-exist"))
    assert not any(path.name.startswith(".missing-source.staging-") for path in (default / "profiles").iterdir())


def test_stripped_oauth_is_not_declared_missing_before_canonical_runtime_probe(profile_homes):
    _default, source = profile_homes
    (source / "auth.json").write_text(
        json.dumps({
            "credential_pool": {
                "anthropic": [{"auth_type": "oauth", "refresh_token": "rotating-token"}],
            }
        }),
        encoding="utf-8",
    )
    entry = resolve_bot_catalog_entry("research-analyst", include_live=False)

    receipt = install_bot_profile(entry, _spec(name="oauth-researcher"))

    assert receipt.oauth_setup_required == ()
    assert receipt.setup_requirements == ()
    assert receipt.setup_state == "needs_setup"  # runtime + first-task proof remain authoritative

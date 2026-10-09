"""Plugins register Automation Blueprints via ``ctx.register_automation_blueprint``.

Contracts: a registered blueprint is listed next to the built-ins (``/blueprint`` and
``/api/cron/blueprints``) under a ``<plugin>:<key>`` key with a plugin source, instantiates
into a normal job, can never shadow a built-in, and is visible only in the profile whose
plugin registered it.
"""

from __future__ import annotations

import json
import logging
import textwrap
from pathlib import Path

import pytest

import hermes_cli.web_models as _web_models
import hermes_cli.web_routers.cron as _rt_cron

_STANDUP = textwrap.dedent(
    """
    def register(ctx):
        ctx.register_automation_blueprint(
            "standup",
            title="Team standup digest",
            description="Weekday summary of what changed in a repo.",
            category="work",
            schedule_template="{minute} {hour} * * 1-5",
            prompt_template="Summarize yesterday's commits for {repo}.",
            slots=[
                {"name": "repo", "type": "text", "label": "Which repo?", "default": "acme/app"},
                {"name": "time", "type": "time", "label": "What time?", "default": "09:15"},
                {"name": "deliver", "type": "enum", "label": "Where?", "default": "origin",
                 "options": ["origin", "local"], "strict": False},
            ],
            tags=["work"],
        )
    """
)


def _write_plugin(home: Path, name: str, body: str, *, enable: bool = True) -> None:
    plugin_dir = home / "plugins" / name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text(f"name: {name}\nversion: 0.1.0\n")
    (plugin_dir / "__init__.py").write_text(body)
    # JSON is valid YAML; avoids a yaml import in the test module.
    plugins = {"enabled": [name]} if enable else {"enabled": [], "disabled": [name]}
    (home / "config.yaml").write_text(json.dumps({"model": "test-model", "plugins": plugins}))


@pytest.fixture()
def homes(tmp_path, monkeypatch):
    """Default profile home (active) plus a second profile home, both isolated."""
    from hermes_cli import profiles

    default_home = tmp_path / ".hermes"
    profiles_root = default_home / "profiles"
    other_home = profiles_root / "other"
    for home in (default_home, other_home):
        (home / "cron").mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("model: test-model\n")
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.setattr("profiles.paths._get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr("profiles.paths._get_profiles_root", lambda: profiles_root)
    return {"default": default_home, "other": other_home}


def _plugin_keys(entries):
    return {e["key"]: e for e in entries if e.get("source") == "plugin"}


def test_plugin_blueprint_is_listed_by_slash_command_and_matches(homes):
    from hermes_cli.blueprint_cmd import handle_blueprint_command, match_blueprint

    _write_plugin(homes["default"], "teamtools", _STANDUP)

    catalog = handle_blueprint_command("")
    assert "teamtools:standup" in catalog.text
    assert "Team standup digest" in catalog.text
    assert match_blueprint("teamtools:standup")[0].key == "teamtools:standup"
    assert match_blueprint("standup")[0].key == "teamtools:standup"


@pytest.mark.asyncio
async def test_plugin_blueprint_in_api_catalog_and_instantiates_real_job(homes):
    from cron.jobs import load_jobs

    _write_plugin(homes["default"], "teamtools", _STANDUP)

    listed = await _rt_cron.list_cron_blueprints(profile="default")
    entry = _plugin_keys(listed["blueprints"])["teamtools:standup"]
    assert entry["plugin"] == "teamtools"
    assert entry["title"] == "Team standup digest"
    assert [f["name"] for f in entry["fields"]] == ["repo", "time", "deliver"]
    builtin = next(e for e in listed["blueprints"] if e["key"] == "morning-brief")
    assert builtin["source"] == "builtin" and builtin["plugin"] is None

    created = await _rt_cron.instantiate_blueprint(
        _web_models.AutomationBlueprintInstantiate(
            blueprint="teamtools:standup",
            values={"repo": "acme/widgets", "time": "10:05", "deliver": "local"},
        ),
        profile="default",
    )
    assert created["name"] == "Team standup digest"
    assert created["profile"] == "default"
    [job] = [j for j in load_jobs() if j["id"] == created["id"]]
    assert job["prompt"] == "Summarize yesterday's commits for acme/widgets."
    assert job["schedule"]["expr"] == "5 10 * * 1-5"
    assert job["deliver"] == "local"


@pytest.mark.asyncio
async def test_plugin_blueprints_are_profile_scoped(homes):
    _write_plugin(homes["default"], "teamtools", _STANDUP)

    # Interleave A -> B -> A: no cached plugin catalog may leak across profiles.
    a1 = _plugin_keys((await _rt_cron.list_cron_blueprints(profile="default"))["blueprints"])
    b = _plugin_keys((await _rt_cron.list_cron_blueprints(profile="other"))["blueprints"])
    a2 = _plugin_keys((await _rt_cron.list_cron_blueprints(profile="default"))["blueprints"])
    assert set(a1) == set(a2) == {"teamtools:standup"}
    assert b == {}

    with pytest.raises(_rt_cron.HTTPException) as exc:
        await _rt_cron.instantiate_blueprint(
            _web_models.AutomationBlueprintInstantiate(blueprint="teamtools:standup", values={}),
            profile="other",
        )
    assert exc.value.status_code == 404


def test_plugin_cannot_shadow_builtin_and_bad_keys_are_rejected(homes, caplog):
    from cron.blueprint_catalog import get_blueprint, list_blueprints

    body = textwrap.dedent(
        """
        def register(ctx):
            common = dict(title="Mine", description="d", schedule_template="0 9 * * *",
                          prompt_template="hi")
            ctx.register_automation_blueprint("morning-brief", **common)
            ctx.register_automation_blueprint("morning-brief", **common)   # duplicate
            ctx.register_automation_blueprint("other:spoof", **common)      # namespace spoof
        """
    )
    _write_plugin(homes["default"], "teamtools", body)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
        keys = [bp.key for bp in list_blueprints()]
    assert keys.count("morning-brief") == 1
    builtin, mine = get_blueprint("morning-brief"), get_blueprint("teamtools:morning-brief")
    assert builtin is not None and builtin.plugin == ""          # still the built-in
    assert mine is not None and mine.title == "Mine"
    assert keys.count("teamtools:morning-brief") == 1
    assert not any(k.endswith("spoof") for k in keys)
    assert "already registered" in caplog.text
    assert "other:spoof" in caplog.text


@pytest.mark.parametrize(
    "fields, reason",
    [
        ({"prompt_template": "hello {nobody}"}, "nobody"),
        ({"slots": [{"name": "x", "type": "colour", "label": "X"}]}, "type"),
        ({"schedule_template": "not a schedule"}, "schedule"),
        ({"schedule_template": "{minute} {hour} * * *"}, "time"),
        ({"title": ""}, "title"),
    ],
)
def test_malformed_plugin_blueprint_is_warned_and_ignored(homes, caplog, fields, reason):
    from cron.blueprint_catalog import list_blueprints

    base = dict(title="T", description="d", schedule_template="0 9 * * *", prompt_template="hi")
    base.update(fields)
    _write_plugin(homes["default"], "teamtools", f"def register(ctx):\n    ctx.register_automation_blueprint('bad', **{base!r})\n")

    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
        keys = [bp.key for bp in list_blueprints()]
    assert "teamtools:bad" not in keys
    assert "teamtools" in caplog.text and reason in caplog.text


@pytest.mark.asyncio
async def test_job_from_plugin_blueprint_survives_plugin_disable(homes):
    from cron.jobs import load_jobs
    from hermes_cli.plugins import _reset_plugin_managers_for_tests

    _write_plugin(homes["default"], "teamtools", _STANDUP)
    created = await _rt_cron.instantiate_blueprint(
        _web_models.AutomationBlueprintInstantiate(blueprint="teamtools:standup", values={}),
        profile="default",
    )

    _write_plugin(homes["default"], "teamtools", _STANDUP, enable=False)
    _reset_plugin_managers_for_tests()

    listed = await _rt_cron.list_cron_blueprints(profile="default")
    assert _plugin_keys(listed["blueprints"]) == {}
    [job] = [j for j in load_jobs() if j["id"] == created["id"]]
    assert job["prompt"] == "Summarize yesterday's commits for acme/app."
    assert job["schedule"]["expr"] == "15 9 * * 1-5"

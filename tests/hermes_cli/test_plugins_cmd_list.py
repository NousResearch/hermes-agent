import argparse
import json
from types import SimpleNamespace

from hermes_cli import plugins_cmd


def _args(**kwargs):
    defaults = {
        "enabled": False,
        "user": False,
        "no_bundled": False,
        "plain": False,
        "json": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _entry(
    name, key=None, *, version="1.0.0", description="", source="bundled", path=None,
    kind="backend",
):
    return name, version, description, source, path, key or name, kind


def _patch_list_state(monkeypatch, entries, *, enabled=(), disabled=()) -> None:
    monkeypatch.setattr(plugins_cmd, "_discover_all_plugins", lambda: entries)
    monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: set(enabled))
    monkeypatch.setattr(plugins_cmd, "_get_disabled_set", lambda: set(disabled))


def _patch_dashboard_state(
    monkeypatch, *, enabled=(), disabled=(), candidates=None, visible=None, resolved=None
):
    saved = {}
    monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: set(enabled))
    monkeypatch.setattr(plugins_cmd, "_get_disabled_set", lambda: set(disabled))
    monkeypatch.setattr(
        plugins_cmd, "_save_enabled_set",
        lambda value: saved.__setitem__("enabled", set(value)),
    )
    monkeypatch.setattr(
        plugins_cmd, "_save_disabled_set",
        lambda value: saved.__setitem__("disabled", set(value)),
    )
    monkeypatch.setattr(plugins_cmd, "_toggle_plugin_toolset", lambda *args, **kwargs: None)
    if candidates is not None:
        monkeypatch.setattr(plugins_cmd, "_discover_plugin_candidates", lambda **_kwargs: candidates)
    if visible is not None:
        monkeypatch.setattr(plugins_cmd, "_discover_all_plugins", lambda: visible)
    if resolved is not None:
        monkeypatch.setattr(
            plugins_cmd, "_resolve_plugin_key_and_source",
            lambda _name, *, for_enable=False: resolved,
        )
    return saved


def test_filter_plugin_entries_enabled_only():
    entries = [
        _entry("disk-cleanup", version="2.0.0", description="Bundled"),
        _entry("web-search-plus", version="2.2.0", description="Search", source="git", kind="standalone"),
        _entry("old-plugin", description="Old", source="user", kind="standalone"),
    ]

    filtered = plugins_cmd._filter_plugin_entries(
        entries,
        _args(enabled=True),
        enabled={"disk-cleanup", "web-search-plus"},
        disabled={"old-plugin"},
    )

    assert [entry[0] for entry in filtered] == ["disk-cleanup", "web-search-plus"]


def test_cmd_list_plain_compact_output(monkeypatch, capsys):
    entries = [
        _entry("disk-cleanup", version="2.0.0", description="Bundled"),
        _entry("web-search-plus", version="2.2.0", description="Search", source="git", kind="standalone"),
    ]
    _patch_list_state(monkeypatch, entries, enabled=("web-search-plus",))

    plugins_cmd.cmd_list(_args(plain=True, no_bundled=True))

    out = capsys.readouterr().out
    assert "web-search-plus" in out
    assert "enabled" in out
    assert "disk-cleanup" not in out
    assert "Search" not in out  # plain mode stays compact, no descriptions


def test_cmd_list_json_preserves_name_and_adds_canonical_key(monkeypatch, capsys):
    entries = [_entry("xai", "image_gen/xai", description="Images")]
    _patch_list_state(monkeypatch, entries)

    plugins_cmd.cmd_list(_args(json=True))

    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["name"] == "xai"
    assert payload[0]["key"] == "image_gen/xai"


def test_cmd_list_plain_disambiguates_duplicate_manifest_names(monkeypatch, capsys):
    entries = [
        _entry("xai", "image_gen/xai", description="Images"),
        _entry("xai", "video_gen/xai", description="Video"),
    ]
    _patch_list_state(monkeypatch, entries)

    plugins_cmd.cmd_list(_args(plain=True))

    out = capsys.readouterr().out
    assert "xai [image_gen/xai]" in out
    assert "xai [video_gen/xai]" in out


def test_dashboard_toggle_response_keeps_input_name_and_adds_key(monkeypatch):
    _patch_dashboard_state(
        monkeypatch, resolved=("image_gen/xai", "user", "xai", "standalone")
    )

    result = plugins_cmd.dashboard_set_agent_plugin_enabled("xai", enabled=True)

    assert result["name"] == "xai"
    assert result["key"] == "image_gen/xai"


def test_dashboard_enable_targets_inactive_higher_precedence_override(monkeypatch):
    key = "shared"
    bundled = _entry("shared-bundled", key)
    user = _entry("shared-user", key, version="2.0.0", source="user")
    saved = _patch_dashboard_state(
        monkeypatch, candidates=[bundled, user], visible=[bundled]
    )

    result = plugins_cmd.dashboard_set_agent_plugin_enabled(key, enabled=True)

    assert result["unchanged"] is False
    assert saved == {"enabled": {key}, "disabled": set()}


def test_dashboard_enable_clears_manifest_deny_without_reviving_colliding_key(
    monkeypatch,
):
    key = "web/firecrawl"
    candidates = [
        _entry("legacy-firecrawl", key),
        _entry("web-firecrawl", key, version="2.0.0", source="user"),
        _entry("video-firecrawl", "legacy-firecrawl"),
    ]
    saved = _patch_dashboard_state(
        monkeypatch,
        enabled=(key, "legacy-firecrawl"),
        disabled=("legacy-firecrawl", "unrelated-plugin"),
        candidates=candidates,
        resolved=(key, "user", "web-firecrawl", "backend"),
    )

    result = plugins_cmd.dashboard_set_agent_plugin_enabled(key, enabled=True)

    assert result["unchanged"] is False
    assert saved == {
        "enabled": {key, "legacy-firecrawl"},
        "disabled": {"unrelated-plugin", "video-firecrawl"},
    }


def test_toggle_group_status_honors_lower_candidate_manifest_deny():
    key = "web/firecrawl"

    status = plugins_cmd._plugin_status(
        "web-firecrawl",
        {key},
        {"legacy-firecrawl"},
        key=key,
        source="user",
        kind="backend",
        aliases={key, "web-firecrawl", "legacy-firecrawl"},
    )

    assert status == "disabled"


def test_discover_all_plugins_includes_entrypoint_plugins(monkeypatch, tmp_path):
    bundled_dir = tmp_path / "bundled"
    user_dir = tmp_path / "user"
    bundled_dir.mkdir()
    user_dir.mkdir()

    dist = SimpleNamespace(
        version="0.1.0",
        metadata={"Summary": "Karpathy-style LLM Wikis for Hermes"},
    )
    entry_point = SimpleNamespace(
        name="wiki",
        value="adapters.hermes.cli_plugin",
        group="hermes_agent.plugins",
        dist=dist,
    )

    monkeypatch.setattr(plugins_cmd, "_plugins_dir", lambda: user_dir)
    monkeypatch.setattr(
        "hermes_cli.plugins.get_bundled_plugins_dir",
        lambda: bundled_dir,
    )
    monkeypatch.setattr(
        plugins_cmd.importlib.metadata,
        "entry_points",
        lambda: [entry_point],
    )

    entries = plugins_cmd._discover_all_plugins()

    assert entries == [
        (
            "wiki",
            "0.1.0",
            "Karpathy-style LLM Wikis for Hermes",
            "entrypoint",
            "adapters.hermes.cli_plugin",
            "wiki",
            "standalone",
        )
    ]



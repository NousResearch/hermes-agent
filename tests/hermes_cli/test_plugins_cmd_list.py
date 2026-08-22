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


def test_filter_plugin_entries_enabled_only():
    entries = [
        ("disk-cleanup", "2.0.0", "Bundled", "bundled", None, "disk-cleanup"),
        ("web-search-plus", "2.2.0", "Search", "git", None, "web-search-plus"),
        ("old-plugin", "1.0.0", "Old", "user", None, "old-plugin"),
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
        ("disk-cleanup", "2.0.0", "Bundled", "bundled", None, "disk-cleanup"),
        ("web-search-plus", "2.2.0", "Search", "git", None, "web-search-plus"),
    ]
    monkeypatch.setattr(plugins_cmd, "_discover_all_plugins", lambda: entries)
    monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: {"web-search-plus"})
    monkeypatch.setattr(plugins_cmd, "_get_disabled_set", lambda: set())

    plugins_cmd.cmd_list(_args(plain=True, no_bundled=True))

    out = capsys.readouterr().out
    assert "web-search-plus" in out
    assert "enabled" in out
    assert "disk-cleanup" not in out
    assert "Search" not in out  # plain mode stays compact, no descriptions


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
        )
    ]


def test_declared_capabilities_for_entrypoint_uses_distribution_metadata(
    monkeypatch, tmp_path
):
    bundled_dir = tmp_path / "bundled"
    user_dir = tmp_path / "user"
    bundled_dir.mkdir()
    user_dir.mkdir()
    plugin_ep = SimpleNamespace(
        name="thread-namer",
        value="thread_namer.plugin:register",
        group="hermes_agent.plugins",
        dist=SimpleNamespace(version="1.0", metadata={"Summary": ""}),
    )
    capability_ep = SimpleNamespace(
        name="thread-namer.gateway.platform_actions",
        value="thread_namer.plugin:register",
        group="hermes_agent.plugin_capabilities",
    )
    monkeypatch.setattr(plugins_cmd, "_plugins_dir", lambda: user_dir)
    monkeypatch.setattr(
        "hermes_cli.plugins.get_bundled_plugins_dir", lambda: bundled_dir
    )
    monkeypatch.setattr(
        plugins_cmd.importlib.metadata,
        "entry_points",
        lambda: [plugin_ep, capability_ep],
    )

    assert plugins_cmd._declared_capabilities_for_key("thread-namer") == [
        "gateway.platform_actions"
    ]


# ---------------------------------------------------------------------------
# Active memory provider status (#82898)
#
# A plugin selected through `memory.provider` is live while appearing in
# neither `plugins.enabled` nor `plugins.disabled`, so it used to render as
# "not enabled" while actively serving memory.
# ---------------------------------------------------------------------------


def test_plugin_status_active_provider_matched_by_directory_not_name(tmp_path):
    """The provider is matched by directory, because the name cannot match.

    `_read_manifest_info` derives `key` from the manifest name at depth 0, so a
    plugin installed at `plugins/mnemosyne/` whose manifest declares
    `name: mnemosyne-hermes` has *neither* a name nor a key equal to
    `memory.provider: mnemosyne`. Only the directory ties them together.
    """
    provider_dir = tmp_path / "plugins" / "mnemosyne"
    provider_dir.mkdir(parents=True)

    status = plugins_cmd._plugin_status(
        "mnemosyne-hermes",
        set(),
        set(),
        key="mnemosyne-hermes",
        dir_path=provider_dir,
        active_provider_dir=provider_dir,
    )

    assert status == plugins_cmd.PLUGIN_STATUS_ACTIVE


def test_plugin_status_does_not_mislabel_unrelated_plugin(tmp_path):
    """A different plugin must not inherit the active-provider label."""
    provider_dir = tmp_path / "plugins" / "mnemosyne"
    other_dir = tmp_path / "plugins" / "unrelated"
    provider_dir.mkdir(parents=True)
    other_dir.mkdir(parents=True)

    status = plugins_cmd._plugin_status(
        "unrelated",
        set(),
        set(),
        key="unrelated",
        dir_path=other_dir,
        active_provider_dir=provider_dir,
    )

    assert status == "not enabled"


def test_plugin_status_without_active_provider_is_unchanged():
    """Existing callers that pass no directory keep the old behaviour."""
    assert plugins_cmd._plugin_status("p", set(), set(), key="p") == "not enabled"
    assert plugins_cmd._plugin_status("p", {"p"}, set(), key="p") == "enabled"
    assert plugins_cmd._plugin_status("p", set(), {"p"}, key="p") == "disabled"


def test_plugin_status_explicit_disable_wins_over_active(tmp_path):
    """An explicit `plugins.disabled` entry still reports disabled."""
    provider_dir = tmp_path / "plugins" / "mnemosyne"
    provider_dir.mkdir(parents=True)

    status = plugins_cmd._plugin_status(
        "mnemosyne-hermes",
        set(),
        {"mnemosyne-hermes"},
        key="mnemosyne-hermes",
        dir_path=provider_dir,
        active_provider_dir=provider_dir,
    )

    assert status == "disabled"


def test_active_memory_provider_dir_none_when_name_unresolvable(monkeypatch):
    """An unresolvable provider name is not serving memory, so not active.

    `load_memory_provider` resolves through the same `find_provider_dir`
    lookup — if that returns None the provider never loads, and nothing
    should be labelled active.
    """
    monkeypatch.setattr(plugins_cmd, "_get_current_memory_provider", lambda: "ghost")
    monkeypatch.setattr("plugins.memory.find_provider_dir", lambda name: None)

    assert plugins_cmd._active_memory_provider_dir() is None


def test_active_memory_provider_dir_none_when_unconfigured(monkeypatch):
    monkeypatch.setattr(plugins_cmd, "_get_current_memory_provider", lambda: "")

    assert plugins_cmd._active_memory_provider_dir() is None


def test_filter_plugin_entries_enabled_includes_active_provider(tmp_path):
    """`--enabled` must show the active provider — it is loaded and running."""
    provider_dir = tmp_path / "plugins" / "mnemosyne"
    other_dir = tmp_path / "plugins" / "unrelated"
    provider_dir.mkdir(parents=True)
    other_dir.mkdir(parents=True)

    entries = [
        ("mnemosyne-hermes", "0.5.0", "Memory", "user", provider_dir, "mnemosyne-hermes"),
        ("unrelated", "1.0.0", "Other", "user", other_dir, "unrelated"),
    ]

    filtered = plugins_cmd._filter_plugin_entries(
        entries,
        _args(enabled=True),
        enabled=set(),
        disabled=set(),
        active_provider_dir=provider_dir,
    )

    assert [entry[0] for entry in filtered] == ["mnemosyne-hermes"]


def test_cmd_list_json_reports_active_provider(monkeypatch, capsys, tmp_path):
    provider_dir = tmp_path / "plugins" / "mnemosyne"
    provider_dir.mkdir(parents=True)
    entries = [
        ("mnemosyne-hermes", "0.5.0", "Memory", "user", provider_dir, "mnemosyne-hermes"),
    ]

    monkeypatch.setattr(plugins_cmd, "_discover_all_plugins", lambda: entries)
    monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: set())
    monkeypatch.setattr(plugins_cmd, "_get_disabled_set", lambda: set())
    monkeypatch.setattr(
        plugins_cmd, "_active_memory_provider_dir", lambda: provider_dir
    )

    plugins_cmd.cmd_list(_args(json=True))

    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["status"] == plugins_cmd.PLUGIN_STATUS_ACTIVE


def _install_real_provider_plugin(home, *, dir_name, manifest_name):
    """Write a real memory-provider plugin into a temp HERMES_HOME.

    Directory name is what ``memory.provider`` refers to; the manifest name is
    what the plugin listing displays. Keeping them different is the point of
    the regression.
    """
    plugin_dir = home / "plugins" / dir_name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text(
        f"name: {manifest_name}\nversion: 0.5.0\ndescription: Memory provider\n",
        encoding="utf-8",
    )
    # plugins.memory._is_memory_provider_dir() scans __init__.py for these
    # markers, so the directory must really look like a provider.
    (plugin_dir / "__init__.py").write_text(
        "from plugins.memory import MemoryProvider\n\n\n"
        "class Provider(MemoryProvider):\n    pass\n",
        encoding="utf-8",
    )
    return plugin_dir


def test_e2e_real_resolution_marks_provider_active(monkeypatch, tmp_path, capsys):
    """End-to-end through the real resolution chain — nothing mocked.

    config.yaml -> _get_current_memory_provider -> find_provider_dir ->
    _is_memory_provider_dir -> directory match -> status, then out through
    the real `cmd_list` JSON renderer. The unit tests above hand
    `active_provider_dir` in directly; this one proves the lookup actually
    resolves against a temp HERMES_HOME.
    """
    home = tmp_path / "hermes_home"
    home.mkdir(parents=True)
    _install_real_provider_plugin(home, dir_name="mnemosyne", manifest_name="mnemosyne-hermes")

    other = home / "plugins" / "unrelated"
    other.mkdir(parents=True)
    (other / "plugin.yaml").write_text(
        "name: unrelated\nversion: 1.0.0\ndescription: Not a provider\n", encoding="utf-8"
    )

    (home / "config.yaml").write_text(
        "memory:\n  provider: mnemosyne\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    # The real lookup must find the plugin's directory from the bare
    # `memory.provider` name.
    resolved = plugins_cmd._active_memory_provider_dir()
    assert resolved is not None
    assert resolved.name == "mnemosyne"

    plugins_cmd.cmd_list(_args(json=True, user=True, no_bundled=True))

    payload = {row["name"]: row for row in json.loads(capsys.readouterr().out)}
    assert payload["mnemosyne-hermes"]["status"] == plugins_cmd.PLUGIN_STATUS_ACTIVE
    assert payload["unrelated"]["status"] == "not enabled"


def test_e2e_unresolvable_provider_is_not_marked_active(monkeypatch, tmp_path):
    """A configured name that does not resolve is not serving memory.

    `load_memory_provider` goes through the same `find_provider_dir` lookup,
    so an unresolvable name must not be labelled active anywhere.
    """
    home = tmp_path / "hermes_home"
    home.mkdir(parents=True)
    _install_real_provider_plugin(home, dir_name="mnemosyne", manifest_name="mnemosyne-hermes")
    (home / "config.yaml").write_text(
        "memory:\n  provider: does-not-exist\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert plugins_cmd._active_memory_provider_dir() is None


def test_e2e_plain_directory_without_provider_markers_is_not_active(monkeypatch, tmp_path):
    """A same-named directory that is not a provider must not resolve.

    Guards the directory match against a plugin that merely shares the
    configured name without being a memory provider at all.
    """
    home = tmp_path / "hermes_home"
    home.mkdir(parents=True)
    impostor = home / "plugins" / "mnemosyne"
    impostor.mkdir(parents=True)
    (impostor / "plugin.yaml").write_text(
        "name: mnemosyne\nversion: 1.0.0\ndescription: Not a provider\n", encoding="utf-8"
    )
    (impostor / "__init__.py").write_text("# no provider markers\n", encoding="utf-8")
    (home / "config.yaml").write_text("memory:\n  provider: mnemosyne\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert plugins_cmd._active_memory_provider_dir() is None


def test_cmd_list_plain_status_column_stays_aligned(monkeypatch, capsys, tmp_path):
    """The active token stays within the 12-char status column."""
    provider_dir = tmp_path / "plugins" / "mnemosyne"
    provider_dir.mkdir(parents=True)
    entries = [
        ("mnemosyne-hermes", "0.5.0", "Memory", "user", provider_dir, "mnemosyne-hermes"),
    ]

    monkeypatch.setattr(plugins_cmd, "_discover_all_plugins", lambda: entries)
    monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: set())
    monkeypatch.setattr(plugins_cmd, "_get_disabled_set", lambda: set())
    monkeypatch.setattr(
        plugins_cmd, "_active_memory_provider_dir", lambda: provider_dir
    )

    plugins_cmd.cmd_list(_args(plain=True))

    line = capsys.readouterr().out.strip()
    assert line.split() == [
        plugins_cmd.PLUGIN_STATUS_ACTIVE,
        "user",
        "0.5.0",
        "mnemosyne-hermes",
    ]


def test_e2e_cmd_show_reports_active_provider(monkeypatch, tmp_path, capsys):
    """`hermes plugins show` is the same surface as `list` and must agree.

    `cmd_show` renders the same activation state for a single plugin, so a
    provider-backed plugin reported as "active" by `cmd_list` must not read
    "not enabled" here — the sibling call path of #82898.
    """
    home = tmp_path / "hermes_home"
    home.mkdir(parents=True)
    _install_real_provider_plugin(home, dir_name="mnemosyne", manifest_name="mnemosyne-hermes")
    (home / "config.yaml").write_text("memory:\n  provider: mnemosyne\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    plugins_cmd.cmd_show("mnemosyne-hermes")

    out = capsys.readouterr().out
    assert f"Status: {plugins_cmd.PLUGIN_STATUS_ACTIVE}" in out
    assert "not enabled" not in out


def test_e2e_cmd_show_unrelated_plugin_still_not_enabled(monkeypatch, tmp_path, capsys):
    """The directory match must not leak onto a plugin that is not the provider."""
    home = tmp_path / "hermes_home"
    home.mkdir(parents=True)
    _install_real_provider_plugin(home, dir_name="mnemosyne", manifest_name="mnemosyne-hermes")
    other = home / "plugins" / "unrelated"
    other.mkdir(parents=True)
    (other / "plugin.yaml").write_text(
        "name: unrelated\nversion: 1.0.0\ndescription: Not a provider\n", encoding="utf-8"
    )
    (home / "config.yaml").write_text("memory:\n  provider: mnemosyne\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    plugins_cmd.cmd_show("unrelated")

    assert "Status: not enabled" in capsys.readouterr().out



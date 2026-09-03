"""`hermes plugins enable` must clear a stale deny entry under any alias.

The loader (``hermes_cli/plugins.py``) denies a plugin when the canonical key
OR the manifest name appears in ``plugins.disabled``, and an explicit disable
always wins over ``plugins.enabled``. An install that moves a plugin to a path
key can leave the old bare name behind in ``disabled``, which silently vetoes
the plugin even though it is listed as enabled.

The enable command is the documented remedy for that state, so its
already-enabled guard has to consider every alias the loader denies on.
Otherwise it reports "already enabled", skips the cleanup, and the deny entry
can never be cleared through the CLI.
"""

import hermes_cli.plugins_cmd as plugins_cmd


def _enable(monkeypatch, *, key, name, enabled, disabled):
    """Run the enable command against in-memory config sets."""
    saved = {"enabled": set(enabled), "disabled": set(disabled)}

    monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: set(saved["enabled"]))
    monkeypatch.setattr(
        plugins_cmd, "_get_disabled_set", lambda: set(saved["disabled"])
    )
    monkeypatch.setattr(
        plugins_cmd,
        "_save_enabled_set",
        lambda value: saved.__setitem__("enabled", set(value)),
    )
    monkeypatch.setattr(
        plugins_cmd,
        "_save_disabled_set",
        lambda value: saved.__setitem__("disabled", set(value)),
    )
    # entry = (name, version, description, source, dir_path, key)
    monkeypatch.setattr(
        plugins_cmd,
        "_discover_all_plugins",
        lambda: [(name, "1.0", "test plugin", "user", "/tmp/plugin", key)],
    )

    monkeypatch.setattr(
        plugins_cmd, "_resolve_plugin_key_and_source", lambda _name: (key, "user")
    )
    # Bundled plugins skip the capability prompt; "user" would prompt, so stop
    # after the config write by declining the privileged grant explicitly.
    plugins_cmd.cmd_enable(key, allow_tool_override=False)
    return saved


def test_enable_clears_a_stale_bare_name_from_disabled(monkeypatch):
    """A bare name left in disabled must not survive enabling the path key."""
    saved = _enable(
        monkeypatch,
        key="superpowers/.hermes-plugin",
        name="superpowers",
        enabled={"superpowers/.hermes-plugin"},
        disabled={"superpowers"},
    )

    # The loader denies on the manifest name, so leaving it behind keeps the
    # plugin off no matter what plugins.enabled says.
    assert "superpowers" not in saved["disabled"]
    assert "superpowers/.hermes-plugin" in saved["enabled"]


def test_enable_is_a_noop_when_no_alias_is_denied(monkeypatch):
    """Enabling a genuinely enabled plugin leaves config untouched."""
    saved = _enable(
        monkeypatch,
        key="web/firecrawl",
        name="web-firecrawl",
        enabled={"web/firecrawl"},
        disabled={"unrelated-plugin"},
    )

    assert saved["enabled"] == {"web/firecrawl"}
    assert saved["disabled"] == {"unrelated-plugin"}

"""Fork-custom toolset regression guard (Kyzcreig fork).

`messaging` (agent-callable ``send_message``) and `moa` (``mixture_of_agents``)
are DELIBERATE fork divergences from upstream — upstream has neither (its stance
is that agents do not get an agent-callable send_message, and it has no MoA tool).

An upstream parity merge that blindly takes upstream's ``toolsets.py`` /
``tools/`` silently DROPS them: ``resolve_toolset()`` then returns ``[]`` for the
name with no error, and every platform that lists the toolset quietly loses the
tool. That exact regression happened in the 2026-06-29 parity sync (PR #119) and
was only caught by the config-migration dry-run's post-migration toolset
validation. This test makes the invariant a HARD CI gate so the next sync can't
reintroduce it.

If this test fails after an upstream merge, you dropped a fork feature — restore
the toolset entry in ``toolsets.py`` (and, for ``moa``, the
``tools/mixture_of_agents_tool.py`` module that self-registers it), do not delete
the test.
"""

import importlib

import pytest

from toolsets import TOOLSETS, resolve_toolset, validate_toolset


class TestForkCustomToolsetsPresent:
    """The two fork-custom toolsets must exist and resolve to their tool(s)."""

    def test_messaging_toolset_resolves_send_message(self):
        # Upstream deliberately omits an agent-callable send_message; the fork
        # re-enables it (origin-routing work #71/#83 depends on it).
        assert "messaging" in TOOLSETS, "fork-custom 'messaging' toolset was dropped (likely an upstream merge)"
        assert validate_toolset("messaging") is True
        tools = resolve_toolset("messaging")
        assert tools, "resolve_toolset('messaging') is empty — the toolset lost its tools"
        assert "send_message" in tools

    def test_moa_toolset_resolves_mixture_of_agents(self):
        assert "moa" in TOOLSETS, "fork-custom 'moa' toolset was dropped (likely an upstream merge)"
        assert validate_toolset("moa") is True
        tools = resolve_toolset("moa")
        assert tools, "resolve_toolset('moa') is empty — the toolset lost its tools"
        assert "mixture_of_agents" in tools


class TestMixtureOfAgentsToolRegistered:
    """The moa toolset is only useful if the tool module self-registers it."""

    def test_module_imports_and_self_registers(self):
        # The module is auto-discovered by tools.registry.discover_builtin_tools()
        # (it contains a registry.register() call). Import it directly here so the
        # test is order-independent, then assert the registration landed.
        #
        # Guard against the editable-install live-tree leak (the very TRAP this
        # whole PR is about): a deleted module can still import from a sibling
        # install. Pin the module's resolved file to THIS repo's tools/ dir, so a
        # phantom import from elsewhere is treated as "missing", not a pass.
        import os
        import toolsets as _toolsets_mod

        repo_root = os.path.dirname(os.path.abspath(_toolsets_mod.__file__))
        expected = os.path.join(repo_root, "tools", "mixture_of_agents_tool.py")
        assert os.path.isfile(expected), (
            "tools/mixture_of_agents_tool.py is missing from THIS repo — the moa "
            "tool module was dropped (likely an upstream parity merge)"
        )

        mod = importlib.import_module("tools.mixture_of_agents_tool")
        mod_file = getattr(mod, "__file__", None)
        assert mod_file is not None and os.path.abspath(mod_file) == expected, (
            f"tools.mixture_of_agents_tool imported from {mod_file}, not this "
            f"repo ({expected}) — editable-install leak masking a dropped module"
        )

        from tools.registry import registry

        entry = registry._tools.get("mixture_of_agents")
        assert entry is not None, "tools/mixture_of_agents_tool.py no longer registers 'mixture_of_agents'"
        assert entry.toolset == "moa"

    def test_tool_schema_shape(self):
        import os
        import toolsets as _toolsets_mod

        repo_root = os.path.dirname(os.path.abspath(_toolsets_mod.__file__))
        expected = os.path.join(repo_root, "tools", "mixture_of_agents_tool.py")
        assert os.path.isfile(expected), "tools/mixture_of_agents_tool.py missing from this repo"

        importlib.import_module("tools.mixture_of_agents_tool")
        from tools.registry import registry

        entry = registry._tools["mixture_of_agents"]
        schema = entry.schema
        assert schema.get("name") == "mixture_of_agents"
        # contract: takes a single required free-text prompt
        params = schema.get("parameters", {})
        assert "user_prompt" in params.get("properties", {})
        assert "user_prompt" in params.get("required", [])


class TestForkToolsetsWiredInToolsUI:
    """`hermes tools` must surface both toolsets so a user can toggle them, and
    moa must declare its OpenRouter credential requirement."""

    def test_tools_config_rows_and_env(self):
        from hermes_cli import tools_config

        # CONFIGURABLE_TOOLSETS rows is a list of (key, label, desc) tuples.
        keys = {row[0] for row in tools_config.CONFIGURABLE_TOOLSETS}
        assert "messaging" in keys, "'messaging' missing from the `hermes tools` UI list"
        assert "moa" in keys, "'moa' missing from the `hermes tools` UI list"

        # moa is an opt-in (default-off) toolset that needs OPENROUTER_API_KEY.
        assert "moa" in tools_config._DEFAULT_OFF_TOOLSETS
        env_req = tools_config.TOOLSET_ENV_REQUIREMENTS.get("moa", [])
        assert any(name == "OPENROUTER_API_KEY" for name, _url in env_req), \
            "moa must declare its OPENROUTER_API_KEY requirement"


def test_send_message_and_moa_not_silently_zero():
    """The headline invariant: neither fork toolset may resolve to zero tools.

    A zero-tool resolve is exactly the silent-drop failure mode (resolve_toolset
    returns [] for an unknown/empty toolset with no error). Assert it can't happen
    for either fork-custom toolset.
    """
    for name in ("messaging", "moa"):
        assert len(resolve_toolset(name)) >= 1, (
            f"fork-custom toolset {name!r} resolves to 0 tools — it was dropped, "
            "probably by an upstream parity merge taking upstream's toolsets.py"
        )

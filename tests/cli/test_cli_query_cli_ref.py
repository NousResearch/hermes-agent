"""Regression tests for ``hermes chat -q`` plugin context.

Previously, ``HermesCLI.run()`` assigned the CLI reference to the plugin
manager, but query mode called ``cli.agent.run_conversation()`` directly
and skipped ``run()``. As a result, ``PluginContext.dispatch_tool()``
could not inject ``parent_agent`` into delegated tools.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from hermes_cli.plugins import get_plugin_manager, PluginContext, PluginManifest


def _capture_cli_and_mock_agent():
    """Return a pair of context managers that:

    1. Capture the ``HermesCLI`` instance created by ``main()``.
    2. Stub ``agent`` on that instance so no real model call runs.
    """
    import cli as cli_mod

    created_clis: list[cli_mod.HermesCLI] = []
    mock_agent = MagicMock()
    mock_agent.run_conversation.return_value = {
        "final_response": "mock reply",
        "failed": False,
    }

    real_init = cli_mod.HermesCLI.__init__

    def _patched_init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        self.agent = mock_agent
        created_clis.append(self)

    return (
        patch.object(cli_mod.HermesCLI, "__init__", _patched_init),
        created_clis,
        mock_agent,
    )


def _stub_cli_methods():
    """Stub every method that ``main()`` calls on the query path so the test
    never touches real credentials, model calls, or I/O."""
    import cli as cli_mod

    return (
        patch.object(cli_mod.HermesCLI, "_claim_active_session", return_value=True),
        patch.object(cli_mod.HermesCLI, "_ensure_runtime_credentials", return_value=True),
        patch.object(cli_mod.HermesCLI, "_print_exit_summary"),
        patch.object(cli_mod.HermesCLI, "_show_security_advisories"),
        # Stub chat() to prevent a real agent conversation — we only need
        # to verify the _cli_ref wiring before that call.
        patch.object(cli_mod.HermesCLI, "chat"),
    )


class TestQueryCliRef:
    """Verify that query-mode CLI execution exposes its agent to plugins."""

    def test_query_path_wires_cli_ref_and_dispatch(self):
        """After ``main(query="...")``, ``_cli_ref`` must point to the
        ``HermesCLI`` instance, and a subsequent ``dispatch_tool()`` call
        must receive that instance's agent as ``parent_agent`` —
        validating the full query-mode lifecycle end-to-end."""
        import cli as cli_mod

        init_patch, created_clis, mock_agent = _capture_cli_and_mock_agent()
        stubs = _stub_cli_methods()

        mgr = get_plugin_manager()
        saved_ref = mgr._cli_ref
        try:
            with init_patch:
                with stubs[0], stubs[1], stubs[2], stubs[3], stubs[4]:
                    with patch.object(cli_mod.HermesCLI, "run") as mock_run:
                        cli_mod.main(query="test query", max_turns=1)

            assert len(created_clis) == 1, "main() must create exactly one HermesCLI"
            created_cli = created_clis[0]

            # Query mode calls chat(), NOT run()
            mock_run.assert_not_called()

            # _cli_ref should point to the created CLI (wired by main())
            assert mgr._cli_ref is created_cli, (
                "_cli_ref should point to the HermesCLI instance after "
                "the query path creates it, but it is None. "
                "main() must set get_plugin_manager()._cli_ref = cli "
                "after HermesCLI() and before the query/interactive branch."
            )

            # Now verify that dispatch_tool forwards parent_agent through
            # the _cli_ref that main() just wired — no manual _cli_ref set.
            manifest = PluginManifest(name="test-plugin", source="user")
            ctx = PluginContext(manifest, mgr)

            mock_registry = MagicMock()
            mock_registry.dispatch.return_value = '{"ok": true}'

            with patch("tools.registry.registry", mock_registry):
                ctx.dispatch_tool("delegate_task", {"goal": "test"})

            call_kwargs = mock_registry.dispatch.call_args
            assert call_kwargs[1].get("parent_agent") is mock_agent, (
                "dispatch_tool must inject the agent from the main()-wired "
                "_cli_ref as parent_agent."
            )
        finally:
            mgr._cli_ref = saved_ref

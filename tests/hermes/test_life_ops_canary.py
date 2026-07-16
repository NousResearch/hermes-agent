"""Canary for the life_ops plugin's coupling to upstream internals.

plugins/life_ops/discord_adapter.py subclasses the bundled Discord adapter
and imports four of its private module-level helpers. Upstream owns those
names and may rename them without notice; this test makes that breakage
loud at test/sync time instead of silent at runtime (a scheduler that
never starts, a button that 500s at 6am).

If a test here fails right after an upstream merge: upstream renamed or
restructured the symbol — re-point the imports in
plugins/life_ops/discord_adapter.py (and plugins/life_ops/__init__.py's
lazy wrappers), do NOT re-add fork code to the upstream adapter file.
See FORK.md.
"""

from __future__ import annotations

import inspect

import pytest


class TestUpstreamAdapterInternals:
    """The private upstream symbols life_ops imports must keep existing."""

    def test_private_helpers_importable(self):
        from plugins.platforms.discord.adapter import (  # noqa: F401
            _DISCORD_SELECT_FIELD_LIMIT,
            _component_check_auth,
            _read_discord_prompt_timeout,
            _truncate_discord_component_text,
        )

    def test_lazy_wrapper_targets_exist(self):
        """Names plugins/life_ops/__init__.py resolves lazily from the
        bundled adapter module at call time."""
        import plugins.platforms.discord.adapter as adapter_mod

        for name in (
            "check_discord_requirements",
            "_is_connected",
            "interactive_setup",
            "_apply_yaml_config",
            "_standalone_send",
        ):
            assert hasattr(adapter_mod, name), (
                f"plugins.platforms.discord.adapter.{name} disappeared — "
                "update the lazy wrappers in plugins/life_ops/__init__.py"
            )

    def test_overridden_lifecycle_methods_exist_on_base(self):
        """LifeOpsDiscordAdapter overrides these and calls super() — they
        must still exist on the upstream base class."""
        from plugins.platforms.discord.adapter import DiscordAdapter

        for name in (
            "_register_slash_commands",
            "_run_post_connect_initialization",
            "cancel_background_tasks",
        ):
            assert callable(getattr(DiscordAdapter, name, None)), (
                f"DiscordAdapter.{name} disappeared — the life_ops override "
                "chain is broken; re-anchor LifeOpsDiscordAdapter"
            )

    def test_client_tree_is_how_commands_register(self):
        """_register_slash_commands must still register via self._client.tree
        (the subclass registers fork commands on the same tree first)."""
        from plugins.platforms.discord.adapter import DiscordAdapter

        src = inspect.getsource(DiscordAdapter._register_slash_commands)
        assert "self._client.tree" in src, (
            "upstream no longer uses self._client.tree in "
            "_register_slash_commands — re-check the life_ops slash "
            "registration override"
        )


class TestLifeOpsSubclassWiring:
    def test_subclass_resolves(self):
        from plugins.life_ops.discord_adapter import LifeOpsDiscordAdapter
        from plugins.platforms.discord.adapter import DiscordAdapter

        assert issubclass(LifeOpsDiscordAdapter, DiscordAdapter)
        # The fork hook methods live on the subclass, not the upstream base.
        for name in (
            "_start_bedtime_scheduler",
            "_start_approvals_scheduler",
            "_start_todo_closure_scheduler",
            "send_journal_dev_todos",
            "send_todo_closure_view",
        ):
            assert callable(getattr(LifeOpsDiscordAdapter, name, None))
            assert getattr(DiscordAdapter, name, None) is None, (
                f"{name} leaked onto the upstream adapter — fork code must "
                "stay in plugins/life_ops (see FORK.md rule 1)"
            )

    def test_view_classes_defined_when_discord_available(self):
        import plugins.life_ops.discord_adapter as mod

        if not mod.DISCORD_AVAILABLE:
            pytest.skip("discord.py not installed in this environment")
        for name in ("JournalApproveView", "BedtimeView", "TodoClosureView"):
            assert getattr(mod, name) is not None

    def test_plugin_register_overrides_discord_platform(self):
        """register(ctx) must re-register the 'discord' platform with the
        life_ops factory (platform_registry is last-writer-wins)."""
        import plugins.life_ops as life_ops

        captured = {}

        class _Ctx:
            def register_platform(self, **kwargs):
                captured.update(kwargs)

        life_ops.register(_Ctx())
        assert captured.get("name") == "discord"
        assert captured["adapter_factory"] is life_ops._build_adapter

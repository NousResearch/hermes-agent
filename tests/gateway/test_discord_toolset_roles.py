"""Per-user Discord toolset roles (adapter.toolsets_for_source).

A Discord platform config may carry ``toolset_roles`` (named toolset
bundles) and ``user_roles`` (numeric Discord user ID -> role name or
list of role names). For runs triggered by an assigned user, the union
of their roles' toolsets REPLACES the platform-level
``platform_toolsets.discord`` resolution. Unassigned users — and users
whose roles are unknown, missing, or resolve to nothing — keep the
platform default. The gateway validates the override through the same
``_get_platform_tools`` path as platform config, so restricted/unknown
toolset names behave identically to a manually configured list.

Models tests/gateway/test_webhook_route_toolsets.py: webhook's per-route
``toolsets`` is the same extension point scoped to routes; this scopes it to
users (use case: a shared Discord bot where the operator wants a second
allowed DM peer to have web access but no terminal/file/config tools).
"""

from types import SimpleNamespace

from gateway.platforms.base import BasePlatformAdapter
from gateway.run import GatewayRunner
from hermes_cli.tools_config import _get_platform_tools


def _ensure_discord_sdk():
    import sys
    import types
    from unittest.mock import MagicMock

    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return
    discord_mod = types.ModuleType("discord")
    discord_mod.Intents = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.Interaction = object
    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod
    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_sdk()

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402

ROLE_USER_ID = "100000000000000001"     # fake numeric Discord user ID
UNASSIGNED_ID = "100000000000000002"    # fake numeric Discord user ID
ROLES = {
    "research": ["web", "skills"],
    "full": ["terminal", "file"],
}


class _Src:
    def __init__(self, user_id):
        self.user_id = user_id
        self.chat_id = f"dm:{user_id}"


def _make_adapter(toolset_roles=None, user_roles=None):
    adapter = object.__new__(DiscordAdapter)
    adapter.config = SimpleNamespace(
        extra={"toolset_roles": toolset_roles, "user_roles": user_roles}
    )
    return adapter


def _make_runner(adapter):
    gr = object.__new__(GatewayRunner)
    gr._adapter_for_source = lambda source: adapter
    return gr


BASE_CONFIG = {
    "platform_toolsets": {"discord": ["hermes-discord"]},
}


class TestDiscordAdapterToolsetsForSource:
    def test_single_role_string_returns_bundle(self):
        adapter = _make_adapter(ROLES, {ROLE_USER_ID: "research"})
        assert adapter.toolsets_for_source(_Src(ROLE_USER_ID)) == ["web", "skills"]

    def test_multiple_roles_union_with_dedupe(self):
        # Overlapping toolsets across roles are additive with one copy.
        adapter = _make_adapter(
            {"a": [" web ", "x"], "b": ["x", "y"]},
            {ROLE_USER_ID: ["a", "b"]},
        )
        assert adapter.toolsets_for_source(_Src(ROLE_USER_ID)) == ["web", "x", "y"]

    def test_unlisted_user_returns_none(self):
        adapter = _make_adapter(ROLES, {ROLE_USER_ID: "research"})
        assert adapter.toolsets_for_source(_Src(UNASSIGNED_ID)) is None

    def test_missing_user_id_returns_none(self):
        adapter = _make_adapter(ROLES, {ROLE_USER_ID: "research"})
        assert adapter.toolsets_for_source(_Src("")) is None

    def test_no_user_roles_config_returns_none(self):
        adapter = _make_adapter(ROLES, None)
        assert adapter.toolsets_for_source(_Src(ROLE_USER_ID)) is None

    def test_unknown_role_returns_none(self):
        # Unknown role names contribute nothing; user keeps platform default.
        adapter = _make_adapter(ROLES, {ROLE_USER_ID: "does_not_exist"})
        assert adapter.toolsets_for_source(_Src(ROLE_USER_ID)) is None

    def test_partial_unknown_roles_still_resolve_known_ones(self):
        adapter = _make_adapter(ROLES, {ROLE_USER_ID: ["ghost", "research"]})
        assert adapter.toolsets_for_source(_Src(ROLE_USER_ID)) == ["web", "skills"]

    def test_missing_toolset_roles_returns_none(self):
        adapter = _make_adapter(None, {ROLE_USER_ID: "research"})
        assert adapter.toolsets_for_source(_Src(ROLE_USER_ID)) is None

    def test_empty_or_non_list_entries_return_none(self):
        adapter = _make_adapter(
            {
                "empty": [],
                "weird": "notalist",
                "blank": ["  "],
            },
            {
                "1": "empty",
                "2": "weird",
                "3": "blank",
                "4": [],
                "5": " ",
            },
        )
        for uid in ("1", "2", "3", "4", "5"):
            assert adapter.toolsets_for_source(_Src(uid)) is None

    def test_missing_extra_returns_none(self):
        adapter = object.__new__(DiscordAdapter)
        adapter.config = SimpleNamespace(extra=None)
        assert adapter.toolsets_for_source(_Src(ROLE_USER_ID)) is None

    def test_base_adapter_default_is_none(self):
        # Adapters without the override (telegram, slack, ...) inherit None.
        adapter = _make_adapter(ROLES, {ROLE_USER_ID: "research"})
        assert (
            BasePlatformAdapter.toolsets_for_source(adapter, _Src(ROLE_USER_ID)) is None
        )


class TestGatewayResolveEnabledToolsetsForSource:
    def test_role_override_replaces_platform_resolution(self):
        adapter = _make_adapter({"research": ["web"]}, {ROLE_USER_ID: "research"})
        gr = _make_runner(adapter)
        res = GatewayRunner._resolve_enabled_toolsets_for_source(
            gr, BASE_CONFIG, _Src(ROLE_USER_ID), "discord"
        )
        assert "web" in res
        # The platform's full-access composite must be gone, not merged.
        for forbidden in ("terminal", "file", "code_execution", "browser",
                          "memory", "skills", "vision", "discord", "discord_admin"):
            assert forbidden not in res, f"unexpected toolset in restricted run: {forbidden}"

    def test_unlisted_user_keeps_platform_default(self):
        adapter = _make_adapter({"research": ["web"]}, {ROLE_USER_ID: "research"})
        gr = _make_runner(adapter)
        res = GatewayRunner._resolve_enabled_toolsets_for_source(
            gr, BASE_CONFIG, _Src(UNASSIGNED_ID), "discord"
        )
        assert res == sorted(_get_platform_tools(BASE_CONFIG, "discord"))
        assert "terminal" in res

    def test_unknown_role_keeps_platform_default(self):
        # Spec: a user assigned a role that doesn't exist gets default perms.
        adapter = _make_adapter({"research": ["web"]}, {ROLE_USER_ID: "no_such_role"})
        gr = _make_runner(adapter)
        res = GatewayRunner._resolve_enabled_toolsets_for_source(
            gr, BASE_CONFIG, _Src(ROLE_USER_ID), "discord"
        )
        assert res == sorted(_get_platform_tools(BASE_CONFIG, "discord"))
        assert "terminal" in res

    def test_role_override_does_not_mutate_caller_config(self):
        adapter = _make_adapter({"research": ["web"]}, {ROLE_USER_ID: "research"})
        gr = _make_runner(adapter)
        cfg = dict(BASE_CONFIG)
        cfg["platform_toolsets"] = dict(BASE_CONFIG["platform_toolsets"])
        GatewayRunner._resolve_enabled_toolsets_for_source(
            gr, cfg, _Src(ROLE_USER_ID), "discord"
        )
        assert cfg["platform_toolsets"]["discord"] == ["hermes-discord"]

    def test_validation_matches_platform_config_path(self):
        # Same-shape invariant as the webhook per-route tests: the per-user
        # role resolution resolves identically to manually saving that list
        # as the platform's toolset config.
        adapter = _make_adapter({"research": ["web"]}, {ROLE_USER_ID: "research"})
        gr = _make_runner(adapter)
        override_res = GatewayRunner._resolve_enabled_toolsets_for_source(
            gr, BASE_CONFIG, _Src(ROLE_USER_ID), "discord"
        )
        manual = sorted(
            _get_platform_tools(
                {**BASE_CONFIG, "platform_toolsets": {**BASE_CONFIG["platform_toolsets"], "discord": ["web"]}},
                "discord",
            )
        )
        assert override_res == manual

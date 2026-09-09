"""Gateway authz must survive the per-turn .env hot-reload when
DISCORD_ALLOWED_USERS contains usernames (Aug 2026 incident).

Sequence under test:
  1. Operator writes usernames into DISCORD_ALLOWED_USERS in ``.env``.
  2. Discord adapter connect resolves usernames -> numeric IDs into adapter
     memory (``_allowed_user_ids``) and mirrors them into ``os.environ``.
  3. The gateway's per-turn env hot-reload
     (``_reload_runtime_env_preserving_config_authority`` ->
     ``load_hermes_dotenv(override=True)``) restores the RAW username strings
     from the file into the process env.
  4. ``GatewayAuthorizationMixin._is_user_authorized`` compares the sender's
     numeric ``user_id`` against the env allowlist.

Before the fix, step 4 found usernames, never matched a numeric ID, and
dropped the operator as "Unauthorized user" from the second agent turn
onward — the bot answered exactly once per reconnect. The fix unions the
adapter's resolved numeric IDs (``resolved_allowlist_user_ids()``) into the
gateway-layer allowlist, so runtime resolution survives env reloads.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.session import Platform, SessionSource

OPERATOR_ID = "387972437901312000"


@pytest.fixture(autouse=True)
def _clean_auth_env(monkeypatch):
    for var in (
        "DISCORD_ALLOWED_USERS",
        "DISCORD_ALLOWED_ROLES",
        "DISCORD_ALLOW_ALL_USERS",
        "DISCORD_ALLOW_BOTS",
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
    ):
        monkeypatch.delenv(var, raising=False)


def _make_runner(adapter=None):
    """Bare GatewayRunner (object.__new__ pattern, AGENTS.md pitfall #17)."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: False)
    runner.adapters = {Platform.DISCORD: adapter} if adapter is not None else {}
    return runner


def _discord_source(user_id: str = OPERATOR_ID):
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="1543941724672368680",
        chat_type="thread",
        user_id=user_id,
        user_name="Teknium",
        is_bot=False,
    )


def _resolved_adapter(ids=frozenset({OPERATOR_ID, "111222333444555666"})):
    """Stand-in for a connected DiscordAdapter after username resolution.

    Mirrors the real accessor's contract: a plain method returning a set of
    numeric-ID strings. SimpleNamespace (not MagicMock) so no unrelated
    attribute can auto-truthy through other authz branches.
    """
    return SimpleNamespace(resolved_allowlist_user_ids=lambda: set(ids))


class TestResolvedAllowlistSurvivesEnvReload:
    def test_username_env_plus_resolved_adapter_authorizes(self, monkeypatch):
        """THE incident shape: env holds usernames (post-reload), adapter holds
        resolved numeric IDs — the operator must stay authorized."""
        monkeypatch.setenv("DISCORD_ALLOWED_USERS", "teknium,123mikeyd")
        runner = _make_runner(_resolved_adapter())
        assert runner._is_user_authorized(_discord_source()) is True

    def test_username_env_without_adapter_still_denies(self, monkeypatch):
        """No live adapter (or resolution never ran): usernames cannot match a
        numeric user_id — deny, exactly as before the fix."""
        monkeypatch.setenv("DISCORD_ALLOWED_USERS", "teknium,123mikeyd")
        runner = _make_runner(adapter=None)
        assert runner._is_user_authorized(_discord_source()) is False

    def test_stranger_denied_despite_resolved_adapter(self, monkeypatch):
        """The union must not widen access: a sender in neither the env list
        nor the resolved set stays denied."""
        monkeypatch.setenv("DISCORD_ALLOWED_USERS", "teknium,123mikeyd")
        runner = _make_runner(_resolved_adapter())
        assert runner._is_user_authorized(_discord_source("666000666000666000")) is False

    def test_empty_env_allowlist_never_consults_adapter(self, monkeypatch):
        """Fail-closed invariant: with NO configured allowlist, adapter memory
        must not become an authorization source (the union is a resolution of
        configured entries, not an independent grant)."""
        consulted = []

        def _resolver():
            consulted.append(True)
            return {OPERATOR_ID}

        adapter = SimpleNamespace(resolved_allowlist_user_ids=_resolver)
        runner = _make_runner(adapter)
        assert runner._is_user_authorized(_discord_source()) is False
        assert consulted == [], (
            "adapter resolved set must not be consulted when no env allowlist "
            "is configured — that would turn stale adapter memory into a grant"
        )

    def test_numeric_env_ids_still_work_without_adapter(self, monkeypatch):
        """Plain numeric-ID configuration is untouched by the fix."""
        monkeypatch.setenv("DISCORD_ALLOWED_USERS", OPERATOR_ID)
        runner = _make_runner(adapter=None)
        assert runner._is_user_authorized(_discord_source()) is True

    def test_non_set_resolver_return_is_ignored(self, monkeypatch):
        """A resolver returning a non-collection (e.g. a MagicMock in a test
        fixture) must be discarded by the isinstance guard, not iterated or
        truthy-tested into an authorization."""
        monkeypatch.setenv("DISCORD_ALLOWED_USERS", "teknium")
        adapter = SimpleNamespace(resolved_allowlist_user_ids=lambda: MagicMock())
        runner = _make_runner(adapter)
        assert runner._is_user_authorized(_discord_source("666000666000666000")) is False

    def test_raising_resolver_fails_closed(self, monkeypatch):
        """An adapter whose accessor raises must not break authz for senders
        the env list already covers, nor authorize anyone else."""
        def _boom():
            raise RuntimeError("adapter mid-reconnect")

        monkeypatch.setenv("DISCORD_ALLOWED_USERS", f"teknium,{OPERATOR_ID}")
        adapter = SimpleNamespace(resolved_allowlist_user_ids=_boom)
        runner = _make_runner(adapter)
        # numeric entry in env still authorizes
        assert runner._is_user_authorized(_discord_source()) is True
        # stranger still denied
        assert runner._is_user_authorized(_discord_source("666000666000666000")) is False


class TestDiscordAdapterResolvedAccessor:
    def _adapter(self, allowed_ids):
        from plugins.platforms.discord.adapter import DiscordAdapter

        adapter = object.__new__(DiscordAdapter)
        adapter._allowed_user_ids = allowed_ids
        return adapter

    def test_returns_numeric_ids_as_strings(self):
        adapter = self._adapter({OPERATOR_ID, 111222333444555666})
        assert adapter.resolved_allowlist_user_ids() == {
            OPERATOR_ID,
            "111222333444555666",
        }

    def test_filters_usernames_and_wildcard(self):
        """Unresolved usernames and the '*' wildcard must not pass through:
        usernames can't match numeric user_ids, and '*' would widen the
        gateway layer to allow-everyone from adapter memory alone."""
        adapter = self._adapter({"teknium", "*", OPERATOR_ID})
        assert adapter.resolved_allowlist_user_ids() == {OPERATOR_ID}

    def test_missing_attribute_yields_empty_set(self):
        from plugins.platforms.discord.adapter import DiscordAdapter

        adapter = object.__new__(DiscordAdapter)  # no _allowed_user_ids at all
        assert adapter.resolved_allowlist_user_ids() == set()

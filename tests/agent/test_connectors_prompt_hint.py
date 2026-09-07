"""What the agent is told about connectors it could set up.

The hint has two sources. ``mcp.connectors`` records apps the user said they
use during desktop onboarding; the reviewed catalog covers everyone else,
including the user who skipped the picker entirely. Nothing in either list is
configured and no credential was collected — that is the point, since sign-in
is deferred to the moment a task needs the app.

The gap only pays off if the agent knows about it. Asked to "check my linear
backlog" with no hint, a model does what any model without that knowledge
does: greps the environment for `LINEAR_API_KEY` and asks the user to paste
one. So these tests pin the properties that make the hint useful without
being harmful:

- it names connectors the user said they use
- it names catalog connectors even when the user never picked any
- it drops one the moment it IS configured, so the agent never offers a
  connector the user already has
- it still names a configured connector whose sign-in never finished, the one
  state that is otherwise invisible: too configured to offer, too parked to
  register any tools
- it forbids the detours (asking for a key, asking which auth method,
  hand-rolling the vendor API) that the card exists to replace
- it only reaches the desktop, the one surface with ``setup_mcp``

Run against the real prompt builder: the hint lives in the byte-stable cached
prefix, and a mock would hide the cache-safety property being verified.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.system_prompt import _pending_connectors_hint, build_system_prompt_parts


@pytest.fixture
def config(monkeypatch):
    """Patch the readonly config loader the hint reads."""

    def apply(payload):
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly", lambda *a, **k: payload
        )

    return apply


@pytest.fixture
def catalog(monkeypatch):
    """Stand in for the on-disk catalog so a test states its own world."""

    def apply(*names):
        monkeypatch.setattr(
            "hermes_cli.mcp_catalog.list_catalog",
            lambda *a, **k: [SimpleNamespace(name=name) for name in names],
        )

    return apply


@pytest.fixture
def tokens(monkeypatch):
    """Name the servers whose OAuth tokens are on disk; the rest are signed out."""

    def apply(*names):
        monkeypatch.setattr(
            "hermes_cli.mcp_config._oauth_tokens_present", lambda name: name in names
        )

    return apply


def _make_agent(platform):
    return SimpleNamespace(
        load_soul_identity=False,
        skip_context_files=False,
        valid_tool_names=[],
        _task_completion_guidance=False,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        _platform_hint_overrides={},
        model="",
        provider="",
        pass_session_id=False,
        session_id="",
        platform=platform,
    )


def _stable_prompt(agent):
    with (
        patch("agent.prompt_builder.load_soul_md", return_value=""),
        patch("agent.prompt_builder.build_environment_hints", return_value=""),
        patch("agent.prompt_builder.build_context_files_prompt", return_value=""),
    ):
        return build_system_prompt_parts(agent)["stable"]


class TestPendingConnectorsHint:
    def test_names_the_connectors_the_user_wants(self, config, catalog):
        config({"mcp": {"connectors": ["linear", "notion"]}})
        catalog()

        hint = _pending_connectors_hint()

        assert "linear" in hint
        assert "notion" in hint
        assert "setup_mcp" in hint

    def test_offers_the_catalog_when_the_user_never_picked_any(self, config, catalog):
        """The regression that sent "check my linear backlog" hunting for an
        API key: a user who skipped onboarding got no hint at all."""
        config({})
        catalog("linear", "notion")

        hint = _pending_connectors_hint()

        assert "linear" in hint
        assert "setup_mcp" in hint

    def test_forbids_asking_for_credentials_instead_of_showing_the_card(
        self, config, catalog
    ):
        config({})
        catalog("linear")

        hint = _pending_connectors_hint()

        assert "API key" in hint and "Do not ask" in hint
        assert "raw HTTP call" in hint

    def test_a_decline_is_scoped_to_the_request_not_the_session(self, config, catalog):
        """A "not now" answers the task in hand. Treating it as a standing ban
        means the next request that needs the same connector gets a workaround
        or a question instead of the card."""
        config({})
        catalog("linear")

        hint = _pending_connectors_hint()

        assert "not permanent" in hint
        assert "fresh ask" in hint

    def test_configured_connectors_drop_off(self, config, catalog):
        """Offering a connector the user already has is the failure mode this
        subtraction exists to prevent."""
        config(
            {
                "mcp": {"connectors": ["linear", "notion"]},
                "mcp_servers": {"notion": {"url": "https://mcp.notion.com/mcp"}},
            }
        )
        catalog("linear", "notion", "figma")

        hint = _pending_connectors_hint()

        assert "linear" in hint
        assert "figma" in hint
        assert "notion" not in hint

    def test_a_configured_connector_that_never_signed_in_is_named(
        self, config, catalog, tokens
    ):
        """An abandoned sign-in leaves an entry that 401s and parks. It is
        excluded from the offers as "already configured" and registers no tools,
        so without this it can never be recovered from chat."""
        config({"mcp_servers": {"linear": {"url": "https://mcp.linear.app/mcp", "auth": "oauth"}}})
        catalog()
        tokens()

        hint = _pending_connectors_hint()

        assert "linear" in hint
        assert "signed out" in hint
        assert "setup_mcp" in hint

    def test_a_signed_in_connector_stays_silent(self, config, catalog, tokens):
        config({"mcp_servers": {"figma": {"url": "https://mcp.figma.com/mcp", "auth": "oauth"}}})
        catalog()
        tokens("figma")

        assert _pending_connectors_hint() == ""

    def test_only_oauth_connectors_can_be_signed_out(self, config, catalog, tokens):
        """A header/API-key server keeps its credential in config, not on disk —
        a missing token file says nothing about it."""
        config({"mcp_servers": {"acme": {"url": "https://acme.test/mcp", "headers": {"X": "y"}}}})
        catalog()
        tokens()

        assert _pending_connectors_hint() == ""

    def test_a_stated_app_is_named_once_not_twice(self, config, catalog):
        """Both lists can hold the same name; the user's own words win."""
        config({"mcp": {"connectors": ["linear"]}})
        catalog("linear", "notion")

        assert _pending_connectors_hint().count("linear") == 1

    def test_nothing_left_to_offer_produces_no_hint(self, config, catalog):
        config(
            {
                "mcp": {"connectors": ["notion"]},
                "mcp_servers": {"notion": {"url": "https://mcp.notion.com/mcp"}},
            }
        )
        catalog("notion")

        assert _pending_connectors_hint() == ""

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"mcp": {}},
            {"mcp": {"connectors": []}},
            # Hand-edited config with the wrong shape must not raise into
            # prompt assembly — a bad key cannot be allowed to break chat.
            {"mcp": {"connectors": "linear"}},
            {"mcp": "not-a-dict"},
        ],
    )
    def test_a_malformed_wishlist_still_offers_the_catalog(
        self, config, catalog, payload
    ):
        config(payload)
        catalog("linear")

        assert "linear" in _pending_connectors_hint()

    def test_malformed_mcp_servers_does_not_suppress_the_hint(self, config, catalog):
        """A broken `mcp_servers` means we can't subtract anything — offering
        a connector the user may already have beats going silent."""
        config({"mcp": {"connectors": ["linear"]}, "mcp_servers": "not-a-dict"})
        catalog()

        assert "linear" in _pending_connectors_hint()

    def test_an_unreadable_catalog_still_names_the_wishlist(self, config, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.mcp_catalog.list_catalog",
            lambda *a, **k: (_ for _ in ()).throw(OSError("bad manifest")),
        )
        config({"mcp": {"connectors": ["linear"]}})

        assert "linear" in _pending_connectors_hint()

    def test_unreadable_config_is_silent(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda *a, **k: (_ for _ in ()).throw(OSError("permission denied")),
        )

        assert _pending_connectors_hint() == ""

    def test_duplicates_and_blanks_are_normalized(self, config, catalog):
        config({"mcp": {"connectors": ["linear", " linear ", "", "notion"]}})
        catalog()

        assert _pending_connectors_hint().count("linear") == 1

    def test_every_offered_name_is_a_real_connector(self, config):
        """One pass against the actual catalog on disk, to prove the wiring —
        the names themselves are free to change."""
        from hermes_cli.mcp_catalog import list_catalog

        config({})
        real = {entry.name for entry in list_catalog()}
        hint = _pending_connectors_hint()

        offered = hint.split("set up on the spot: ", 1)[-1].split(".")[0].split(", ")

        assert real
        assert set(offered) <= real


class TestHintReachesOnlyTheDesktopPrefix:
    def test_desktop_prompt_carries_the_hint_in_the_stable_prefix(self, config, catalog):
        """Stable prefix, not a per-turn injection: both lists are static for
        the life of a session, so they cannot break the prompt cache."""
        config({"mcp": {"connectors": ["linear"]}})
        catalog()

        assert "not connected yet: linear" in _stable_prompt(_make_agent("desktop"))

    @pytest.mark.parametrize("platform", ["cli", "tui", "telegram", "discord"])
    def test_other_surfaces_never_see_it(self, config, catalog, platform):
        """setup_mcp is desktop-only, so naming un-connected connectors
        anywhere else would advertise an action the agent cannot take."""
        config({"mcp": {"connectors": ["linear"]}})
        catalog("notion")

        prompt = _stable_prompt(_make_agent(platform))

        assert "not connected yet" not in prompt
        assert "setup_mcp" not in prompt

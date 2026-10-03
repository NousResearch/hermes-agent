"""``/models`` is a read-only, one-provider model listing (#3500).

The invariant is a RELATION, never a catalog value: whatever the shared lister
(``list_authenticated_providers`` — the same one ``/model`` and ``hermes model`` read)
returns for a provider is exactly what ``/models <provider>`` shows, and a provider the
lister does not return is not listable. A routine catalog update therefore cannot break
these tests, while a lister that stops being scoped to one provider does.
"""

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


_ROWS = [
    {"slug": "openrouter", "name": "OpenRouter", "is_current": False,
     "models": ["router/alpha", "router/beta"], "total_models": 2},
    {"slug": "anthropic", "name": "Anthropic", "is_current": True,
     "models": ["claude-x", "claude-y"], "total_models": 2},
    {"slug": "custom:lmstudio", "name": "LM Studio", "is_current": False,
     "models": ["local-z"], "total_models": 1},
]


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._running_agents = {}
    return runner


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"),
    )


@pytest.fixture
def _isolated_config(tmp_path, monkeypatch):
    """Empty isolated home so config loading is cheap and deterministic."""
    import gateway.run as gateway_run

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "model:\n  default: claude-x\n  provider: anthropic\nproviders: {}\n", encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    return hermes_home


@pytest.fixture
def _stub_lister(monkeypatch):
    """Pin the shared lister's output so the assertions are about SCOPING, not catalog data."""
    calls: list[dict] = []

    def _fake(**kwargs):
        calls.append(kwargs)
        return [dict(r) for r in _ROWS]

    monkeypatch.setattr("hermes_cli.model_switch_providers.list_authenticated_providers", _fake)
    return calls


@pytest.mark.asyncio
async def test_named_provider_lists_only_that_provider(_isolated_config, _stub_lister):
    """``/models openrouter`` must answer about openrouter even though anthropic is the CURRENT
    provider — the requested provider wins over the configured one, and no other provider's models
    leak in."""
    runner = _make_runner()

    reply = await runner._handle_models_command(_make_event("/models openrouter"))

    assert "router/alpha" in reply and "router/beta" in reply
    assert "claude-y" not in reply and "local-z" not in reply


@pytest.mark.asyncio
async def test_custom_provider_alias_reaches_its_own_row(_isolated_config, _stub_lister):
    """A configured custom endpoint answers under its ``custom:`` slug AND its bare display name
    (``lmstudio``), because that is how the endpoint is spelled in config.yaml."""
    runner = _make_runner()

    by_slug = await runner._handle_models_command(_make_event("/models custom:lmstudio"))
    by_name = await runner._handle_models_command(_make_event("/models lmstudio"))

    assert "local-z" in by_slug and "local-z" in by_name
    assert "claude-y" not in by_slug and "router/alpha" not in by_name


@pytest.mark.asyncio
async def test_bare_listing_scopes_to_the_current_provider(_isolated_config, _stub_lister):
    """``/models`` with no argument lists the configured provider's models. The relation is
    'the current provider is in the listing, the others are not' — which holds whatever the
    catalog happens to contain."""
    runner = _make_runner()

    reply = await runner._handle_models_command(_make_event("/models"))

    assert "claude-x" in reply
    assert "router/alpha" not in reply and "local-z" not in reply


@pytest.mark.asyncio
async def test_provider_the_lister_omits_is_reported_not_invented(_isolated_config, _stub_lister):
    """A provider absent from the listing gets a clear message, not a neighbouring
    provider's models and not a fabricated catalog."""
    runner = _make_runner()

    reply = await runner._handle_models_command(_make_event("/models nosuchprovider"))

    lowered = reply.lower()
    assert "nosuchprovider" in lowered
    for row in _ROWS:
        for model in row["models"]:
            assert model not in reply, f"listed {model} for an unconfigured provider"


@pytest.mark.asyncio
async def test_models_answers_from_the_declared_inventory_with_models_dev_held(
    _isolated_config, monkeypatch
):
    """Cold/held models.dev: the reply is built from the DECLARED inventory, never from a registry
    round-trip the reply would have to wait on. ``/models`` asks the shared lister for cache-only
    catalogs (``non_blocking_catalogs=True``), so a degraded models.dev — one that never answers
    inside its 15s timeout — must not stand between the command and its answer.

    The held fake raises rather than hangs: an attempted network read from the reply path then
    shows up as a failure here instead of as a stall in production. (#118518 carries the same
    cache-only policy into the remaining label/aggregator reads; those tolerate a held registry
    and fall back to the plain provider id.)"""
    import agent.models_dev as models_dev
    import hermes_cli.model_switch as model_switch_mod
    from hermes_cli.model_switch_providers import list_authenticated_providers as real_lister

    # ``hermes_cli.model_switch`` binds the lister at import time and the sibling tests above stub
    # it there first, so re-pin the real one: this test is about the live read path, in any order.
    monkeypatch.setattr(model_switch_mod, "list_authenticated_providers", real_lister)

    def _held_models_dev(force_refresh=False, *, allow_network=True):
        if allow_network:
            raise RuntimeError("models.dev is held")
        return {}

    monkeypatch.setattr(models_dev, "fetch_models_dev", _held_models_dev)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    runner = _make_runner()

    reply = await runner._handle_models_command(_make_event("/models"))

    assert "anthropic" in reply.lower()
    assert "\n- " in reply, f"no model lines rendered from the declared inventory: {reply!r}"

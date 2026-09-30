"""The gateway ``/model`` inline picker must not truncate a provider's catalog below what the
platform adapters can actually render.

The gateway asked :func:`list_picker_providers` for ``max_models=50``, but every adapter already
bounds what it renders and reports the remainder: Discord partitions the list across up to three
25-option selects (``_DISCORD_MODEL_SELECT_CAPACITY`` = 75) and appends an "N more available" note,
Slack renders up to 100, Telegram paginates, Matrix keys off its reaction budget. The shared 50
therefore hid models those surfaces could have shown — and hid them *silently*, because the row's
``total_models`` was sliced to the cap's worth of survivors downstream, so the "N more available"
note could never fire either.

These tests go through the real slicing path (``list_picker_providers`` -> ``list_authenticated_
providers`` -> ``_PickerBuild.add_row`` -> ``_cap_models``), so RED comes from the truncation
itself rather than from an assertion about a keyword argument.
"""

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource

# 86 models on the reported profile; the model the user was looking for sits at rank 74.
_MODELS = [f"vendor/model-{i:02d}" for i in range(86)]
_TARGET = _MODELS[73]
_SLUG = "deepseek"  # ordinary (non-aggregator) lane, so the shared cap applied to it


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._running_agents = {}
    return runner


def _make_event():
    return MessageEvent(
        text="/model",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"),
    )


@pytest.fixture
def _isolated_config(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "model:\n  default: gpt-x\n  provider: openrouter\nproviders: {}\n", encoding="utf-8"
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    return hermes_home


def _seed_large_catalog(monkeypatch, models):
    """Report *models* as ``_SLUG``'s live catalog, so the real ``_cap_models`` slice runs."""
    import hermes_cli.models as hm
    from hermes_cli import model_switch

    monkeypatch.setattr(hm, "cached_provider_model_ids", lambda provider, **kw: list(models))
    monkeypatch.setattr(hm, "provider_model_ids", lambda provider, **kw: list(models))
    # A configured API-key provider is what makes the row appear at all.
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek")
    monkeypatch.setattr(model_switch, "_load_model_config", lambda *a, **kw: {}, raising=False)


class _FakePickerResult:
    success = True


class _RecordingPickerAdapter:
    """Adapter that records the providers payload the gateway handed it."""

    def __init__(self):
        self.providers = None

    async def send_model_picker(self, *, providers=None, **kwargs):
        self.providers = providers
        return _FakePickerResult()


async def _open_picker(runner, monkeypatch, models):
    """Run the bare ``/model`` picker branch and return the adapter that received the payload."""
    import hermes_cli.model_switch_providers as msp

    _seed_large_catalog(monkeypatch, models)
    monkeypatch.setattr(msp, "fetch_openrouter_models", lambda *a, **kw: [], raising=False)
    import hermes_cli.models as hm
    monkeypatch.setattr(hm, "fetch_openrouter_models", lambda *a, **kw: [], raising=False)

    adapter = _RecordingPickerAdapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    monkeypatch.setattr(runner, "_thread_metadata_for_source", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(runner, "_reply_anchor_for_event", lambda *a, **k: None, raising=False)
    assert await runner._handle_model_command(_make_event()) is None
    return adapter


def _row_for(providers, slug=_SLUG):
    row = next((p for p in providers or [] if p.get("slug") == slug), None)
    assert row is not None, f"{slug} missing from picker payload: {[p.get('slug') for p in providers or []]}"
    return row


# --------------------------------------------------------------------------- #
# Invariants
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_picker_surfaces_a_model_past_rank_fifty(_isolated_config, monkeypatch):
    """A model at rank 74 of an 86-model catalog must survive to the picker payload.

    Red before the fix: the gateway's ``max_models=50`` made ``_cap_models`` slice the row, so
    rank 74 was dropped before any adapter saw it.
    """
    runner = _make_runner()
    adapter = await _open_picker(runner, monkeypatch, _MODELS)

    assert adapter.providers, "picker payload never reached the adapter"
    row = _row_for(adapter.providers)
    assert _TARGET in row["models"], (
        f"model at rank 74 was truncated out of the picker payload "
        f"(row carried {len(row['models'])} of {len(_MODELS)} models)"
    )


@pytest.mark.asyncio
async def test_picker_keeps_the_true_total_so_adapters_can_hint_the_remainder(_isolated_config, monkeypatch):
    """``total_models`` must stay the full count so adapters render "N more available".

    Both Discord (``_on_provider_selected``) and Slack (``_build_model_picker_model_blocks``)
    compute that note from ``total_models``.
    """
    runner = _make_runner()
    adapter = await _open_picker(runner, monkeypatch, _MODELS)

    row = _row_for(adapter.providers)
    assert row["total_models"] == len(_MODELS) == 86
    assert len(row["models"]) > 50, (
        "the payload must carry more than the old 50-model cap, or the picker is still truncated"
    )


@pytest.mark.asyncio
async def test_text_fallback_keeps_its_preview_cap(_isolated_config, monkeypatch):
    """Uncapping the picker must not blow up the plain-text listing, which is a chat message.

    The text path keeps its own 5-model preview cap via ``_TEXT_LISTING_MODELS``.
    """
    from gateway import slash_commands_model

    captured = {}

    def _fake_list_authenticated(**kwargs):
        captured["max_models"] = kwargs.get("max_models")
        return [{
            "slug": _SLUG, "name": "DeepSeek", "is_current": True, "is_user_defined": False,
            "models": list(_MODELS), "total_models": len(_MODELS), "source": "built-in",
        }]

    monkeypatch.setattr("hermes_cli.model_switch.list_authenticated_providers", _fake_list_authenticated)
    runner = _make_runner()
    # No picker-capable adapter => the text-fallback branch runs.
    runner.adapters = {Platform.TELEGRAM: object()}

    ctx = slash_commands_model._ModelSwitchContext(
        session_key="s1", source=_make_event().source, config_path=None, persist_global=False,
        current_provider=_SLUG, current_base_url="", current_model=_MODELS[0],
    )
    reply = await runner._model_listing_reply(_make_event(), ctx, profile_home=None)

    assert captured["max_models"] == slash_commands_model._TEXT_LISTING_MODELS
    assert reply is not None
    assert len(reply.splitlines()) < len(_MODELS), "text listing must stay a short preview"


@pytest.mark.asyncio
async def test_an_exploding_picker_listing_falls_through_to_the_text_listing(_isolated_config, monkeypatch):
    """Uncapping must not remove the guard: a failing listing still degrades to the text path.

    ``list_picker_providers`` does not guard the inventory call — the gateway's ``try/except`` is
    load-bearing. Exercise it by failing only the picker lane (``for_picker=True``) and leaving the
    text lane working, which is exactly how the two call sites differ.
    """
    from gateway import slash_commands_model

    calls: list[dict] = []

    def _selective(**kwargs):
        calls.append(kwargs)
        if kwargs.get("for_picker"):
            raise RuntimeError("provider listing exploded")
        return [{
            "slug": _SLUG, "name": "DeepSeek", "is_current": True, "is_user_defined": False,
            "models": list(_MODELS), "total_models": len(_MODELS), "source": "built-in",
        }]

    monkeypatch.setattr("hermes_cli.model_switch.list_authenticated_providers", _selective)
    runner = _make_runner()

    class _ExplodingAdapter:
        async def send_model_picker(self, **kwargs):
            raise AssertionError("picker must not be reached when the listing fails")

    runner.adapters = {Platform.TELEGRAM: _ExplodingAdapter()}

    ctx = slash_commands_model._ModelSwitchContext(
        session_key="s1", source=_make_event().source, config_path=None, persist_global=False,
        current_provider=_SLUG, current_base_url="", current_model=_MODELS[0],
    )
    reply = await runner._model_listing_reply(_make_event(), ctx, profile_home=None)

    assert any(c.get("for_picker") for c in calls), "the picker lane never ran, guard untested"
    assert reply is not None, "a failing picker listing produced no text fallback"
    text_calls = [c for c in calls if not c.get("for_picker")]
    assert text_calls and text_calls[0].get("max_models") == slash_commands_model._TEXT_LISTING_MODELS


def test_adapter_ceilings_still_bound_an_oversized_row():
    """Dropping the gateway cap must not let an unbounded row overflow a platform control.

    The gateway no longer slices, so the adapters are the only bound. Discord's select menu is the
    hard one (25 options, 5 rows): prove a pathologically large row still renders within limits
    and is capped by the adapter rather than by the gateway.
    """
    from plugins.platforms.discord.adapter import ModelPickerView

    huge = [f"vendor/model-{i}" for i in range(5000)]
    view = ModelPickerView(
        providers=[{
            "slug": "huge", "name": "Huge", "models": huge,
            "total_models": len(huge), "is_current": False,
        }],
        current_model=huge[0], current_provider="huge", session_key="s1",
        on_model_selected=lambda *a, **k: None, allowed_user_ids={"1"},
    )
    view._selected_provider = "huge"
    view._build_model_select("huge")

    selects = [
        child for child in view.children
        if str(getattr(child, "custom_id", "")).startswith("model_model_select")
    ]
    assert selects, "no model select rendered"
    # Discord caps one select at 25 options and a view at 5 action rows (2 reserved for
    # Back/Cancel), so the adapter partitions into at most 3 selects of at most 25.
    assert all(len(sel.options) <= 25 for sel in selects), (
        f"a select overflowed Discord's 25-option cap: {[len(s.options) for s in selects]}"
    )
    assert len(view.children) <= 5, f"view overflowed 5 action rows: {len(view.children)}"
    # The row is bounded here, and ``total_models`` still reports the true count so the
    # adapter's "N more available" note can tell the user what the tail holds.
    assert sum(len(sel.options) for sel in selects) < len(huge)

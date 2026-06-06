from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = ROOT / "plugins" / "producers-triage"
SPEC = importlib.util.spec_from_file_location(
    "hermes_plugins.producers_triage_test",
    PLUGIN_DIR / "__init__.py",
    submodule_search_locations=[str(PLUGIN_DIR)],
)
producers_triage = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = producers_triage
assert SPEC.loader is not None
SPEC.loader.exec_module(producers_triage)


class DummyCtx:
    def __init__(self) -> None:
        self.hooks = []

    def register_hook(self, name, callback):
        self.hooks.append((name, callback))


class DummyAdapter:
    def __init__(self) -> None:
        self.send = AsyncMock()


@pytest.fixture(autouse=True)
def producers_home(monkeypatch, tmp_path):
    monkeypatch.setattr(producers_triage, "_is_producers_profile", lambda: True)
    monkeypatch.setattr(producers_triage, "BREV_REQUESTS_FILE", tmp_path / "brev_generation_requests.json")


def make_event(text: str, author: str = "a meobius", chat_id: str = "1509389598923559053"):
    return SimpleNamespace(
        text=text,
        message_id="msg-1",
        source=SimpleNamespace(
            platform="discord",
            chat_id=chat_id,
            user_id="user-1",
            user_name=author,
            thread_id=None,
        ),
    )


def make_gateway(adapter: DummyAdapter):
    return SimpleNamespace(adapters={"discord": adapter})


def test_register_adds_pre_gateway_dispatch_hook():
    ctx = DummyCtx()
    producers_triage.register(ctx)
    assert len(ctx.hooks) == 1
    assert ctx.hooks[0][0] == "pre_gateway_dispatch"


def test_hook_returns_sync_skip_and_schedules_worker(monkeypatch):
    scheduled = []

    def fake_create_task(coro):
        scheduled.append(coro)
        coro.close()
        return object()

    monkeypatch.setattr(producers_triage.asyncio, "create_task", fake_create_task)
    ctx = DummyCtx()
    producers_triage.register(ctx)
    result = ctx.hooks[0][1](event=make_event("кработ статус роста"), gateway=make_gateway(DummyAdapter()))
    assert result == {"action": "skip", "reason": "producers-triage-fast-path"}
    assert len(scheduled) == 1


def test_hook_ignores_non_trigger_messages(monkeypatch):
    monkeypatch.setattr(producers_triage.asyncio, "create_task", lambda coro: (_ for _ in ()).throw(AssertionError("should not schedule")))
    ctx = DummyCtx()
    producers_triage.register(ctx)
    result = ctx.hooks[0][1](event=make_event("обычный текст", chat_id="other"), gateway=make_gateway(DummyAdapter()))
    assert result is None


@pytest.mark.asyncio
async def test_prompt_doctor_fast_path_sends_offline_reply(monkeypatch):
    adapter = DummyAdapter()
    event = make_event("кработ почини промпт\nпромпт: dark neurodance\nцель: плотный клубный грув\nпроблема: мутный микс")

    monkeypatch.setattr(producers_triage, "run_sanitizer", lambda value: value)
    fallback = SimpleNamespace(run_prompt_doctor_offline=lambda prompt, goal, failure: f"diagnosed:{prompt}|{goal}|{failure}")
    monkeypatch.setitem(sys.modules, f"{SPEC.name}.prompt_doctor_fallback", fallback)

    result = await producers_triage.pre_gateway_dispatch(event=event, gateway=make_gateway(adapter))

    assert result == {"action": "skip"}
    adapter.send.assert_awaited_once()
    args, kwargs = adapter.send.await_args
    assert args[0] == "1509389598923559053"
    assert "diagnosed:dark neurodance|плотный клубный грув|мутный микс" in args[1]
    assert kwargs["reply_to"] == "msg-1"


@pytest.mark.asyncio
async def test_consent_command_uses_source_fields_and_reply_adapter(monkeypatch):
    adapter = DummyAdapter()
    event = make_event("кработ согласие", author="artist one")
    recorded = {}
    monkeypatch.setattr(producers_triage, "set_user_consent", lambda user, consent, channel: recorded.update(user=user, consent=consent, channel=channel))
    monkeypatch.setattr(producers_triage, "run_sanitizer", lambda value: value)

    result = await producers_triage.pre_gateway_dispatch(event=event, gateway=make_gateway(adapter))

    assert result == {"action": "skip"}
    assert recorded == {"user": "artist one", "consent": True, "channel": "1509389598923559053"}
    adapter.send.assert_awaited_once()
    assert "запомнил, artist one" in adapter.send.await_args.args[1]


def test_format_queue_refresh_result_summarizes_sources(monkeypatch):
    monkeypatch.setattr(producers_triage, "run_sanitizer", lambda value: value)
    reply = producers_triage.format_queue_refresh_result({
        "ok": True,
        "summary": {
            "candidate_count": 4,
            "denied_count": 2,
            "source_counts": {"gitdb": 2, "trendshift": 1, "telegram": 1, "x": 1},
            "top": [{"full_name": "demo/audio-tool", "score": 8.5, "sources": ["x", "gitdb"]}],
        },
    })

    assert "очередь инструментов обновлена" in reply
    assert "кандидатов: 4" in reply
    assert "tg 1" in reply
    assert "x 1" in reply
    assert "demo/audio-tool" in reply


@pytest.mark.asyncio
async def test_brev_request_schedules_background_generation(monkeypatch):
    adapter = DummyAdapter()
    event = make_event("кработ трек\nprompt: dark neurodance\nстиль: binaural pulse")
    request = {
        "request_id": "brev-auto-1",
        "prompt": "dark neurodance",
        "style": "binaural pulse",
        "lyrics": "",
        "instrumental": True,
        "status": "queued",
    }
    # We patch the helper function imported inside the triage run_gen
    # The plugin modules are executed dynamically under 'hermes_plugins.producers_triage_test' spec
    # namespace or they are inserted into sys.modules.
    import sys
    # Find the loaded module name
    p_triage_name = [name for name in sys.modules if "producers_triage_test" in name][0]
    p_triage = sys.modules[p_triage_name]

    # Let's mock the imported module run_brev_generation inside the plugin's own module dict!
    # Because 'from .brev_runner_helper import run_brev_generation' creates a local name or module ref.
    # We can inject/mock a mock helper module in sys.modules.
    fake_helper = SimpleNamespace(
        run_brev_generation=lambda request_id: {
            "ok": True,
            "final_status": "completed",
            "request_id": request_id,
            "asset_urls": ["https://cdn.example/track.mp3"],
        }
    )
    sys.modules[p_triage.__package__ + ".brev_runner_helper"] = fake_helper

    monkeypatch.setattr(p_triage, "run_sanitizer", lambda value: value)
    monkeypatch.setattr(p_triage, "create_brev_generation_request", lambda raw, author, channel: (request, []))

    async def fake_to_thread(func):
        return func()

    monkeypatch.setattr(p_triage.asyncio, "to_thread", fake_to_thread)

    result = await p_triage.pre_gateway_dispatch(event=event, gateway=make_gateway(adapter))
    for _ in range(5):
        await asyncio.sleep(0)

    assert result == {"action": "skip"}
    assert adapter.send.await_count == 2
    assert "запускаю генерацию" in adapter.send.await_args_list[0].args[1]
    assert "трек готов" in adapter.send.await_args_list[1].args[1]
    assert "https://cdn.example/track.mp3" in adapter.send.await_args_list[1].args[1]


@pytest.mark.asyncio
async def test_queue_refresh_command_is_admin_only(monkeypatch):
    adapter = DummyAdapter()
    event = make_event("кработ обнови очередь", author="artist one")
    monkeypatch.setattr(producers_triage, "run_sanitizer", lambda value: value)

    result = await producers_triage.pre_gateway_dispatch(event=event, gateway=make_gateway(adapter))

    assert result == {"action": "skip"}
    adapter.send.assert_awaited_once()
    assert "только администраторам" in adapter.send.await_args.args[1]


@pytest.mark.asyncio
async def test_queue_refresh_command_schedules_background_refresh(monkeypatch):
    adapter = DummyAdapter()
    event = make_event("кработ обнови очередь без трендшифт", author="a meobius")
    called = {}

    monkeypatch.setattr(producers_triage, "run_sanitizer", lambda value: value)
    monkeypatch.setattr(producers_triage, "trigger_queue_refresh", lambda **kwargs: called.update(kwargs) or {
        "ok": True,
        "summary": {"candidate_count": 1, "denied_count": 0, "source_counts": {"gitdb": 1}, "top": []},
    })

    async def fake_to_thread(func):
        func()

    monkeypatch.setattr(producers_triage.asyncio, "to_thread", fake_to_thread)

    result = await producers_triage.pre_gateway_dispatch(event=event, gateway=make_gateway(adapter))
    for _ in range(5):
        await asyncio.sleep(0)

    assert result == {"action": "skip"}
    assert called == {"no_trendshift": True, "no_llm": True}
    assert adapter.send.await_count == 2
    assert "обновляю очередь" in adapter.send.await_args_list[0].args[1]
    assert "очередь инструментов обновлена" in adapter.send.await_args_list[1].args[1]


def test_brev_card_fields_include_title_alias_and_friendly_reply(monkeypatch):
    monkeypatch.setattr(producers_triage, "run_sanitizer", lambda value: value)
    raw = """кработ трек
название: nocturnal pulse
описание: instrumental neurodance with binaural ASMR pulses
стиль: neurodance, organic percussion
лирика:
опции:
alias: /tag/neurofunk
model: auto
"""
    request, errors = producers_triage.create_brev_generation_request(raw, "artist", "1509389598923559053")

    assert errors == []
    assert request is not None
    assert request["title"] == "nocturnal pulse"
    assert request["prompt"] == "instrumental neurodance with binaural ASMR pulses"
    assert request["style"] == "neurodance, organic percussion"
    assert request["requested_alias"] == "/tag/neurofunk"
    assert request["instrumental"] is True
    assert request["execution"] == "discord_auto_run"

    reply = producers_triage.format_brev_request_reply(request)
    assert "карточка трека принята" in reply
    assert "название: nocturnal pulse" in reply
    assert "статус: запускаю генерацию" in reply
    assert "ручную оператором" not in reply

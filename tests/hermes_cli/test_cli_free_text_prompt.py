"""Slash commands that ask for typed text (a /topup amount, a bare /hatch description) take the answer
from the composer while the interactive app runs.

The command runs on a worker thread, as the REPL's process_loop dispatches it, against a real
prompt_toolkit Application whose keys arrive through a pipe and the CLI's own key bindings. Only the
network calls (billing state, charge, auto-reload PATCH, image generation) are replaced.
"""

import asyncio
from decimal import Decimal

import pytest

from tests.waiting import async_wait_until


@pytest.fixture
def cli(monkeypatch):
    from cli import HermesCLI

    monkeypatch.setenv("HERMES_DEFER_AGENT_STARTUP", "1")
    cli = HermesCLI(model="fixture", provider="openai-compat", api_key="fixture",
                    base_url="http://127.0.0.1:1/v1")
    cli._tui_init_run_state()
    return cli


@pytest.fixture
def billing(monkeypatch):
    """Logged-in admin with a saved card; records what would be sent to the billing service."""
    from agent import billing_view
    from hermes_cli import nous_billing
    from agent.billing_view import BillingState, CardInfo

    state = BillingState(logged_in=True, role="OWNER", cli_billing_enabled=True, balance_usd=Decimal(5),
                         charge_presets=(Decimal(10), Decimal(50)), min_usd=Decimal(5),
                         max_usd=Decimal(1000), card=CardInfo(brand="Visa", last4="4242"))
    sent = {}
    monkeypatch.setattr(billing_view, "build_billing_state", lambda: state)
    monkeypatch.setattr(nous_billing, "post_charge",
                        lambda *, amount_usd, idempotency_key, **kw: sent.setdefault("charge", amount_usd) and {})
    monkeypatch.setattr(nous_billing, "patch_auto_top_up", lambda **kw: sent.setdefault("auto_reload", kw))
    return sent


def _drive(cli, command, script):
    """Run ``command`` on a worker thread while ``script(pipe, done)`` types into a live Application."""
    from prompt_toolkit.application import Application
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.output import DummyOutput

    editor = cli._tui_build_input_area()
    cli._input_area = editor

    async def run():
        with create_pipe_input() as pipe:
            app = Application(layout=Layout(editor), key_bindings=cli._tui_build_key_bindings(),
                              input=pipe, output=DummyOutput())
            cli._app = app
            painted = asyncio.Event()
            app.after_render += lambda _: painted.set()
            task = asyncio.create_task(app.run_async())
            await asyncio.wait_for(painted.wait(), 3)
            done = asyncio.Event()

            async def dispatch():
                try:
                    await asyncio.to_thread(cli.process_command, command)
                finally:
                    done.set()
            worker = asyncio.create_task(dispatch())
            try:
                await script(pipe, done)
                await asyncio.wait_for(worker, 5)
            finally:
                app.exit()
                await task
    asyncio.run(run())


_answered = []  # panels already answered; each panel is a fresh state dict


async def _answer(cli, pipe, done, keys, *, free_text):
    """Wait for the next panel (a choice list, or a free-text prompt), then type ``keys``."""
    def next_panel():
        state = cli._slash_confirm_state
        return (state is not None and all(state is not seen for seen in _answered)
                and bool(state.get("choices")) != free_text)
    await async_wait_until(lambda: done.is_set() or next_panel(), timeout=5, interval=0.01)
    assert not done.is_set(), f"the command returned before its {'text' if free_text else 'choice'} prompt opened"
    _answered.append(cli._slash_confirm_state)
    pipe.send_text(keys)


def test_topup_custom_amount_typed_in_the_composer_is_charged(cli, billing):
    async def script(pipe, done):
        await _answer(cli, pipe, done, "1", free_text=False)        # Add funds
        await _answer(cli, pipe, done, "3", free_text=False)        # Custom amount (after the two presets)
        await _answer(cli, pipe, done, "25\r", free_text=True)      # the amount
        await _answer(cli, pipe, done, "1", free_text=False)        # Pay now
    _drive(cli, "/topup", script)
    assert billing.get("charge") == Decimal(25)


def test_topup_auto_reload_amounts_typed_in_the_composer_are_saved(cli, billing):
    async def script(pipe, done):
        await _answer(cli, pipe, done, "2", free_text=False)        # Auto-reload
        await _answer(cli, pipe, done, "10\r", free_text=True)      # threshold
        await _answer(cli, pipe, done, "50\r", free_text=True)      # reload to
        await _answer(cli, pipe, done, "1", free_text=False)        # Agree
    _drive(cli, "/topup", script)
    assert billing.get("auto_reload") == {"enabled": True, "threshold": 10.0, "top_up_amount": 50.0}


def test_bare_hatch_takes_the_description_typed_in_the_composer(cli, monkeypatch):
    from agent.pet.generate import orchestrate
    concepts = []
    monkeypatch.setattr(orchestrate, "generate_base_drafts", lambda concept, n=1: concepts.append(concept) or [])

    async def script(pipe, done):
        await _answer(cli, pipe, done, "a small blue dragon\r", free_text=True)
    _drive(cli, "/hatch", script)
    assert concepts == ["a small blue dragon"]


def test_escape_on_a_text_prompt_cancels_without_an_answer(cli, monkeypatch):
    from agent.pet.generate import orchestrate
    concepts = []
    monkeypatch.setattr(orchestrate, "generate_base_drafts", lambda concept, n=1: concepts.append(concept) or [])

    async def script(pipe, done):
        await _answer(cli, pipe, done, "cancel\x1b", free_text=True)
    _drive(cli, "/hatch", script)
    assert concepts == []

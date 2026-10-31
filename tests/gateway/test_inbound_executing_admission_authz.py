"""An admission the authority is executing is not re-authorized against the transport allowlist.

When a durably-admitted admission is claimed and executed, ``_hm_admit_event`` re-ran the inbound
transport allowlist and bot-loop gate on the event. A cold Desktop row rebuilt as an API-family
source (gateway/session_desktop_recovery) has no allowlistable messaging identity — its row
``user_id`` is a backend stamp like ``basic:admin`` — so the re-run verdict denied the executing
turn ('Unauthorized user ... on api_server') and the row settled terminal/completed with no
transcript row, no turn and no reply. The executing admission was already authorized where it was
admitted, so its ContextVar now skips both gates; live inbound traffic keeps its verdict.
"""
import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


@pytest.fixture(autouse=True)
def _clean_allowlist_env(monkeypatch):
    for var in ('GATEWAY_ALLOW_ALL_USERS', 'GATEWAY_ALLOWED_USERS', 'API_SERVER_ALLOWED_USERS',
                'API_SERVER_ALLOW_ALL_USERS', 'TELEGRAM_ALLOWED_USERS', 'TELEGRAM_ALLOW_ALL_USERS'):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def runner(monkeypatch, tmp_path):
    from gateway.run import GatewayRunner

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    gw = GatewayRunner(GatewayConfig())
    ran = []

    async def turn(event, source, quick_key, run_generation):
        ran.append((event.text, source.platform))
        return 'reply ' + event.text

    monkeypatch.setattr(gw, '_handle_message_with_agent', turn)
    gw._ran = ran
    return gw


def _rebuilt_desktop_source():
    """The source session_desktop_recovery rebuilds: API-family, backend-stamp user_id, no
    messaging identity any allowlist could name."""
    return SessionSource(platform=Platform.API_SERVER, chat_id='20261030_100000_execdesk',
                         user_id='basic:admin', chat_type='dm')


async def _admit(runner, source, *, executing):
    from gateway.session_ingress import executing_admission

    event = MessageEvent(text='probe', message_id=f'p-{executing}', source=source)
    token = executing_admission.set(True) if executing else None
    try:
        return await runner._hm_admit_event(event)
    finally:
        if token is not None:
            executing_admission.reset(token)


@pytest.mark.asyncio
async def test_an_executing_admission_reaches_the_turn_path(runner):
    """The claimed viewer turn on a remounted desktop row is admitted into the turn path instead of
    being default-denied by the transport allowlist."""
    admitted = await _admit(runner, _rebuilt_desktop_source(), executing=True)
    assert admitted is not None
    event, source, internal = admitted
    assert event.text == 'probe'
    assert source.user_id == 'basic:admin'
    assert internal is False


@pytest.mark.asyncio
async def test_live_inbound_on_the_same_source_keeps_its_refusal(runner):
    """The exemption moves only executing admissions: the identical source arriving live (not
    through the durable FIFO) is still refused."""
    assert await _admit(runner, _rebuilt_desktop_source(), executing=False) is None
    assert runner._ran == []

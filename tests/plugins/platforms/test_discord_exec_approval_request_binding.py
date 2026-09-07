"""Request-identity regression tests for Discord exec-approval cards (#104915).

A card's buttons must resolve only the approval request that card was issued
for. Historically ``ExecApprovalView`` carried no request id, so
``resolve_gateway_approval`` fell back to its FIFO branch and a click resolved
the session's OLDEST pending request — approving a different command than the
one displayed (overlapping prompts, or a stale card still clickable after its
request was removed or expired).

The tests drive the real ``ExecApprovalView._resolve`` against synthetic queue
entries (``data={"request_id": ...}``, ``result=None``, ``threading.Event()``)
with a mocked interaction — no shell command runs and no Discord transport is
touched.
"""

import ast
import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# This file lives next to its owner module (plugins/platforms/discord/), away
# from tests/gateway/, so the shared comprehensive discord mock must be
# triggered explicitly before importing the production module.
from tests.gateway.conftest import _ensure_discord_mock  # noqa: E402

_ensure_discord_mock()

from plugins.platforms.discord.adapter import ExecApprovalView  # noqa: E402


SESSION_KEY = "sess-104915"
ALLOWED_UID = "11111"


def _entry(request_id: str, command: str = "cmd"):
    from tools.approval_gateway_wait import _ApprovalEntry

    return _ApprovalEntry({"request_id": request_id, "command": command})


def _view(request_id, *, allowed_user_ids=None):
    return ExecApprovalView(
        session_key=SESSION_KEY,
        allowed_user_ids={ALLOWED_UID} if allowed_user_ids is None else allowed_user_ids,
        request_id=request_id,
    )


def _interaction(user_id=ALLOWED_UID):
    response = MagicMock()
    response.send_message = AsyncMock()
    response.edit_message = AsyncMock()
    return SimpleNamespace(
        user=SimpleNamespace(id=user_id, display_name="op"),
        response=response,
        message=SimpleNamespace(embeds=[]),
    )


def _click(view, choice="once", *, user_id=ALLOWED_UID):
    interaction = _interaction(user_id)
    asyncio.run(view._resolve(interaction, choice, 1, "label"))
    return interaction


@pytest.fixture(autouse=True)
def _pairing_store_never_approves():
    mock_store = MagicMock()
    mock_store.is_approved.return_value = False
    with patch("gateway.pairing.PairingStore", return_value=mock_store):
        yield


@pytest.fixture(autouse=True)
def _color_dark_grey_available(monkeypatch):
    """The shared discord fake has no Color.dark_grey (only the expired-card path uses it)."""
    monkeypatch.setattr(
        "plugins.platforms.discord.adapter.discord.Color",
        SimpleNamespace(dark_grey=lambda: 0),
    )


@pytest.fixture()
def queue():
    """Own the session's gateway approval queue and always clean it up."""
    from tools import approval

    approval._gateway_queues.pop(SESSION_KEY, None)
    yield approval._gateway_queues
    approval._gateway_queues.pop(SESSION_KEY, None)


def test_card_resolves_only_its_own_request(queue):
    """Clicking card B must settle B, never the older request A."""
    entry_a = _entry("req-A", "echo A")
    entry_b = _entry("req-B", "echo B")
    queue[SESSION_KEY] = [entry_a, entry_b]

    _click(_view("req-B"), "once")

    assert entry_b.result == "once"
    assert entry_b.event.is_set()
    assert entry_a.result is None
    assert not entry_a.event.is_set()
    assert queue[SESSION_KEY] == [entry_a]


@pytest.mark.parametrize("choice", ["once", "session", "always", "deny"])
def test_each_choice_binds_to_its_request(queue, choice):
    """Every button choice resolves the bound request (not the queue head)."""
    entry = _entry("req-1")
    queue[SESSION_KEY] = [entry]

    _click(_view("req-1"), choice)

    assert entry.result == choice
    assert entry.event.is_set()
    assert SESSION_KEY not in queue


def test_stale_card_does_not_resolve_newer_request(queue):
    """A card whose request was removed must not settle a newer queued one."""
    entry_b = _entry("req-B")
    queue[SESSION_KEY] = [entry_b]

    _click(_view("req-A"), "once")

    assert entry_b.result is None
    assert not entry_b.event.is_set()
    assert queue[SESSION_KEY] == [entry_b]


def test_unbound_card_fails_closed(queue):
    """A card with no request id must not fall back to FIFO resolution."""
    entry = _entry("req-1")
    queue[SESSION_KEY] = [entry]

    _click(_view(None), "once")

    assert entry.result is None
    assert not entry.event.is_set()
    assert queue[SESSION_KEY] == [entry]


def test_duplicate_click_is_inert(queue):
    """After a card resolves, a second click is rejected without touching the queue."""
    entry = _entry("req-1")
    queue[SESSION_KEY] = [entry]
    view = _view("req-1")

    _click(view, "once")
    assert entry.result == "once"
    assert SESSION_KEY not in queue

    interaction = _click(view, "once")
    assert interaction.response.send_message.await_count == 1
    assert "already been resolved" in interaction.response.send_message.await_args.args[0]


def test_unauthorized_click_leaves_queue_untouched(queue):
    """An unauthorized click never reaches the resolver."""
    entry = _entry("req-1")
    queue[SESSION_KEY] = [entry]

    interaction = _click(_view("req-1"), "once", user_id="99999")

    assert interaction.response.send_message.await_count == 1
    assert "not authorized" in interaction.response.send_message.await_args.args[0]
    assert entry.result is None
    assert not entry.event.is_set()
    assert queue[SESSION_KEY] == [entry]


def test_gateway_passes_request_id_to_the_card():
    """The notify path must forward the entry's immutable request id (#104915).

    AST wiring guard in the style of TestApprovalCommandWiring: pins that
    ``_approval_notify_sync`` passes ``request_id=approval_data.get("request_id")``
    to every ``adapter.send_exec_approval`` that accepts the converged keyword
    (external/plugin adapters keep the legacy call shape), so each rendered card
    can bind to its request generation.
    """
    import gateway.run_turn_runner as run

    source = inspect.getsource(run)
    tree = ast.parse(source)
    notify_fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_approval_notify_sync"
    )
    send_calls = [
        node
        for node in ast.walk(notify_fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "send_exec_approval"
    ]
    assert send_calls, "send_exec_approval call not found in _approval_notify_sync"
    # request_id rides through the extra kwargs dict, gated on _accepts_keyword so
    # adapters predating the converged signature are not broken by the wider call.
    guarded_assigns = [
        node
        for node in ast.walk(notify_fn)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Subscript)
        and ast.unparse(node.targets[0]) == "extra['request_id']"
    ]
    assert guarded_assigns, "extra['request_id'] assignment not found in _approval_notify_sync"
    assert (
        ast.unparse(guarded_assigns[0].value) == "approval_data.get('request_id')"
    ), "request_id must come from approval_data"
    keyword_gates = [
        node
        for node in ast.walk(notify_fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_accepts_keyword"
    ]
    assert keyword_gates, "_accepts_keyword gate not found in _approval_notify_sync"

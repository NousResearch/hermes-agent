"""``gateway/session_bot.py``'s Bot-Mode DM admission gate: a delivery must offer the receiving
profile's ``pre_gateway_dispatch`` plugins the trusted sender identity before it commits (the same
pre-inference authorization boundary platform messages get), and a ``skip`` must refuse the
delivery with nothing written — never silently admit it. Exercises ``_admit``/``_maybe_retry``
directly against fakes, so it never depends on the full gateway/model bootstrap a real Bot Chat
creation would pull in (out of scope for this unit)."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.session_bot import _admit, _maybe_retry
from gateway.session_contract import SessionActor, SessionRef
from hermes_state_runtime import RuntimeStoreError


class _FakeSource:
    platform = None


class _FakeLive:
    def __init__(self):
        self.source = _FakeSource()
        self.route = 'route-1'


class _FakeEntry:
    session_id = 'sess-1'


class _FakeAuthority:
    def __init__(self, runner):
        self.runner = runner
        self.profile_id = 'default'
        self.waiters = {}

    async def admit_automation(self, adapter, event, identity):
        raise AssertionError('admit_automation must not run once the gate refused the delivery')


def _actor():
    return SessionActor('uid:1000', 'default', frozenset({'session:submit', 'session:read'}), 'test')


@pytest.mark.asyncio
async def test_skip_refuses_before_admission_and_commits_nothing(tmp_path, monkeypatch):
    async def deny(event, source):
        return None

    runner = SimpleNamespace(_hm_pre_gateway_dispatch_hook=deny)
    authority = _FakeAuthority(runner)
    ref = SessionRef('default', 'sess-1')
    home = tmp_path
    root = tmp_path / 'mailbox'
    root.mkdir()

    with pytest.raises(RuntimeStoreError) as exc:
        await _admit(authority, _actor(), home, root, 'd' * 64, 'hello teammate', ref, _FakeLive(), _FakeEntry(),
                     author={'id': 'bot:unknown', 'name': 'x', 'is_bot': True})

    assert exc.value.reason == 'authorization_refused'
    assert list(root.iterdir()) == [], 'nothing committed when the gate refuses'


@pytest.mark.asyncio
async def test_allowed_delivery_reaches_the_gate_with_the_trusted_author_then_admits(tmp_path, monkeypatch):
    seen = []

    async def allow(event, source):
        seen.append((event, source))
        return event

    class _Receipt:
        admission_id = 'adm-1'

    class _Authority(_FakeAuthority):
        async def admit_automation(self, adapter, event, identity):
            return _Receipt()

    runner = SimpleNamespace(_hm_pre_gateway_dispatch_hook=allow, _adapter_for_source=lambda source: None)
    authority = _Authority(runner)
    monkeypatch.setattr('gateway.session_bot._result', lambda authority, record: {'status': 'queued', 'reply': ''})
    ref = SessionRef('default', 'sess-1')
    home = tmp_path
    root = tmp_path / 'mailbox'
    root.mkdir()
    author = {'id': 'bot:watch_tracker', 'name': 'watch_tracker', 'is_bot': True}

    result = await _admit(authority, _actor(), home, root, 'e' * 64, 'hello teammate', ref, _FakeLive(), _FakeEntry(),
                          author=author)

    assert result['status'] == 'queued', result
    [(event, source)] = seen
    assert event.metadata['turn_author'] == author
    assert event.text == 'hello teammate'
    assert source is not None


@pytest.mark.asyncio
async def test_trusted_producer_report_has_no_author_in_the_gate(tmp_path, monkeypatch):
    """A delivery with no author (a cron/producer report) must not be misrepresented to the gate
    as a bot-authored message — the metadata key is simply absent, never a forged claim."""
    seen = []

    async def allow(event, source):
        seen.append(event)
        return event

    class _Receipt:
        admission_id = 'adm-2'

    class _Authority(_FakeAuthority):
        async def admit_automation(self, adapter, event, identity):
            return _Receipt()

    runner = SimpleNamespace(_hm_pre_gateway_dispatch_hook=allow, _adapter_for_source=lambda source: None)
    authority = _Authority(runner)
    monkeypatch.setattr('gateway.session_bot._result', lambda authority, record: {'status': 'queued', 'reply': ''})
    ref = SessionRef('default', 'sess-1')
    root = tmp_path / 'mailbox'
    root.mkdir()

    await _admit(authority, _actor(), tmp_path, root, 'f' * 64, 'status report', ref, _FakeLive(), _FakeEntry())

    [event] = seen
    assert 'turn_author' not in event.metadata


@pytest.mark.asyncio
async def test_a_bare_runner_with_no_hook_machinery_fails_open(tmp_path, monkeypatch):
    """A runner that never wired the gateway hook mixin (bare test harnesses, stripped-down
    deployments) must not break bot-chat delivery — the gate only enforces when it is wired."""
    class _Receipt:
        admission_id = 'adm-3'

    class _Authority(_FakeAuthority):
        async def admit_automation(self, adapter, event, identity):
            return _Receipt()

    runner = SimpleNamespace(_adapter_for_source=lambda source: None)  # no _hm_pre_gateway_dispatch_hook
    authority = _Authority(runner)
    monkeypatch.setattr('gateway.session_bot._result', lambda authority, record: {'status': 'queued', 'reply': ''})
    ref = SessionRef('default', 'sess-1')
    root = tmp_path / 'mailbox'
    root.mkdir()

    result = await _admit(authority, _actor(), tmp_path, root, 'a' * 64, 'hello', ref, _FakeLive(), _FakeEntry())

    assert result['status'] == 'queued', result


@pytest.mark.asyncio
async def test_maybe_retry_skip_refuses_without_admitting(tmp_path, monkeypatch):
    async def deny(event, source):
        return None

    runner = SimpleNamespace(_hm_pre_gateway_dispatch_hook=deny)

    class _Authority(_FakeAuthority):
        def __init__(self, runner):
            super().__init__(runner)
            self.db = SimpleNamespace(latest_conversation_role=lambda sid: 'assistant')

        def _require_admission_open(self):
            return None

    authority = _Authority(runner)
    home = tmp_path
    path = tmp_path / ('d' * 64)
    record = dict(delivery_id='d' * 64, profile_home=str(home), session_id='sess-1', principal_id='uid:1000',
                 message='hi', status='ambiguous',
                 author={'id': 'bot:unknown', 'name': 'x', 'is_bot': True})
    from gateway import session_bot as sb

    async def fake_target(authority, actor):
        return SessionRef('default', 'sess-1'), _FakeLive(), _FakeEntry()

    monkeypatch.setattr(sb, '_target', fake_target)
    monkeypatch.setattr(sb, '_retry_eligible', lambda authority, record: True)
    monkeypatch.setattr(sb, '_retry_identity', lambda delivery_id: 'retry:' + delivery_id)

    await _maybe_retry(authority, home, path, record)

    assert record['retry'] == {'identity': 'retry:' + record['delivery_id'], 'refused': 'authorization_refused'}

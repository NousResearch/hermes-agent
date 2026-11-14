"""``agent.adaptive_reasoning`` on a LOCAL session turn (classic CLI, Ink, Desktop): the owner's TurnRunner
and a managed worker both run it, from the session's policy config, and neither adjusts a level the session
chose.

The classic CLI no longer builds the agent that runs its turn (``hermes_cli/gateway_agent.py`` is a proxy), so
the shell-side provenance never reaches it: the owner reads the session's own policy. One ordinary daemon,
an in-process and a ``gateway.managed_workers`` session from the same profile; the loopback provider records
the ``reasoning_effort`` each turn sent.
"""
import pytest

from tests.gateway.test_managed_worker_construct import run_pair

DEBUG = 'Why does the gateway keep failing after I restart it?'  # lone code signal: high
PLAIN = 'An ambiguous follow-up'


def _adaptive_profile(home):
    return {'agent': {'reasoning_effort': 'medium', 'adaptive_reasoning': {'enabled': True, 'max_effort': 'high'}}}


async def _two_turns(ws, sid, name, settle, servers):
    efforts = []
    for index, text in enumerate((DEBUG, PLAIN)):
        start = len(servers.primary.requests)
        assert (await settle(ws, sid, f'{name}-{index}', text))['final_response'] == 'PRIMARY_OK'
        sent = [r for r in servers.primary.requests[start:]
                if any(text in str(m.get('content')) for m in r['messages'] if m.get('role') == 'user')]
        efforts.append([r.get('reasoning_effort') for r in sent])
    return {'efforts': efforts}


@pytest.mark.platforms("linux")
def test_local_session_turn_adapts_then_restores_the_baseline(tmp_path):
    results = run_pair(tmp_path, _adaptive_profile, _two_turns)
    for name in ('inproc', 'managed'):
        assert results[name]['efforts'] == [['high'], ['medium']], (name, results)
    assert (results['inproc']['workers'], results['managed']['workers']) == (0, 2), results


@pytest.mark.platforms("linux")
def test_local_session_reasoning_pick_stops_adjusting_and_reset_resumes(tmp_path):
    """A committed session ``/reasoning <level>`` (``config.set reasoning``, what Ink, Desktop and the classic
    CLI send) pins the session: the owner clears the typed override once the pick is the policy's, so only
    the policy's own record says so. ``/reasoning reset`` gives the level, and adaptation, back."""
    from tests.gateway.fixtures.local_recovery_probe import rpc

    async def turn(ws, sid, name, settle, servers):
        async def debug_turn(tag):
            start = len(servers.primary.requests)
            assert (await settle(ws, sid, f'{name}-{tag}', DEBUG))['final_response'] == 'PRIMARY_OK'
            return [r.get('reasoning_effort') for r in servers.primary.requests[start:]]

        efforts = [await debug_turn('adapt')]
        for value, tag in (('low', 'pinned'), ('reset', 'reset')):
            changed = await rpc(ws, 'config.set', session_id=sid, key='reasoning', value=value)
            assert 'result' in changed, changed
            efforts.append(await debug_turn(tag))
        return {'efforts': efforts}
    results = run_pair(tmp_path, _adaptive_profile, turn)
    for name in ('inproc', 'managed'):
        assert results[name]['efforts'] == [['high'], ['low'], ['high']], (name, results)


@pytest.mark.platforms("linux")
def test_local_session_launch_level_is_never_adjusted(tmp_path):
    """``--reasoning low`` / a Desktop pick distinct from the default (``session.create reasoning``) is the
    session's own choice: the debugging turn stays at it."""
    results = run_pair(tmp_path, _adaptive_profile, _two_turns, reasoning='low')
    for name in ('inproc', 'managed'):
        assert results[name]['efforts'] == [['low'], ['low']], (name, results)

"""A key-less ``ctx.inject_message(text)`` from plugin code the gateway runs (#134889 PLG-7, B2-11).

Before the one-gateway runtime each classic CLI loaded its own plugins, so a key-less inject meant that
CLI's own chat. The gateway's plugin copy serves every session of its profile, so the gateway picks the
target: the local session the plugin is working for (its command, a hook fired for it, a hook in its
turn); with no session in the calling context (the plugin's own thread), the one attached classic CLI
session of the plugin's profile; otherwise a named refusal and nothing injected.

A real daemon (``gateway.run``) loads a probe plugin that notes each ``inject_message`` result; a landed
inject is a queued admission in the profile's ``state.db`` that runs once.
"""
from contextlib import contextmanager
import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import signal
import sqlite3
import threading

import pytest

from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon
from tests.gateway.test_cli_plugin_command_in_gateway import _Wire
from tests.gateway.test_pre_command_in_gateway import _ProfileWire
from tests.waiting import wait_until

_PLUGIN = '''import json
import threading


def register(ctx):
    def inject(where, text, **kw):
        ok = ctx.inject_message(text, **kw)
        with open(%r, "a") as log:
            log.write(json.dumps({"where": where, "text": text, "ok": ok}) + "\\n")

    def keyed(arg):
        key, _, text = arg.partition(" ")
        inject("keyed", text, session_key=key)

    def own_thread(arg):
        threading.Thread(target=inject, args=("thread", arg), daemon=True).start()
        return "started"

    def pre_command(command="", args_raw="", **_kw):
        if command == "help" and args_raw.startswith("inject "):
            inject("hook", args_raw[len("inject "):])

    def pre_llm_call(user_message="", **_kw):
        if str(user_message).startswith("TURN_INJECT "):
            inject("turn", str(user_message)[len("TURN_INJECT "):])

    ctx.register_command("inj", lambda arg: inject("command", arg), description="Probe inject")
    ctx.register_command("inj-key", keyed, description="Probe keyed inject")
    ctx.register_command("inj-thread", own_thread, description="Probe inject from a thread")
    ctx.register_hook("pre_command", pre_command)
    ctx.register_hook("pre_llm_call", pre_llm_call)
'''


def _probe_plugin(home, log):
    plugin = home / 'plugins' / 'probe-inj'
    plugin.mkdir(parents=True)
    (plugin / 'plugin.yaml').write_text("name: probe-inj\nversion: '1.0'\n")
    (plugin / '__init__.py').write_text(_PLUGIN % str(log))


@contextmanager
def keyless_gateway(tmp_path, *, isolation=None, beta=False):
    """A real daemon whose profile loads the probe plugin (``allow_gateway_injection`` granted):
    ``(desc, home, work, log, peer)``. ``beta`` also serves profile ``beta`` (multiplexed) with its own
    copy of the plugin, noting to ``beta.jsonl``."""
    root = Path(__file__).resolve().parents[2]
    home, user, work = tmp_path / 'state', tmp_path / 'user', tmp_path / 'work'
    home.mkdir(mode=0o700)
    user.mkdir()
    work.mkdir()
    log = tmp_path / 'inject.jsonl'
    _probe_plugin(home, log)
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    plugins = {'enabled': ['probe-inj'], 'entries': {'probe-inj': {'allow_gateway_injection': True}},
               **({'isolation': isolation} if isolation else {})}
    profile = {'model': {'provider': 'custom', 'default': 'plugin-model', 'base_url': url},
               'auxiliary': {'title_generation': {'enabled': False}}, 'plugins': plugins}
    (home / 'config.yaml').write_text(json.dumps({'gateway': {'multiplex_profiles': beta}, **profile}))
    if beta:
        (home / 'profiles' / 'beta').mkdir(parents=True)
        (home / 'profiles' / 'beta' / 'config.yaml').write_text(json.dumps(profile))
        _probe_plugin(home / 'profiles' / 'beta', tmp_path / 'beta.jsonl')
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1')
    try:
        with daemon(root, home, env, barrier=False) as (proc, desc):
            yield desc, home, work, log, peer
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=20)
    finally:
        peer.release.set()
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)


def noted(log, where, text):
    """The plugin's ``inject_message`` result for ``text`` (its note lands after the call returns)."""
    def note():
        lines = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return next(([r['ok']] for r in lines if (r['where'], r['text']) == (where, text)), None)
    return wait_until(note, timeout=20, what=f'the {where} note for {text!r}')[0]


def admissions(home, text):
    """``(session id, status)`` of every admission in ``home``'s ledger whose text is ``text``."""
    with sqlite3.connect(home / 'state.db') as db:
        rows = db.execute('SELECT target_session_id, status, payload_json FROM session_admissions ORDER BY seq').fetchall()
    return [(sid, status) for sid, status, payload in rows if json.loads(payload).get('text') == text]


def landed_once(home, peer, text):
    """The one session ``text`` was queued into, once its turn ran; the model saw it exactly once."""
    def ran():
        rows = admissions(home, text)
        return rows if rows and all(status == 'terminal' for _, status in rows) else None
    rows = wait_until(ran, timeout=30, what=f'the turn for {text!r}')
    asked = [r for r in peer.requests
             if next((m.get('content') for m in reversed(r['messages']) if m['role'] == 'user'), None) == text]
    assert len(rows) == 1 and len(asked) == 1, (rows, len(asked))
    return rows[0][0]


def gateway_log(home):
    return (home / 'restart.log').read_text()


@pytest.mark.platforms("linux")
@pytest.mark.parametrize('isolation', [None, 'host'])
def test_a_key_less_inject_goes_to_the_session_the_plugin_is_working_for(tmp_path, isolation):
    """Rule 1, with two classic CLI sessions (and a TUI one) attached: a plugin command, a ``pre_command``
    hook fired for a session and a hook in a session's turn each land in that session only, once, as a
    queued input; in the plugin host too (the forwarded call keeps its caller). A keyed inject is
    unchanged: accepted for the gateway's keyed path, which never falls back to the calling session."""
    with keyless_gateway(tmp_path, isolation=isolation) as (_desc, home, work, log, peer):
        wire = _Wire(home, _desc)
        try:
            cli1, cli2, tui = (wire.call('session.create', request_id=f'r1-{source}-{n}', source=source, toolsets=[],
                                         cwd=str(work))['session_id'] for n, source in enumerate(('cli', 'cli', 'tui')))
            wire.call('slash.exec', session_id=cli2, command='/inj CMD_TEXT')
            wire.call('slash.exec', session_id=tui, command='/inj TUI_TEXT')
            wire.call('slash.exec', session_id=cli1, command='/help inject HOOK_TEXT')
            wire.call('prompt.submit', session_id=cli1, input_id='turn', text='TURN_INJECT TURN_TEXT')
            wire.call('slash.exec', session_id=cli1, command='/inj-key agent:main:telegram:dm:nobody KEYED_TEXT')
            results = {text: noted(log, where, text) for where, text in (
                ('command', 'CMD_TEXT'), ('command', 'TUI_TEXT'), ('hook', 'HOOK_TEXT'), ('turn', 'TURN_TEXT'),
                ('keyed', 'KEYED_TEXT'))}
            targets = {text: landed_once(home, peer, text) for text in ('CMD_TEXT', 'TUI_TEXT', 'HOOK_TEXT', 'TURN_TEXT')}
            wait_until(lambda: 'plugin=probe-inj session=agent:main:telegram:dm:nobody' in gateway_log(home),
                       timeout=20, what='the keyed inject reaching the keyed path')
            keyed = admissions(home, 'KEYED_TEXT')
        finally:
            wire.close()
    assert results == dict.fromkeys(results, True), results
    assert targets == {'CMD_TEXT': cli2, 'TUI_TEXT': tui, 'HOOK_TEXT': cli1, 'TURN_TEXT': cli1}, targets
    assert keyed == []


@pytest.mark.platforms("linux")
def test_with_no_calling_session_only_the_one_attached_classic_cli_session_is_picked(tmp_path):
    """Rules 2 and 3 from the plugin's own thread (started by a command, it inherits no session): with
    only TUI, Desktop and another profile's classic CLI session attached it is refused by name and
    nothing lands anywhere; with one classic CLI session of its profile it lands there once; with two it
    is refused again (one warning per kind, not one per call)."""
    with keyless_gateway(tmp_path, beta=True) as (desc, home, work, log, peer):
        beta_home = (home / 'profiles' / 'beta').resolve()
        root, beta = _Wire(home, desc), _ProfileWire(home, desc, beta_home)
        try:
            beta.call('session.create', request_id='beta-cli', source='cli', toolsets=[], cwd=str(work))
            tui, _gui = (root.call('session.create', request_id=f'r2-{source}', source=source, toolsets=[],
                                   cwd=str(work))['session_id'] for source in ('tui', 'gui'))
            root.call('slash.exec', session_id=tui, command='/inj-thread ZERO_TEXT')
            zero = noted(log, 'thread', 'ZERO_TEXT')
            zero_rows = admissions(home, 'ZERO_TEXT') + admissions(beta_home, 'ZERO_TEXT')
            cli1 = root.call('session.create', request_id='r2-cli1', source='cli', toolsets=[], cwd=str(work))['session_id']
            root.call('slash.exec', session_id=tui, command='/inj-thread ONE_TEXT')
            one = noted(log, 'thread', 'ONE_TEXT')
            one_target = landed_once(home, peer, 'ONE_TEXT')
            root.call('session.create', request_id='r2-cli2', source='cli', toolsets=[], cwd=str(work))
            for text in ('TWO_TEXT_A', 'TWO_TEXT_B'):
                root.call('slash.exec', session_id=tui, command=f'/inj-thread {text}')
            two = [noted(log, 'thread', text) for text in ('TWO_TEXT_A', 'TWO_TEXT_B')]
            two_rows = [row for text in ('TWO_TEXT_A', 'TWO_TEXT_B')
                        for row in admissions(home, text) + admissions(beta_home, text)]
            log_text = gateway_log(home)
        finally:
            root.close()
            beta.close()
    assert (zero, zero_rows) == (False, [])
    assert (one, one_target) == (True, cli1)
    assert (two, two_rows) == ([False, False], [])
    refusals = [line for line in log_text.splitlines()
                if 'WARNING' in line and 'plugin probe-inj has no session key' in line and 'Pass session_key=' in line]
    assert len(refusals) == 2, refusals
    assert 'no classic CLI session' in refusals[0] and '2 classic CLI sessions' in refusals[1], refusals


def test_a_refusal_warns_once_per_window_and_expired_marks_are_dropped(monkeypatch, caplog):
    """One warning per (plugin, kind) per window; a mark older than the window is dropped on the next
    refusal, so the map holds only the plugins refused within the last window."""
    from types import SimpleNamespace
    from gateway import plugin_keyless_inject as keyless
    clock = [1000.0]
    monkeypatch.setattr(keyless, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(keyless, '_warned', {})
    with caplog.at_level('WARNING', logger='gateway.run'):
        keyless._refuse('old', 'none', 'no classic CLI session is attached in its profile')
        keyless._refuse('old', 'none', 'no classic CLI session is attached in its profile')
        clock[0] += keyless._WARN_WINDOW_S
        keyless._refuse('new', 'several', '2 classic CLI sessions are attached in its profile')
    assert [r.getMessage().split(' has ')[0] for r in caplog.records if r.levelname == 'WARNING'] == [
        'inject_message from plugin old', 'inject_message from plugin new']
    assert keyless._warned == {('new', 'several'): clock[0]}

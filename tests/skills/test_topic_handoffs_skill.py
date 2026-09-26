"""Tests for the shipped topic_handoffs.py script.

These tests drive the REAL script exactly as shipped:

- Configuration is read from environment variables at import time, so every
  test reloads the module via importlib.util.spec_from_file_location /
  module_from_spec AFTER monkeypatching the environment.
- There is no run() function; tests call main() and check its integer return
  value (0 = success, 1 = at least one topic failed).
- Files land in OUT_DIR/<date>/<tid>-<slug>.md and OUT_DIR/latest/<tid>.md
  (HANDOFF_CHAT_ID is set in every test, so the stem is just the thread id).
- redact(text) returns a (clean_text, dropped_count) tuple.
- The send path is send_message(...) -> urllib.request.urlopen; tests mock
  urllib.request.urlopen so no live network is ever touched.
- The fixture database uses the real schema:
      sessions(id, chat_id, thread_id, title, last_activity_at)  -- unix ts int
      messages(session_id, role, content, timestamp)
  and is opened by the script read-only via a file: URI.
"""

import datetime
import importlib.util
import json
import os
import sqlite3
import urllib.request
from pathlib import Path
from unittest import mock

CHAT_ID = -100123
BASE_TS = 1_700_000_000  # fixed, valid unix timestamp for last_activity_at


# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------

def _find_script() -> Path:
    """Locate the topic_handoffs.py helper script.

    In the repo layout the test file lives at
    <repo>/tests/skills/test_topic_handoffs_skill.py and the script lives at
    <repo>/skills/productivity/telegram-topic-handoffs/scripts/topic_handoffs.py,
    so the repo root is Path(__file__).resolve().parents[2]. If the expected
    path does not exist, fall back to an rglob of the repo root before giving
    up with a clear FileNotFoundError.
    """
    repo_root = Path(__file__).resolve().parents[2]
    expected = (
        repo_root
        / "skills"
        / "productivity"
        / "telegram-topic-handoffs"
        / "scripts"
        / "topic_handoffs.py"
    )
    if expected.is_file():
        return expected

    candidates = sorted(repo_root.rglob("topic_handoffs.py"))
    if candidates:
        preferred = [p for p in candidates if "telegram-topic-handoffs" in p.parts]
        return preferred[0] if preferred else candidates[0]

    raise FileNotFoundError(
        "Could not locate topic_handoffs.py. Expected it at "
        f"{expected}, and an rglob of the repo root ({repo_root}) found no "
        "file named topic_handoffs.py. Confirm that the telegram-topic-handoffs "
        "skill is checked out under skills/productivity/."
    )


SCRIPT_PATH = _find_script()


def load_module(monkeypatch, env):
    """Re-import topic_handoffs fresh with a fully controlled environment.

    The shipped script reads ALL of its configuration at import time, so the
    environment must be monkeypatched before exec_module runs.
    """
    for key in list(os.environ):
        if key.startswith('HANDOFF_') or key in ('TELEGRAM_BOT_TOKEN', 'DRY_RUN'):
            monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    spec = importlib.util.spec_from_file_location('topic_handoffs', SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def base_env(tmp_path, **overrides):
    """Environment shared by all tests; tests override what they exercise."""
    env = {
        'HANDOFF_CHAT_ID': str(CHAT_ID),
        'HANDOFF_DB': str(tmp_path / 'state.db'),
        'HANDOFF_OUT_DIR': str(tmp_path / 'out'),
        # Point the .env fallback at a file that does not exist so no real
        # token can leak in from ~/.hermes/.env on the host.
        'HANDOFF_ENV_FILE': str(tmp_path / 'does-not-exist.env'),
        'HANDOFF_SEND_DELAY': '0',  # keep the post-send pause out of tests
    }
    env.update(overrides)
    return env


def seed_db(db_path, sessions, messages):
    """Create the real schema the script queries, then close cleanly.

    sessions:  (id, chat_id, thread_id, title, last_activity_at)
    messages:  (session_id, role, content, timestamp)
    """
    con = sqlite3.connect(str(db_path))
    con.execute(
        'CREATE TABLE sessions ('
        'id INTEGER PRIMARY KEY, chat_id INTEGER, thread_id INTEGER, '
        'title TEXT, last_activity_at INTEGER)')
    con.execute(
        'CREATE TABLE messages ('
        'id INTEGER PRIMARY KEY, session_id INTEGER, role TEXT, '
        'content TEXT, timestamp INTEGER)')
    con.executemany(
        'INSERT INTO sessions (id, chat_id, thread_id, title, last_activity_at) '
        'VALUES (?, ?, ?, ?, ?)', sessions)
    con.executemany(
        'INSERT INTO messages (session_id, role, content, timestamp) '
        'VALUES (?, ?, ?, ?)', messages)
    con.commit()
    con.close()


def day_dir(tmp_path):
    return tmp_path / 'out' / datetime.date.today().isoformat()


def latest_dir(tmp_path):
    return tmp_path / 'out' / 'latest'


# --------------------------------------------------------------------------
# 1. Handoff files are created
# --------------------------------------------------------------------------

def test_handoff_files_are_created(tmp_path, monkeypatch):
    seed_db(
        tmp_path / 'state.db',
        sessions=[(1, CHAT_ID, 42, 'release planning', BASE_TS)],
        messages=[
            (1, 'user', 'when is the release cut?', BASE_TS + 100),
            (1, 'assistant', 'the release cut is on friday', BASE_TS + 200),
        ],
    )
    names_file = tmp_path / 'names.json'
    names_file.write_text(json.dumps({'42': 'release planning'}),
                          encoding='utf-8')
    mod = load_module(monkeypatch, base_env(
        tmp_path, DRY_RUN='1', HANDOFF_TOPIC_NAMES=str(names_file)))

    rc = mod.main()

    assert rc == 0
    # Real layout: OUT_DIR/<date>/<tid>-<slug>.md and OUT_DIR/latest/<tid>.md
    day_file = day_dir(tmp_path) / '42-release-planning.md'
    latest_file = latest_dir(tmp_path) / '42.md'
    assert day_file.is_file()
    assert latest_file.is_file()
    body = day_file.read_text(encoding='utf-8')
    assert '# Handoff: release planning (topic 42)' in body
    assert '- User: when is the release cut?' in body
    assert '- Assistant: the release cut is on friday' in body
    # latest/ holds the identical body
    assert latest_file.read_text(encoding='utf-8') == body


# --------------------------------------------------------------------------
# 2. Dry run makes no network call
# --------------------------------------------------------------------------

def test_dry_run_makes_no_network_call(tmp_path, monkeypatch, capsys):
    seed_db(
        tmp_path / 'state.db',
        sessions=[(1, CHAT_ID, 7, 'infra', BASE_TS)],
        messages=[(1, 'user', 'hello world', BASE_TS + 100)],
    )
    # A token IS present; DRY_RUN must still skip the send path entirely.
    mod = load_module(monkeypatch, base_env(
        tmp_path, DRY_RUN='1',
        TELEGRAM_BOT_TOKEN='fake-bot-token-for-tests'))

    with mock.patch.object(urllib.request, 'urlopen') as urlopen_mock:
        rc = mod.main()

    assert rc == 0
    assert urlopen_mock.call_count == 0
    assert 'DRY RUN' in capsys.readouterr().out
    # Dry run still writes files (name falls back to "topic <tid>").
    assert (day_dir(tmp_path) / '7-topic-7.md').is_file()
    assert (latest_dir(tmp_path) / '7.md').is_file()


# --------------------------------------------------------------------------
# 3. Secret lines are redacted from output files
# --------------------------------------------------------------------------

def test_secret_lines_are_redacted_from_output_files(tmp_path, monkeypatch):
    seed_db(
        tmp_path / 'state.db',
        sessions=[(1, CHAT_ID, 5, 'ops chat', BASE_TS)],
        messages=[
            (1, 'user', 'totally benign status update', BASE_TS + 100),
            (1, 'user',
             'my api_key = sk-ant-aaaabbbbccccdddd1234 do not share',
             BASE_TS + 200),
        ],
    )
    mod = load_module(monkeypatch, base_env(tmp_path, DRY_RUN='1'))

    # Unit-level contract: redact returns a (clean_text, dropped_count) tuple.
    result = mod.redact('keep me\npassword: hunter2\nkeep me too')
    assert isinstance(result, tuple)
    assert len(result) == 2
    clean, dropped = result
    assert dropped == 1
    assert clean == 'keep me\nkeep me too'

    rc = mod.main()

    assert rc == 0
    body = (day_dir(tmp_path) / '5-topic-5.md').read_text(encoding='utf-8')
    assert 'totally benign status update' in body
    # The entire line carrying the secret is dropped before it hits disk.
    assert 'sk-ant' not in body
    assert 'api_key' not in body
    # ...and the script appends its redaction note to the file.
    assert '1 line(s) omitted by the secret-redaction pass' in body


# --------------------------------------------------------------------------
# 4. Messages starting with the handoff prefix are filtered out
# --------------------------------------------------------------------------

def test_handoff_prefix_messages_are_filtered_out(tmp_path, monkeypatch):
    seed_db(
        tmp_path / 'state.db',
        sessions=[(1, CHAT_ID, 9, 'echo check', BASE_TS)],
        messages=[
            # This script's own earlier post: must never be quoted back.
            (1, 'assistant',
             'Daily handoff for echo check (yesterday). ECHO-MARKER-123',
             BASE_TS + 100),
            # Leading whitespace must not defeat the lstrip()+startswith filter.
            (1, 'assistant',
             '   Daily handoff for echo check. ECHO-MARKER-WS',
             BASE_TS + 150),
            (1, 'user', 'REAL-QUESTION-456 what shipped?', BASE_TS + 200),
            (1, 'assistant', 'REAL-ANSWER-789 the parser shipped',
             BASE_TS + 300),
        ],
    )
    mod = load_module(monkeypatch, base_env(
        tmp_path, DRY_RUN='1', HANDOFF_PREFIX='Daily handoff for'))

    rc = mod.main()

    assert rc == 0
    body = (day_dir(tmp_path) / '9-topic-9.md').read_text(encoding='utf-8')
    assert 'ECHO-MARKER-123' not in body
    assert 'ECHO-MARKER-WS' not in body
    assert 'Daily handoff for' not in body
    assert 'REAL-QUESTION-456' in body
    assert 'REAL-ANSWER-789' in body


# --------------------------------------------------------------------------
# 5. One malformed topic does not stop the others
# --------------------------------------------------------------------------

def test_malformed_topic_does_not_stop_other_topics(tmp_path, monkeypatch,
                                                    capsys):
    # Topic 2 is malformed: last_activity_at is TEXT, so the script's
    # datetime.datetime.fromtimestamp(last_act) raises TypeError, which is
    # caught by the per-topic isolation handler (the inner try only catches
    # OverflowError/OSError/ValueError).
    seed_db(
        tmp_path / 'state.db',
        sessions=[
            (1, CHAT_ID, 1, 'good topic', BASE_TS),
            (2, CHAT_ID, 2, 'bad topic', 'not-a-timestamp'),
        ],
        messages=[
            (1, 'user', 'good topic message', BASE_TS + 100),
            (2, 'user', 'bad topic message', BASE_TS + 100),
        ],
    )
    mod = load_module(monkeypatch, base_env(tmp_path, DRY_RUN='1'))

    rc = mod.main()

    # main() returns 1 when any topic failed...
    assert rc == 1
    # ...but the healthy topic was still extracted and filed.
    good_file = day_dir(tmp_path) / '1-topic-1.md'
    assert good_file.is_file()
    assert 'good topic message' in good_file.read_text(encoding='utf-8')
    # The malformed topic wrote nothing (it failed before the file writes).
    assert not (day_dir(tmp_path) / '2-topic-2.md').exists()
    out = capsys.readouterr().out
    assert 'FAILURES: 1' in out
    assert '- topic 2 topic 2: error:' in out


# --------------------------------------------------------------------------
# 6. Live mode posts through the mocked transport
# --------------------------------------------------------------------------

def test_live_mode_posts_through_mocked_transport(tmp_path, monkeypatch):
    token = 'fake-bot-token-for-tests'
    seed_db(
        tmp_path / 'state.db',
        sessions=[(1, CHAT_ID, 42, 'live topic', BASE_TS)],
        messages=[
            (1, 'user', 'live excerpt question', BASE_TS + 100),
            (1, 'assistant', 'live excerpt answer', BASE_TS + 200),
        ],
    )
    # No DRY_RUN: live mode. send_message() will call urllib.request.urlopen.
    mod = load_module(monkeypatch, base_env(
        tmp_path, TELEGRAM_BOT_TOKEN=token))

    fake_response = mock.MagicMock(name='fake-response')
    fake_response.read.return_value = b'{"ok": true}'
    fake_response.__enter__.return_value = fake_response
    fake_response.__exit__.return_value = False

    with mock.patch.object(urllib.request, 'urlopen',
                           return_value=fake_response) as urlopen_mock:
        rc = mod.main()

    assert rc == 0
    assert urlopen_mock.call_count == 1
    req = urlopen_mock.call_args[0][0]
    assert isinstance(req, urllib.request.Request)
    assert req.full_url == f'https://api.telegram.org/bot{token}/sendMessage'
    assert req.get_header('Content-type') == 'application/json'
    payload = json.loads(req.data.decode('utf-8'))
    assert payload['chat_id'] == CHAT_ID
    assert payload['message_thread_id'] == 42
    # The posted text opens with the handoff prefix and carries excerpts.
    assert payload['text'].startswith('Daily handoff for topic 42 (')
    assert 'live excerpt question' in payload['text']
    assert 'live excerpt answer' in payload['text']
    # Live mode also writes the handoff files before posting.
    assert (day_dir(tmp_path) / '42-topic-42.md').is_file()
    assert (latest_dir(tmp_path) / '42.md').is_file()

#!/usr/bin/env python3
"""Nightly Telegram forum-topic handoffs (generalized, config-by-environment).

For every forum topic in a Telegram supergroup, this script reads the topic's
real session history from the agent's state database (read-only), writes a
markdown handoff file, and posts a visible handoff message into the topic so
the agent (and its human) can pick up context after a restart.

PRIVACY
-------
Handoff files contain private conversation text: verbatim excerpts of user
and assistant messages stored in the local state database. Treat the output
directory as sensitive data:

- Files are written to local disk and accumulate forever (nothing is deleted).
- If HANDOFF_MIRROR_DIR / HANDOFF_PUBLIC_BASE_URL are set, copies may be
  served over a network reachable by anyone who can reach that host.
- Posted excerpts are visible to every member of the Telegram group.

A best-effort redaction pass drops lines matching common secret shapes (API
key patterns, bearer tokens, private-key blocks, .env-style KEY=value lines)
BEFORE anything is written to disk or posted. Redaction is a safety net, NOT
a guarantee: review retention, file permissions, and group membership
accordingly. Do not point HANDOFF_PUBLIC_BASE_URL at a host you do not
control, and do not commit handoff output to a repository.

Design rules:
- Read-only access to the state database. This script never writes to it.
- Handoff content comes only from real stored messages. Nothing is invented.
- One bad topic never kills the run; failures are isolated and reported.

Expected database schema (subset):
    sessions(id, chat_id, thread_id, title, last_activity_at)
    messages(session_id, role, content, timestamp)

Configuration (environment variables; every value has a sensible default):
    HANDOFF_CHAT_ID          Supergroup id (e.g. -100...). Empty = all chats.
    HANDOFF_DB               Path to state.db            [~/.hermes/state.db]
    HANDOFF_OUT_DIR          Handoff output dir          [~/.hermes/topic-handoffs]
    HANDOFF_MIRROR_DIR       Optional second output dir (e.g. web-served copy)
    HANDOFF_PUBLIC_BASE_URL  Optional public base URL for the mirror; posts
                             link here, never to a local file path
    HANDOFF_USER_LABEL       Label for user messages            [User]
    HANDOFF_BOT_LABEL        Label for assistant messages       [Assistant]
    HANDOFF_PREFIX           Prefix of handoff posts; also used to filter
                             this script's own earlier posts    [Daily handoff for]
    HANDOFF_TOPIC_NAMES      Path to a JSON object mapping thread ids to
                             names; missing/invalid file falls back to
                             'topic <id>'
    TELEGRAM_BOT_TOKEN       Bot token (or HANDOFF_BOT_TOKEN, or via
                             HANDOFF_ENV_FILE). No token = file-only mode.
    HANDOFF_ENV_FILE         .env fallback for the token        [~/.hermes/.env]
    HANDOFF_API_BASE         Telegram API base URL     [https://api.telegram.org]
    HANDOFF_RECENT_LIMIT     Messages pulled per topic          [12]
    HANDOFF_POST_EXCERPTS    Excerpt lines in the posted message [6]
    HANDOFF_FILE_TRUNC       Per-message truncation in files    [400]
    HANDOFF_POST_TRUNC       Per-message truncation in posts    [250]
    HANDOFF_MAX_POST_LEN     Posted message length cap          [3800]
    HANDOFF_DB_TIMEOUT       SQLite busy timeout, seconds       [30]
    HANDOFF_SEND_DELAY       Pause between posts, seconds       [1.2]
    HANDOFF_MAX_RETRIES      Send retries after first attempt   [4]
    HANDOFF_BACKOFF_BASE     Backoff multiplier                 [2.0]
    DRY_RUN                  If truthy, write files but skip posting

Exit status: 0 on success; 1 if any topic failed or a fatal error occurred.
"""

import datetime
import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def _env(name, default=''):
    value = os.environ.get(name)
    return value if value not in (None, '') else default


def _env_int(name, default):
    try:
        return int(os.environ.get(name, '') or default)
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    try:
        return float(os.environ.get(name, '') or default)
    except (TypeError, ValueError):
        return default


def _env_flag(name):
    return (os.environ.get(name) or '').strip().lower() in ('1', 'true', 'yes', 'on')


CHAT_ID = _env('HANDOFF_CHAT_ID')
DB_PATH = os.path.expanduser(_env('HANDOFF_DB', '~/.hermes/state.db'))
OUT_DIR = os.path.expanduser(_env('HANDOFF_OUT_DIR', '~/.hermes/topic-handoffs'))
MIRROR_DIR = _env('HANDOFF_MIRROR_DIR')
if MIRROR_DIR:
    MIRROR_DIR = os.path.expanduser(MIRROR_DIR)
PUBLIC_BASE_URL = _env('HANDOFF_PUBLIC_BASE_URL').rstrip('/')
USER_LABEL = _env('HANDOFF_USER_LABEL', 'User')
BOT_LABEL = _env('HANDOFF_BOT_LABEL', 'Assistant')
HANDOFF_PREFIX = _env('HANDOFF_PREFIX', 'Daily handoff for')
TOPIC_NAMES_PATH = _env('HANDOFF_TOPIC_NAMES')
if TOPIC_NAMES_PATH:
    TOPIC_NAMES_PATH = os.path.expanduser(TOPIC_NAMES_PATH)
API_BASE = _env('HANDOFF_API_BASE', 'https://api.telegram.org').rstrip('/')
ENV_FILE = os.path.expanduser(_env('HANDOFF_ENV_FILE', '~/.hermes/.env'))

RECENT_LIMIT = _env_int('HANDOFF_RECENT_LIMIT', 12)
POST_EXCERPTS = _env_int('HANDOFF_POST_EXCERPTS', 6)
FILE_TRUNC = _env_int('HANDOFF_FILE_TRUNC', 400)
POST_TRUNC = _env_int('HANDOFF_POST_TRUNC', 250)
MAX_POST_LEN = _env_int('HANDOFF_MAX_POST_LEN', 3800)
DB_TIMEOUT = _env_float('HANDOFF_DB_TIMEOUT', 30)
SEND_DELAY = _env_float('HANDOFF_SEND_DELAY', 1.2)
MAX_RETRIES = _env_int('HANDOFF_MAX_RETRIES', 4)
BACKOFF_BASE = _env_float('HANDOFF_BACKOFF_BASE', 2.0)
DRY_RUN = _env_flag('DRY_RUN')


# --------------------------------------------------------------------------
# Secret redaction
# --------------------------------------------------------------------------

# Lines matching any of these patterns are dropped before files are written
# or messages are posted. This is best-effort defense in depth, not a
# substitute for keeping secrets out of conversation history.
PEM_BEGIN = re.compile(r'-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----')
PEM_END = re.compile(r'-----END [A-Z0-9 ]*PRIVATE KEY-----')

SECRET_PATTERNS = [
    # Bearer tokens (Authorization-style)
    re.compile(r'(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}'),
    # OpenAI / Anthropic style keys
    re.compile(r'\bsk-(?:ant-)?[A-Za-z0-9_-]{16,}\b'),
    # GitHub tokens (old and fine-grained formats)
    re.compile(r'\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b'),
    re.compile(r'\bgithub_pat_[A-Za-z0-9_]{20,}\b'),
    # Slack tokens
    re.compile(r'\bxox[baprs]-[A-Za-z0-9-]{10,}\b'),
    # AWS access key id
    re.compile(r'\bAKIA[0-9A-Z]{16}\b'),
    # Google API key
    re.compile(r'\bAIza[0-9A-Za-z_-]{35}\b'),
    # Telegram bot token (123456789:AA...)
    re.compile(r'\b\d{8,10}:[A-Za-z0-9_-]{30,}\b'),
    # JSON Web Tokens (three base64url segments, header starts with eyJ)
    re.compile(r'\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b'),
    # .env-style KEY=value lines (whole line, optionally quoted/exported)
    re.compile(r"^\s*(?:export\s+)?[A-Za-z_][A-Za-z0-9_]*\s*=\s*"
               r"(\"[^\"]*\"|'[^']*'|\S{6,})\s*$"),
    # Named secrets anywhere in a line: "password: ...", "api_key = ...", etc.
    re.compile(r'(?i)\b(?:password|passwd|secret|api[ _-]?key|access[ _-]?token|'
               r'auth[ _-]?token|client[ _-]?secret)\b\s*[:=]\s*\S+'),
]


def redact(text):
    """Drop lines that match common secret shapes.

    Returns (clean_text, dropped_count). A PEM private-key block drops every
    line from BEGIN through END inclusive. Unterminated blocks drop through
    the end of the text.
    """
    kept = []
    dropped = 0
    in_private_key = False
    for line in text.splitlines():
        if in_private_key:
            dropped += 1
            if PEM_END.search(line):
                in_private_key = False
            continue
        if PEM_BEGIN.search(line):
            in_private_key = True
            dropped += 1
            continue
        if any(pattern.search(line) for pattern in SECRET_PATTERNS):
            dropped += 1
            continue
        kept.append(line)
    return '\n'.join(kept), dropped


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def load_bot_token():
    """Token from the environment, then HANDOFF_ENV_FILE. '' = file-only mode."""
    token = os.environ.get('TELEGRAM_BOT_TOKEN') or os.environ.get('HANDOFF_BOT_TOKEN') or ''
    if token:
        return token.strip()
    if os.path.exists(ENV_FILE):
        try:
            with open(ENV_FILE, encoding='utf-8') as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    key, value = line.split('=', 1)
                    if key.strip() in ('TELEGRAM_BOT_TOKEN', 'HANDOFF_BOT_TOKEN'):
                        return value.strip().strip('"').strip("'")
        except OSError:
            pass
    return ''


def load_topic_names(path):
    """Optional JSON mapping of thread id -> display name. Always degrades."""
    if not path:
        return {}
    try:
        with open(path, encoding='utf-8') as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
        print(f'NOTE: topic names file {path} is not a JSON object; '
              f'falling back to "topic <id>"')
    except (OSError, ValueError) as e:
        print(f'NOTE: could not load topic names from {path}: {e}; '
              f'falling back to "topic <id>"')
    return {}


def truncate_words(text, limit):
    """Collapse whitespace, then truncate on a word boundary with an ellipsis."""
    text = re.sub(r'\s+', ' ', text or '').strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(' ', 1)[0]
    return (cut if cut else text[:limit]) + '...'


def slugify(name):
    slug = re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')
    return slug[:40] or 'topic'


def send_message(token, chat_id, thread_id, text):
    """Send one message, retrying transient failures with exponential backoff.

    HTTP 429 responses honor Telegram's retry_after parameter. Raises
    RuntimeError when the send cannot be completed.
    """
    url = f'{API_BASE}/bot{token}/sendMessage'
    chat = str(chat_id)
    payload = {
        'chat_id': int(chat) if re.fullmatch(r'-?\d+', chat) else chat,
        'message_thread_id': int(thread_id),
        'text': text,
    }
    data = json.dumps(payload).encode('utf-8')
    delay = 1.0
    last_error = 'unknown error'
    for attempt in range(MAX_RETRIES + 1):
        req = urllib.request.Request(
            url, data=data, headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.load(resp)
            if result.get('ok'):
                return
            last_error = f"telegram not-ok: {result.get('description', 'no description')}"
            retry_after = (result.get('parameters') or {}).get('retry_after')
            if retry_after is None:
                raise RuntimeError(last_error)  # non-retryable API rejection
            wait = max(float(retry_after), delay)
        except urllib.error.HTTPError as e:
            description = ''
            retry_after = None
            try:
                err_body = json.loads(e.read() or b'{}')
                description = err_body.get('description', '')
                retry_after = (err_body.get('parameters') or {}).get('retry_after')
            except (ValueError, UnicodeDecodeError, AttributeError):
                pass
            last_error = f'HTTP {e.code}: {description or e.reason}'
            if e.code == 429:
                wait = max(float(retry_after or 0), delay)
            elif 500 <= e.code < 600:
                wait = delay
            else:
                raise RuntimeError(last_error)  # 4xx other than 429: do not retry
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_error = f'network error: {e}'
            wait = delay
        if attempt >= MAX_RETRIES:
            break
        time.sleep(wait)
        delay *= BACKOFF_BASE
    raise RuntimeError(f'giving up after {MAX_RETRIES + 1} attempt(s): {last_error}')


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    today = datetime.date.today().isoformat()
    day_dir = os.path.join(OUT_DIR, today)
    latest_dir = os.path.join(OUT_DIR, 'latest')
    os.makedirs(day_dir, exist_ok=True)
    os.makedirs(latest_dir, exist_ok=True)
    mirror_day_dir = ''
    if MIRROR_DIR:
        mirror_day_dir = os.path.join(MIRROR_DIR, today)
        os.makedirs(mirror_day_dir, exist_ok=True)

    token = load_bot_token()
    names = load_topic_names(TOPIC_NAMES_PATH)

    # Read-only connection with an explicit busy timeout. Never writes.
    uri = 'file:' + urllib.request.pathname2url(DB_PATH) + '?mode=ro'
    con = sqlite3.connect(uri, uri=True, timeout=DB_TIMEOUT)
    cur = con.cursor()
    cur.execute(f'PRAGMA busy_timeout = {int(DB_TIMEOUT * 1000)}')
    journal = cur.execute('PRAGMA journal_mode').fetchone()[0]
    if str(journal).lower() != 'wal':
        print(f'NOTE: database journal_mode is {journal}, expected wal')

    if CHAT_ID:
        threads = cur.execute(
            'SELECT DISTINCT chat_id, thread_id FROM sessions '
            'WHERE chat_id=? AND thread_id IS NOT NULL', (CHAT_ID,)).fetchall()
    else:
        threads = cur.execute(
            'SELECT DISTINCT chat_id, thread_id FROM sessions '
            'WHERE thread_id IS NOT NULL').fetchall()
    if not threads:
        print('NOTE: no forum topics found in the database')

    posted, failed, skipped = [], [], []
    extracted = 0
    redacted_total = 0

    for chat_id, tid in threads:
        name = names.get(str(tid), f'topic {tid}')
        # Per-topic error isolation: one bad row must not kill the run.
        try:
            if chat_id is not None:
                sessions = cur.execute(
                    'SELECT id, title, last_activity_at FROM sessions '
                    'WHERE chat_id=? AND thread_id=? '
                    'ORDER BY last_activity_at DESC', (chat_id, tid)).fetchall()
            else:
                sessions = cur.execute(
                    'SELECT id, title, last_activity_at FROM sessions '
                    'WHERE thread_id=? '
                    'ORDER BY last_activity_at DESC', (tid,)).fetchall()
            if not sessions:
                continue
            sess_id, title, last_act = sessions[0]
            try:
                last_date = (datetime.datetime.fromtimestamp(last_act)
                             .strftime('%Y-%m-%d %H:%M')) if last_act else 'unknown'
            except (OverflowError, OSError, ValueError):
                last_date = 'unknown'

            msgs = cur.execute(
                'SELECT role, content FROM messages WHERE session_id=? '
                "AND role IN ('user','assistant') "
                "AND content IS NOT NULL AND content != '' "
                'ORDER BY timestamp DESC LIMIT ?', (sess_id, RECENT_LIMIT)).fetchall()
            msgs.reverse()
            # Self-post filtering: never quote this script's earlier handoffs.
            if HANDOFF_PREFIX:
                msgs = [(r, c) for r, c in msgs
                        if not (c or '').lstrip().startswith(HANDOFF_PREFIX)]

            lines = [
                f'# Handoff: {name} (topic {tid})',
                f'Date: {today}',
                '',
                f"Sessions in this topic: {len(sessions)}. "
                f"Latest session title: {title or 'untitled'}.",
                f'Latest activity: {last_date}.',
                '',
                '## Recent conversation, in order, real stored words:',
            ]
            for role, content in msgs:
                text = truncate_words(content, FILE_TRUNC)
                if not text:
                    continue
                who = USER_LABEL if role == 'user' else BOT_LABEL
                lines.append(f'- {who}: {text}')
            if not msgs:
                lines.append('- No recent messages in the latest session.')

            # Redaction pass runs BEFORE anything is written to disk.
            body, dropped = redact('\n'.join(lines))
            redacted_total += dropped
            if dropped:
                body += f'\n\n_Note: {dropped} line(s) omitted by the ' \
                        f'secret-redaction pass._'

            stem = str(tid) if CHAT_ID else f'{chat_id}-{tid}'
            slug = slugify(name)
            with open(os.path.join(day_dir, f'{stem}-{slug}.md'), 'w',
                      encoding='utf-8') as fh:
                fh.write(body + '\n')
            with open(os.path.join(latest_dir, f'{stem}.md'), 'w',
                      encoding='utf-8') as fh:
                fh.write(body + '\n')
            if mirror_day_dir:
                with open(os.path.join(mirror_day_dir, f'{stem}.md'), 'w',
                          encoding='utf-8') as fh:
                    fh.write(body + '\n')
            extracted += 1

            if DRY_RUN:
                skipped.append((str(tid), name, 'dry run'))
                continue
            if not token:
                # File-only mode is a deliberate configuration, not a failure.
                skipped.append((str(tid), name, 'file-only mode: no bot token'))
                continue
            if chat_id in (None, ''):
                skipped.append((str(tid), name, 'file-only mode: no chat id'))
                continue

            excerpt_lines = []
            for role, content in msgs[-POST_EXCERPTS:]:
                text = truncate_words(content, POST_TRUNC)
                if text:
                    who = USER_LABEL if role == 'user' else BOT_LABEL
                    excerpt_lines.append(f'{who}: {text}')

            header = (f'{HANDOFF_PREFIX} {name} ({today}).\n'
                      f'Sessions so far: {len(sessions)}. '
                      f'Last activity: {last_date}.')
            # Only ever post a public URL, never a local file path.
            footer = (f'Full handoff: {PUBLIC_BASE_URL}/{today}/{stem}.md'
                      if PUBLIC_BASE_URL else '')

            while True:
                parts = [header]
                if excerpt_lines:
                    parts.append('Recent words from this topic:\n'
                                 + '\n'.join(excerpt_lines))
                if footer:
                    parts.append(footer)
                post = '\n\n'.join(parts)
                if len(post) <= MAX_POST_LEN or not excerpt_lines:
                    break
                excerpt_lines.pop(0)  # drop oldest excerpt until it fits

            # Redaction pass runs BEFORE anything is posted.
            post, dropped = redact(post)
            redacted_total += dropped
            if dropped:
                post += f'\n({dropped} line(s) omitted by the ' \
                        f'secret-redaction pass.)'
            if len(post) > MAX_POST_LEN:
                post = post[:MAX_POST_LEN]

            send_message(token, chat_id, tid, post)
            posted.append((str(tid), name, 'posted'))
            time.sleep(SEND_DELAY)
        except Exception as e:
            failed.append((str(tid), name, f'error: {e}'))

    con.close()

    if DRY_RUN:
        print(f'DRY RUN: {extracted} topic(s) extracted and filed, posting skipped')
    else:
        print(f'POSTED: {len(posted)} of {extracted} topic(s)')
    if skipped and not DRY_RUN:
        print(f'SKIPPED: {len(skipped)}')
        for tid, name, why in skipped:
            print(f'- topic {tid} {name}: {why}')
    if redacted_total:
        print(f'REDACTED: {redacted_total} line(s) omitted as possible secrets')
    locations = f'{day_dir} and {latest_dir}'
    if mirror_day_dir:
        locations += f' and {mirror_day_dir}'
    print(f'Files written under {locations}')
    if failed:
        print(f'FAILURES: {len(failed)}')
        for tid, name, status in failed:
            print(f'- topic {tid} {name}: {status}')
    return 1 if failed else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except sqlite3.OperationalError as e:
        print(f'DATABASE UNAVAILABLE: {e}', file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print('INTERRUPTED', file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f'TOPIC HANDOFF RUN FAILED: {e}', file=sys.stderr)
        sys.exit(1)

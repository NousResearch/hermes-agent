"""Canonical local webhook Bot Chat targets and conservative store-incarnation pins."""
from pathlib import Path
import sqlite3

from hermes_state_runtime import RuntimeStoreError


def is_bot_chat(target):
    return isinstance(target, str) and (target == 'bot-chat' or target.startswith('bot-chat:'))


def target_profile(target):
    """None means the effective producer, never the receiving adapter's launch profile."""
    from hermes_constants import PROFILE_ID_RE
    if target == 'bot-chat':
        return None
    if not is_bot_chat(target):
        raise ValueError('not a bot-chat target')
    profile = target[len('bot-chat:'):]
    if not PROFILE_ID_RE.fullmatch(profile):
        raise ValueError('bot-chat requires a canonical local profile name')
    return profile


def target_home(target, producer_home):
    from hermes_cli.profiles import get_profile_dir, profile_exists
    profile = target_profile(target)
    if profile is None:
        return Path(producer_home).resolve()
    path = get_profile_dir(profile)
    if not profile_exists(profile) or path.is_symlink() or not path.is_dir():
        raise ValueError('bot-chat target profile unavailable')
    return path.resolve()


def file_identity(path):
    stat = Path(path).stat()
    if not stat.st_ino:
        raise ValueError('store continuity unavailable')
    return [stat.st_dev, stat.st_ino]


def capture_target(home):
    """Never creates or migrates a recipient DB, even for a profile's first delivery."""
    home = Path(home).resolve()
    directory = file_identity(home)
    path = home / 'state.db'
    store = None
    if path.exists():
        identity = file_identity(path)
        with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as conn:
            row = conn.execute("SELECT value FROM state_meta WHERE key='store_instance_id'").fetchone()
        if identity != file_identity(path) or directory != file_identity(home):
            raise ValueError('target incarnation changed')
        store = {'file': identity, 'instance': row[0] if row else None}
    return {'home': str(home), 'directory': directory, 'store': store}


def refresh_target(pin):
    current = capture_target(pin['home'])
    if current['directory'] != pin['directory'] or (pin['store'] is not None and current['store'] != pin['store']):
        raise ValueError('target incarnation changed')
    return current  # a missing store can freeze once, under the same directory


def require_incarnation(authority, pin, conn=None):
    """Recipient-side check of its OPEN writer, not a freshly reopened path alone.

    Checked inside the mailbox operation before target lookup and again before admission.
    Old callers without an expected incarnation retain their native contract.
    """
    if pin is None:
        return
    home = Path(authority.db.db_path).resolve().parent
    try:
        if (not isinstance(pin, dict) or set(pin) != {'home', 'directory', 'store'}
                or pin['home'] != str(home) or pin['store'] is None):
            raise ValueError('invalid target pin')
        current = refresh_target(pin)
        db = authority.db
        opened = getattr(db, '_db_file_identity', None)
        # The admission transaction passes its own connection; an early mailbox check
        # takes the writer lock rather than borrowing a newly opened read-pool handle.
        if conn is None:
            with db._lock:
                row = db._conn.execute("SELECT value FROM state_meta WHERE key='store_instance_id'").fetchone()
        else:
            row = conn.execute("SELECT value FROM state_meta WHERE key='store_instance_id'").fetchone()
        instance = row[0] if row else None
        if opened is None or list(opened) != current['store']['file'] or instance != current['store']['instance']:
            raise ValueError('open store differs from target')
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error):
        raise RuntimeStoreError('admission_conflict') from None

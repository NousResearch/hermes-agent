"""Opt-in conversation browser lease. No credentials or CDP URLs are persisted."""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from datetime import datetime

from hermes_constants import get_hermes_home


def settings():
    from hermes_cli.config import load_config
    return (load_config().get('browser') or {}).get('browser_use') or {}


def owner(task_id):
    """Bind continuity to a runtime conversation, not to a secret-collection policy."""
    from gateway.session_context import get_session_env as get
    platform = get('HERMES_SESSION_PLATFORM')
    chat, user = get('HERMES_SESSION_CHAT_ID'), get('HERMES_SESSION_USER_ID')
    key, thread = get('HERMES_SESSION_KEY'), get('HERMES_SESSION_THREAD_ID')
    if get('HERMES_CRON_SESSION'):
        if not task_id or not task_id.startswith('cron:'):
            raise RuntimeError('Persistent cron browser requires its execution task ID')
        value = ['cron', task_id]
    elif platform == 'telegram':
        if not chat or not user:
            raise RuntimeError('Persistent Telegram browser requires an authenticated conversation')
        if get('HERMES_SESSION_CHAT_TYPE') in ('dm', 'private'):
            if chat != user:
                raise RuntimeError('Private Telegram owner does not match its chat')
            # Preserve existing private leases across this upgrade.
            value = [chat, user, thread, key]
        elif get('HERMES_SESSION_CHAT_TYPE') in ('group', 'supergroup'):
            value = ['telegram-group', chat, thread, key]
        else:
            raise RuntimeError('Unsupported Telegram conversation type')
    else:
        if not task_id or task_id == 'default':
            raise RuntimeError('Persistent browser requires a runtime task ID')
        value = [platform or 'local', chat, thread, key, task_id]
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


class Lease:
    """The OS lease lives as long as the browser is tracked by Hermes' idle reaper."""
    def __init__(self, profile_id, owner_id):
        # The production target is Linux; fail explicitly rather than omit exclusion elsewhere.
        import fcntl
        root = get_hermes_home() / 'browser' / 'cloud'
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        name = hashlib.sha256(profile_id.encode()).hexdigest()
        self.path = root / (name + '.json')
        self.fd = os.open(root / (name + '.lock'), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self.fd)
            self.fd = None
            raise RuntimeError('Browser profile is already in use') from None
        self.profile_id, self.owner_id = profile_id, owner_id
        self.record = {}
        try:
            if self.path.exists():
                self.record = json.loads(self.path.read_text())
        except Exception:
            self.release()
            raise

    def write(self):
        tmp = self.path.with_name(self.path.name + '.' + uuid.uuid4().hex)
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(self.record, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        finally:
            tmp.unlink(missing_ok=True)

    def acquire(self, provider, config):
        import requests
        headers = provider._headers(config)
        base = config['base_url']
        old = self.record
        reconciling = old.get('status') == 'creating'
        if reconciling and not config.get('tenant_gateway'):
            raise RuntimeError('Browser creation outcome unknown; reconcile the recorded operation before retrying')
        if old and not reconciling:
            response = requests.get(base + '/browsers/' + old['browser_id'], headers=headers, timeout=15)
            if response.status_code == 404:
                current = {'status': 'stopped'}
            else:
                response.raise_for_status()
                current = response.json()
            if current.get('status') == 'active':
                if old.get('owner') != self.owner_id:
                    raise RuntimeError('Browser profile belongs to another pending conversation')
                expiry = datetime.fromisoformat(current['timeoutAt'].replace('Z', '+00:00')).timestamp()
                if time.time() < expiry and time.time() - old['last_used_at'] < 600:
                    return current
                stopped = provider._release(config, old['browser_id'], 15)
                stopped.raise_for_status()
                check = requests.get(base + '/browsers/' + old['browser_id'], headers=headers, timeout=15)
                check.raise_for_status()
                if check.json().get('status') != 'stopped':
                    raise RuntimeError('Previous browser is not confirmed stopped')
            elif current.get('status') != 'stopped':
                raise RuntimeError('Browser status unknown; no replacement was created')
        if reconciling and old.get('owner') != self.owner_id:
            raise RuntimeError('Browser creation belongs to another pending conversation')
        if not reconciling:
            self.record = {'status': 'creating', 'profile_id': self.profile_id, 'owner': self.owner_id,
                           'operation_id': uuid.uuid4().hex, 'last_used_at': time.time()}
            self.write()
        response = provider._post_create(base + '/browsers', headers,
            {'profileId': self.profile_id, 'timeout': 30, 'solveCaptchas': True,
             'enableRecording': False, 'metadata': {'hermes_operation': self.record['operation_id']}})
        if 400 <= response.status_code < 500 and response.status_code not in (408, 409, 429):
            self.path.unlink(missing_ok=True)
        provider._check_created(response)
        current = response.json()
        if not current.get('id') or not (current.get('cdpUrl') or current.get('connectUrl')):
            raise RuntimeError('Browser creation response incomplete; reconcile before retrying')
        self.record.update(status='active', browser_id=current['id'], timeout_at=current['timeoutAt'])
        self.write()
        return current

    def checkpoint(self):
        self.record['last_used_at'] = time.time()
        self.write()

    def release(self, *, stopped=False):
        if stopped:
            self.path.unlink(missing_ok=True)
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

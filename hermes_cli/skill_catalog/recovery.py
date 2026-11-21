"""Writer-local recovery decisions; no provider callback or lock crosses RPC."""
import hashlib
import json
import os
import stat
import time
from contextlib import contextmanager

from hermes_cli.skill_catalog.locks import writer_hold
from hermes_cli.skill_catalog.projection import _target
from hermes_cli.skill_catalog.validation import CatalogError
from hermes_cli.skill_catalog.writer import Commit


def _changed():
    return CatalogError('review_changed', 'Catalog recovery evidence changed. Review again.')


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def observe(target):
    """SHA-256 of canonical relative regular-file hashes, or null for absence.

    Hash facts include executable intent. Redirects, special files, racing reads
    and over-budget trees refuse, never produce a partial digest. This explicit
    recovery observation does not infer ownership or run during ordinary reads.
    """
    if not target.exists():
        return None
    deadline = time.monotonic() + 3
    facts = {}
    for root, directories, files in os.walk(target, followlinks=False):
        for name in directories + files:
            path = target.__class__(root) / name
            before = path.lstat()
            if stat.S_ISLNK(before.st_mode) or getattr(before, 'st_file_attributes', 0) & 0x400:
                raise _changed()
            if stat.S_ISDIR(before.st_mode):
                continue
            if not stat.S_ISREG(before.st_mode) or len(facts) >= 2000:
                raise _changed()
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(65536), b''):
                    if time.monotonic() >= deadline:
                        raise _changed()
                    digest.update(chunk)
                if _identity(os.fstat(stream.fileno())) != _identity(before):
                    raise _changed()
            if _identity(path.lstat()) != _identity(before):
                raise _changed()
            facts[path.relative_to(target).as_posix()] = (
                ('exec:' if before.st_mode & 0o111 else '') + 'sha256:' + digest.hexdigest())
    if not target.is_dir() or time.monotonic() >= deadline:
        raise _changed()
    data = json.dumps(facts, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


@contextmanager
def recover(facade, home, operation_id, acquired):
    """Yield current local facts only after the original binding is validated."""
    try:
        with writer_hold(home, [acquired['target_relative_path']]):
            facade.validate(dict(permit_id=acquired['permit_id']))
            target = _target(home, acquired['target_relative_path'])
            recovery = Recovery(facade, acquired, operation_id, observe(target))
            yield recovery
            if recovery.decision is None:
                raise _changed()
    finally:
        facade.release(dict(permit_id=acquired['permit_id']))


class Recovery(Commit):
    """One decision under one hold. Provider evidence never supplies a target."""

    def __init__(self, facade, acquired, operation_id, observed):
        super().__init__(facade, acquired, acquired['admission'], operation_id)
        self.observed_revision = observed
        # The installation revision is provider-opaque. Only presence is a core
        # precondition; providers check their journal's complete tree/identity.
        self.pre_revision_matches = ((observed is not None) ==
                                     (self.expected_installation_revision is not None))
        self.decision = None

    def _fail(self, reason):
        result = dict(operation_id=self._operation_id, identifier=self._acquired['identifier'],
            action=self._acquired['action'], state='failed', outcome='refused',
            managed_state='unmanaged_local', message=reason, installation_revision=None)
        return super().settle(result)

    def decide(self, answer):
        if self.decision is not None or type(answer) is not dict:
            raise _changed()
        kind = answer.get('kind')
        allowed = {'reconciled': {'kind', 'journal_digest', 'result'},
                   'retry_effect': {'kind'}, 'abandon': {'kind', 'reason'}}
        if type(kind) is not str or kind not in allowed or set(answer) != allowed[kind]:
            raise _changed()
        if self._acquired.get('local_result') is not None:
            if kind != 'reconciled' or answer['result'] != self._acquired['local_result']:
                raise _changed()
        if kind == 'reconciled':
            if answer['journal_digest'] != self.observed_revision:
                self._fail('Catalog recovery digest differs. Local files were preserved.')
                self.decision = 'abandon'
                raise _changed()
            if type(answer['result']) is not dict or answer['result'].get('state') != 'pending':
                raise _changed()
            result = super().settle(answer['result'])
        elif kind == 'abandon':
            if type(answer['reason']) is not str or not 1 <= len(answer['reason']) <= 2048:
                raise _changed()
            result = self._fail(answer['reason'])
        else:
            result = None
        self.decision = kind
        return result

    def settle(self, result):
        if self.decision != 'retry_effect':
            raise _changed()
        return super().settle(result)

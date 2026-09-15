"""Read-only Graph date-range scanner for Himalaya CLI 2.1.0. Python 3.9+.

Never invents offsets or follows a URL with credentials. Nonterminal ranges are
bisected until each leaf is terminal; dense/failed/stopped work stays incomplete.
The callable collector accepts a fake fetch adapter for offline regression tests.
"""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

FIELDS = ('id', 'parentFolderId', 'subject', 'from', 'receivedDateTime', 'isRead',
          'isDraft', 'hasAttachments', 'internetMessageId', 'importance', 'flag', 'categories')


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError('Expected ISO timestamp string')
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Timestamp needs an explicit timezone')
    return result.astimezone(timezone.utc)


def iso(value):
    return value.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


@dataclass(frozen=True)
class Scope:
    account: str
    folder: str
    top: int = 200
    config: str = ''
    extra_filter: str = ''

    def __post_init__(self):
        for field in ('account', 'folder'):
            if not isinstance(getattr(self, field), str) or not getattr(self, field).strip():
                raise ValueError('An explicit account and folder ID/well-known name are required')
        if type(self.top) is not int or not 1 <= self.top <= 1000:
            raise ValueError('top must be 1..1000')
        if not isinstance(self.config, str) or not isinstance(self.extra_filter, str):
            raise ValueError('Config and extra filter must be strings')


def list_argv(scope, start, end):
    if start >= end:
        raise ValueError('Range must be increasing')
    predicate = f'receivedDateTime ge {iso(start)} and receivedDateTime lt {iso(end)}'
    if scope.extra_filter:
        predicate += f' and ({scope.extra_filter})'
    args = ['himalaya']
    if scope.config:
        args += ['--config', scope.config]
    return args + ['--account', scope.account, '--backend', 'msgraph', '--json',
                   'msgraph', 'message', 'list', '--folder', scope.folder,
                   '--top', str(scope.top), '--select', ','.join(FIELDS),
                   '--orderby', 'receivedDateTime asc', '--filter', predicate]


def read_argv(scope, message_id):
    if not isinstance(message_id, str) or not message_id:
        raise ValueError('Use the unchanged nonempty ID from parsed JSON')
    args = ['himalaya']
    if scope.config:
        args += ['--config', scope.config]
    return args + ['--account', scope.account, '--backend', 'msgraph',
                   'message', 'read', '--mailbox', scope.folder, '--raw', '--', message_id]


def supported_version(version):
    """Only the source-verified released CLI, not similarly named dev versions."""
    return (isinstance(version, str) and version.split()[:2] in
            (['himalaya', 'v2.1.0'], ['himalaya', '2.1.0']))


def snapshot_hash(message):
    data = json.dumps(message, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def _metadata_evidence(scope, lower, upper, message, cli_version):
    # Captured by the collector after a successful parsed terminal response.
    # This records a CLI projection, not original HTTP response bytes.
    return {'schema_version': 1, 'source': 'himalaya-native-msgraph-list',
            'cli_version': cli_version, 'account': scope.account,
            'scope': asdict(scope), 'start': iso(lower), 'end_exclusive': iso(upper),
            'argv': list_argv(scope, lower, upper), 'snapshot_sha256': snapshot_hash(message),
            'observed_at': iso(datetime.now(timezone.utc))}


def _selected_list_evidence_matches(message, evidence, account):
    if not isinstance(evidence, dict) or not isinstance(account, str) or not account:
        return False
    try:
        scope = Scope(**evidence['scope'])
        lower, upper = timestamp(evidence['start']), timestamp(evidence['end_exclusive'])
        timestamp(evidence['observed_at'])
        return (evidence.get('schema_version') == 1
                and evidence.get('source') == 'himalaya-native-msgraph-list'
                and supported_version(evidence.get('cli_version'))
                and account == evidence.get('account') == scope.account
                and evidence.get('argv') == list_argv(scope, lower, upper)
                and isinstance(message.get('id'), str) and bool(message['id'])
                and isinstance(message.get('parentFolderId'), str) and bool(message['parentFolderId'])
                and lower <= timestamp(message['receivedDateTime']) < upper
                and evidence.get('snapshot_sha256') == snapshot_hash(message))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def category_check(message, metadata_evidence=None, account=None):
    """Separate interpretation; never add categories to the original snapshot.

    Omission is accepted only under the selected-field CLI contract. The hash
    binds evidence to a snapshot; it is not proof of authenticity or freshness.
    """
    if 'categories' in message:
        value = message['categories']
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return {'state': 'empty' if not value else 'protected', 'basis': 'explicit_array'}
        return {'state': 'unknown', 'basis': 'invalid_categories_value'}
    if _selected_list_evidence_matches(message, metadata_evidence, account):
        return {'state': 'empty', 'basis': 'selected_cli_contract',
                'wire_shape': 'unknown'}
    return {'state': 'unknown', 'basis': 'missing_without_supported_evidence'}


def parse_page(payload):
    if not isinstance(payload, dict) or 'error' in payload:
        raise ValueError('Command returned an error or non-object JSON')
    if not isinstance(payload.get('messages'), list):
        raise ValueError('Expected Himalaya messages array; this is not a Gmail/provider schema')
    if any(key in payload for key in ('@odata.nextLink', 'nextLink', 'next_page_token')):
        raise ValueError('Unexpected continuation schema; inspect installed output')
    continuation = payload.get('next_page')
    if continuation is not None and (not isinstance(continuation, str) or not continuation):
        raise ValueError('next_page must be absent, null or a nonempty string')
    rows = payload['messages']
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id']:
            raise ValueError('Message ID is missing or not a string')
        timestamp(row.get('receivedDateTime'))
    return rows, continuation


def collect(scope, start, end, fetch, stopped=lambda: False, max_requests=200,
            checkpoint=lambda result: None, cli_version=None):
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise ValueError('Use an increasing timezone-aware interval')
    if start.microsecond or end.microsecond:
        raise ValueError('Scan boundaries must be whole seconds')
    if type(max_requests) is not int or max_requests < 1:
        raise ValueError('max_requests must be positive')
    queue = [(start, end)]
    if cli_version is not None and not supported_version(cli_version):
        raise ValueError('Metadata evidence requires the supported CLI 2.1.0 version output')
    result = {'schema_version': 2, 'account': scope.account, 'backend': 'msgraph',
              'folder': scope.folder, 'start': iso(start), 'end_exclusive': iso(end),
              'date_basis': 'receivedDateTime', 'extra_filter': scope.extra_filter,
              'complete': False, 'reason': 'running', 'requests': 0,
              'terminal_intervals': [], 'unresolved_intervals': [], 'messages': [],
              'cli_version': cli_version, 'metadata_evidence': {}}
    by_id = {}

    def save(reason):
        result['reason'] = reason
        result['messages'] = list(by_id.values())
        result['unique_messages'] = len(by_id)
        result['unresolved_intervals'] = [[iso(a), iso(b)] for a, b in queue]
        checkpoint(result)
        return result

    while queue:
        if stopped():
            return save('stopped')
        if result['requests'] >= max_requests:
            return save('request_budget')
        lower, upper = queue[0]
        try:
            result['requests'] += 1
            rows, continuation = parse_page(fetch(scope, lower, upper))
            if any(not lower <= timestamp(row['receivedDateTime']) < upper for row in rows):
                raise ValueError('Returned message lies outside the requested interval')
            if continuation is None:
                pending = dict(by_id)
                pending_evidence = dict(result['metadata_evidence'])
                for row in rows:
                    if row['id'] in pending and pending[row['id']] != row:
                        raise ValueError('Same ID returned with conflicting records; mailbox may have changed')
                    pending[row['id']] = row
                    if cli_version is not None:
                        pending_evidence[row['id']] = _metadata_evidence(scope, lower, upper, row, cli_version)
                by_id = pending
                result['metadata_evidence'] = pending_evidence
                queue.pop(0)
                result['terminal_intervals'].append([iso(lower), iso(upper)])
            elif not stopped():
                seconds = int((upper-lower).total_seconds())
                if seconds <= 1:
                    return save('dense_interval_requires_continuation')
                middle = lower + timedelta(seconds=seconds//2)
                queue[:1] = [(lower, middle), (middle, upper)]
            # Nonterminal parent-page rows are deliberately not counted as a
            # completed subset. They will be rediscovered in terminal leaves.
            if stopped():
                return save('stopped')
            save('running')
        except (ValueError, RuntimeError, OSError) as exc:
            result['error'] = str(exc)
            return save('stopped' if stopped() else 'error')
    result['complete'] = True
    return save('terminal_coverage')


class Runner:
    """Direct native process launch. No shell, token lookup, retries or mutations."""
    def __init__(self, timeout=30, stopped=lambda: False):
        self.timeout, self.stopped = timeout, stopped

    def run(self, args, binary=False):
        if self.stopped():
            raise RuntimeError('Stopped before command launch')
        if self.timeout <= 0:
            raise ValueError('Timeout must be positive')
        with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              shell=False) as process:
            elapsed = 0.0
            try:
                while True:
                    if self.stopped():
                        raise RuntimeError('Stopped during command')
                    try:
                        output, _stderr = process.communicate(timeout=min(.2, self.timeout-elapsed))
                        break
                    except subprocess.TimeoutExpired:
                        elapsed += min(.2, self.timeout-elapsed)
                        if elapsed >= self.timeout:
                            raise RuntimeError('Command timed out; no automatic retry')
            except BaseException:
                process.kill()
                process.communicate()
                raise
        if process.returncode:
            # stderr can contain tokens/message text. Do not echo it automatically.
            raise RuntimeError(f'Command failed with exit {process.returncode}; inspect locally with redaction')
        if binary:
            return output
        try:
            value = json.loads(output)
        except (ValueError, UnicodeError) as exc:
            raise RuntimeError('Command did not return valid JSON') from exc
        return value

    def fetch(self, scope, lower, upper):
        return self.run(list_argv(scope, lower, upper))


def atomic_write(path, result):
    path = Path(path)
    fd, temp = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--account', required=True)
    parser.add_argument('--folder', required=True)
    parser.add_argument('--start', required=True, type=timestamp)
    parser.add_argument('--end', required=True, type=timestamp)
    parser.add_argument('--top', type=int, default=200)
    parser.add_argument('--config', default='')
    parser.add_argument('--extra-filter', default='')
    parser.add_argument('--max-requests', type=int, default=200)
    parser.add_argument('--timeout', type=float, default=30)
    parser.add_argument('--output', required=True, help='New absolute native path; never overwrites an existing file')
    parser.add_argument('--stop-file', help='Optional absolute native cancellation file path')
    args = parser.parse_args()
    out = Path(args.output)
    if not out.is_absolute() or (args.stop_file and not Path(args.stop_file).is_absolute()):
        parser.error('Use absolute paths understood by this Python runtime (Windows: C:/...)')
    if args.timeout <= 0 or args.max_requests < 1:
        parser.error('Timeout and max-requests must be positive')
    scope = Scope(args.account, args.folder, args.top, args.config, args.extra_filter)
    stop_state = [False]
    def stop_handler(*_):
        stop_state[0] = True
    signal.signal(signal.SIGINT, stop_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, stop_handler)
    stopped = lambda: stop_state[0] or bool(args.stop_file and Path(args.stop_file).exists())
    runner = Runner(args.timeout, stopped)
    # Version/help checks are local; only this pinned interface is supported here.
    version = runner.run(['himalaya', '--version'], binary=True).decode('utf-8', 'replace')
    if not supported_version(version):
        parser.error('This scanner is validated for CLI 2.1.0; inspect/adapt other versions first')
    help_text = runner.run(['himalaya', 'msgraph', 'message', 'list', '--help'], binary=True)
    if not all(flag in help_text for flag in (b'--filter', b'--folder', b'--select', b'--orderby')):
        parser.error('Installed command help does not match the scanner')
    # Exclusively reserve this run's path. Each checkpoint replaces only that file.
    with out.open('x', encoding='utf-8') as stream:
        json.dump({'complete': False, 'reason': 'starting'}, stream)
    try:
        result = collect(scope, args.start, args.end, runner.fetch, stopped,
                         args.max_requests, lambda state: atomic_write(out, state), cli_version=version.strip())
    except (ValueError, RuntimeError, OSError) as exc:
        atomic_write(out, {'complete': False, 'reason': 'error', 'error': str(exc)})
        parser.exit(2, 'Scan incomplete; inspect the output file.\n')
    print(json.dumps({k: result[k] for k in ('complete', 'reason', 'requests', 'unique_messages')}))
    raise SystemExit(0 if result['complete'] else 2)


if __name__ == '__main__':
    main()

"""F-4 regression (t_6e177f6d): fetch read-timeout must fail CLEANLY.

_default_transport reads under urllib's 30s timeout; a READ timeout
raises bare TimeoutError/socket.timeout, which the C8 skip clause (built
for HTTPError/URLError) does not catch — pre-fix it escaped main() as an
uncaught traceback (review F-4, petasos/pipeline.py:194-201). Owner
decision D3 scoped the fix narrowly: timeout → ONE '[fatal]' line in the
module's existing error vocabulary + rc 1, nothing written; the skip
path and 5xx behavior stay untouched (do NOT widen the catch).

Transport is deliberately NOT injected here: urlopen is MOCKED so the
real seam — _default_transport → fetch() → main()'s catch — runs with
zero network. socket.timeout is an alias of TimeoutError on ≥3.10 and a
distinct OSError subclass before it; both spellings are pinned.
"""
import contextlib
import io
import json
import os
import socket
import sys
import unittest
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_pipeline import Vault, acb_thread  # noqa: E402
from petasos.pipeline import main  # noqa: E402


class _FakeResponse:
    """urlopen()-shaped context manager yielding one JSON payload."""

    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def run_default_transport(vault):
    """main() with NO injected transport — the real _default_transport
    seam with urlopen mocked (no network). An escaped timeout becomes a
    clean test failure naming the F-4 regression, not a test error."""
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main([], environ=vault.env())
    except (TimeoutError, socket.timeout) as e:
        raise AssertionError(
            f'uncaught timeout escaped main(): {e!r} — F-4 regression')
    return code, out.getvalue(), err.getvalue()


class ReadTimeoutFailsClean(unittest.TestCase):
    """Read-timeout anywhere in the fetch path → one [fatal] line, rc 1,
    no traceback, vault byte-identical (F-4 / D3 contract)."""

    def _assert_clean_failure(self, code, out, err, vault, before):
        text = out + err
        self.assertEqual(code, 1, text)                  # pinned: rc 1
        fatal = [ln for ln in text.splitlines() if '[fatal]' in ln]
        self.assertEqual(len(fatal), 1, text)            # ONE readable line
        self.assertIn('timed out', fatal[0].lower())     # names the cause
        self.assertNotIn('Traceback', text)              # no raw traceback
        self.assertEqual(vault.snapshot(), before)       # touched NOTHING

    def test_threads_list_read_timeout_fails_clean(self):
        v = Vault()
        before = v.snapshot()
        with mock.patch('urllib.request.urlopen',
                        side_effect=TimeoutError('timed out')):
            code, out, err = run_default_transport(v)
        self._assert_clean_failure(code, out, err, v, before)

    def test_thread_detail_read_timeout_fails_clean(self):
        v = Vault()
        before = v.snapshot()
        t = acb_thread('tttt0001', 2000, 500)
        listing = _FakeResponse({'threads': [
            {'thread_id': t['thread_id'], 'subject': t['subject'],
             'timestamp': t['timestamp']}]})
        with mock.patch('urllib.request.urlopen',
                        side_effect=[listing, socket.timeout('timed out')]):
            code, out, err = run_default_transport(v)
        self._assert_clean_failure(code, out, err, v, before)


if __name__ == '__main__':
    unittest.main()

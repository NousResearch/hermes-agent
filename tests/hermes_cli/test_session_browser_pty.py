"""Exercise the classic picker with actual curses reads on a private UTF-8 PTY."""

import json
import os
import select
import subprocess
import sys
import textwrap

import pytest


# Instrument only completed draws; keyboard decoding and rendering stay real.
_CHILD = textwrap.dedent("""
    import fcntl, json, os, sys, termios
    from hermes_cli.sessions_cmd_browse import _CursesBrowser, _session_browse_picker
    fcntl.ioctl(0, termios.TIOCSCTTY, 0)  # this child's private controlling terminal
    report = int(sys.argv[1])
    original_draw = _CursesBrowser._draw
    ready = False
    def draw(self, *args):
        global ready
        original_draw(self, *args)
        if not ready:
            os.write(report, b'READY\\n')
            ready = True
    _CursesBrowser._draw = draw
    sessions = [{'id': f'saved_{i}', 'title': f'Project {i}', 'preview': ''} for i in range(30)]
    sessions[1]['title'] = 'cafeteria'
    sessions[17]['title'] = '界 j/k café Ă 👩‍💻 کار‌ها'
    result = _session_browse_picker(sessions, keyboard_navigation=True)
    os.write(report, (json.dumps({'selected': result}) + '\\n').encode())
    line = sys.stdin.readline()
    os.write(report, (json.dumps({'next_prompt': line}) + '\\n').encode())
""")


def _probe(payload):
    import fcntl
    import pty
    import struct
    import termios

    master, slave = pty.openpty()
    report_r, report_w = os.pipe()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 12, 64, 0, 0))
    child = subprocess.Popen(
        [sys.executable, '-c', _CHILD, str(report_w)],
        stdin=slave, stdout=slave, stderr=slave, pass_fds=(report_w,), start_new_session=True,
        env={**os.environ, 'TERM': 'xterm-256color', 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8'},
    )
    os.close(slave)
    os.close(report_w)
    pending = bytearray()
    terminal = bytearray()

    def receipt():
        import time
        deadline = time.monotonic() + 10
        while b'\n' not in pending:
            remaining = deadline - time.monotonic()
            assert remaining > 0, f'PTY timeout: {terminal.decode(errors="replace")}'
            readers, _, _ = select.select([report_r, master], [], [], remaining)
            for fd in readers:
                try:
                    data = os.read(fd, 65536)
                except OSError:
                    data = b''
                if fd == report_r:
                    assert data, f'PTY child exited: {terminal.decode(errors="replace")}'
                    pending.extend(data)
                else:
                    terminal.extend(data)
        line, _, rest = pending.partition(b'\n')
        pending[:] = rest
        return line

    try:
        assert receipt() == b'READY'
        os.write(master, payload)
        result = json.loads(receipt())
        os.write(master, b'next prompt\n')
        following = json.loads(receipt())
        assert child.wait(timeout=10) == 0
        return result['selected'], following['next_prompt']
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        os.close(master)
        os.close(report_r)


@pytest.mark.platforms('posix')
@pytest.mark.parametrize('payload,index', [
    (b'\x1b[B\r', 1),
    (b'\x1bOB\r', 1),
    (b'\x1b[1;129B\r', 1),  # NumLock Down
    (b'\x1b[1;193B\x1b[1;65A\r', 0),  # both locks, CapsLock Up
    (b'\x1b[6;129~\r', 5),
    (b'\x1b[5~\r', 0),
    (b'\x1b[F\x1b[Hj\r', 1),
    (b'\x1bOF\r', 29),
    (b'\x1b[4~\x1b[1~j\r', 1),
    (b'\x1b[103;2u\r', 29),  # Shift+g = G
    (b'\x1b[103;194u\r', 29),
    (b'\x1b[27;2;71~\r', 29),  # modifyOtherKeys Shift+G
    (b'\x1b[27;2;71u\r', 29),
    (b'/j/kX\x1b[127;129u\r', 17),
    (b'/j/kX\x1b[27;1;127~\r', 17),
    (b'\x1b[99;197:3u' + b'j\r', 1),  # Ctrl+C release is inert
    (b'\x1b[106;129:3u\r', 0),  # j release is inert
    (b'\x1b[106;65:2u\x1b[13;193u', 1),  # repeat + lock-state Enter
    (b'\x1b[999;129u' + b'j\r', 1),  # unknown key must not cancel or type
])
def test_escape_encodings_keep_navigation_and_consume_the_entire_key(payload, index):
    selected, next_prompt = _probe(payload)
    assert selected == f'saved_{index}'
    assert next_prompt == 'next prompt\n'


@pytest.mark.platforms('posix')
@pytest.mark.parametrize('payload', [
    b'\x1b', b'q', b'\x03', b'\x1b[99;5u', b'\x1b[99;197u', b'\x1b[27;5;99~', b'\x1b[27;193u',
    b'\x1b[99;5u\x1b[99;5:3u', b'\x1b[27;193u\x1b[27;193:3u',
])
def test_cancel_does_not_leave_sequence_bytes_in_the_next_prompt(payload):
    selected, next_prompt = _probe(payload)
    assert selected is None
    assert next_prompt == 'next prompt\n'


@pytest.mark.platforms('posix')
@pytest.mark.parametrize('query', ['界', 'café', 'Ă', '👩‍💻', 'کار‌ها', '\x1b[258;129u', '界X\x7f', '界 j/k', '界🙂\x7f', '界\x1b[127;129:3u'])
def test_utf8_search_edits_codepoints_and_selects_the_filtered_identity(query):
    selected, next_prompt = _probe(('/' + query + '\r').encode('utf-8'))
    assert selected == 'saved_17'
    assert next_prompt == 'next prompt\n'

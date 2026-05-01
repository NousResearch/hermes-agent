#!/usr/bin/env python3
"""Simple PTY echo test + capture."""
import asyncio
import os
import pty
import fcntl
import json
import time
import signal
import termios
import sys
import subprocess
import threading

buf = bytearray()
ready = threading.Event()

def reader(master_fd):
    """Read in a thread."""
    os.set_blocking(master_fd, True)
    try:
        while True:
            d = os.read(master_fd, 4096)
            if not d:
                break
            buf.extend(d)
            sys.stderr.write(f'THREAD READ {len(d)}: {bytes(d[:80])!r}\n')
            sys.stderr.flush()
    except OSError as e:
        sys.stderr.write(f'Thread error: {e}\n')
        sys.stderr.flush()
    finally:
        sys.stderr.write(f'Thread done, total={len(buf)}\n')
        sys.stderr.flush()
        ready.set()

async def main():
    master_fd, slave_fd = pty.openpty()
    sys.stderr.write(f'PTY: {master_fd}\n')
    sys.stderr.flush()
    
    pid = os.fork()
    if pid == 0:
        os.close(master_fd)
        os.setsid()
        fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)
        os.dup2(slave_fd, 0)
        os.dup2(slave_fd, 1)
        os.dup2(slave_fd, 2)
        if slave_fd > 2:
            os.close(slave_fd)
        os.execvpe('/usr/local/bin/claude', ['claude', 'mcp', 'serve'], {
            'HOME': '/root',
            'TERM': 'xterm-256color',
            'PATH': os.environ.get('PATH', ''),
            'CLAUDE_CODE_SIMPLE': '1'
        })
        os._exit(1)
    
    os.close(slave_fd)
    
    # Start reader thread BEFORE writing anything
    t = threading.Thread(target=reader, args=(master_fd,), daemon=True)
    t.start()
    
    # Wait for startup
    sys.stderr.write('Waiting for startup...\n')
    sys.stderr.flush()
    await asyncio.sleep(3)
    
    # Check process
    res = os.waitpid(pid, os.WNOHANG)
    sys.stderr.write(f'Process: {res}\n')
    sys.stderr.flush()
    
    sys.stderr.write(f'Buffer before send: {len(buf)} bytes\n')
    if buf:
        sys.stderr.write(f'Data: {bytes(buf[:200])!r}\n')
        sys.stderr.flush()
        buf.clear()
    
    # Send initialize
    init = json.dumps({
        'jsonrpc': '2.0',
        'id': 1,
        'method': 'initialize',
        'params': {
            'protocolVersion': '2025-03-26',
            'capabilities': {},
            'clientInfo': {'name': 'test', 'version': '1.0'}
        }
    }).encode() + b'\n'
    
    sys.stderr.write(f'Sending {len(init)} bytes...\n')
    sys.stderr.flush()
    
    # Use subprocess.run to do the write synchronously
    # (the thread reader will capture the response)
    result = os.write(master_fd, init)
    sys.stderr.write(f'Wrote {result} bytes\n')
    sys.stderr.flush()
    
    # Wait for response
    sys.stderr.write('Waiting for response...\n')
    sys.stderr.flush()
    
    # Wait for ready event (reader detected data) or timeout
    got_response = ready.wait(timeout=5)
    sys.stderr.write(f'Got response event: {got_response}, buf={len(buf)}\n')
    sys.stderr.flush()
    
    if buf:
        sys.stderr.write(f'Buffer: {bytes(buf[:500])!r}\n')
        sys.stderr.flush()
    else:
        # Check if process is still alive
        res2 = os.waitpid(pid, os.WNOHANG)
        sys.stderr.write(f'Process now: {res2}\n')
        sys.stderr.flush()
    
    # Give more time and check again
    await asyncio.sleep(2)
    sys.stderr.write(f'Buffer final: {len(buf)} bytes\n')
    if buf:
        sys.stderr.write(f'Final: {bytes(buf[:500])!r}\n')
        sys.stderr.flush()
    
    # Cleanup
    if t.is_alive():
        os.kill(pid, signal.SIGTERM)
    os.waitpid(pid, 0)

asyncio.run(main())
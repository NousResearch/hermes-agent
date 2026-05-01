#!/usr/bin/env python3
import asyncio, os, pty, fcntl, json, time, signal, termios, sys

async def main():
    master_fd, slave_fd = pty.openpty()
    print(f'PTY: {master_fd}', file=sys.stderr, flush=True)

    pid = os.fork()
    if pid == 0:
        os.close(master_fd)
        os.setsid()
        fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)
        os.dup2(slave_fd, 0)
        os.dup2(slave_fd, 1)
        if slave_fd > 2: os.close(slave_fd)
        os.execvpe('/usr/local/bin/claude', ['claude', 'mcp', 'serve', '--verbose'], {
            'HOME': '/root', 'TERM': 'xterm-256color',
            'PATH': os.environ.get('PATH',''),
            'CLAUDE_CODE_SIMPLE': '1'
        })
        os._exit(127)

    os.close(slave_fd)

    flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
    fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    await asyncio.sleep(1)

    # Drain startup output
    for _ in range(20):
        try:
            d = os.read(master_fd, 4096)
            if d:
                sys.stderr.write(f'STARTUP {len(d)}: {d[:100]!r}\n')
                sys.stderr.flush()
            else:
                break
        except BlockingIOError:
            break

    # Send initialize
    init = json.dumps({
        'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
        'params': {
            'protocolVersion': '2025-03-26',
            'capabilities': {},
            'clientInfo': {'name': 'test', 'version': '1.0'}
        }
    }).encode() + b'\n'
    sys.stderr.write(f'SENDING: {len(init)} bytes\n')
    sys.stderr.flush()
    os.write(master_fd, init)

    # Read responses for 5 seconds
    start = time.time()
    while time.time() - start < 5:
        await asyncio.sleep(0.2)
        try:
            d = os.read(master_fd, 4096)
            if d:
                sys.stderr.write(f'RECV {len(d)}: {d!r}\n')
                sys.stderr.flush()
        except BlockingIOError:
            pass

    os.kill(pid, signal.SIGTERM)
    os.waitpid(pid, 0)

asyncio.run(main())
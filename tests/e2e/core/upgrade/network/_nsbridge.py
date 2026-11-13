"""First process inside the sandbox's private network namespace (run with ``python -I``).

``_nsbridge.py <socket-dir> <port,port,...> -- <argv...>``: listen on ``127.0.0.1:<port>`` for
each port, relay every connection to the host's unix socket ``<socket-dir>/<port>``, run argv,
and exit with its status. Stdlib only; it must not import anything from the checkout.
"""

import os
import socket
import subprocess
import sys
import threading


def _pump(src, dst):
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def _relay(client, path):
    upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        upstream.connect(path)
    except OSError:
        client.close()
        upstream.close()
        return
    t = threading.Thread(target=_pump, args=(upstream, client), daemon=True)
    t.start()
    _pump(client, upstream)
    t.join(600)
    client.close()
    upstream.close()


def _serve(listener, path):
    while True:
        try:
            client, _ = listener.accept()
        except OSError:
            return
        threading.Thread(target=_relay, args=(client, path), daemon=True).start()


def main():
    sock_dir, ports, sep, *argv = sys.argv[1:]
    assert sep == "--" and argv, "usage: _nsbridge.py <dir> <ports> -- argv"
    dirfd = os.open(sock_dir, os.O_RDONLY | os.O_DIRECTORY)
    for port in filter(None, ports.split(",")):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", int(port)))
        listener.listen(128)
        threading.Thread(target=_serve, args=(listener, f"/proc/self/fd/{dirfd}/{port}"), daemon=True).start()
    proc = subprocess.Popen(argv, pass_fds=())
    try:
        sys.exit(proc.wait())
    except KeyboardInterrupt:
        proc.terminate()
        sys.exit(proc.wait())


if __name__ == "__main__":
    main()

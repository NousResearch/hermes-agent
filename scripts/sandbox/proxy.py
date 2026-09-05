"""MITM proxy backing the dev sandbox's fake Internet.

Listens on 127.0.0.1:8080 and is pointed at by http_proxy/https_proxy inside
the sandbox. For each request it either serves a fixture from the filesystem or
forwards to the real host:

* ``<root>/<host>/<path>`` exists -> serve it. This is how the sandbox answers
  the canonical install URL with the installer under test, so the payload can
  run the true ``curl -fsSL https://…/install.sh | bash`` one-liner.
* otherwise -> forward upstream, verifying against the real CA bundle. The
  sandbox is isolated from the *host*, not from the internet: a real install
  still has to reach PyPI and npm.

HTTPS is intercepted by minting a per-host certificate from the sandbox's own
throwaway CA, which the payload trusts via CURL_CA_BUNDLE / SSL_CERT_FILE.

Usage: proxy.py <fixture-root> <certs-dir> <real-ca-bundle>
"""

import os
import pathlib
import socket
import ssl
import subprocess
import sys
import threading
from urllib.parse import unquote, urlsplit

ROOT, CERTS, REAL_CA = map(pathlib.Path, sys.argv[1:])

LISTEN_ADDRESS = ('127.0.0.1', 8080)
MAX_REQUEST_BYTES = 65536
UPSTREAM_TIMEOUT_SECONDS = 30
CERT_VALIDITY_DAYS = 2


def read_request(conn):
    data = b""
    while b"\r\n\r\n" not in data and len(data) < MAX_REQUEST_BYTES:
        part = conn.recv(4096)
        if not part:
            return b""
        data += part
    return data


def run_openssl(args):
    """Run openssl, raising with its stderr when it fails.

    Discarding stderr here costs real debugging time: the caller sees only a
    dropped connection (``curl: (35) Recv failure``) and the log holds nothing
    but the argv, so an unwritable directory, a missing CA key, and an option
    the host's openssl rejects all look identical.
    """
    done = subprocess.run(
        ['openssl', *args], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    if done.returncode != 0:
        detail = done.stderr.decode('utf-8', 'replace').strip()
        raise RuntimeError(
            f'openssl {args[0]} failed (exit {done.returncode}): {detail}'
        )


_CERT_LOCK = threading.Lock()


def cert_for(host):
    """Return a (cert, key) pair for host, minting it from the sandbox CA.

    Minting is serialized and published atomically. The proxy is threaded, so
    two concurrent requests for the same host would otherwise both run openssl
    into the same paths, and a reader could pick up a finished certificate
    beside a key from the other writer -- which TLS rejects as
    ``[X509: KEY_VALUES_MISMATCH] key values mismatch``.
    """
    safe = ''.join(char if char.isalnum() or char in '.-' else '_' for char in host)
    cert, key = CERTS / f'{safe}.pem', CERTS / f'{safe}.key'
    if cert.exists() and key.exists():
        return cert, key
    with _CERT_LOCK:
        # Re-check: another thread may have finished while we waited.
        if cert.exists() and key.exists():
            return cert, key
        # Build under unique temp names, then rename into place. os.replace is
        # atomic, so a reader sees either the old pair or the new one, never a
        # half-written mix. The key lands first: the certificate's existence is
        # what everything else keys off.
        stamp = f'{os.getpid()}.{threading.get_ident()}'
        tmp_key = CERTS / f'{safe}.key.{stamp}'
        tmp_cert = CERTS / f'{safe}.pem.{stamp}'
        csr = CERTS / f'{safe}.csr.{stamp}'
        run_openssl([
            'req', '-newkey', 'rsa:2048', '-nodes',
            '-subj', f'/CN={host}',
            '-addext', f'subjectAltName=DNS:{host}',
            '-keyout', str(tmp_key), '-out', str(csr),
        ])
        run_openssl([
            'x509', '-req', '-days', str(CERT_VALIDITY_DAYS), '-in', str(csr),
            '-CA', str(CERTS / 'ca.pem'), '-CAkey', str(CERTS / 'ca.key'),
            '-CAcreateserial', '-copy_extensions', 'copy', '-out', str(tmp_cert),
        ])
        csr.unlink(missing_ok=True)
        os.replace(tmp_key, key)
        os.replace(tmp_cert, cert)
    return cert, key


def file_for(host, target):
    """Resolve a request to a fixture file, or None to forward upstream."""
    path = urlsplit(target).path or '/'
    parts = pathlib.PurePosixPath(unquote(path)).parts
    if '..' in parts:
        return None
    candidate = ROOT / host / pathlib.PurePosixPath(*[p for p in parts if p != '/'])
    if candidate.is_dir():
        candidate /= 'index.html'
    return candidate if candidate.is_file() else None


def respond_fixture(conn, found):
    body = found.read_bytes()
    headers = (
        f'Content-Length: {len(body)}\r\nConnection: close\r\n\r\n'.encode()
    )
    conn.sendall(b'HTTP/1.1 200 OK\r\n' + headers + body)


def close_request(request, target=None):
    """Rewrite a proxied request for a direct upstream connection."""
    headers, separator, body = request.partition(b'\r\n\r\n')
    lines = headers.split(b'\r\n')
    if target is not None:
        method, _, version = lines[0].split(b' ', 2)
        lines[0] = b' '.join((method, target.encode(), version))
    lines = [
        line for line in lines
        if not line.lower().startswith(b'proxy-connection:')
    ]
    lines.append(b'Connection: close')
    return b'\r\n'.join(lines) + separator + body


def relay(source, destination):
    while True:
        chunk = source.recv(MAX_REQUEST_BYTES)
        if not chunk:
            return
        destination.sendall(chunk)


def forward_https(conn, host, port, request):
    context = ssl.create_default_context(cafile=str(REAL_CA))
    with socket.create_connection((host, port), timeout=UPSTREAM_TIMEOUT_SECONDS) as raw:
        with context.wrap_socket(raw, server_hostname=host) as upstream:
            upstream.sendall(close_request(request))
            relay(upstream, conn)


def forward_http(conn, host, port, request, target):
    parsed = urlsplit(target)
    path = parsed.path or '/'
    if parsed.query:
        path += f'?{parsed.query}'
    with socket.create_connection((host, port), timeout=UPSTREAM_TIMEOUT_SECONDS) as upstream:
        upstream.sendall(close_request(request, path))
        relay(upstream, conn)


def handle_connect(conn, target):
    """Intercept a CONNECT tunnel, terminating TLS with a minted cert."""
    host, _, port_text = target.rpartition(':')
    port = int(port_text or '443')
    conn.sendall(b'HTTP/1.1 200 Connection Established\r\n\r\n')
    cert, key = cert_for(host)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    with context.wrap_socket(conn, server_side=True) as tls:
        nested = read_request(tls)
        if not nested:
            return
        line = nested.split(b'\r\n', 1)[0].decode('iso-8859-1')
        nested_target = line.split(' ', 2)[1]
        found = file_for(host, nested_target)
        if found is not None:
            respond_fixture(tls, found)
        else:
            forward_https(tls, host, port, nested)


def host_from_headers(request):
    for header in request.split(b'\r\n')[1:]:
        if header.lower().startswith(b'host:'):
            value = header.split(b':', 1)[1].strip().decode()
            return value.split(':', 1)[0]
    return None


def handle_request(conn):
    with conn:
        request = read_request(conn)
        if not request:
            return
        line = request.split(b'\r\n', 1)[0].decode('iso-8859-1')
        method, target, _ = line.split(' ', 2)
        if method.upper() == 'CONNECT':
            handle_connect(conn, target)
            return
        parsed = urlsplit(target)
        host = parsed.hostname or host_from_headers(request) or 'unknown'
        found = file_for(host, target)
        if found is not None:
            respond_fixture(conn, found)
        else:
            forward_http(conn, host, parsed.port or 80, request, target)


def handle(conn):
    try:
        handle_request(conn)
    except Exception as error:
        print(f'proxy request failed: {error!r}', file=sys.stderr, flush=True)


def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(LISTEN_ADDRESS)
        server.listen()
        while True:
            conn, _ = server.accept()
            threading.Thread(target=handle, args=(conn,), daemon=True).start()


if __name__ == '__main__':
    main()

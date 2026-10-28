"""Owned v1 issuer redemption endpoint, mutually authenticated TLS only.

Trusted operator supplies the Issuer and TLS context. No network listener starts
on import, plugin discovery or gateway startup. No header/JSON identity fallback.
"""
import hashlib
import json
import ssl
import socket
import threading
from socketserver import ThreadingMixIn
from http.server import BaseHTTPRequestHandler, HTTPServer

from hermes_cli.private_invocation_store import canonical


def serve(issuer, address, tls_context):
    if tls_context.verify_mode != ssl.CERT_REQUIRED:
        raise ValueError('issuer requires authenticated client certificates')
    if tls_context.minimum_version < ssl.TLSVersion.TLSv1_2:
        raise ValueError('issuer requires TLS 1.2 or newer')

    class Redemption(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            # Absolute header/body admission deadline also defeats a slow byte trickle.
            def close_stalled():
                try:
                    self.connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            self._admission_timer = threading.Timer(5, close_stalled)
            self._admission_timer.daemon = True
            self._admission_timer.start()

        def finish(self):
            self._admission_timer.cancel()
            super().finish()

        def log_message(self, format, *args):
            # Neither URL nor request/result/private exceptions enter access logs.
            return

        def do_POST(self):
            self.connection.settimeout(5)
            if self.path not in ('/private-invocation/v1/redeem', '/private-invocation/v1/forward'):
                self.send_error(404)
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 32768 or self.headers.get('Transfer-Encoding'):
                    raise ValueError('bounded request required')
                certificate = self.connection.getpeercert(binary_form=True)
                if not certificate:
                    raise PermissionError('authenticated peer required')
                request = json.loads(self.rfile.read(length))
                self._admission_timer.cancel()
                peer = hashlib.sha256(certificate).hexdigest()
                if self.path.endswith('/redeem'):
                    if set(request) != {'version', 'reference', 'binding'} or request['version'] != 1:
                        raise ValueError('invalid redemption envelope')
                    result = issuer.redeem(peer, request['reference'], request['binding'])
                else:
                    if set(request) != {'version', 'lease', 'binding', 'bindingId', 'input'} or request['version'] != 1:
                        raise ValueError('invalid forwarding envelope')
                    result = issuer.forward(peer, request['lease'], request['binding'],
                                            request['bindingId'], request['input'])
                body = json.dumps(result, ensure_ascii=False, allow_nan=False).encode('utf-8')
            except (ValueError, TypeError, KeyError, PermissionError):
                body = b'{"error":"private invocation refused"}'
                self.send_response(403)
            except Exception:
                body = b'{"error":"private invocation unavailable; outcome unknown"}'
                self.send_response(503)
            else:
                self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    class BoundedServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True
        request_queue_size = 8

        def __init__(self, *args):
            self.slots = threading.BoundedSemaphore(4)
            super().__init__(*args)

        def process_request(self, request, client_address):
            if not self.slots.acquire(blocking=False):
                request.close()
                return
            try:
                super().process_request(request, client_address)
            except BaseException:
                self.slots.release()
                raise

        def process_request_thread(self, request, client_address):
            wrapped = None
            try:
                # Accept is raw. A peer cannot block the listener in TLS accept.
                request.settimeout(5)
                wrapped = tls_context.wrap_socket(request, server_side=True,
                                                  do_handshake_on_connect=False)
                wrapped.do_handshake()
                # Header parsing has the same finite inactivity deadline.
                wrapped.settimeout(5)
                super().process_request_thread(wrapped, client_address)
            except (OSError, ssl.SSLError):
                pass
            finally:
                (wrapped or request).close()
                self.slots.release()

    with BoundedServer(address, Redemption) as server:
        server.serve_forever(poll_interval=0.5)


def forward_https(url, tls_context, payload):
    """One configured authenticated HTTPS destination; refuse every redirect."""
    from urllib.request import Request, HTTPSHandler, HTTPRedirectHandler, build_opener
    from urllib.parse import urlsplit
    parsed = urlsplit(url)
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
            or parsed.fragment or tls_context.verify_mode != ssl.CERT_REQUIRED
            or not tls_context.check_hostname):
        raise PermissionError('verified fixed HTTPS forwarding required')

    class NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            raise PermissionError('private forwarding redirect refused')

    # No ambient proxy credentials or cookies; operator supplies authenticated TLS.
    from urllib.request import ProxyHandler
    opener = build_opener(ProxyHandler({}), HTTPSHandler(context=tls_context), NoRedirect())
    request = Request(url, data=canonical(payload).encode('utf-8'),
                      headers={'Content-Type': 'application/json'}, method='POST')
    with opener.open(request, timeout=30) as response:
        data = response.read(1048577)
        if len(data) > 1048576:
            raise ValueError('private forwarding result exceeds limit')
        result = json.loads(data)
        if type(result) is not dict or set(result) != {'result'}:
            raise ValueError('ordinary result envelope required')
        return result['result']


def main():
    import argparse
    import importlib
    parser = argparse.ArgumentParser(description='Private invocation issuer (owned development host)')
    parser.add_argument('--factory', required=True, help='Trusted operator module:function -> (issuer,address,TLSContext)')
    args = parser.parse_args()
    module, name = args.factory.split(':', 1)
    issuer, address, context = getattr(importlib.import_module(module), name)()
    serve(issuer, address, context)


if __name__ == '__main__':
    main()

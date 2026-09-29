"""HTTPS federation endpoint in front of the disposable homeserver.

The authorisation service verifies OpenID tokens through the federation API over HTTPS.
Without a server delegation it connects to port 8448 of the server name, while the
test homeserver serves plain HTTP, so this forwards TLS requests on 8448 to it.
"""

from __future__ import annotations

import http.server
import ssl
import subprocess
import urllib.error
import urllib.request


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            with urllib.request.urlopen("http://synapse:8008" + self.path, timeout=5) as response:
                self.reply(response.status, response.read())
        except urllib.error.HTTPError as error:
            self.reply(error.code, error.read())

    def reply(self, status: int, body: bytes) -> None:
        # rustls rejects a body that ends at a TLS connection close without close_notify,
        # so the length has to be explicit.
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print(format % args, flush=True)


def main() -> None:
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                    "-keyout", "/tmp/rtc.key", "-out", "/tmp/rtc.crt", "-days", "1",
                    "-subj", "/CN=matrix.test", "-addext", "subjectAltName=DNS:matrix.test"],
                   check=True, capture_output=True)
    server = http.server.ThreadingHTTPServer(("0.0.0.0", 8448), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain("/tmp/rtc.crt", "/tmp/rtc.key")
    server.socket = context.wrap_socket(server.socket, server_side=True)
    print("RTC federation proxy ready", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

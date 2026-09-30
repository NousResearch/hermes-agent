"""A connected but stalled origin must release its pinned mirror and partial owner.

Regression for the transport stalls reported on #98049: socket inactivity
timeouts do not bound a peer that trickles bytes below a useful transfer rate.
"""
import hashlib
import errno
import socketserver
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import ip_address
import re
import ssl
import threading
import urllib.request

import pytest

from pm import downloader

@contextmanager
def _server(handler, tmp_path, monkeypatch, tls):
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        server.daemon_threads = True
        if tls:
            from cryptography import x509
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import ec
            from cryptography.x509.oid import NameOID

            key = ec.generate_private_key(ec.SECP256R1())
            name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "download loopback")])
            now = datetime.now(timezone.utc)
            cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
                    .public_key(key.public_key()).serial_number(x509.random_serial_number())
                    .not_valid_before(now - timedelta(days=1)).not_valid_after(now + timedelta(days=1))
                    .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ip_address("127.0.0.1"))]),
                                   critical=False).sign(key, hashes.SHA256()))
            bundle, private = tmp_path / "server.pem", tmp_path / "server.key"
            bundle.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
            private.write_bytes(key.private_bytes(serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(bundle, private)
            server.socket = context.wrap_socket(server.socket, server_side=True)
            # Keep verification enabled; the fixture cert is the sole added CA.
            context = ssl.create_default_context(cafile=str(bundle))
            monkeypatch.setattr(downloader, "_OPENER", urllib.request.build_opener(
                downloader._HttpsRedirectHandler(), downloader._GuardedHTTPHandler(),
                downloader._GuardedHTTPSHandler(context=context)))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server
        finally:
            server.shutdown()
            thread.join(5)


@pytest.mark.parametrize("phase", ["headers", "probe-body", "ranged", "parallel", "single", "active"])
@pytest.mark.parametrize("tls", [False, True])
def test_bounded_stall_reaches_mirror_without_limiting_active_download(tmp_path, monkeypatch, phase, tls):
    monkeypatch.setattr(downloader, "_NETWORK_BUDGET", .3, raising=False)
    monkeypatch.setattr(downloader, "_READ_QUANTUM", 128, raising=False)
    payload = b"pinned" * (400_000 if phase == "parallel" else 100)
    release = threading.Event()
    mirror = threading.Event()
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            bounds = self.headers.get("Range")
            probe = bounds == "bytes=0-0"
            primary = self.path == "/primary"
            requests.append((self.path, bounds))
            if not primary:
                mirror.set()
            stall = primary and (
                phase in ("headers", "probe-body") or
                (not probe and phase in ("ranged", "parallel", "single")))
            try:
                if stall and phase == "headers":
                    for byte in b"HTTP/1.0 200 OK\r\n":
                        self.wfile.write(bytes([byte]))
                        self.wfile.flush()
                        if release.wait(.08):
                            break
                    return
                ranged = bounds is not None and phase != "single"
                start, end = (map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", bounds).groups())
                              if ranged else (0, len(payload) - 1))
                body = payload[start:end + 1]
                self.send_response(206 if ranged else 200)
                if stall and phase == "probe-body":
                    self.send_header("Transfer-Encoding", "chunked")
                else:
                    self.send_header("Content-Length", str(len(body)))
                if ranged:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
                self.send_header("ETag", '"pinned-object"')
                self.end_headers()
                if stall:
                    # Bytes keep arriving well inside the socket inactivity
                    # timeout, yet no complete useful read can finish.
                    while not release.is_set():
                        # Probe-body trickles the chunk size line: no body
                        # byte completes, although recv continues succeeding.
                        self.wfile.write(b"0" if phase == "probe-body" else body[:1])
                        self.wfile.flush()
                        release.wait(.08)
                    return
                if primary and phase == "active" and not probe:
                    for offset in range(0, len(body), 128):
                        self.wfile.write(body[offset:offset + 128])
                        self.wfile.flush()
                        release.wait(.12)
                else:
                    self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    destination = tmp_path / "artifact"
    errors = []
    complete = threading.Event()
    with _server(Handler, tmp_path, monkeypatch, tls) as server:
        origin = f"{'https' if tls else 'http'}://127.0.0.1:{server.server_port}"
        dl = downloader.Download([downloader.Source(origin + "/primary", destination,
            hashlib.sha256(payload).hexdigest(), (origin + "/mirror",))],
            partials_dir=tmp_path / "partials", connections=2)

        def run():
            try:
                dl.run()
            except BaseException as exc:
                errors.append(exc)
            finally:
                complete.set()

        task = threading.Thread(target=run, daemon=True)
        task.start()
        try:
            finished_without_release = complete.wait(3)
        finally:
            release.set()
            task.join(10)
        assert finished_without_release, "a stalled peer still prevents mirror fallback"
        assert not task.is_alive() and not errors, errors
        assert destination.read_bytes() == payload
        assert dl.connections == 2
        if phase == "active":
            assert not mirror.is_set(), "ongoing useful progress hit a whole-download deadline"
        else:
            assert mirror.is_set()
            failing = [entry for entry in requests if entry[0] == "/primary" and
                       (phase in ("headers", "probe-body") or entry[1] != "bytes=0-0")]
            assert len(failing) <= (2 if phase == "parallel" else 1)


@pytest.mark.parametrize("phase", ["headers", "probe-body", "ranged", "single"])
@pytest.mark.parametrize("tls", [False, True])
def test_pause_shuts_stalled_socket_before_returning_partial_owner(tmp_path, monkeypatch, phase, tls):
    # A long network budget makes cancellation, rather than the deadline,
    # responsible for waking the blocked real HTTP read.
    monkeypatch.setattr(downloader, "_NETWORK_BUDGET", 20, raising=False)
    entered = threading.Event()
    release = threading.Event()
    payload = b"partial" * 100

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            bounds = self.headers.get("Range")
            probe = bounds == "bytes=0-0"
            stalled = phase in ("headers", "probe-body") or not probe
            try:
                if phase == "headers":
                    entered.set()
                    release.wait(10)
                    return
                ranged = bounds is not None and phase != "single"
                start, end = (map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", bounds).groups())
                              if ranged else (0, len(payload) - 1))
                self.send_response(206 if ranged else 200)
                self.send_header("Content-Length", str(end - start + 1))
                if ranged:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
                self.send_header("ETag", '"unchanged"')
                self.end_headers()
                if stalled:
                    entered.set()
                    release.wait(10)
                self.wfile.write(payload[start:end + 1])
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    errors = []
    complete = threading.Event()
    existing_threads = {thread.ident for thread in threading.enumerate()}
    with _server(Handler, tmp_path, monkeypatch, tls) as server:
        destination = tmp_path / "artifact"
        dl = downloader.Download([downloader.Source(
            f"{'https' if tls else 'http'}://127.0.0.1:{server.server_port}/artifact", destination,
            hashlib.sha256(payload).hexdigest())], partials_dir=tmp_path / "partials")

        def run():
            try:
                dl.run()
            except BaseException as exc:
                errors.append(exc)
            finally:
                complete.set()

        task = threading.Thread(target=run, daemon=True)
        task.start()
        try:
            assert entered.wait(5)
            dl.pause()
            finished_before_peer_release = complete.wait(3)
            before = {path.name: path.read_bytes() for path in dl.partials_dir.glob("*.*") if path.is_file()}
        finally:
            release.set()
            task.join(10)
        assert finished_before_peer_release, "pause still waits for the blocked origin"
        assert len(errors) == 1 and isinstance(errors[0], downloader.DownloadPaused), errors
        assert not task.is_alive() and not destination.exists()
        after = {path.name: path.read_bytes() for path in dl.partials_dir.glob("*.*") if path.is_file()}
        assert after == before, "a cancelled worker kept writing after run() returned"
        assert not [thread for thread in threading.enumerate() if thread.ident not in existing_threads
                    and thread.name.startswith(("hermes-download", "hermes-request"))]


@pytest.mark.parametrize("proxy", [False, True])
def test_pause_interrupts_a_tls_handshake_and_releases_request_threads(tmp_path, monkeypatch, proxy):
    # Neither a silent TLS peer nor a proxy that completed CONNECT should
    # keep pause waiting for the normal network timeout.
    monkeypatch.setattr(downloader, "_NETWORK_BUDGET", 20)
    entered, release, complete = threading.Event(), threading.Event(), threading.Event()
    errors = []
    existing_threads = {thread.ident for thread in threading.enumerate()}

    class Peer(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.settimeout(10)
            try:
                if proxy:
                    headers = b""
                    while not headers.endswith(b"\r\n\r\n"):
                        block = self.request.recv(4096)
                        if not block:
                            return
                        headers += block
                    assert headers.startswith(b"CONNECT origin.invalid:443 ")
                    self.request.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
                if not self.request.recv(4096):
                    return
                entered.set()  # The real client has sent its TLS ClientHello.
                release.wait(10)
            except (BrokenPipeError, ConnectionResetError):
                pass

    with socketserver.ThreadingTCPServer(("127.0.0.1", 0), Peer) as server:
        server.daemon_threads = True
        peer = threading.Thread(target=server.serve_forever, daemon=True)
        peer.start()
        host = f"127.0.0.1:{server.server_address[1]}"
        proxy_handler = urllib.request.ProxyHandler({"https": f"http://{host}"} if proxy else {})
        monkeypatch.setattr(downloader, "_OPENER", urllib.request.build_opener(
            proxy_handler, downloader._HttpsRedirectHandler(), downloader._GuardedHTTPHandler(),
            downloader._GuardedHTTPSHandler()))
        destination = tmp_path / "artifact"
        source = f"https://{'origin.invalid' if proxy else host}/artifact"
        dl = downloader.Download([downloader.Source(source, destination, hashlib.sha256(b"payload").hexdigest())],
                                 partials_dir=tmp_path / "partials")

        def run():
            try:
                dl.run()
            except BaseException as exc:
                errors.append(exc)
            finally:
                complete.set()

        task = threading.Thread(target=run, daemon=True)
        task.start()
        try:
            assert entered.wait(5), "TLS handshake did not start"
            dl.pause()
            returned_before_peer_release = complete.wait(3)
        finally:
            release.set()
            task.join(10)
            server.shutdown()
            peer.join(5)
        assert returned_before_peer_release, "pause left the TLS handshake waiting on an unowned socket"
        assert len(errors) == 1 and isinstance(errors[0], downloader.DownloadPaused), errors
        assert not task.is_alive() and not destination.exists()
        assert not [thread for thread in threading.enumerate() if thread.ident not in existing_threads
                    and thread.name.startswith(("hermes-download", "hermes-request"))]


@pytest.mark.parametrize("ranged", [False, True])
@pytest.mark.parametrize("failure", ["observer-timeout", "observer-reset", "local-sync-timeout"])
def test_local_failures_never_retry_the_origin_or_start_its_mirror(tmp_path, monkeypatch, ranged, failure):
    # Network-shaped errno/types also occur in local UI/IPC and network-backed
    # filesystems. Their origin, rather than their class, decides fallback.
    payload = b"complete pinned artifact"
    requests = []
    original = TimeoutError("local operation timed out") if failure != "observer-reset" else OSError(
        errno.ECONNRESET, "local progress IPC reset")

    class Peer(BaseHTTPRequestHandler):
        def do_GET(self):
            bounds = self.headers.get("Range")
            requests.append((self.path, bounds))
            if ranged and bounds:
                start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", bounds).groups())
            else:
                start, end = 0, len(payload) - 1
            self.send_response(206 if ranged and bounds else 200)
            self.send_header("Content-Length", str(end - start + 1))
            if ranged and bounds:
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.send_header("ETag", '"unchanged"')
            self.end_headers()
            self.wfile.write(payload[start:end + 1])

        def log_message(self, *args):
            pass

    monkeypatch.setattr(downloader.Download, "_wait_retry", lambda self, delay: None)
    if failure == "local-sync-timeout":
        def failed_sync(fd):
            raise original
        monkeypatch.setattr(downloader.os, "fsync", failed_sync)

    def observer(done, total, coverage):
        if done and failure.startswith("observer"):
            raise original

    with _server(Peer, tmp_path, monkeypatch, False) as server:
        origin = f"http://127.0.0.1:{server.server_port}"
        destination = tmp_path / "artifact"
        dl = downloader.Download([downloader.Source(origin + "/primary", destination,
            hashlib.sha256(payload).hexdigest(), (origin + "/mirror",))], partials_dir=tmp_path / "partials")
        with pytest.raises(type(original)) as caught:
            dl.run(observer)
        assert caught.value is original
        assert not any(path == "/mirror" for path, _ in requests), requests
        transfers = [request for request in requests if request[0] == "/primary"
                     and (not ranged and request[1] is None or ranged and request[1] != "bytes=0-0")]
        assert len(transfers) == 1, "a local failure was retried as an origin outage"
        assert not destination.exists()


def test_guarded_tls_connection_keeps_certificate_verification_enabled(tmp_path, monkeypatch):
    reached = threading.Event()

    class Peer(BaseHTTPRequestHandler):
        def do_GET(self):
            reached.set()
            self.send_response(200)
            self.send_header("Content-Length", "7")
            self.end_headers()
            self.wfile.write(b"payload")

        def log_message(self, *args):
            pass

    with _server(Peer, tmp_path, monkeypatch, True) as server:
        # The server uses an untrusted fixture CA. Unlike the positive verified
        # TLS tests, this client must keep the ordinary default trust roots.
        monkeypatch.setattr(downloader, "_OPENER", urllib.request.build_opener(
            urllib.request.ProxyHandler({}), downloader._HttpsRedirectHandler(),
            downloader._GuardedHTTPHandler(), downloader._GuardedHTTPSHandler()))
        destination = tmp_path / "artifact"
        dl = downloader.Download([downloader.Source(f"https://127.0.0.1:{server.server_port}/artifact",
            destination, hashlib.sha256(b"payload").hexdigest())], partials_dir=tmp_path / "partials")
        with pytest.raises(downloader.DownloadTransportError) as caught:
            dl.run()
        assert caught.value.fallback_allowed is False
        assert "CERTIFICATE_VERIFY_FAILED" in str(caught.value)
        assert not reached.is_set() and not destination.exists()


def test_pause_before_watchdog_start_restores_request_context(monkeypatch):
    paused = threading.Event()
    original_arm = downloader._RequestGuard.arm

    def pause_before_arm(guard):
        paused.set()
        original_arm(guard)

    monkeypatch.setattr(downloader._RequestGuard, "arm", pause_before_arm)
    before = downloader._REQUEST_GUARD.get()
    with pytest.raises(downloader.DownloadPaused):
        with downloader._open_request(urllib.request.Request("https://origin.invalid/artifact"), paused):
            pytest.fail("a paused request should not reach its body")
    assert downloader._REQUEST_GUARD.get() is before

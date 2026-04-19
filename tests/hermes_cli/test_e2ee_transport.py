"""Tests for E2EETransport — httpx-level replacement for the proxy server."""
import json
import secrets

import httpx
import pytest
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from hermes_cli.e2ee_proxy import _decrypt_ecdsa, _generate_ecdsa_keypair
from hermes_cli.e2ee_transport import E2EETransport


def _make_model_keypair():
    priv = ec.generate_private_key(ec.SECP256K1(), default_backend())
    pub_bytes = priv.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )
    return priv, pub_bytes[1:].hex()


def _ecies_encrypt_to_client(plaintext: bytes, client_pub_hex: str) -> bytes:
    pub_bytes = bytes.fromhex(client_pub_hex)
    if len(pub_bytes) == 65 and pub_bytes[0] == 0x04:
        pub_bytes = pub_bytes[1:]
    client_pub = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256K1(), b"\x04" + pub_bytes)
    eph = ec.generate_private_key(ec.SECP256K1(), default_backend())
    shared = eph.exchange(ec.ECDH(), client_pub)
    aes_key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"ecdsa_encryption", backend=default_backend()).derive(shared)
    nonce = secrets.token_bytes(12)
    ct = AESGCM(aes_key).encrypt(nonce, plaintext, None)
    eph_pub = eph.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    return eph_pub + nonce + ct


class _FakeUpstream(httpx.BaseTransport):
    """Inner transport that plays the role of the attested model CVM."""
    def __init__(self, model_priv, response_text: str = "4", stream: bool = False):
        self.model_priv = model_priv
        self.response_text = response_text
        self.stream = stream
        self.received_plaintexts: list[str] = []
        self.last_headers: httpx.Headers | None = None

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.last_headers = request.headers
        body = json.loads(request.content)
        for msg in body.get("messages", []):
            ct = msg.get("content", "")
            if ct:
                pt = _decrypt_ecdsa(bytes.fromhex(ct), self.model_priv)
                self.received_plaintexts.append(pt.decode())

        client_pub_hex = request.headers.get("x-client-pub-key", "")
        enc = _ecies_encrypt_to_client(self.response_text.encode(), client_pub_hex).hex()

        if self.stream:
            sse_lines = [
                "data: " + json.dumps({"choices": [{"delta": {"content": enc}}]}),
                "data: [DONE]",
                "",
            ]
            body_bytes = ("\n".join(sse_lines) + "\n").encode()
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=body_bytes,
                request=request,
            )

        resp_body = json.dumps({"choices": [{"message": {"content": enc}}]}).encode()
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=resp_body,
            request=request,
        )


class TestE2EETransportNonStreaming:
    def test_encrypts_request_and_decrypts_response(self):
        model_priv, model_pub_hex = _make_model_keypair()
        upstream = _FakeUpstream(model_priv, response_text="4")
        transport = E2EETransport(model_pub_hex, "ecdsa", inner=upstream)

        with httpx.Client(transport=transport, base_url="https://model.example") as client:
            resp = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "what is 2+2?"}], "stream": False},
            )

        assert resp.status_code == 200
        assert upstream.received_plaintexts == ["what is 2+2?"]
        assert resp.json()["choices"][0]["message"]["content"] == "4"

    def test_e2ee_headers_injected(self):
        model_priv, model_pub_hex = _make_model_keypair()
        upstream = _FakeUpstream(model_priv)
        transport = E2EETransport(model_pub_hex, "ecdsa", inner=upstream)

        with httpx.Client(transport=transport, base_url="https://model.example") as client:
            client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})

        h = upstream.last_headers
        assert h["x-signing-algo"] == "ecdsa"
        assert h["x-model-pub-key"] == model_pub_hex
        assert h["x-client-pub-key"]  # ephemeral, nonempty

    def test_non_chat_path_passes_through(self):
        class _Echo(httpx.BaseTransport):
            def handle_request(self, req):
                return httpx.Response(200, json={"models": ["a", "b"]}, request=req)

        _, model_pub_hex = _make_model_keypair()
        transport = E2EETransport(model_pub_hex, "ecdsa", inner=_Echo())
        with httpx.Client(transport=transport, base_url="https://model.example") as client:
            resp = client.get("/v1/models")
        assert resp.json() == {"models": ["a", "b"]}


class TestE2EETransportStreaming:
    def test_streaming_decrypts_on_the_fly(self):
        model_priv, model_pub_hex = _make_model_keypair()
        upstream = _FakeUpstream(model_priv, response_text="streaming-answer", stream=True)
        transport = E2EETransport(model_pub_hex, "ecdsa", inner=upstream)

        collected: list[str] = []
        with httpx.Client(transport=transport, base_url="https://model.example") as client:
            with client.stream(
                "POST", "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "stream pls"}], "stream": True},
            ) as resp:
                for line in resp.iter_lines():
                    if line.startswith("data: {"):
                        delta = json.loads(line[6:])["choices"][0]["delta"]
                        collected.append(delta.get("content", ""))

        assert upstream.received_plaintexts == ["stream pls"]
        assert "".join(collected) == "streaming-answer"

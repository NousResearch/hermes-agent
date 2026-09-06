"""The sandbox payload must trust the sandbox CA, not the real bundle.

Every TLS client inside the sandbox talks to the MITM proxy, which
terminates HTTPS with a cert minted by the sandbox CA and validates
upstream against the real CA bundle itself. Node is the one payload
client that reads NODE_EXTRA_CA_CERTS instead of SSL_CERT_FILE; pointing
it at the real bundle made npm reject the proxy's minted certs
(UNABLE_TO_VERIFY_LEAF_SIGNATURE) while curl/git/uv kept working, which
surfaced as npm-install failures in the Install & Update E2E workflow
and as the proxy logging each aborted handshake as an SSLEOFError
(#87093).
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE2 = REPO_ROOT / "scripts" / "sandbox" / "stage2-run.sh"


def test_payload_clients_trust_the_sandbox_ca() -> None:
    text = STAGE2.read_text()

    # Every payload-side CA variable points at the sandbox CA...
    assert "--setenv CURL_CA_BUNDLE /work/certs/ca.pem \\" in text
    assert "--setenv SSL_CERT_FILE /work/certs/ca.pem \\" in text
    assert "--setenv GIT_SSL_CAINFO /work/certs/ca.pem \\" in text
    # ...including Node's, which reads NODE_EXTRA_CA_CERTS instead of
    # SSL_CERT_FILE.
    assert "--setenv NODE_EXTRA_CA_CERTS /work/certs/ca.pem \\" in text
    # The real bundle must never be handed to a payload client as its
    # trust anchor.
    assert "--setenv NODE_EXTRA_CA_CERTS /work/certs/real-ca.pem" not in text


def test_proxy_still_validates_upstream_against_the_real_ca() -> None:
    """The trust split: payload trusts the sandbox CA, proxy trusts real CAs."""
    text = STAGE2.read_text()

    # proxy.py's third argument is the real CA bundle.
    assert (
        "python3 /work/proxy.py /work/http /work/certs /work/certs/real-ca.pem"
        in text
    )

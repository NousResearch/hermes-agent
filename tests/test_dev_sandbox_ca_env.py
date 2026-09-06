"""Regression coverage for the dev sandbox's CA wiring.

Every client inside the sandbox reaches the network through
``scripts/sandbox/proxy.py``, which terminates TLS with a leaf minted from the
sandbox's own throwaway CA (``certs/ca.pem``). ``certs/real-ca.pem`` is a copy
of the *host's* public bundle and belongs to the proxy alone, for verifying the
upstream it forwards to.

Pointing a client at ``real-ca.pem`` therefore breaks it: the public bundle
carries no sandbox CA, so the client rejects the proxy's leaf and hangs up. The
only trace on the proxy side is ``SSLEOFError: UNEXPECTED_EOF_WHILE_READING``,
which names neither the client nor the CA -- which is what made this expensive
to find when ``NODE_EXTRA_CA_CERTS`` was wired to ``real-ca.pem`` and every
``npm install`` in the Install & Update E2E installer legs failed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE2_RUN = REPO_ROOT / "scripts" / "sandbox" / "stage2-run.sh"

SANDBOX_CA = "/work/certs/ca.pem"
REAL_CA = "/work/certs/real-ca.pem"

# Every runtime the sandbox installs through has to trust the proxy's leaf.
# NODE_EXTRA_CA_CERTS *adds* to Node's compiled-in roots, the others replace
# the store outright, but all four have to name the sandbox CA either way.
CLIENT_CA_VARS = (
    "CURL_CA_BUNDLE",
    "SSL_CERT_FILE",
    "GIT_SSL_CAINFO",
    "NODE_EXTRA_CA_CERTS",
)


@pytest.fixture(scope="module")
def stage2_text() -> str:
    if not STAGE2_RUN.exists():
        pytest.skip("scripts/sandbox/stage2-run.sh not present in this checkout")
    return STAGE2_RUN.read_text(encoding="utf-8")


def _setenv_values(text: str, name: str) -> list[str]:
    return re.findall(rf"--setenv\s+{re.escape(name)}\s+(\S+)", text)


@pytest.mark.parametrize("var", CLIENT_CA_VARS)
def test_client_ca_vars_point_at_the_sandbox_ca(stage2_text: str, var: str) -> None:
    values = _setenv_values(stage2_text, var)

    assert values, f"{var} is not set in the sandbox environment"
    assert values == [SANDBOX_CA], (
        f"{var} must name the sandbox CA ({SANDBOX_CA}) that proxy.py mints "
        f"leaf certificates from, not {values}. Pointing it at {REAL_CA} makes "
        "the client reject the proxy's certificate and hang up mid-handshake."
    )


def test_real_ca_stays_the_proxys_upstream_bundle(stage2_text: str) -> None:
    """real-ca.pem still has a job -- just not as a client trust store.

    Guards the other direction of the same confusion: keeping the clients on
    ca.pem by taking real-ca.pem away from proxy.py would leave the proxy
    unable to verify the real upstreams it forwards to.
    """
    assert re.search(rf"proxy\.py\s+\S+\s+\S+\s+{re.escape(REAL_CA)}", stage2_text), (
        "proxy.py must still receive real-ca.pem as its upstream CA bundle"
    )

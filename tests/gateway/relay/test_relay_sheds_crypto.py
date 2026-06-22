"""Invariant: the relay path sheds platform crypto — it re-validates nothing.

Under the A2 trust model (see docs/relay-connector-contract.md §6), the
*connector* is the sole crypto/identity boundary: it verifies/decrypts every
inbound platform payload at the edge (it holds the tenant secrets), normalizes
it to a tenant-scoped ``MessageEvent``, and forwards only the sanitized event.
The gateway re-validates nothing — it cannot, without being handed the shared
signing secret, which would itself be the leak on a shared bot.

The relay package therefore MUST NOT import or call platform signature/crypto
verification (Discord ed25519, Twilio HMAC, WeCom BizMsgCrypt, generic webhook
signature checks). Those live in the *direct* platform adapters
(``gateway/platforms/*``) which serve non-relay deployments; the relay receives
already-trusted events. This test fails if someone bolts re-validation onto the
relay path, re-coupling the gateway to platform secrets it must never hold.

It is an invariant (asserts the *relation* "relay imports no crypto"), not a
change-detector snapshot of a frozen import list.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

# gateway/relay package directory: tests/gateway/relay/ -> repo root parents[3].
_REPO_ROOT = Path(__file__).resolve().parents[3]
_RELAY_PKG = _REPO_ROOT / "gateway" / "relay"

# Modules / symbols that mean "platform crypto re-validation". If the relay path
# imports any of these it has re-coupled the gateway to a platform secret.
_FORBIDDEN_MODULE_TOKENS = (
    "wecom_crypto",
    "wecom_callback",
    "webhook",  # gateway.platforms.webhook holds signature verification
)
_FORBIDDEN_SYMBOL_RE = re.compile(
    r"(ed25519|verify_key|verifykey|verify_signature|verify_ed25519|"
    r"verify_webhook|bizmsg|hmac|x[-_]signature)",
    re.IGNORECASE,
)


def _relay_py_files() -> list[Path]:
    assert _RELAY_PKG.is_dir(), f"relay package missing at {_RELAY_PKG}"
    return sorted(_RELAY_PKG.glob("*.py"))


# ``auth.py`` is the connector⇄gateway CHANNEL authenticator (the gateway's WS
# upgrade bearer). It is net-new, intended, and the whole point of
# authenticating an untrusted/disposable gateway — it is NOT platform crypto.
# It uses HMAC over the connector's per-gateway secret (NOT any platform's
# signing secret), so it is exempt from the platform-crypto symbol scan below.
# The module-import ban (platform-crypto modules) still applies to every file
# including this one — it imports only stdlib hmac/hashlib, never a
# platform-crypto module, so it stays clean there.
_CHANNEL_AUTH_FILES = {"auth.py"}


def test_relay_package_imports_no_platform_crypto():
    """No module in gateway/relay imports a platform-crypto / verification module."""
    offenders: list[str] = []
    for path in _relay_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            mods: list[str] = []
            if isinstance(node, ast.Import):
                mods = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [node.module or ""]
                mods += [f"{node.module or ''}.{a.name}" for a in node.names]
            for mod in mods:
                if any(tok in mod for tok in _FORBIDDEN_MODULE_TOKENS):
                    offenders.append(f"{path.name}: imports '{mod}'")
    assert not offenders, (
        "The relay path must re-validate NOTHING (A2: connector is the sole "
        "crypto boundary). Found platform-crypto imports in the relay package:\n  "
        + "\n  ".join(offenders)
        + "\nMove verification to the connector edge; the gateway trusts the "
        "normalized MessageEvent. See docs/relay-connector-contract.md §6."
    )


def test_relay_package_calls_no_signature_verification():
    """No relay module references a PLATFORM signature/crypto-verification symbol.

    Scoped to platform crypto (Discord ed25519, Twilio/WeCom HMAC, webhook
    signature checks). The connector⇄gateway channel authenticator (``auth.py``)
    is exempt: its HMAC is over the connector's own per-gateway/per-tenant
    secrets to authenticate the relay channel itself — the gateway holds NO
    platform secret and re-validates NO platform payload. See ``auth.py`` and
    docs/connector-gateway-auth-design.md.
    """
    offenders: list[str] = []
    for path in _relay_py_files():
        if path.name in _CHANNEL_AUTH_FILES:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            # Skip comments / docstrings-as-prose: only flag code-like usage.
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            m = _FORBIDDEN_SYMBOL_RE.search(line)
            if m:
                offenders.append(f"{path.name}:{lineno}: '{m.group(0)}' in: {stripped[:80]}")
    assert not offenders, (
        "The relay path must not perform platform signature/crypto verification "
        "(A2). Found verification-symbol references:\n  "
        + "\n  ".join(offenders)
        + "\nThe connector verifies at the edge; the gateway re-validates nothing."
    )


def test_channel_auth_uses_only_stdlib_crypto_not_platform_modules():
    """auth.py (channel authenticator) imports only stdlib crypto, no platform crypto.

    Positive guard: the connector⇄gateway channel auth is allowed to do HMAC,
    but it must do so with stdlib primitives over connector-owned secrets — it
    must never reach for a platform-crypto module. This keeps the exemption
    above honest (auth.py can't smuggle in platform verification).
    """
    auth_py = _RELAY_PKG / "auth.py"
    assert auth_py.is_file(), "gateway/relay/auth.py (channel authenticator) is missing"
    tree = ast.parse(auth_py.read_text(encoding="utf-8"), filename=str(auth_py))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    # No platform-crypto module import.
    assert not [m for m in imported if any(tok in m for tok in _FORBIDDEN_MODULE_TOKENS)], (
        f"auth.py must not import platform crypto; imports={imported}"
    )
    # It does use stdlib hmac/hashlib (that's how it authenticates the channel).
    assert "hmac" in imported and "hashlib" in imported, (
        f"auth.py should authenticate the channel with stdlib hmac/hashlib; imports={imported}"
    )

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
    """No relay module references a signature/crypto-verification symbol by name."""
    offenders: list[str] = []
    for path in _relay_py_files():
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

"""Merged Windows CA bundle builder — fail-open (Task 1 of 2 for issue #43294).

WHY: On corporate-managed Windows, the OS trust store holds the private root CA,
but httpx's default ``verify=True`` pins certifi's frozen Mozilla bundle
(``ssl.create_default_context(cafile=certifi.where())``). TLS to internal
corporate FQDNs then fails with CERTIFICATE_VERIFY_FAILED. This module builds a
MERGED PEM bundle — Windows ROOT + CA store certs (filtered for server auth) +
certifi Mozilla roots — so callers (Task 2) can point CA seams at it on win32.

FAIL-OPEN CONTRACT: Every failure logs a warning via
``logging.getLogger(__name__)`` and returns ``None``. Callers treat ``None``
as "keep current behavior".
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
import ssl
import sys
import tempfile
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# Module-level memo: build once per process. Fresh on each new process so
# store state (cert rotation, revocation) is re-read.
_bundle_path: str | None = None
_bundle_lock = threading.Lock()

# Security-relevant server-auth OID from CPython ssl.
_SERVER_AUTH_OID = "1.3.6.1.5.5.7.3.1"


def _pem_from_store_entries(entries: list[tuple[bytes, str, bool | frozenset | tuple]]) -> list[str]:
    """Pure helper: build PEM blocks from enum-style store entry tuples.

    Args:
        entries: (cert_der_bytes, encoding_str, trust) as yielded by
            ssl.enum_certificates() on Windows — trust is True or a
            frozenset of OID strings (CPython Modules/_ssl.c parseKeyUsage
            builds it with PyFrozenSet_New).

    Returns:
        List of PEM-formatted certificate strings (no file I/O, no sys.gate).
    """
    seen: set[str] = set()
    blocks: list[str] = []
    for der_bytes, encoding, trust in entries:
        # Mirror CPython ssl.SSLContext.load_default_certs(Purpose.SERVER_AUTH):
        # skip pkcs_7_asn; accept only x509_asn.
        if encoding != "x509_asn":
            continue
        # Trust filtering mirrors Lib/ssl.py _load_windows_store_certs:
        # `trust is True or purpose.oid in trust`. trust is True or a
        # frozenset of OIDs (never a tuple — Modules/_ssl.c:5487); membership
        # in any iterable of OIDs is what matters, so no isinstance gate here.
        trusted = False
        if trust is True:
            trusted = True
        else:
            try:
                trusted = _SERVER_AUTH_OID in trust
            except TypeError:
                trusted = False
        if not trusted:
            continue
        # Dedup by SHA-256 of the DER bytes.
        digest = hashlib.sha256(der_bytes).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        # DER -> base64 -> PEM block.
        b64_data = base64.b64encode(der_bytes).decode("ascii")
        lines = ["-----BEGIN CERTIFICATE-----"]
        for i in range(0, len(b64_data), 64):
            lines.append(b64_data[i:i + 64])
        lines.append("-----END CERTIFICATE-----")
        lines.append("")
        blocks.append("\n".join(lines) + "\n")
    return blocks


def windows_merged_ca_bundle() -> str | None:
    """Return cached merged PEM bundle path, or None on any failure (fail-open).

    Gate: ``sys.platform == "win32"`` FIRST — never builds on non-Windows.
    Cache written to ``Path(get_hermes_home()) / "cache" / "windows-ca-bundle.pem"``.
    Module-level memo + threading.Lock ensures concurrent callers get one build.
    """
    global _bundle_path

    # Gate FIRST — monkeypatchable because we read sys.platform inside the
    # function each call (not at import time).
    if sys.platform != "win32":
        return None

    with _bundle_lock:
        if _bundle_path is not None:
            return _bundle_path

        try:
            from hermes_constants import get_hermes_home
        except Exception as exc:
            logger.warning("agent.win_ca_bundle: cannot import get_hermes_home: %s", exc)
            return None

        try:
            # 1. Enumerate Windows store certs.
            store_entries: list[tuple[bytes, str, bool | frozenset | tuple]] = []
            try:
                for store in ("ROOT", "CA"):
                    for cert_der, encoding, trust in ssl.enum_certificates(store):
                        # Note: ssl.enum_certificates returns (der_bytes, encoding, trust)
                        # where trust is True or a frozenset of OID strings.
                        store_entries.append((cert_der, encoding, trust))
            except Exception as exc:
                logger.warning("agent.win_ca_bundle: store enumeration failed: %s", exc)
                # Empty store entries is acceptable only if we can fall back to
                # certifi-only; but the contract says return None when store
                # yields nothing *usable*. We'll still try certifi append.

            # 2. Build PEM from store entries via pure helper.
            pem_blocks = _pem_from_store_entries(store_entries)

            # 3. Append certifi Mozilla roots (already PEM). If import fails,
            # a store-only bundle (possibly empty if store yielded nothing)
            # is acceptable per spec — but if empty and certifi missing,
            # validation will catch it.
            certifi_content = ""
            try:
                import certifi
                cert_path = Path(certifi.where())
                certifi_content = cert_path.read_text(encoding="utf-8")
            except Exception as exc:
                logger.warning("agent.win_ca_bundle: certifi import/read failed: %s", exc)

            # Assemble: store PEM blocks first, then raw certifi PEM content.
            assembled_parts = pem_blocks.copy()
            if certifi_content:
                assembled_parts.append(certifi_content)
            if not assembled_parts:
                logger.warning("agent.win_ca_bundle: no PEM blocks assembled (empty store + no certifi)")
                return None

            bundle_text = "".join(assembled_parts)

            # 4. Resolve cache path inside function (never module-level constant).
            cache_path = Path(get_hermes_home()) / "cache" / "windows-ca-bundle.pem"
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                logger.warning("agent.win_ca_bundle: cannot create cache dir: %s", exc)
                return None

            # 5-6. Write + validate via a private temp file, then os.replace() into
            # place. Concurrent Hermes processes share this cache path; a process's
            # memoized path must never dangle, so validation failure may only unlink
            # the TEMP file (never a final path another process may have loaded),
            # and os.replace() makes the swap atomic on the final name.
            fd, tmp_name = tempfile.mkstemp(
                dir=str(cache_path.parent), prefix=".windows-ca-bundle-", suffix=".pem.tmp"
            )
            tmp_path = Path(tmp_name)
            published = False
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
                    tmp_file.write(bundle_text)
                try:
                    ctx = ssl.create_default_context(cafile=str(tmp_path))
                    # Content check BEFORE publishing: a syntactically valid but
                    # empty bundle must never reach the final path.
                    try:
                        loaded = ctx.get_ca_certs()
                        if not loaded and not pem_blocks:
                            logger.warning(
                                "agent.win_ca_bundle: bundle validated but loaded 0 certificates — refusing to publish"
                            )
                            return None  # finally unlinks the private temp; final path untouched
                    except NotImplementedError:
                        pass  # truststore-backed; creation success is sufficient validation
                except Exception as exc:
                    logger.warning(
                        "agent.win_ca_bundle: validation with ssl.create_default_context failed: %s", exc
                    )
                    return None  # finally unlinks the private temp; final path untouched
                os.replace(tmp_name, cache_path)
                published = True
            except OSError as exc:
                logger.warning("agent.win_ca_bundle: cannot publish bundle to %s: %s", cache_path, exc)
                return None
            finally:
                if not published:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except OSError:
                        pass

            # Cache memo. Log honestly: a certifi-only bundle (store enum failed or
            # yielded nothing) is NOT the merged bundle the caller asked for — warn
            # so a corporate box can tell the difference in agent.log.
            _bundle_path = str(cache_path)
            if pem_blocks:
                logger.info("agent.win_ca_bundle: merged CA bundle built at %s", _bundle_path)
            else:
                logger.warning(
                    "agent.win_ca_bundle: certifi-only bundle written to %s — "
                    "no certificates loaded from the Windows store", _bundle_path
                )
            return _bundle_path

        except Exception as exc:
            logger.warning("agent.win_ca_bundle: unexpected failure building merged CA bundle: %s", exc)
            return None

"""End-to-end test for Matrix cross-signing auto-bootstrap.

Spins a real Continuwuity homeserver in docker, registers a fresh bot,
runs the patched ``MatrixAdapter.connect()`` against it, and asserts:

  1. cross-signing keys get published with **unpadded** base64 keyids
     (the bug this PR fixes — padded keyids are silently rejected by
     matrix-rust-sdk in Element);
  2. on a second startup with the same crypto store, bootstrap is
     skipped (``get_own_cross_signing_public_keys`` finds the keys);
  3. the bot's current device is signed by the new SSK, so Element
     considers the device "verified by its owner".

Self-contained: ``docker compose up -d`` brings up Continuwuity on
127.0.0.1:26167; this script registers a fresh bot using the
homeserver's one-time admin registration token (printed once at first
boot, parsed from the container logs); then drives the gateway code.

Run from repo root::

    docker compose -f tests/e2e/matrix_xsign_bootstrap/docker-compose.yml up -d
    python tests/e2e/matrix_xsign_bootstrap/test_bootstrap.py
    docker compose -f tests/e2e/matrix_xsign_bootstrap/docker-compose.yml down -v

Skipped automatically if mautrix isn't installed or the homeserver
isn't reachable.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

HS = os.environ.get("E2E_MATRIX_HS", "http://127.0.0.1:26167")
COMPOSE_DIR = Path(__file__).parent
CONTAINER_NAME = "matrix_xsign_bootstrap-homeserver-1"


def _hs_reachable() -> bool:
    try:
        urllib.request.urlopen(f"{HS}/_matrix/client/versions", timeout=2).read()
        return True
    except Exception:
        return False


def _first_time_token() -> str | None:
    """Continuwuity prints a one-time registration token on first boot.

    The configured CONTINUWUITY_REGISTRATION_TOKEN does NOT activate
    until an account exists, so we have to pull this token out of the
    docker logs to bootstrap the very first user.
    """
    try:
        out = subprocess.run(
            ["docker", "logs", CONTAINER_NAME],
            capture_output=True, text=True, check=True,
        ).stdout + subprocess.run(
            ["docker", "logs", CONTAINER_NAME],
            capture_output=True, text=True, check=True,
        ).stderr
    except Exception:
        return None
    cleaned = re.sub(r"\x1b\[[0-9;]*m", "", out)
    m = re.search(r"registration token ([A-Za-z0-9]+)", cleaned)
    return m.group(1) if m else None


def _post_json(url: str, body: dict, headers: dict | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        r = urllib.request.urlopen(req)
        return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


CONFIG_REG_TOKEN = "testreg"  # matches docker-compose.yml


def _register_bot(*, prefer_token: str = CONFIG_REG_TOKEN, fallback_token: str | None = None) -> dict:
    """Register a fresh bot. Tries the configured token first; falls back to
    the homeserver's one-time admin token (only valid until the first user
    is created)."""
    user = "bot" + secrets.token_hex(3)
    password = secrets.token_urlsafe(20)
    last_err = None
    for tok in (prefer_token, fallback_token):
        if tok is None:
            continue
        st, b = _post_json(f"{HS}/_matrix/client/v3/register", {})
        if st != 401 or "session" not in b:
            last_err = (st, b); continue
        session = b["session"]
        st, b = _post_json(f"{HS}/_matrix/client/v3/register", {
            "auth": {"type": "m.login.registration_token", "token": tok, "session": session},
            "username": user, "password": password,
            "initial_device_display_name": "e2e-bootstrap-test",
        })
        if st == 200:
            return b
        last_err = (st, b)
    raise AssertionError(f"register failed for both tokens: {last_err}")


def _query_keys(token: str, mxid: str) -> dict:
    return _post_json(
        f"{HS}/_matrix/client/v3/keys/query",
        {"device_keys": {mxid: []}},
        headers={"Authorization": f"Bearer {token}"},
    )[1]


@unittest.skipUnless(_hs_reachable(), f"homeserver not reachable at {HS}")
class XsignBootstrapE2E(unittest.IsolatedAsyncioTestCase):
    """Drive the patched MatrixAdapter.connect() against real continuwuity."""

    @classmethod
    def setUpClass(cls):
        try:
            import mautrix  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("mautrix not installed")
        cls.first_tok = _first_time_token()
        # If no user has ever been created, the configured `testreg` token
        # won't activate yet — burn the one-time admin token first to
        # bootstrap the homeserver into a usable state.
        if cls.first_tok:
            try:
                _register_bot(prefer_token=cls.first_tok, fallback_token=None)
            except AssertionError:
                pass  # Already burnt previously; testreg should now work.

    async def _connect_with_bootstrap(self, creds: dict, store_dir: Path) -> tuple[list[str], str | None]:
        """Drive matrix.py's bootstrap branch directly.

        We import the gateway module and execute the same OlmMachine init +
        bootstrap sequence, capturing log lines so we can assert what fired.
        Returns (log_lines, recovery_key_or_None).
        """
        from mautrix.api import HTTPAPI
        from mautrix.client import Client
        from mautrix.client.state_store.memory import MemoryStateStore
        from mautrix.crypto import OlmMachine, PgCryptoStore
        from mautrix.types import TrustState
        from mautrix.util.async_db import Database

        # The actual bootstrap snippet from gateway/platforms/matrix.py
        # (copied so we can run it without importing the full hermes
        # gateway and its many deps). If the source code drifts from this,
        # the test should be updated to match.
        log_lines: list[str] = []
        captured_recovery_key: str | None = None

        class _Capture(logging.Handler):
            def emit(self, record):
                log_lines.append(self.format(record))

        logger = logging.getLogger("e2e.bootstrap")
        logger.setLevel(logging.DEBUG)
        handler = _Capture()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)

        api = HTTPAPI(base_url=creds["homeserver"], token=creds["access_token"])
        client = Client(
            mxid=creds["user_id"], api=api,
            device_id=creds["device_id"], state_store=MemoryStateStore(),
        )
        client.api.token = creds["access_token"]

        store_dir.mkdir(parents=True, exist_ok=True)
        db_path = store_dir / "crypto.db"
        crypto_db = Database.create(f"sqlite:///{db_path}", upgrade_table=PgCryptoStore.upgrade_table)
        await crypto_db.start()
        crypto_store = PgCryptoStore(account_id=creds["user_id"], pickle_key="e2e-test", db=crypto_db)
        await crypto_store.open()

        olm = OlmMachine(client, crypto_store, MemoryStateStore())
        olm.share_keys_min_trust = TrustState.UNVERIFIED
        olm.send_keys_min_trust = TrustState.UNVERIFIED
        await olm.load()

        # --- The patched bootstrap block, mirrored from matrix.py ---
        recovery_key = os.getenv("MATRIX_RECOVERY_KEY", "").strip()
        if recovery_key:
            try:
                await olm.verify_with_recovery_key(recovery_key)
                logger.info("Matrix: cross-signing verified via recovery key")
            except Exception as exc:
                logger.warning("Matrix: recovery key verification failed: %s", exc)
        else:
            try:
                own_xsign = await olm.get_own_cross_signing_public_keys()
            except Exception as exc:
                own_xsign = None
                logger.warning("Matrix: cross-signing key lookup failed: %s", exc)
            if own_xsign is None:
                try:
                    new_recovery_key = await olm.generate_recovery_key()
                    captured_recovery_key = new_recovery_key
                    logger.warning(
                        "Matrix: bootstrapped cross-signing for %s. "
                        "SAVE THIS RECOVERY KEY: %s",
                        client.mxid, new_recovery_key,
                    )
                except Exception as exc:
                    logger.warning("Matrix: cross-signing bootstrap failed: %s", exc)

        # --- /end patched block ---
        # Clean teardown — without this the asyncio loop never exits.
        await crypto_db.stop()
        await api.session.close()
        return log_lines, captured_recovery_key

    async def asyncSetUp(self):
        self.creds = _register_bot(prefer_token=CONFIG_REG_TOKEN, fallback_token=self.first_tok)
        self.creds["homeserver"] = HS
        self.tmp = Path(tempfile.mkdtemp(prefix="e2e-xsign-"))
        # mautrix.generate_recovery_key requires account.shared, which means
        # we must share device keys (one-time keys) first. Do that via a
        # short bootstrap to publish device keys.
        await self._publish_device_keys(self.creds, self.tmp)

    async def _publish_device_keys(self, creds, store_dir):
        """Tiny helper: open OlmMachine, share device keys, close."""
        from mautrix.api import HTTPAPI
        from mautrix.client import Client
        from mautrix.client.state_store.memory import MemoryStateStore
        from mautrix.crypto import OlmMachine, PgCryptoStore
        from mautrix.util.async_db import Database

        api = HTTPAPI(base_url=creds["homeserver"], token=creds["access_token"])
        client = Client(mxid=creds["user_id"], api=api, device_id=creds["device_id"],
                        state_store=MemoryStateStore())
        store_dir.mkdir(parents=True, exist_ok=True)
        crypto_db = Database.create(f"sqlite:///{store_dir / 'crypto.db'}",
                                    upgrade_table=PgCryptoStore.upgrade_table)
        await crypto_db.start()
        crypto_store = PgCryptoStore(account_id=creds["user_id"], pickle_key="e2e-test", db=crypto_db)
        await crypto_store.open()
        olm = OlmMachine(client, crypto_store, MemoryStateStore())
        await olm.load()
        await olm.share_keys()  # publishes device keys (precondition for generate_recovery_key)
        await crypto_db.stop()
        await api.session.close()

    async def asyncTearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    async def test_bootstrap_publishes_unpadded_keys(self):
        """Fresh bot → bootstrap fires, keys published unpadded, device signed."""
        log_lines, rec_key = await self._connect_with_bootstrap(self.creds, self.tmp)
        # 1. Bootstrap must have produced a recovery key
        self.assertIsNotNone(rec_key, "expected recovery key from bootstrap")
        self.assertTrue(any("bootstrapped cross-signing" in l for l in log_lines),
                        f"expected bootstrap log line, got: {log_lines}")
        # 2. Homeserver should now serve a master + ssk for the bot
        d = _query_keys(self.creds["access_token"], self.creds["user_id"])
        self.assertIn(self.creds["user_id"], d.get("master_keys", {}),
                      "no master_keys after bootstrap")
        self.assertIn(self.creds["user_id"], d.get("self_signing_keys", {}),
                      "no self_signing_keys after bootstrap")
        # 3. The keyids must be UNPADDED (this is the bug this PR exists to fix)
        master_kid = next(iter(d["master_keys"][self.creds["user_id"]]["keys"]))
        ssk_kid = next(iter(d["self_signing_keys"][self.creds["user_id"]]["keys"]))
        self.assertFalse(master_kid.endswith("="),
                         f"master keyid is padded: {master_kid!r}")
        self.assertFalse(ssk_kid.endswith("="),
                         f"ssk keyid is padded: {ssk_kid!r}")
        # 4. The current device must be signed by the new SSK
        dev = d["device_keys"][self.creds["user_id"]][self.creds["device_id"]]
        sig_kids = list(dev["signatures"][self.creds["user_id"]].keys())
        self.assertIn(ssk_kid, sig_kids,
                      f"device {self.creds['device_id']} not signed by new SSK; "
                      f"signatures: {sig_kids}")

    async def test_second_startup_skips_bootstrap(self):
        """Second startup with same crypto store → no second recovery key."""
        # First connect bootstraps.
        _, rec1 = await self._connect_with_bootstrap(self.creds, self.tmp)
        self.assertIsNotNone(rec1, "first connect should have bootstrapped")
        # Second connect on same crypto store should NOT re-bootstrap.
        log2, rec2 = await self._connect_with_bootstrap(self.creds, self.tmp)
        self.assertIsNone(rec2, f"second connect re-bootstrapped! logs: {log2}")
        self.assertFalse(any("bootstrapped cross-signing" in l for l in log2),
                         f"second connect re-bootstrapped! logs: {log2}")

    async def test_recovery_key_path_takes_precedence(self):
        """If MATRIX_RECOVERY_KEY is set, no fresh bootstrap happens."""
        # First, bootstrap to get a real recovery key.
        _, rec_key = await self._connect_with_bootstrap(self.creds, self.tmp)
        self.assertIsNotNone(rec_key)
        # Fresh store directory + recovery key set in env: must take the
        # verify_with_recovery_key path, NOT bootstrap a new identity.
        fresh_store = Path(tempfile.mkdtemp(prefix="e2e-xsign-fresh-"))
        try:
            await self._publish_device_keys(self.creds, fresh_store)
            os.environ["MATRIX_RECOVERY_KEY"] = rec_key
            try:
                log, rec2 = await self._connect_with_bootstrap(self.creds, fresh_store)
                self.assertIsNone(rec2, "bootstrap fired despite MATRIX_RECOVERY_KEY being set")
                self.assertTrue(
                    any("verified via recovery key" in l for l in log),
                    f"expected recovery-key verify log, got: {log}",
                )
            finally:
                del os.environ["MATRIX_RECOVERY_KEY"]
        finally:
            shutil.rmtree(fresh_store, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Tests for keyed pseudonymization of the shared-metrics install identity.

The load-bearing property: install_id must never be transmitted, and the
value that IS transmitted must stay stable for a package even across a salt
rotation, or a retry would change the body under an already-used package_id.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from hermes_cli.observability.shared_metrics_identity import (
    ROTATION_INTERVAL,
    SALT_ISSUED_AT_KEY,
    SALT_KEY,
    current_salt,
    derive_install_id,
    substitute_install_id,
)

INSTALL_ID = "12a73e97-4de9-4766-830d-9ca1192c0420"
T0 = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def connection():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE telemetry_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    yield conn
    conn.close()


class TestSaltLifecycle:
    def test_first_call_generates_a_salt(self, connection):
        salt = current_salt(connection, now=T0)
        assert len(salt) == 64  # 32 bytes hex
        assert int(salt, 16) >= 0  # valid hex

    def test_salt_is_stable_within_the_window(self, connection):
        first = current_salt(connection, now=T0)
        later = current_salt(connection, now=T0 + timedelta(days=29, hours=23))
        assert first == later

    def test_salt_rotates_after_the_interval(self, connection):
        first = current_salt(connection, now=T0)
        after = current_salt(connection, now=T0 + ROTATION_INTERVAL + timedelta(seconds=1))
        assert first != after

    def test_salt_is_persisted(self, connection):
        salt = current_salt(connection, now=T0)
        stored = connection.execute(
            "SELECT value FROM telemetry_state WHERE key = ?", (SALT_KEY,)
        ).fetchone()[0]
        assert stored == salt

    def test_issued_at_is_recorded(self, connection):
        current_salt(connection, now=T0)
        stored = connection.execute(
            "SELECT value FROM telemetry_state WHERE key = ?", (SALT_ISSUED_AT_KEY,)
        ).fetchone()[0]
        assert stored.startswith("2026-08-26T12:00")

    def test_two_installs_get_different_salts(self):
        salts = set()
        for _ in range(5):
            conn = sqlite3.connect(":memory:")
            conn.execute(
                "CREATE TABLE telemetry_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            salts.add(current_salt(conn, now=T0))
            conn.close()
        assert len(salts) == 5, "salts must be random per install, not derived"

    def test_clock_rollback_reissues_rather_than_trusting_the_stamp(self, connection):
        """A future issued_at means the clock moved; the age is unknowable.

        Reissuing is the safe direction — it shortens linkability rather than
        extending it, and packages already prepared keep their frozen id.
        """
        first = current_salt(connection, now=T0)
        rolled_back = current_salt(connection, now=T0 - timedelta(days=5))
        assert rolled_back != first

    def test_corrupt_issued_at_reissues_rather_than_crashing(self, connection):
        current_salt(connection, now=T0)
        connection.execute(
            "UPDATE telemetry_state SET value = 'not-a-date' WHERE key = ?",
            (SALT_ISSUED_AT_KEY,),
        )
        assert current_salt(connection, now=T0) is not None


class TestDerivation:
    def test_derivation_is_deterministic(self):
        salt = "a" * 64
        assert derive_install_id(INSTALL_ID, salt) == derive_install_id(INSTALL_ID, salt)

    def test_derivation_hides_the_install_id(self):
        derived = derive_install_id(INSTALL_ID, "a" * 64)
        assert INSTALL_ID not in derived
        assert derived != INSTALL_ID

    def test_different_salts_give_different_values(self):
        assert derive_install_id(INSTALL_ID, "a" * 64) != derive_install_id(
            INSTALL_ID, "b" * 64
        )

    def test_different_installs_give_different_values(self):
        salt = "a" * 64
        assert derive_install_id(INSTALL_ID, salt) != derive_install_id("other", salt)

    def test_output_shape_is_sha256_hex(self):
        derived = derive_install_id(INSTALL_ID, "a" * 64)
        assert len(derived) == 64
        int(derived, 16)


class TestSubstitution:
    def _package(self):
        return {
            "schema_version": "hermes.shared_metrics.v2",
            "package_id": "3a63d27e-f170-4d4c-8c4d-ebd80feac592",
            "install_id": INSTALL_ID,
            "generated_at": "2026-08-26T01:01:25.311956Z",
            "period_start": "2026-08-26T00:00:00Z",
            "period_end": "2026-08-27T00:00:00Z",
            "resource": {"hermes_version": "0.20.5", "os_family": "macos"},
            "metrics": [{"name": "hermes.client.active", "type": "counter", "value": 1}],
        }

    def test_install_id_is_replaced(self):
        result = substitute_install_id(self._package(), "derived-value")
        assert result["install_id"] == "derived-value"

    def test_no_other_field_changes(self):
        original = self._package()
        result = substitute_install_id(original, "derived-value")
        for key in original:
            if key != "install_id":
                assert result[key] == original[key]

    def test_the_caller_dict_is_not_mutated(self):
        original = self._package()
        substitute_install_id(original, "derived-value")
        assert original["install_id"] == INSTALL_ID

    def test_no_fields_are_added_or_removed(self):
        original = self._package()
        assert set(substitute_install_id(original, "x")) == set(original)

    def test_the_raw_install_id_never_survives_substitution(self):
        import json

        body = json.dumps(substitute_install_id(self._package(), "derived-value"))
        assert INSTALL_ID not in body


class TestRetryStability:
    """The property that keeps retries contract-compliant."""

    def test_a_frozen_derived_id_survives_a_rotation(self, connection):
        salt_before = current_salt(connection, now=T0)
        frozen = derive_install_id(INSTALL_ID, salt_before)

        # Time passes, the salt rotates, and the package is retried.
        salt_after = current_salt(connection, now=T0 + ROTATION_INTERVAL + timedelta(days=1))
        assert salt_after != salt_before

        # Rebuilding from the FROZEN value reproduces identical bytes; deriving
        # afresh would not.
        assert substitute_install_id({"install_id": INSTALL_ID}, frozen) == {
            "install_id": frozen
        }
        assert derive_install_id(INSTALL_ID, salt_after) != frozen

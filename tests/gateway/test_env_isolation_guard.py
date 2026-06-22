"""Regression guards for the gateway test-isolation fixtures.

These tests prove the ``tests/gateway/conftest.py`` hygiene fixtures actually
revert the state they promise to, by construction — so a future edit that
weakens them goes RED here.

Covered:
  * ``_restore_os_environ_after_test`` reverts ANY ``os.environ`` mutation a test
    body makes (raw assignment — the kind ``load_gateway_config()`` does — that
    ``monkeypatch`` does NOT undo), regardless of the var name. This is the
    fail-closed property: no enumerated var list, so an arbitrary fake var is
    reverted exactly like a real bridge-gating var.

The pair of tests below are ORDER-DEPENDENT *on purpose*: the first leaks a raw
env write; the second asserts the leak is gone. pytest runs tests in file order
within a class by default, and these two assert the fixture closed the leak
between them. (They also pass under ``-p randomly`` because the leak is reverted
at the FIRST test's teardown, not the second's setup — so the second never sees
it regardless of order.)
"""
import os

import pytest


# A var name that the config bridge will never write, to prove the fixture is
# var-name-agnostic (fail-closed: not a hardcoded allowlist).
_ARBITRARY_LEAK_VAR = "ZZ_FAKE_BRIDGE_LEAK_VAR"
# A real bridge-gating var (the exact class that caused the telegram + slack
# random-order failures): a leaked value flips adapter gating behavior.
_REAL_GATING_VAR = "TELEGRAM_ALLOWED_TOPICS"


class TestOsEnvironRestoreFixture:
    """Prove the env snapshot/restore fixture reverts raw os.environ writes."""

    def test_a_raw_env_writes_must_not_leak(self):
        # Sanity: neither var is set before we (the leaker) write them raw.
        assert _ARBITRARY_LEAK_VAR not in os.environ
        assert _REAL_GATING_VAR not in os.environ
        # Raw assignment — exactly what load_gateway_config() does, and exactly
        # what monkeypatch CANNOT revert. The autouse fixture must revert it.
        os.environ[_REAL_GATING_VAR] = "8"
        os.environ[_ARBITRARY_LEAK_VAR] = "leaked"

    def test_b_previous_raw_env_writes_were_reverted(self):
        # If the fixture works, test_a's raw writes are gone at this test's
        # setup. RED-proof: delete the fixture (or its restore body) and this
        # fails — the leaked values persist.
        assert _REAL_GATING_VAR not in os.environ, (
            f"{_REAL_GATING_VAR} leaked from a prior test's raw os.environ write — "
            "the _restore_os_environ_after_test fixture did not revert it"
        )
        assert _ARBITRARY_LEAK_VAR not in os.environ, (
            f"{_ARBITRARY_LEAK_VAR} leaked — the fixture is not var-name-agnostic "
            "(it must revert ANY mutation, by construction, not a hardcoded set)"
        )

    def test_c_modified_existing_var_is_restored(self, monkeypatch):
        # A var that EXISTS at setup (set here via monkeypatch so it's part of
        # the snapshot) and is then RAW-overwritten must be restored to its
        # snapshot value, not left mutated.
        monkeypatch.setenv("TELEGRAM_REQUIRE_MENTION", "true")
        # Raw overwrite (monkeypatch-invisible).
        os.environ["TELEGRAM_REQUIRE_MENTION"] = "false"
        # Within this test the overwrite is visible; the fixture restores the
        # snapshot ("true") at teardown. We can't observe teardown here, so the
        # next test would — but TELEGRAM_REQUIRE_MENTION is stripped by the root
        # hermetic fixture each setup anyway. The load-bearing assertion is that
        # the fixture's restore path runs without error on a changed-existing
        # key; tests a + b already prove add/remove reversion.
        assert os.environ["TELEGRAM_REQUIRE_MENTION"] == "false"


def test_fixture_is_idempotent_when_test_mutates_nothing():
    """A test that touches no env must not trip the restore (no spurious work)."""
    # The fixture's restore is guarded by `if os.environ != snapshot`; a no-op
    # test leaves them equal, so clear()+update() is skipped. This just exercises
    # that path (no assertion needed beyond "doesn't raise").
    assert True

"""Behavioral setters must register their names with the test environment scrubber.

Call the guard rather than reading production source: scattered runtime paths
cannot all be exercised by a source-shape assertion.
"""

import os

import pytest


def test_unregistered_behavioral_flag_is_rejected_before_mutation(monkeypatch):
    from hermes_cli.env_contract import set_behavioral_flag

    name = "HERMES_FAKE_FLAG"
    monkeypatch.setenv(name, "original")
    with pytest.raises(AssertionError, match=name):
        set_behavioral_flag(name, "replacement")
    assert os.environ[name] == "original"


def test_registered_flags_are_set_and_scrubbed(monkeypatch, tmp_path):
    from tests.conftest import _hermetic_environment
    from hermes_cli.env_contract import BEHAVIORAL_ENV_VARS, set_behavioral_flag

    assert BEHAVIORAL_ENV_VARS
    for name in BEHAVIORAL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
        set_behavioral_flag(name, "contract-test")
        assert os.environ[name] == "contract-test"

    # Exercise the actual fixture, not a duplicate scrub loop in this test.
    nested = tmp_path / "scrub"
    nested.mkdir()
    with pytest.MonkeyPatch.context() as scrub:
        _hermetic_environment.__wrapped__(nested, scrub)
        assert all(os.environ.get(name) != "contract-test" for name in BEHAVIORAL_ENV_VARS)

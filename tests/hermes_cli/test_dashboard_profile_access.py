import pytest
from fastapi import HTTPException

from hermes_cli.dashboard_auth import profile_access
from hermes_cli.dashboard_auth.profile_access import Identity


@pytest.fixture
def checker():
    def only_own(identity, profile):
        if identity is None:
            return None
        return profile == identity.user_id

    profile_access.register_checker(only_own)
    yield only_own
    profile_access.unregister_checker(only_own)


def test_no_checker_allows_everything():
    assert profile_access.is_allowed("anything")


def test_bound_identity_is_checked(checker):
    token = profile_access.bind_identity(Identity(user_id="alice", provider="test"))
    try:
        assert profile_access.is_allowed("alice")
        assert not profile_access.is_allowed("bob")
        assert profile_access.filter_names(["alice", "bob"]) == ["alice"]
        with pytest.raises(HTTPException) as exc:
            profile_access.require("bob")
        assert exc.value.status_code == 403
    finally:
        profile_access.reset_identity(token)


def test_abstaining_checker_allows(checker):
    assert profile_access.is_allowed("bob")


def test_failing_checker_denies():
    def broken(identity, profile):
        raise RuntimeError("store down")

    profile_access.register_checker(broken)
    try:
        assert not profile_access.is_allowed("alice")
    finally:
        profile_access.unregister_checker(broken)


def test_ws_stamp_identity():
    assert profile_access.identity_from_ws_stamp(None) is None
    assert profile_access.identity_from_ws_stamp({"user_id": "u1", "provider": "p"}).user_id == "u1"

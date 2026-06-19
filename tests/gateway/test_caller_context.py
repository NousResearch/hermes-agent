from gateway.caller_context import get_caller, reset_caller, set_caller


def test_caller_context_roundtrip():
    token = set_caller("slack", "U07TCQBDPMJ")
    try:
        caller = get_caller()
        assert caller is not None
        assert caller.provider == "slack"
        assert caller.external_id == "U07TCQBDPMJ"
    finally:
        reset_caller(token)

    assert get_caller() is None


def test_empty_caller_is_anonymous():
    token = set_caller("slack", "")
    try:
        assert get_caller() is None
    finally:
        reset_caller(token)

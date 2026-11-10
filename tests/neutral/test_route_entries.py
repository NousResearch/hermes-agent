"""The shared rules' session source and key builder keep main's ``shared_session`` flag (``gateway.session``)."""
from session_rules.route_entries import Platform, SessionSource, build_session_key


def _group(**extra):
    return SessionSource(platform=Platform.TELEGRAM, chat_id="-100", chat_type="group", user_id="111", **extra)


def test_shared_session_round_trips_and_is_absent_by_default():
    shared = _group(shared_session=True)
    assert shared.to_dict()["shared_session"] is True
    assert SessionSource.from_dict(shared.to_dict()).shared_session is True
    assert "shared_session" not in _group().to_dict()
    assert SessionSource.from_dict(_group().to_dict()).shared_session is False


def test_a_shared_group_source_keys_the_group_session_though_it_keeps_its_sender():
    from gateway.session import SessionSource as GatewaySource, build_session_key as gateway_key

    assert build_session_key(_group()) == "agent:main:telegram:group:-100:111"
    assert build_session_key(_group(shared_session=True)) == "agent:main:telegram:group:-100"
    main_source = GatewaySource.from_dict(_group(shared_session=True).to_dict())
    assert build_session_key(_group(shared_session=True)) == gateway_key(main_source)

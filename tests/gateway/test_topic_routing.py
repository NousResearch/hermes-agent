from gateway.config import Platform
from gateway.session import SessionSource
from gateway.topic_routing import TopicRoute, route_from_session_source


def test_route_from_telegram_topic_is_strict():
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="413",
        chat_topic="舆情监测",
    )

    route = route_from_session_source(source)

    assert route == TopicRoute(
        chat_id="-1001",
        thread_id="413",
        topic_name="舆情监测",
        boundary="strict",
    )
    assert route.is_strict is True
    assert route.to_metadata() == {
        "thread_id": "413",
        "topic_name": "舆情监测",
        "topic_boundary": "strict",
    }


def test_route_from_non_threaded_group_is_soft():
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
    )

    route = route_from_session_source(source)

    assert route == TopicRoute(
        chat_id="-1001",
        thread_id=None,
        topic_name=None,
        boundary="soft",
    )
    assert route.is_strict is False
    assert route.to_metadata() == {"topic_boundary": "soft"}


def test_route_from_discord_thread_is_soft():
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="123",
        chat_type="thread",
        thread_id="abc",
        chat_topic="dev thread",
    )

    route = route_from_session_source(source)

    assert route == TopicRoute(
        chat_id="123",
        thread_id="abc",
        topic_name="dev thread",
        boundary="soft",
    )
    assert route.is_strict is False


def test_route_from_telegram_dm_topic_is_strict():
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="42",
        chat_type="dm",
        thread_id="9",
        chat_topic="私人项目",
    )

    route = route_from_session_source(source)

    assert route == TopicRoute(
        chat_id="42",
        thread_id="9",
        topic_name="私人项目",
        boundary="strict",
    )
    assert route.is_strict is True

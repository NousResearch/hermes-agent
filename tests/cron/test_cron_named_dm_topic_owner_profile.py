import asyncio
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from cron.scheduler import _deliver_result
from gateway.config import Platform, PlatformConfig


def test_shared_bot_named_topic_is_persisted_in_the_bot_owners_config(tmp_path, monkeypatch):
    """A credentialless satellite's cron job delivers through the primary's bot (exact profile route).
    The topic that bot creates, and the one the router recreates after thread-not-found, belong to the
    bot's profile: the satellite's config.yaml must not gain Telegram dm_topics."""
    import hermes_yaml as yaml
    from cron.scheduler_preflight import SharedRouteAdapters
    from gateway.profile_routing import ProfileRoute
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tests.gateway.conftest import _ensure_telegram_mock

    _ensure_telegram_mock()
    from plugins.platforms.telegram.adapter import TelegramAdapter

    owner, satellite = tmp_path / "owner", tmp_path / "satellite"
    for home in (owner, satellite):
        home.mkdir()
        (home / "config.yaml").write_text("platforms: {}\n")
    monkeypatch.setenv("HERMES_HOME", str(owner))  # the primary bot is built in its own profile
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake", extra={}))
    topic_ids = iter([4242, 5151])

    async def create_topic(chat_id, name, **_):
        return next(topic_ids)

    adapter._create_dm_topic = create_topic
    sent = []

    async def send(chat_id, content, reply_to=None, metadata=None):
        sent.append(metadata["thread_id"])
        ok = len(sent) > 1  # the saved topic was deleted: the router recreates it and resends
        return SimpleNamespace(success=ok, message_id="1" if ok else None, raw_response=None,
                               error=None if ok else "Bad Request: message thread not found")

    adapter.send = send
    shared = SharedRouteAdapters({Platform.TELEGRAM: adapter}, [
        ProfileRoute(name="sat", platform="telegram", chat_id="722341991", profile="satellite")])
    job = {"id": "j-shared", "name": "Brief", "deliver": "telegram:722341991:Debug"}
    loop = MagicMock()
    loop.is_running.return_value = True

    def run_now(coro, _loop):
        future = Future()
        try:
            future.set_result(asyncio.run(coro))
        except BaseException as exc:
            future.set_exception(exc)
        return future

    config = MagicMock()
    config.platforms = {}
    config.get_home_channel = lambda p: None
    token = set_hermes_home_override(satellite)  # the multiplexed cron tick runs the job in its profile
    try:
        with patch("gateway.config.load_gateway_config", return_value=config), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
             patch("asyncio.run_coroutine_threadsafe", side_effect=run_now):
            error = _deliver_result(job, "daily brief", adapters=shared, loop=loop)
    finally:
        reset_hermes_home_override(token)

    assert error is None and sent == ["4242", "5151"]
    owner_cfg = yaml.safe_load((owner / "config.yaml").read_text())
    topics = owner_cfg["platforms"]["telegram"]["extra"]["dm_topics"]
    assert topics == [{"chat_id": 722341991, "topics": [{"name": "Debug", "thread_id": 5151}]}]
    assert yaml.safe_load((satellite / "config.yaml").read_text()) == {"platforms": {}}

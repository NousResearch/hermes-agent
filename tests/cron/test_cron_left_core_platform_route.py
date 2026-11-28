"""A cron route naming a platform that left core (``deliver: <platform>:<chat>``, saved while core shipped
it) keeps delivering once the plugin registers the platform, and names the install when it does not."""

from unittest.mock import patch

import pytest

from cron.scheduler import _deliver_result
from gateway.config import Platform, PlatformConfig
from gateway.platform_registry import PlatformEntry, platform_registry
from hermes_cli.left_core_migration import LEFT_CORE

# Platforms that left core with no static ``Platform`` member: only the plugin can register them.
_MOVED = [f for f in LEFT_CORE if f.platform and f.platform not in Platform._value2member_map_]


@pytest.fixture(autouse=True)
def _no_discovery(monkeypatch):
    # The test home has no plugins; discovery must not register (or drop) the stand-in entry below.
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda *a, **k: None)


def _config(name=None):
    """A gateway config enabling *name* only; ``None`` is a credentialless satellite's (nothing connected)."""
    cfg = type("Cfg", (), {})()
    cfg.platforms = {Platform(name): PlatformConfig(enabled=True)} if name else {}
    cfg.get_home_channel = lambda _platform: None
    cfg.get_connected_platforms = lambda: list(cfg.platforms)
    return cfg


@pytest.mark.parametrize("feature", _MOVED, ids=[f.platform for f in _MOVED])
def test_route_delivers_through_the_plugins_sender(feature):
    sent = []

    async def sender(pconfig, chat_id, message, **_kw):
        sent.append((chat_id, message))
        return {"success": True, "platform": feature.platform, "chat_id": chat_id, "message_id": "m1"}

    platform_registry.register(PlatformEntry(
        name=feature.platform, label=feature.label, adapter_factory=lambda cfg: None, check_fn=lambda: True,
        standalone_sender_fn=sender, cron_deliver_env_var=f"{feature.platform.upper()}_HOME_CHANNEL"))
    try:
        with patch("gateway.config.load_gateway_config", return_value=_config(feature.platform)):
            error = _deliver_result({"id": "j1", "name": "Report", "deliver": f"{feature.platform}:chat-1"},
                                    "the report", adapters=None, loop=None)
    finally:
        platform_registry.unregister(feature.platform)
    assert error is None
    assert [chat for chat, _ in sent] == ["chat-1"] and "the report" in sent[0][1]


@pytest.mark.parametrize("feature", _MOVED, ids=[f.platform for f in _MOVED])
def test_route_without_the_plugin_names_the_install(feature):
    assert platform_registry.get(feature.platform) is None
    error = _deliver_result({"id": "j1", "name": "Report", "deliver": f"{feature.platform}:chat-1"},
                            "the report", adapters=None, loop=None)
    assert error and f"hermes plugins install {feature.plugin}" in error


@pytest.mark.parametrize("feature", _MOVED, ids=[f.platform for f in _MOVED])
def test_webhook_route_stays_loaded_before_the_plugin_registers(feature):
    """Route load checks the launch registry; a secondary profile's plugin is not in it, and a skipped route is
    not re-read until its file changes."""
    from gateway.webhook_targets import is_dispatchable_target
    assert platform_registry.get(feature.platform) is None
    assert is_dispatchable_target(feature.platform) is True


@pytest.mark.parametrize("route", ["{p}:chat-1", "{p}"], ids=["chat", "home"])
@pytest.mark.parametrize("feature", _MOVED, ids=[f.platform for f in _MOVED])
def test_route_beside_a_delivered_one_still_names_the_install(feature, route):
    """One destination resolving must not hide the moved one it skipped: the run reports the install."""
    sent = []

    async def sender(pconfig, chat_id, message, **_kw):
        sent.append(chat_id)
        return {"success": True, "platform": "standin", "chat_id": chat_id, "message_id": "m1"}

    platform_registry.register(PlatformEntry(
        name="standin", label="Stand-in", adapter_factory=lambda cfg: None, check_fn=lambda: True,
        standalone_sender_fn=sender, cron_deliver_env_var="STANDIN_HOME_CHANNEL"))
    try:
        with patch("gateway.config.load_gateway_config", return_value=_config("standin")):
            error = _deliver_result(
                {"id": "j1", "name": "Report", "deliver": "standin:chat-0," + route.format(p=feature.platform)},
                "the report", adapters=None, loop=None)
    finally:
        platform_registry.unregister("standin")
    assert sent == ["chat-0"]
    assert error and f"hermes plugins install {feature.plugin}" in error


@pytest.mark.parametrize("feature", _MOVED, ids=[f.platform for f in _MOVED])
def test_origin_on_a_moved_platform_names_the_install(feature):
    error = _deliver_result(
        {"id": "j1", "name": "Report", "deliver": "origin",
         "origin": {"platform": feature.platform, "chat_id": "chat-1"}},
        "the report", adapters=None, loop=None)
    assert error and f"hermes plugins install {feature.plugin}" in error


@pytest.mark.parametrize("deliver", ["all,{p}", "{p},all", "{p}", "origin", "origin,{p}"])
@pytest.mark.parametrize("feature", _MOVED, ids=[f.platform for f in _MOVED])
def test_a_saved_moved_platform_home_is_reported_once(feature, deliver):
    """A home saved for a platform that left core, its plugin not loaded: ``all``/``origin`` report the skipped
    home and a bare token addresses that same home, so a job naming it both ways still names the install once."""
    from hermes_constants import get_hermes_home
    (get_hermes_home() / "config.yaml").write_text(
        f"platforms:\n  {feature.platform}:\n    home_channel: {{chat_id: home-1}}\n", encoding="utf-8")
    assert platform_registry.get(feature.platform) is None
    error = _deliver_result({"id": "j1", "name": "Report", "deliver": deliver.format(p=feature.platform)},
                            "the report", adapters=None, loop=None)
    assert error and error.count(f"hermes plugins install {feature.plugin}") == 1, error


def _feishu_style_parser(ref):
    """A plugin's native ref parser in the Feishu shape: ``oc_<chat>[:omt_<topic>]``; anything else is not its syntax."""
    chat, _, thread = ref.partition(":")
    if not chat.startswith("oc_") or (thread and not thread.startswith("omt_")):
        return None
    return chat, thread or None


# The generated corpus: every destination shape cron accepts, against every satellite / primary state.
_SHAPES = ["{p}", "{p}:chat-1", "{p}:chat-2", "{p}:chat-1:t9", "{p}:oc_chat", "{p}:oc_chat:omt_topic",
           "{p}:oc_chat:omt_other", "origin", "all"]
_ORIGINS = [None, {"chat_id": "chat-1"}, {"chat_id": "chat-2"}, {"chat_id": "oc_chat", "thread_id": "omt_topic"}]
_ROUTES = {"routed": [{"chat_id": "chat-1"}, {"chat_id": "oc_chat", "thread_id": "omt_topic"}], "unrouted": []}
# ``own_bot``: the satellite runs a bot of its own on another platform, so cron sends through its own
# adapters and never borrows the primary's.
_CORPUS = [(shape, origin, routes, loaded, primary_parser, False)
           for shape in _SHAPES for origin in _ORIGINS for routes in _ROUTES
           for loaded in (False, True) for primary_parser in (False, True)] + [
           (shape, origin, routes, False, False, True) for shape in _SHAPES for origin in _ORIGINS for routes in _ROUTES] + [
           # ``home``: the satellite's own ``<PLATFORM>_HOME_CHANNEL``, read for a bare platform destination.
           (shape, origin, routes, loaded, False, False, home) for shape in ("{p}", "origin", "all")
           for origin in _ORIGINS for routes in _ROUTES for loaded in (False, True) for home in ("chat-1", "chat-2")] + [
           # ``failure_only``: the shape is the ``failure_deliver`` lane of a job whose regular report goes to
           # ``local``; preflight never names the install, the failed run's delivery does.
           (shape, origin, routes, loaded, False, False, None, True) for shape in _SHAPES
           for origin in _ORIGINS for routes in _ROUTES for loaded in (False, True)] + [
           # ``home_saved``: a bare destination's home saved only as ``platforms.<p>.home_channel`` in the satellite's
           # config.yaml (what ``/sethome`` writes), the plugin not loaded there. (``all``/``origin`` falling back to
           # that home is #160701.)
           ("{p}", origin, routes, False, False, False, home, False, True)
           for origin in _ORIGINS for routes in _ROUTES for home in ("chat-1", "chat-2")]


def _run_case(root, feature, shape, origin, routes, loaded, primary_parser, own_bot=False, home_chat=None,
              failure_only=False, home_saved=False):
    """``(preflight error, delivery error, [(chat, thread)] the primary's bot sent, origin read as absent)`` for
    one corpus case: a credentialless ``fitness`` profile the primary's bot serves through ``profile_routes``."""
    import asyncio
    from concurrent.futures import Future
    from unittest.mock import MagicMock

    import hermes_yaml as yaml
    from cron.scheduler_delivery import _origin_unresolved
    from cron.scheduler_preflight import _preflight_check_delivery, shared_route_adapters_for_current_home
    from hermes_constants import hermes_home_key, reset_hermes_home_override, set_hermes_home_override

    home = root / "profiles" / "fitness"
    home.mkdir(parents=True)
    (root / "config.yaml").write_text(yaml.safe_dump({"gateway": {"multiplex_profiles": True, "profile_routes": [
        {"platform": feature.platform, "profile": "fitness", **route} for route in _ROUTES[routes]]}}), encoding="utf-8")
    sent = []

    async def send(chat_id, content, metadata=None):
        sent.append((chat_id, (metadata or {}).get("thread_id")))
        return {"success": True, "message_id": "m1"}

    def run_now(coro, _loop):
        future = Future()
        future.set_result(asyncio.run(coro))
        return future

    primary = MagicMock()
    primary.send = send
    loop = MagicMock()
    loop.is_running.return_value = True
    job = ({"id": "j1", "name": "Report", "deliver": "local", "failure_deliver": shape.format(p=feature.platform)}
           if failure_only else {"id": "j1", "name": "Report", "deliver": shape.format(p=feature.platform)})
    if origin is not None:
        job["origin"] = {"platform": feature.platform, **origin}

    def entry():
        return PlatformEntry(name=feature.platform, label=feature.label, adapter_factory=lambda cfg: None,
                             check_fn=lambda: True, cron_deliver_env_var=f"{feature.platform.upper()}_HOME_CHANNEL",
                             parse_target_ref_fn=_feishu_style_parser if primary_parser else None)
    # The primary loads the plugin in its own scope; the satellite loads it only when ``loaded``.
    scopes = [hermes_home_key(root)] + ([hermes_home_key(home)] if loaded else [])
    for scope in scopes:
        platform_registry.register(entry(), scope=scope)
    token = set_hermes_home_override(str(root))
    primary_adapters = {Platform(feature.platform): primary}
    reset_hermes_home_override(token)
    token = set_hermes_home_override(str(home))
    home_env = {f"{feature.platform.upper()}_HOME_CHANNEL": home_chat} if home_chat and not home_saved else {}
    saved = {"platforms": {feature.platform: {"home_channel": {"chat_id": home_chat}}}} if home_saved else {}
    (home / "config.yaml").write_text(yaml.safe_dump(saved), encoding="utf-8")
    try:
        with patch.dict("os.environ", home_env), \
             patch("gateway.config.load_gateway_config", return_value=_config("telegram" if own_bot else None)):
            preflight, unresolved = _preflight_check_delivery(job), _origin_unresolved(job, for_failure=failure_only)
        with patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}, **saved}), \
             patch("gateway.config.load_gateway_config", return_value=_config("telegram" if own_bot else None)), \
             patch.dict("os.environ", home_env), \
             patch("asyncio.run_coroutine_threadsafe", side_effect=run_now):
            delivered = _deliver_result(job, "the report", adapters={Platform.TELEGRAM: MagicMock()} if own_bot else (
                shared_route_adapters_for_current_home(primary_adapters)), loop=loop, for_failure=failure_only)
    finally:
        reset_hermes_home_override(token)
        for scope in scopes:
            platform_registry.unregister(feature.platform, scope=scope)
    return preflight, delivered, sent, unresolved


@pytest.mark.parametrize("feature", _MOVED, ids=[f.platform for f in _MOVED])
def test_preflight_is_a_dry_run_of_delivery_for_a_moved_platform(tmp_path, monkeypatch, feature):
    """Property over the generated corpus: preflight names the install exactly when delivery does, never blocks
    a destination delivery reaches, and never passes one delivery fails to reach without saying why. Where the
    satellite lacks the plugin both read the ref with the primary's parser (else core's ``chat[:thread]``
    split) and send only where an exact route serves the parsed chat."""
    root_dir = {}
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root_dir["root"])
    install = f"plugins install {feature.plugin}"
    counterexamples, outcomes = [], set()
    for index, case in enumerate(_CORPUS):
        root_dir["root"] = tmp_path / f"case{index}"
        preflight, delivered, sent, unresolved = _run_case(root_dir["root"], feature, *case)
        verdict = "install" if preflight and install in preflight else ("pass" if preflight is None else "blocked")
        if len(case) > 7 and case[7]:
            # A failure-only destination the move leaves unreachable never blocks the run (``origin``/``all`` pass
            # as main skipped them; an explicit platform keeps main's NS-788 credential check); the failed run's
            # delivery is what names the install.
            if verdict == "install" or (case[0] in ("origin", "all") and verdict != "pass") or (
                    # (an origin-less job's ``origin`` names no platform: nothing to report for this one)
                    not case[3] and sent == [] and delivered is None and not (case[0] == "origin" and case[1] is None)):
                counterexamples.append((case, preflight, delivered, sent))
            continue
        outcome = ("install" if delivered and install in delivered else
                   "reached" if sent and delivered is None else "other")
        outcomes.add((verdict, outcome, case[3]))
        # Unloaded, a pass must not hide a failure to reach this platform (``all`` with no home addresses none),
        # and a run with preflight off must not file an origin delivery names the install as "no origin".
        if (verdict == "install") != (outcome == "install") or (outcome == "reached" and verdict != "pass") or (
                not case[3] and verdict == "pass" and outcome != "reached" and feature.platform in (delivered or "")
        ) or (outcome == "install" and unresolved) or (
                # A bare destination's home address is a parsed chat like any other: an exact route serves it.
                case[0] == "{p}" and len(case) > 6 and {"chat_id": case[6]} in _ROUTES[case[2]] and outcome != "reached"):
            counterexamples.append((case, preflight, delivered, sent))
    assert counterexamples == []
    # The corpus exercises both verdicts on the unloaded side, and the loaded side never names the install.
    assert {("pass", "reached", False), ("install", "install", False), ("pass", "reached", True)} <= outcomes
    assert not any(verdict == "install" for verdict, _outcome, loaded in outcomes if loaded)


@pytest.mark.parametrize("primary_parser", [False, True], ids=["generic-split", "primary-parser"])
@pytest.mark.parametrize("feature", _MOVED, ids=[f.platform for f in _MOVED])
def test_satellite_topic_rides_the_primarys_bot_without_the_plugin(tmp_path, monkeypatch, feature, primary_parser):
    """``<platform>:oc_chat:omt_topic`` on a satellite without the plugin reaches the routed topic through the
    primary's bot: the ref is parsed into chat + thread before the exact route lookup, not matched whole."""
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: tmp_path)
    preflight, delivered, sent, _unresolved = _run_case(
        tmp_path, feature, "{p}:oc_chat:omt_topic", None, "routed", False, primary_parser)
    assert (preflight, delivered, sent) == (None, None, [("oc_chat", "omt_topic")])


@pytest.mark.parametrize("feature", _MOVED, ids=[f.platform for f in _MOVED])
def test_satellite_home_channel_rides_the_primarys_bot_without_the_plugin(tmp_path, monkeypatch, feature):
    """Bare ``deliver: <platform>`` on a satellite without the plugin reads its own home address and sends it
    through the primary's bot where an exact route serves that chat, as an explicit ``<platform>:<chat>`` does."""
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: tmp_path)
    preflight, delivered, sent, _unresolved = _run_case(
        tmp_path, feature, "{p}", None, "routed", False, False, home_chat="chat-1")
    assert (preflight, delivered, sent) == (None, None, [("chat-1", None)])


_HOME_PLATFORMS = [f for f in LEFT_CORE if f.platform and f"{f.platform.upper()}_HOME_CHANNEL" in f.private_env]


@pytest.mark.parametrize("saved_in", ["env", "config.yaml", "gateway.platforms", "gateway.<platform> over platforms"])
@pytest.mark.parametrize("feature", _HOME_PLATFORMS, ids=[f.platform for f in _HOME_PLATFORMS])
def test_worker_without_any_registration_reads_the_satellite_home(tmp_path, monkeypatch, feature, saved_in):
    """A managed cron worker holds no registration for the platform (not this profile's, not the primary's): a bare
    ``deliver: <platform>`` still reads the profile's ``<PLATFORM>_HOME_CHANNEL`` and passes preflight exactly when
    a primary route serves that chat, as an explicit ``<platform>:<chat>`` does; the gateway's bot sends it."""
    import hermes_yaml as yaml
    from cron.scheduler_delivery import resolve_cron_destination
    from cron.scheduler_preflight import _preflight_check_delivery
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: tmp_path)
    home = tmp_path / "profiles" / "fitness"
    home.mkdir(parents=True)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"gateway": {"multiplex_profiles": True, "profile_routes": [
        {"platform": feature.platform, "profile": "fitness", "chat_id": "chat-1"}]}}), encoding="utf-8")
    monkeypatch.setattr("gateway.platform_registry.platform_registry.get", lambda _name: None)
    job = {"id": "j1", "name": "Report", "deliver": feature.platform}
    verdicts = {}
    token = set_hermes_home_override(str(home))
    try:
        with patch("gateway.config.load_gateway_config", return_value=_config(None)):
            for chat in ("chat-1", "chat-2"):
                if saved_in == "env":
                    monkeypatch.setenv(f"{feature.platform.upper()}_HOME_CHANNEL", chat)
                else:  # what ``/sethome`` writes (greptile #159651), and the gateway loader's other blocks
                    saved = {"home_channel": {"chat_id": chat}}
                    (home / "config.yaml").write_text(yaml.safe_dump({
                        "config.yaml": {"platforms": {feature.platform: saved}},
                        "gateway.platforms": {"gateway": {"platforms": {feature.platform: saved}}},
                        # an older home under platforms: the gateway reads gateway.<platform>'s, and so does cron
                        "gateway.<platform> over platforms": {
                            "platforms": {feature.platform: {"home_channel": {"chat_id": "chat-1"}}},
                            "gateway": {feature.platform: saved}},
                    }[saved_in]), encoding="utf-8")
                target, error = resolve_cron_destination(job, feature.platform)
                verdicts[chat] = (target and target["chat_id"], error is None, _preflight_check_delivery(job) is None)
    finally:
        reset_hermes_home_override(token)
    assert verdicts == {"chat-1": ("chat-1", True, True), "chat-2": (None, False, False)}

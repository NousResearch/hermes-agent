"""``cron.plugin_api``: the published cron calls a plugin makes without a ``ctx``.

Each test drives the two public functions against a real cron store in the test's own home.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

import hermes_yaml as yaml
from agent.secret_scope import current_secret_scope, get_secret, reset_secret_scope, set_secret_scope
from cron import jobs
from cron.execution_identity import enter_cron_execution, exit_cron_execution
from cron.plugin_api import AmbiguousJobReference, delivery_platforms, ensure_named_job
from cron.scheduler import CronSchedulerRegistrationError
from hermes_cli.plugin_validate import validate_plugin_dir
from hermes_constants import get_hermes_home

TELEGRAM_CHAT = "-100111222"
DISCORD_CHAT = "987654321"


@pytest.fixture()
def registered(monkeypatch, make_cron_provider):
    """Job ids the active scheduler provider was asked to register a trigger for."""
    seen: list[str] = []
    provider = make_cron_provider(register_job=lambda job: seen.append(job["id"]))
    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", lambda: provider)
    return seen


@pytest.fixture()
def home_channels(monkeypatch):
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", TELEGRAM_CHAT)
    monkeypatch.setenv("DISCORD_HOME_CHANNEL", DISCORD_CHAT)


@pytest.fixture()
def caller_scope():
    """The caller's own secret scope, holding a Telegram home channel."""
    mapping = {"TELEGRAM_HOME_CHANNEL": TELEGRAM_CHAT, "REVIEW_SLOT": "caller"}
    token = set_secret_scope(mapping, profile_home=str(get_hermes_home()))
    try:
        yield mapping
    finally:
        reset_secret_scope(token)


@pytest.fixture()
def target(tmp_path):
    """Another profile's home whose ``.env`` holds a Discord home channel."""
    home = tmp_path / "other-profile"
    home.mkdir()
    (home / ".env").write_text(f"DISCORD_HOME_CHANNEL={DISCORD_CHAT}\nREVIEW_SLOT=owner\n", encoding="utf-8")
    return home


def _all_jobs() -> list[dict]:
    return jobs.list_jobs(include_disabled=True)


def test_missing_job_is_created_once_with_the_given_fields(registered):
    other_case = jobs.create_job(prompt="other", schedule="every 1h", name="Follow-Up", paused=True)

    job = ensure_named_job(
        "follow-up", prompt="check in", schedule="every 2h", repeat=3, deliver="telegram",
        enabled_toolsets=["web"])

    [stored] = [j for j in _all_jobs() if j["id"] != other_case["id"]]
    assert stored["id"] == job["id"]
    assert stored["name"] == "follow-up"
    assert stored["prompt"] == "check in"
    assert stored["schedule_display"] == "every 120m"
    assert stored["repeat"]["times"] == 3
    assert stored["deliver"] == "telegram"
    assert stored["enabled_toolsets"] == ["web"]
    assert registered == [job["id"]]

    again = ensure_named_job("follow-up", prompt="other", schedule="every 5h")
    assert again["id"] == job["id"]
    assert len(_all_jobs()) == 2


def test_existing_enabled_job_is_registered_again_and_left_unchanged(registered):
    existing = jobs.create_job(prompt="original", schedule="every 1h", name="follow-up")

    job = ensure_named_job("follow-up", prompt="new prompt", schedule="every 9h")

    assert job["id"] == existing["id"]
    assert registered == [existing["id"]]
    [stored] = _all_jobs()
    assert stored["prompt"] == "original"
    assert stored["schedule"] == existing["schedule"]


def test_existing_disabled_job_is_returned_without_registering(registered):
    existing = jobs.create_job(prompt="original", schedule="every 1h", name="follow-up", paused=True)

    job = ensure_named_job("follow-up", prompt="new prompt", schedule="every 9h")

    assert job["id"] == existing["id"]
    assert job["enabled"] is False
    assert job["prompt"] == "original"
    assert registered == []


def test_an_existing_job_comes_back_in_a_new_jobs_shape_without_reading_run_history(monkeypatch, registered):
    ledger_reads: list[str] = []
    monkeypatch.setattr("cron.executions.latest_executions", lambda ids: ledger_reads.append("latest") or {})
    monkeypatch.setattr("cron.executions.live_executions", lambda ids: ledger_reads.append("live") or {})

    created = ensure_named_job("follow-up", prompt="check in", schedule="every 1h")
    found = ensure_named_job("follow-up", prompt="check in", schedule="every 1h")

    assert found["id"] == created["id"]
    assert set(found) == set(created)
    assert ledger_reads == []


def test_two_jobs_with_the_name_raise_and_create_nothing(registered):
    jobs.create_job(prompt="a", schedule="every 1h", name="follow-up")
    jobs.create_job(prompt="b", schedule="every 1h", name="follow-up", paused=True)

    with pytest.raises(AmbiguousJobReference):
        ensure_named_job("follow-up", prompt="c", schedule="every 1h")

    assert len(_all_jobs()) == 2
    assert registered == []


def test_failed_first_registration_keeps_the_job_and_the_next_call_registers_it(monkeypatch, make_cron_provider):
    attempts: list[str] = []

    def register(job):
        attempts.append(job["id"])
        if len(attempts) <= 2:
            raise OSError("provider unreachable")

    provider = make_cron_provider(register_job=register)
    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", lambda: provider)

    with pytest.raises(CronSchedulerRegistrationError) as failed:
        ensure_named_job("follow-up", prompt="check in", schedule="every 1h")
    [stored] = _all_jobs()
    assert failed.value.job["id"] == stored["id"]
    with pytest.raises(CronSchedulerRegistrationError) as still_down:  # the saved job, same error
        ensure_named_job("follow-up", prompt="check in", schedule="every 1h")
    assert still_down.value.job["id"] == stored["id"]
    assert isinstance(still_down.value.cause, OSError)

    job = ensure_named_job("follow-up", prompt="check in", schedule="every 1h")

    assert job["id"] == stored["id"]
    assert attempts == [stored["id"]] * 3
    assert len(_all_jobs()) == 1


def test_home_targets_that_store_and_restores_the_active_one(tmp_path, registered):
    other = tmp_path / "other-profile"
    other.mkdir()
    active_home, active_store = get_hermes_home(), jobs.get_cron_output_dir()

    job = ensure_named_job("follow-up", prompt="check in", schedule="every 1h", home=other)

    assert (get_hermes_home(), jobs.get_cron_output_dir()) == (active_home, active_store)
    assert _all_jobs() == []
    with jobs.use_cron_store(other):
        assert [j["id"] for j in _all_jobs()] == [job["id"]]
        jobs.create_job(prompt="dup", schedule="every 1h", name="follow-up")

    with pytest.raises(AmbiguousJobReference):
        ensure_named_job("follow-up", prompt="check in", schedule="every 1h", home=other)
    assert (get_hermes_home(), jobs.get_cron_output_dir()) == (active_home, active_store)


def test_names_cron_cannot_store_verbatim_are_refused_before_anything_is_written(registered):
    for name in ("", "  ", " follow-up ", 42):
        with pytest.raises(ValueError, match="non-empty with no leading or trailing whitespace"):
            ensure_named_job(name, prompt="check in", schedule="every 1h")

    assert _all_jobs() == []
    assert registered == []


def test_missing_home_is_refused_and_not_created(tmp_path, registered):
    missing = tmp_path / "never-existed"

    with pytest.raises(FileNotFoundError, match=str(missing)):
        ensure_named_job("follow-up", prompt="check in", schedule="every 1h", home=missing)

    assert not missing.exists()
    assert registered == []


def test_home_reads_that_profiles_home_channels_not_the_callers(target, caller_scope, registered):
    job = ensure_named_job("follow-up", prompt="check in", schedule="every 1h", deliver="all", home=target)

    assert delivery_platforms(job["id"], home=target) == ("discord",)
    assert current_secret_scope() is caller_scope


def test_provider_registers_under_that_profiles_secrets(target, caller_scope, monkeypatch, make_cron_provider):
    slots: list[str | None] = []
    provider = make_cron_provider(register_job=lambda job: slots.append(get_secret("REVIEW_SLOT")))
    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", lambda: provider)

    ensure_named_job("follow-up", prompt="check in", schedule="every 1h", home=target)

    assert slots == ["owner"]
    assert current_secret_scope() is caller_scope
    with jobs.use_cron_store(target):
        jobs.create_job(prompt="dup", schedule="every 1h", name="follow-up")
    with pytest.raises(AmbiguousJobReference):
        ensure_named_job("follow-up", prompt="check in", schedule="every 1h", home=target)
    assert current_secret_scope() is caller_scope


def test_home_schedules_in_that_profiles_time_zone_not_the_processes(target, monkeypatch, registered):
    monkeypatch.setenv("HERMES_TIMEZONE", "America/New_York")
    (target / "config.yaml").write_text("timezone: Asia/Tokyo\n", encoding="utf-8")

    def local_hour(job, zone):
        return datetime.fromisoformat(job["next_run_at"]).astimezone(ZoneInfo(zone)).hour

    theirs = ensure_named_job("follow-up", prompt="check in", schedule="0 9 * * *", home=target)
    mine = ensure_named_job("follow-up", prompt="check in", schedule="0 9 * * *")

    assert local_hour(theirs, "Asia/Tokyo") == 9
    assert local_hour(mine, "America/New_York") == 9


def test_a_tilde_home_is_the_directory_it_expands_to(tmp_path, monkeypatch, registered):
    other = tmp_path / "other-profile"
    other.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    job = ensure_named_job("follow-up", prompt="check in", schedule="every 1h", deliver="local",
                           home="~/other-profile")

    assert delivery_platforms(job["id"], home="~/other-profile") == ()
    with jobs.use_cron_store(other):
        assert [j["id"] for j in _all_jobs()] == [job["id"]]


def test_home_is_read_the_way_hermes_reads_a_home_path(tmp_path, monkeypatch, registered):
    monkeypatch.setenv("SEAM_PROFILE_DIR", "other-profile")
    literal, named = tmp_path / "$SEAM_PROFILE_DIR", tmp_path / "other-profile"
    literal.mkdir()

    with pytest.raises(FileNotFoundError, match="other-profile"):
        ensure_named_job("follow-up", prompt="check in", schedule="every 1h", home=literal)
    assert not named.exists()

    named.mkdir()
    job = ensure_named_job("follow-up", prompt="check in", schedule="every 1h", home=literal)
    with jobs.use_cron_store(named):
        assert [j["id"] for j in _all_jobs()] == [job["id"]]
    assert not (literal / "cron").exists()


def test_home_expands_variables_once_for_store_and_timezone(tmp_path, monkeypatch, registered):
    monkeypatch.setenv("CRON_PROFILE_OUTER", "$CRON_PROFILE_INNER")
    monkeypatch.setenv("CRON_PROFILE_INNER", "other-profile")
    literal, other = tmp_path / "$CRON_PROFILE_INNER", tmp_path / "other-profile"
    for home, zone in [(literal, "Asia/Tokyo"), (other, "America/New_York")]:
        home.mkdir()
        (home / "config.yaml").write_text(f"timezone: {zone}\n", encoding="utf-8")

    job = ensure_named_job("follow-up", prompt="check in", schedule="0 9 * * *",
                           home=tmp_path / "$CRON_PROFILE_OUTER")

    with jobs.use_cron_store(literal):
        assert [j["id"] for j in _all_jobs()] == [job["id"]]
    assert not (other / "cron").exists()
    assert datetime.fromisoformat(job["next_run_at"]).astimezone(ZoneInfo("Asia/Tokyo")).hour == 9


def test_home_wins_over_a_cron_store_the_caller_is_already_inside(tmp_path, registered):
    # A plugin hook running inside one profile's cron fire is already routed to that profile's store.
    mine, other = tmp_path / "mine", tmp_path / "other-profile"
    mine.mkdir()
    other.mkdir()

    with jobs.use_cron_store(mine):
        job = ensure_named_job("follow-up", prompt="check in", schedule="every 1h", home=other)
        assert _all_jobs() == []
    with jobs.use_cron_store(other):
        assert [j["id"] for j in _all_jobs()] == [job["id"]]


def test_unknown_job_and_no_cron_run_answer_none():
    assert delivery_platforms("no-such-job") is None
    assert delivery_platforms() is None


def test_a_hosted_activity_cannot_target_another_profile(target, registered):
    from agent.runtime_scope import bind_state
    from hermes_scope import Principal, Scope, ScopeSwitchRefused

    with bind_state(Scope("acme", "agent-7", Principal("u1"), owner=Principal("u1"), can_administer=True)):
        with pytest.raises(ScopeSwitchRefused):
            ensure_named_job("follow-up", prompt="check in", schedule="every 1h", home=target)
        with pytest.raises(ScopeSwitchRefused):
            delivery_platforms("no-such-job", home=target)

    assert registered == []
    assert not (target / "cron").exists()


def test_missing_home_answers_none_and_is_not_created(tmp_path):
    missing = tmp_path / "never-existed"

    assert delivery_platforms("no-such-job", home=missing) is None
    assert not missing.exists()


def test_no_job_id_inside_a_cron_run_answers_for_that_run(home_channels):
    job = jobs.create_job(prompt="p", schedule="every 1h", name="run", deliver="telegram")

    token = enter_cron_execution(job, "exec-1", {"source": "direct"})
    try:
        assert delivery_platforms() == ("telegram",)
    finally:
        exit_cron_execution(token)
    assert delivery_platforms() is None


def test_firing_all_delivery_requires_job_id_for_explicit_home(target, caller_scope, registered):
    other_job = ensure_named_job("other", prompt="p", schedule="every 1h", deliver="all", home=target)
    job = jobs.create_job(prompt="p", schedule="every 1h", name="run", deliver="all")
    token = enter_cron_execution(job, "exec-1", {"source": "direct"})
    try:
        assert delivery_platforms() == ("telegram",)
        assert delivery_platforms(other_job["id"], home=target) == ("discord",)
        with pytest.raises(ValueError, match="job_id is required"):
            delivery_platforms(home=target)
    finally:
        exit_cron_execution(token)


def test_no_job_id_uses_firing_job_settings_even_after_disk_edit_or_delete(home_channels):
    from agent.delegation_context import delegated_child_context

    job = jobs.create_job(prompt="p", schedule="every 1h", name="run", deliver="telegram")

    token = enter_cron_execution(job, "exec-1", {"source": "direct"})
    try:
        jobs.update_job(job["id"], {"deliver": "local"})
        assert delivery_platforms() == ("telegram",)
        assert delivery_platforms(job["id"]) == ()

        jobs.remove_job(job["id"])
        assert delivery_platforms() == ("telegram",)
        assert delivery_platforms(job["id"]) is None

        with delegated_child_context("child"):
            assert delivery_platforms() is None
    finally:
        exit_cron_execution(token)
    assert delivery_platforms() is None


def test_no_job_id_uses_firing_origin_even_if_mutated_or_edited_mid_run(home_channels):
    job = jobs.create_job(
        prompt="p", schedule="every 1h", name="run", deliver="origin",
        origin={"platform": "telegram", "chat_id": TELEGRAM_CHAT},
    )

    token = enter_cron_execution(job, "exec-1", {"source": "direct"})
    try:
        job["origin"]["platform"] = "discord"
        jobs.update_job(job["id"], {"deliver": "discord"})
        assert delivery_platforms() == ("telegram",)
        assert delivery_platforms(job["id"]) == ("discord",)
    finally:
        exit_cron_execution(token)


HOSTED_HOOK_PLUGIN = '''
from cron.plugin_api import delivery_platforms


def register(ctx):
    def hook(**kwargs):
        run = ctx.current_cron_execution()
        firing = ctx.cron_delivery_platforms()
        preview = ctx.cron_delivery_platforms(run.job_id) if run is not None else None
        return {
            "local_no_id": delivery_platforms(),
            "firing": firing,
            "firing_is_tuple": isinstance(firing, tuple),
            "preview": preview,
            "preview_is_tuple": isinstance(preview, tuple),
        }
    ctx.register_hook("post_llm_call", hook)
'''


@pytest.mark.platforms("any")  # the host is a child process
def test_a_hook_in_a_plugin_host_reads_firing_delivery_platforms_through_ctx(tmp_path, monkeypatch):
    from hermes_cli.plugins import PluginManager
    from tests.hermes_cli.plugin_host_support import _home_with_plugins

    _home_with_plugins(tmp_path / "plugins", monkeypatch, {"hostedcron": HOSTED_HOOK_PLUGIN}, isolation="host")
    job = jobs.create_job(
        prompt="p", schedule="every 1h", name="run", deliver="origin",
        origin={"platform": "telegram", "chat_id": TELEGRAM_CHAT}, paused=True,
    )
    manager = PluginManager()
    manager.discover_and_load()
    try:
        assert manager._plugins["hostedcron"].error is None
        token = enter_cron_execution(job, "exec-1", {"source": "direct"})
        try:
            jobs.update_job(job["id"], {"deliver": "local"})
            # The host process has no run bound; ctx.cron_delivery_platforms() runs in the gateway
            # and returns the firing run's platforms (as a tuple) even after the saved job was edited.
            assert manager.invoke_hook("post_llm_call", session_id="s") == [{
                "local_no_id": None,
                "firing": ["telegram"],
                "firing_is_tuple": True,
                "preview": [],
                "preview_is_tuple": True,
            }]
        finally:
            exit_cron_execution(token)
        assert manager.invoke_hook("post_llm_call", session_id="s") == [{
            "local_no_id": None,
            "firing": None,
            "firing_is_tuple": False,
            "preview": None,
            "preview_is_tuple": False,
        }]
    finally:
        manager.unload()
        manager._plugin_host().shutdown()


def test_one_platform_is_lower_cased_and_local_delivers_nowhere(home_channels):
    named = jobs.create_job(prompt="p", schedule="every 1h", deliver="Telegram")
    local = jobs.create_job(prompt="p", schedule="every 1h", deliver="local")

    assert delivery_platforms(named["id"]) == ("telegram",)
    assert delivery_platforms(local["id"]) == ()


def test_all_lists_each_home_channel_platform_once_in_resolver_order(home_channels):
    # A second Telegram chat is its own cron target; the platform still appears once.
    job = jobs.create_job(prompt="p", schedule="every 1h", deliver="all,telegram:-100999")

    assert delivery_platforms(job["id"]) == ("telegram", "discord")


def test_result_holds_only_platform_names(home_channels):
    job = jobs.create_job(
        prompt="p", schedule="every 1h", deliver=f"all,telegram:{TELEGRAM_CHAT}")

    result = delivery_platforms(job["id"])

    assert result == ("telegram", "discord")
    assert TELEGRAM_CHAT not in repr(result) and DISCORD_CHAT not in repr(result)


def test_plugin_importing_the_module_gets_no_private_import_warning(tmp_path):
    from cron.plugin_api import CronSchedulerRegistrationError as PublicRegError

    assert PublicRegError is CronSchedulerRegistrationError
    plugin = tmp_path / "follow_up"
    plugin.mkdir()
    (plugin / "plugin.yaml").write_text(
        yaml.safe_dump({"name": "follow_up", "version": "1.0.0", "description": "x"}), encoding="utf-8")
    Path(plugin / "__init__.py").write_text(
        "from cron.plugin_api import (\n"
        "    AmbiguousJobReference, CronSchedulerRegistrationError, delivery_platforms, ensure_named_job,\n"
        ")\n"
        "def register(ctx):\n"
        "    return ensure_named_job, delivery_platforms, AmbiguousJobReference, CronSchedulerRegistrationError\n",
        encoding="utf-8")

    report = validate_plugin_dir(plugin)

    assert report.ok, report.failures
    assert not [w for w in report.warnings if w.startswith("private core imports")]


def test_ctx_cron_delivery_platforms_in_process(home_channels):
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    ctx = PluginContext(PluginManifest(name="demo"), PluginManager())
    job = jobs.create_job(prompt="p", schedule="every 1h", name="run", deliver="telegram")
    assert ctx.cron_delivery_platforms() is None
    assert ctx.cron_delivery_platforms(job["id"]) == ("telegram",)

    token = enter_cron_execution(job, "exec-1", {"source": "direct"})
    try:
        jobs.update_job(job["id"], {"deliver": "local"})
        assert ctx.cron_delivery_platforms() == ("telegram",)
        assert ctx.cron_delivery_platforms(job["id"]) == ()
    finally:
        exit_cron_execution(token)
    assert ctx.cron_delivery_platforms() is None

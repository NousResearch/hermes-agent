"""Phase 4: lifecycle guard + per-profile observability."""
import pytest

from gateway.config import GatewayConfig


class TestServedProfilesStatus:
    def test_write_and_read_served_profiles(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import importlib
        import gateway.status as status
        importlib.reload(status)
        try:
            status.write_runtime_status(
                gateway_state="running", served_profiles=["default", "coder"]
            )
            rec = status.read_runtime_status()
            assert rec.get("served_profiles") == ["default", "coder"]
        finally:
            importlib.reload(status)


def test_cron_profile_homes_follow_allowlist(tmp_path, monkeypatch):
    """The helper wired into in-process cron returns only selected profiles."""
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    for name in ("worker", "guest"):
        (default_home / "profiles" / name).mkdir(parents=True)

    import gateway.run as gateway_run

    homes = gateway_run._multiplex_profile_homes(
        GatewayConfig(
            multiplex_profiles=True,
            multiplex_profile_allowlist=["worker"],
        )
    )

    assert [name for name, _home in homes] == ["default", "worker"]


class _FakeRunner:
    """Minimal GatewayRunner stand-in for _build_cron_start_kwargs tests."""

    def __init__(self, profile_adapters=None, draining=False):
        self.adapters = {"telegram": "default-telegram-adapter"}
        self.config = GatewayConfig(multiplex_profiles=True)
        self._profile_adapters = profile_adapters
        self._draining = draining
        self._external_drain_active = False


class _ExternalProvider:
    """A provider that is NOT the built-in in-process ticker (Chronos-like).

    Its ``start`` mirrors the external provider contract:
    ``start(stop_event, *, adapters, loop, interval)`` — no
    ``profile_adapters``/``profile_homes``/``can_dispatch`` kwargs.
    """

    def start(self, stop_event, *, adapters=None, loop=None, interval=60):
        pass


def test_cron_start_kwargs_external_provider_gets_no_profile_kwargs():
    """Review #83197 blocker 1: external providers (Chronos) must NEVER receive
    ``profile_adapters`` (or the other in-process-only kwargs) in
    ``cron_start_kwargs`` — their ``start()`` signature does not accept them
    and a multiplex deployment would crash with a deterministic TypeError."""
    import asyncio

    import gateway.run as gateway_run
    from cron.scheduler_provider import InProcessCronScheduler

    runner = _FakeRunner(profile_adapters={"coder": {"telegram": "coder-adapter"}})
    external = _ExternalProvider()

    async def _build():
        return gateway_run._build_cron_start_kwargs(
            runner, external, multiplex_cron=True
        )

    kwargs = asyncio.run(_build())
    assert "profile_adapters" not in kwargs
    assert "profile_homes" not in kwargs
    assert "can_dispatch" not in kwargs
    assert "adapters" in kwargs
    assert "loop" in kwargs


def test_cron_start_kwargs_inprocess_gets_profile_adapters_under_multiplex():
    """Review #1 blocker 1 positive path: when the resolved provider IS the
    in-process scheduler and the runner has a per-profile adapter registry,
    ``profile_adapters`` is injected (plus the multiplex profile homes and the
    drain gate) so multiplex cron delivery routes through the owning profile's
    bot/chat."""
    import asyncio

    import gateway.run as gateway_run
    from cron.scheduler_provider import InProcessCronScheduler

    pa = {"coder": {"telegram": "coder-telegram-adapter"}}
    runner = _FakeRunner(profile_adapters=pa)
    provider = InProcessCronScheduler()
    gateway_run._multiplex_profile_homes = lambda config: [("default", None)]

    async def run():
        return gateway_run._build_cron_start_kwargs(
            runner, provider, multiplex_cron=True
        )

    kwargs = asyncio.run(run())
    assert kwargs["profile_adapters"] is pa
    assert kwargs["profile_homes"] == [("default", None)]
    assert callable(kwargs["can_dispatch"])
    assert kwargs["can_dispatch"]() is True


def test_cron_start_kwargs_inprocess_without_registry_omits_key():
    """Review #1: an in-process provider whose gateway has NO per-profile
    registry must simply omit ``profile_adapters`` — the key only exists when
    there is an actual per-profile adapter map to thread through."""
    import asyncio

    import gateway.run as gateway_run
    from cron.scheduler_provider import InProcessCronScheduler

    runner = _FakeRunner(profile_adapters=None)
    provider = InProcessCronScheduler()
    gateway_run._multiplex_profile_homes = lambda config: []

    async def run():
        return gateway_run._build_cron_start_kwargs(
            runner, provider, multiplex_cron=True
        )

    kwargs = asyncio.run(run())
    assert "profile_adapters" not in kwargs
    # No profile homes resolved -> the key is omitted entirely (matches the
    # original start_gateway behavior: empty homes list never sets the kwarg).
    assert "profile_homes" not in kwargs
    assert callable(kwargs["can_dispatch"])


class TestNamedProfileMultiplexerGuard:
    """_guard_named_profile_under_multiplexer is inert unless all conditions hold."""


    def test_force_bypasses(self, monkeypatch):
        from hermes_cli import gateway as gw
        # Even if it looks like a named profile, force returns immediately.
        monkeypatch.setattr(gw, "_profile_suffix", lambda: "coder")
        gw._guard_named_profile_under_multiplexer(force=True)

    def test_inert_when_no_default_gateway_running(self, monkeypatch, tmp_path):
        from hermes_cli import gateway as gw
        monkeypatch.setattr(gw, "_profile_suffix", lambda: "coder")
        monkeypatch.setattr(
            "hermes_constants.get_default_hermes_root", lambda: tmp_path
        )
        # No gateway.pid in tmp_path => no running default gateway => no raise.
        gw._guard_named_profile_under_multiplexer(force=False)

    def _fake_running_default_gateway(self, monkeypatch, tmp_path):
        """Make the guard believe a live default gateway exists at tmp_path."""
        from hermes_cli import gateway as gw
        import gateway.status as status

        monkeypatch.setattr(gw, "_profile_suffix", lambda: "coder")
        monkeypatch.setattr(
            "hermes_constants.get_default_hermes_root", lambda: tmp_path
        )
        (tmp_path / "gateway.pid").write_text("12345", encoding="utf-8")
        monkeypatch.setattr(status, "_read_pid_record", lambda p: {"pid": 12345})
        monkeypatch.setattr(status, "_pid_from_record", lambda rec: 12345)
        monkeypatch.setattr(status, "_pid_exists", lambda pid: True)

    def test_unset_allowlist_preserves_historical_guard(self, monkeypatch, tmp_path):
        self._fake_running_default_gateway(monkeypatch, tmp_path)
        (tmp_path / "config.yaml").write_text(
            "gateway:\n  multiplex_profiles: true\n",
            encoding="utf-8",
        )

        from hermes_cli import gateway as gw

        with pytest.raises(SystemExit, match="1"):
            gw._guard_named_profile_under_multiplexer(force=False)

    def test_served_profile_is_still_guarded(self, monkeypatch, tmp_path):
        self._fake_running_default_gateway(monkeypatch, tmp_path)
        (tmp_path / "config.yaml").write_text(
            "gateway:\n"
            "  multiplex_profiles: true\n"
            "  multiplex_profile_allowlist:\n"
            "    - Coder\n",
            encoding="utf-8",
        )

        from hermes_cli import gateway as gw

        with pytest.raises(SystemExit, match="1"):
            gw._guard_named_profile_under_multiplexer(force=False)

    @pytest.mark.parametrize(
        "allowlist_yaml",
        ["[]", "[worker]", "coder"],
    )
    def test_unserved_profile_may_run_standalone(
        self, monkeypatch, tmp_path, allowlist_yaml
    ):
        self._fake_running_default_gateway(monkeypatch, tmp_path)
        (tmp_path / "config.yaml").write_text(
            "gateway:\n"
            "  multiplex_profiles: true\n"
            f"  multiplex_profile_allowlist: {allowlist_yaml}\n",
            encoding="utf-8",
        )

        from hermes_cli import gateway as gw

        gw._guard_named_profile_under_multiplexer(force=False)



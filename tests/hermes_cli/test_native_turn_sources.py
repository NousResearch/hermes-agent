from __future__ import annotations

from dataclasses import FrozenInstanceError
import threading

import pytest

from hermes_cli.native_turn_sources import (
    NativeSessionView,
    NativeTurnAdmission,
    NativeTurnLease,
    RegisteredNativeTurnSource,
    fence_native_turn_sources,
    fence_registered_sources,
    poll_native_turn,
    poll_registered_sources,
    register_profile_source,
)
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def view(tmp_path, surface="cli"):
    return NativeSessionView(
        profile="default",
        profile_home=tmp_path,
        session_id="tip",
        surface=surface,
        compression_lineage=("root", "tip", "root"),
        session_key="route",
        owner_token="owner",
        metadata={"transport": "test"},
    )


class Source:
    name = "external-source"

    def __init__(self, lease=None):
        self.lease = lease
        self.polled = []
        self.boundaries = []

    def poll(self, session):
        self.polled.append(session)
        return self.lease

    def on_user_boundary(self, session, reason):
        self.boundaries.append((session, reason))


def registration(source, surfaces=("cli",)):
    return {
        source.name: RegisteredNativeTurnSource(
            source=source, surfaces=frozenset(surfaces), plugin_id="plugin"
        )
    }


def test_session_view_is_canonical_and_deeply_immutable(tmp_path):
    original = {"transport": "test"}
    session = NativeSessionView(
        profile=" default ",
        profile_home=tmp_path / ".." / tmp_path.name,
        session_id=" tip ",
        surface="CLI",
        compression_lineage=("root", "tip", "root"),
        metadata=original,
    )
    original["transport"] = "changed"

    assert session.profile == "default"
    assert session.session_id == "tip"
    assert session.surface == "cli"
    assert session.compression_lineage == ("root", "tip")
    assert session.metadata["transport"] == "test"
    with pytest.raises(FrozenInstanceError):
        session.session_id = "other"
    with pytest.raises(TypeError):
        session.metadata["transport"] = "changed"


def test_plugin_context_registration_is_scoped_and_disposed(tmp_path):
    manager = PluginManager(scope_key=str(tmp_path))
    source = Source()
    ctx = PluginContext(PluginManifest(name="test-plugin"), manager)

    handle = ctx.register_native_turn_source(source, surfaces=("cli", "gateway"))

    assert handle.active is True
    assert manager._native_turn_sources[source.name].source is source
    assert manager._native_turn_sources[source.name].surfaces == {"cli", "gateway"}
    assert poll_native_turn(view(tmp_path)) is None
    assert source.polled == [view(tmp_path)]
    handle.dispose()
    assert handle.active is False
    assert source.name not in manager._native_turn_sources
    assert poll_native_turn(view(tmp_path)) is None
    assert source.polled == [view(tmp_path)]


def test_duplicate_name_and_invalid_source_fail_loudly(tmp_path):
    manager = PluginManager(scope_key=str(tmp_path))
    ctx = PluginContext(PluginManifest(name="one"), manager)
    ctx.register_native_turn_source(Source(), surfaces=("cli",))
    with pytest.raises(ValueError, match="already registered"):
        PluginContext(PluginManifest(name="two"), manager).register_native_turn_source(
            Source(), surfaces=("cli",)
        )
    class MissingPoll:
        name = "invalid"

        def on_user_boundary(self, _session, _reason):
            return None

    with pytest.raises(TypeError, match="poll"):
        ctx.register_native_turn_source(MissingPoll(), surfaces=("cli",))


def test_list_profile_homes_uses_canonical_public_profile_api(
    tmp_path, monkeypatch
):
    ctx = PluginContext(
        PluginManifest(name="profiles"), PluginManager(scope_key=str(tmp_path))
    )
    monkeypatch.setattr("hermes_cli.profiles.list_profile_names", lambda: ["default", "worker"])
    monkeypatch.setattr(
        "hermes_cli.profiles.get_profile_dir",
        lambda name: tmp_path / ("root" if name == "default" else f"profiles/{name}"),
    )

    homes = ctx.list_profile_homes()

    assert homes == (
        ("default", (tmp_path / "root").resolve()),
        ("worker", (tmp_path / "profiles/worker").resolve()),
    )


def test_dispatch_tool_can_use_explicit_profile_scope(
    tmp_path, monkeypatch
):
    ctx = PluginContext(
        PluginManifest(name="dispatch"), PluginManager(scope_key=str(tmp_path))
    )
    target = (tmp_path / "profiles/worker").resolve()
    observed = {}

    def fake_dispatch(name, args, **kwargs):
        from hermes_constants import get_hermes_home

        observed.update(name=name, args=args, kwargs=kwargs, home=get_hermes_home())
        return {"success": True}

    monkeypatch.setattr("tools.registry.discover_builtin_tools", lambda: None)
    monkeypatch.setattr("tools.registry.registry.dispatch", fake_dispatch)

    result = ctx.dispatch_tool(
        "cronjob_manage",
        {"action": "list", "include_disabled": True},
        profile_home=target,
    )

    assert result == {"success": True}
    assert observed["home"] == target
    assert observed["kwargs"]["scope"] == str(target)


def test_no_source_and_surface_mismatch_are_inert(tmp_path):
    source = Source()
    assert poll_registered_sources({}, view(tmp_path)) is None
    assert poll_registered_sources(registration(source, ("gateway",)), view(tmp_path)) is None
    assert source.polled == []


def test_poll_returns_bound_exactly_once_admission(tmp_path):
    calls = []
    session = view(tmp_path)
    lease = NativeTurnLease(
        lease_id="lease-1",
        prompt="continue",
        commit=lambda seen: calls.append(("commit", seen)) or True,
        abort=lambda outcome, reason: calls.append((outcome, reason)),
    )
    admission = poll_registered_sources(registration(Source(lease)), session)

    assert isinstance(admission, NativeTurnAdmission)
    assert admission.commit() is True
    assert admission.commit() is True
    assert admission.abort_if_open("canceled", "late") is False
    assert calls == [("commit", session)]


def test_false_commit_aborts_not_sent_once(tmp_path):
    calls = []
    admission = NativeTurnAdmission(
        NativeTurnLease(
            "lease-false",
            "continue",
            lambda _session: False,
            lambda outcome, reason: calls.append((outcome, reason)),
        ),
        view(tmp_path),
    )

    assert admission.commit() is False
    assert admission.commit() is False
    assert admission.abort_if_open("canceled", "late") is False
    assert calls == [("not_sent", "native lease commit declined admission")]


def test_uncertain_commit_exception_is_unknown_and_never_retried(tmp_path):
    calls = []

    def uncertain(_session):
        calls.append("commit")
        raise TimeoutError("lost acknowledgement")

    admission = NativeTurnAdmission(
        NativeTurnLease(
            "lease-unknown",
            "continue",
            uncertain,
            lambda outcome, reason: calls.append((outcome, reason)),
        ),
        view(tmp_path),
    )

    assert admission.commit() is False
    assert admission.commit() is False
    assert calls[0] == "commit"
    assert calls[1][0] == "unknown"
    assert "lost acknowledgement" in calls[1][1]


def test_boundary_is_synchronous_and_source_failures_are_isolated(tmp_path):
    session = view(tmp_path)
    source = Source()
    broken = Source()

    def fail(_session, _reason):
        raise RuntimeError("broken")

    broken.on_user_boundary = fail
    registrations = {
        "broken": RegisteredNativeTurnSource(broken, frozenset({"cli"}), "broken"),
        "ok": RegisteredNativeTurnSource(source, frozenset({"cli"}), "ok"),
    }

    fence_registered_sources(registrations, session, "session_switch")

    assert source.boundaries == [(session, "session_switch")]


def test_poll_failure_falls_through_to_next_source(tmp_path):
    session = view(tmp_path)
    broken = Source()
    broken.poll = lambda _session: (_ for _ in ()).throw(RuntimeError("broken"))
    lease = NativeTurnLease("next", "continue", lambda _session: True, lambda *_: None)
    healthy = Source(lease)
    registrations = {
        "broken": RegisteredNativeTurnSource(broken, frozenset({"cli"}), "broken"),
        "healthy": RegisteredNativeTurnSource(healthy, frozenset({"cli"}), "healthy"),
    }

    admission = poll_registered_sources(registrations, session)

    assert admission is not None
    assert admission.lease is lease


def test_boundary_waits_for_inflight_pre_model_commit(tmp_path):
    session = view(tmp_path)
    commit_started = threading.Event()
    release_commit = threading.Event()
    order = []

    def commit(_session):
        commit_started.set()
        assert release_commit.wait(timeout=2)
        order.append("commit")
        return True

    source = Source(NativeTurnLease("race", "continue", commit, lambda *_: None))
    original_boundary = source.on_user_boundary

    def boundary(session, reason):
        original_boundary(session, reason)
        order.append("boundary")

    source.on_user_boundary = boundary
    unregister = register_profile_source(
        tmp_path,
        RegisteredNativeTurnSource(source, frozenset({"cli"}), "test:race"),
    )
    try:
        admission = poll_native_turn(session)
        assert admission is not None
        commit_thread = threading.Thread(target=admission.commit)
        commit_thread.start()
        assert commit_started.wait(timeout=2)

        boundary_thread = threading.Thread(
            target=fence_native_turn_sources,
            args=(session, "stop"),
        )
        boundary_thread.start()
        assert boundary_thread.is_alive()
        release_commit.set()
        commit_thread.join(timeout=2)
        boundary_thread.join(timeout=2)
        assert not commit_thread.is_alive()
        assert not boundary_thread.is_alive()
        assert order == ["commit", "boundary"]
    finally:
        release_commit.set()
        unregister()

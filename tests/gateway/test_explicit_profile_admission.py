"""Strict admission for explicit multiplex runtime profiles.

These tests use the real profile-directory predicates under a temporary Hermes
home.  The legacy resolver may retain its fallback for its compatibility
callers; canonical admission must refuse an explicit name that is absent or
invalid before that fallback can be pinned.
"""

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource
from gateway.session_identity import identity_of


@pytest.fixture
def multiplex_runner(tmp_path, monkeypatch):
    """A real default home with one live named profile and no route fixtures."""
    default_home = tmp_path / ".hermes"
    served_home = default_home / "profiles" / "served"
    served_home.mkdir(parents=True)
    (served_home / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(default_home))

    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._primary_profile_name = "default"
    runner.adapters = {}
    runner._profile_adapters = {}
    return runner, default_home, served_home


def _source(profile):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="neutral-chat",
        chat_type="dm",
        user_id="neutral-user",
        profile=profile,
    )


@pytest.mark.parametrize(
    ("profile", "remove_before_legacy"),
    [("missing", False), ("../invalid", False), ("served", True)],
    ids=["absent", "invalid", "removed-during-legacy-fallback"],
)
def test_primary_admission_refuses_an_explicit_runtime_without_a_live_profile(
    multiplex_runner, profile, remove_before_legacy,
):
    runner, default_home, served_home = multiplex_runner
    source = _source(profile)
    if remove_before_legacy:
        native_resolver = runner._resolve_profile_home_for_source

        def resolve_after_removal(resolved_source):
            shutil.rmtree(served_home)
            return native_resolver(resolved_source)

        runner._resolve_profile_home_for_source = resolve_after_removal

    assert runner._admit_primary_source(source, default_home) is None
    assert source.profile_route_rejected is True
    assert identity_of(source) is None
    assert not hasattr(source, "_rejected_route_profile")


@pytest.mark.parametrize(
    ("profile", "remove_before_legacy"),
    [("missing", False), ("../invalid", False), ("served", True)],
    ids=["absent", "invalid", "removed-during-legacy-fallback"],
)
def test_restored_source_refuses_an_explicit_runtime_without_a_live_profile(
    multiplex_runner, profile, remove_before_legacy,
):
    runner, _default_home, served_home = multiplex_runner
    source = _source(profile)
    if remove_before_legacy:
        native_resolver = runner._resolve_profile_home_for_source

        def resolve_after_removal(resolved_source):
            shutil.rmtree(served_home)
            return native_resolver(resolved_source)

        runner._resolve_profile_home_for_source = resolve_after_removal

    restored = runner._restored_source(
        SimpleNamespace(origin=source, transport_profile="default"),
    )

    assert restored is None
    assert source.profile_route_rejected is True
    assert identity_of(source) is None
    assert not hasattr(source, "_rejected_route_profile")

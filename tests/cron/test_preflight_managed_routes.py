"""Managed-scope gateway.profile_routes must reach the satellite preflight (#121212)."""

from types import SimpleNamespace

import cron.scheduler_preflight as preflight


def _route(profile="sat"):
    return SimpleNamespace(enabled=True, profile=profile)


def _setup(monkeypatch, tmp_path, user_raw, managed_raw):
    import hermes_constants
    import hermes_cli.config as cfg
    import hermes_cli.managed_scope as ms
    import hermes_cli.profiles as profiles
    import gateway.profile_routing as pr

    primary = tmp_path / "primary"
    primary.mkdir()
    (primary / "config.yaml").write_text("gateway: {}\n")
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: primary)
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: str(tmp_path / "satellite"))
    monkeypatch.setattr(cfg, "read_user_config_raw", lambda path=None: user_raw)
    monkeypatch.setattr(ms, "load_managed_config", lambda: managed_raw)
    monkeypatch.setattr(pr, "parse_profile_routes", lambda routes: [_route()])
    monkeypatch.setattr(profiles, "profile_matches_home", lambda p: True)


def test_managed_scope_routes_resolve(monkeypatch, tmp_path):
    _setup(
        monkeypatch, tmp_path,
        user_raw={"gateway": {}},
        managed_raw={"gateway": {"profile_routes": [{"profile": "sat", "platforms": ["whatsapp"]}]}},
    )
    assert len(preflight._primary_profile_routes_for_current_home()) == 1


def test_user_file_routes_still_resolve_without_managed_scope(monkeypatch, tmp_path):
    _setup(
        monkeypatch, tmp_path,
        user_raw={"profile_routes": [{"profile": "sat", "platforms": ["whatsapp"]}]},
        managed_raw={},
    )
    assert len(preflight._primary_profile_routes_for_current_home()) == 1


def test_managed_scope_wins_over_user_file(monkeypatch, tmp_path):
    calls = []
    import gateway.profile_routing as pr

    _setup(
        monkeypatch, tmp_path,
        user_raw={"profile_routes": [{"profile": "from-user"}]},
        managed_raw={"gateway": {"profile_routes": [{"profile": "from-managed"}]}},
    )
    monkeypatch.setattr(pr, "parse_profile_routes", lambda routes: (calls.append(routes) or [_route()]))
    assert len(preflight._primary_profile_routes_for_current_home()) == 1
    assert calls == [[{"profile": "from-managed"}]]

"""Behavior contracts for profile-scoped ``hermes auth reset``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


PROVIDER = "openai-codex"
STATUS_FIELDS = (
    "last_status",
    "last_status_at",
    "last_error_code",
    "last_error_reason",
    "last_error_message",
    "last_error_reset_at",
)


def _exhausted_entry(entry_id: str) -> dict:
    """Return status-only fixture data: no credential or token material."""
    return {
        "id": entry_id,
        "label": entry_id,
        "priority": 0,
        "marker": f"preserve-{entry_id}",
        "last_status": "exhausted",
        "last_status_at": 1_711_230_000.0,
        "last_error_code": 429,
        "last_error_reason": "usage_limit_reached",
        "last_error_message": "The usage limit has been reached",
        "last_error_reset_at": 1_711_233_600.0,
    }


def _write_store(home: Path, entries: list[dict] | None) -> None:
    home.mkdir(parents=True, exist_ok=True)
    pool = {} if entries is None else {PROVIDER: entries}
    (home / "auth.json").write_text(
        json.dumps({"version": 1, "providers": {}, "credential_pool": pool}),
        encoding="utf-8",
    )


def _read_store(home: Path) -> dict:
    return json.loads((home / "auth.json").read_text(encoding="utf-8"))


def _assert_reset(home: Path) -> None:
    entry = _read_store(home)["credential_pool"][PROVIDER][0]
    assert all(entry[field] is None for field in STATUS_FIELDS)
    assert entry["marker"] == f"preserve-{entry['id']}"


@pytest.fixture()
def profile_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Build an isolated real root/profiles layout; never touch user auth."""
    fake_home = tmp_path / "home"
    root = fake_home / ".hermes"
    alpha = root / "profiles" / "alpha"
    beta = root / "profiles" / "beta"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.setenv("HERMES_HOME", str(root))
    return {"root": root, "alpha": alpha, "beta": beta}


def test_default_reset_reaches_each_named_profile(
    profile_tree: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    from hermes_cli.auth_commands import auth_reset_command

    for name, home in profile_tree.items():
        _write_store(home, [_exhausted_entry(name)])

    auth_reset_command(SimpleNamespace(provider=PROVIDER))

    assert "across 3 profiles" in capsys.readouterr().out
    for home in profile_tree.values():
        _assert_reset(home)


def test_named_reset_updates_root_fallback_without_materializing_it(
    profile_tree: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.auth_commands import auth_reset_command

    root = profile_tree["root"]
    current = profile_tree["alpha"]
    _write_store(root, [_exhausted_entry("root")])
    _write_store(current, None)
    monkeypatch.setenv("HERMES_HOME", str(current))
    current_before = (current / "auth.json").read_bytes()

    auth_reset_command(SimpleNamespace(provider=PROVIDER))

    _assert_reset(root)
    assert _read_store(current)["credential_pool"] == {}
    assert (current / "auth.json").read_bytes() == current_before


def test_current_profile_only_does_not_modify_fallback_root_store(
    profile_tree: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.auth_commands import auth_reset_command

    root = profile_tree["root"]
    current = profile_tree["alpha"]
    _write_store(root, [_exhausted_entry("root")])
    _write_store(current, [_exhausted_entry("alpha")])
    monkeypatch.setenv("HERMES_HOME", str(current))
    root_before = (root / "auth.json").read_bytes()

    auth_reset_command(
        SimpleNamespace(
            provider=PROVIDER,
            all_profiles=False,
            current_profile_only=True,
        )
    )

    _assert_reset(current)
    assert (root / "auth.json").read_bytes() == root_before


def test_all_profiles_from_named_profile_reaches_every_profile(
    profile_tree: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.auth_commands import auth_reset_command

    for name, home in profile_tree.items():
        _write_store(home, [_exhausted_entry(name)])
    monkeypatch.setenv("HERMES_HOME", str(profile_tree["alpha"]))

    auth_reset_command(
        SimpleNamespace(
            provider=PROVIDER,
            all_profiles=True,
            current_profile_only=False,
        )
    )

    for home in profile_tree.values():
        _assert_reset(home)


def test_auth_reset_parser_rejects_conflicting_profile_scopes() -> None:
    from hermes_cli.subcommands.auth import build_auth_parser

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_auth_parser(subparsers, cmd_auth=lambda _args: None)

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(
            [
                "auth",
                "reset",
                PROVIDER,
                "--all-profiles",
                "--current-profile-only",
            ]
        )

    assert exc_info.value.code == 2
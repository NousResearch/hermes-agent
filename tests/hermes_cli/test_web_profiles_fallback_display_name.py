"""Regression tests: the ``/api/profiles`` fallback payload keeps ``display_name``.

``list_profiles_endpoint`` serves ``_profile_to_dict`` rows on the happy path,
which have carried ``display_name`` since the payload grew one. When
``profiles_mod.list_profiles`` raises, it falls back to
``_fallback_profile_dicts`` — a directory scan whose entries previously
omitted ``display_name``, so a renamed profile (``hermes profile rename
default "Legal writing"``) would lose its label in the dashboard profile
switcher exactly when the server was already degraded (#103251).

These tests pin the fallback entry's contract: ``display_name`` is read from
``profile.yaml`` via the same ``read_profile_meta`` the CLI uses, and is an
empty string (not a missing key) when unset.
"""

from pathlib import Path
from types import SimpleNamespace

from hermes_cli.profiles import read_profile_meta
from hermes_cli.web_server_profiles import _fallback_profile_entry


def _fake_profiles_mod():
    return SimpleNamespace(
        _read_config_model=lambda home: (None, None),
        read_profile_meta=read_profile_meta,
        _count_skills=lambda home: 0,
        _check_gateway_running=lambda home: False,
    )


def test_fallback_entry_reads_display_name_from_profile_yaml(tmp_path: Path) -> None:
    (tmp_path / "profile.yaml").write_text("display_name: Legal writing\n")

    entry = _fallback_profile_entry(
        _fake_profiles_mod(), "default", tmp_path,
        is_default=True, has_env=False, gateway_running=lambda: False,
    )

    assert entry["display_name"] == "Legal writing"


def test_fallback_entry_defaults_display_name_to_empty_string(tmp_path: Path) -> None:
    entry = _fallback_profile_entry(
        _fake_profiles_mod(), "work", tmp_path,
        is_default=False, has_env=False, gateway_running=lambda: False,
    )

    assert entry["display_name"] == ""

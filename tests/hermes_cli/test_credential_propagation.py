"""Multi-profile fan-out for a rotated credential (dashboard path).

Rotating a key used to update ONLY the active profile's ``.env``; every sibling kept the
dead value and kept 401'ing (real incident: 11 profiles, 10 stale copies, every agent
401'ing plus a memory backend that could no longer authenticate).

These tests drive the real propagation core and the real ``PUT /api/env`` handler against
temp installs (``tests/conftest.py`` isolation): no real profile ``.env`` is ever touched.
"""

import json
import os
import re

import pytest
from fastapi.testclient import TestClient

from hermes_cli import credential_lifecycle
from hermes_cli.web_server import _SESSION_TOKEN, app

client = TestClient(app)
HEADERS = {"X-Hermes-Session-Token": _SESSION_TOKEN}

KEY = "ZAI_API_KEY"
# Runtime-constructed fake credentials: no key-shaped literal ever lands in the repo.
OLD_KEY = "zk-" + "a" * 24
NEW_KEY = "zk-" + "b" * 24
OTHER_KEY = "zk-" + "c" * 24


def _invalidate():
    from hermes_cli.config import invalidate_env_cache

    invalidate_env_cache()


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read(path):
    return path.read_text(encoding="utf-8")


def _snapshot(root):
    """Every file's (relative path, mtime_ns, size, content) under ``root``."""
    out = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            stat = os.stat(path)
            with open(path, "rb") as handle:
                out[os.path.relpath(path, root)] = (stat.st_mtime_ns, stat.st_size, handle.read())
    return out


@pytest.fixture
def install(tmp_path, monkeypatch):
    """Temp HERMES_HOME + sibling profiles:

    alpha  defines KEY (old) and mirrors it in config.yaml
    beta   defines KEY (old), no config.yaml mirror
    gamma  deliberately never had KEY (only OTHER) — must never be provisioned
    delta  defines KEY (old) but is TOMBSTONED (profiles/.deleted/delta)
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    _invalidate()

    _write(home / ".env", f"{KEY}={OLD_KEY}\n")
    _write(
        home / "config.yaml",
        "model:\n  provider: custom\n  default: active-model\n"
        f"  api_key: {OLD_KEY}\n",
    )

    profiles = home / "profiles"
    _write(profiles / "alpha" / ".env", f"{KEY}={OLD_KEY}\nOTHER=keep\n")
    _write(
        profiles / "alpha" / "config.yaml",
        "model:\n  provider: custom\n"
        "  default: my-model\n"
        f"  api_key: {OLD_KEY}\n",
    )
    _write(profiles / "beta" / ".env", f"{KEY}={OLD_KEY}\n")
    _write(profiles / "beta" / "config.yaml", "model:\n  default: beta-model\n")
    _write(profiles / "gamma" / ".env", "OTHER=keep\n")
    _write(profiles / "gamma" / "config.yaml", "model:\n  default: gamma-model\n")
    _write(profiles / "delta" / ".env", f"{KEY}={OLD_KEY}\n")
    _write(profiles / "delta" / "config.yaml", "model:\n  default: delta-model\n")
    (profiles / ".deleted").mkdir(parents=True, exist_ok=True)
    (profiles / ".deleted" / "delta").write_text("deleted\n", encoding="utf-8")

    return home


# ---------------------------------------------------------------------------
# Core: enumeration
# ---------------------------------------------------------------------------


def test_only_siblings_that_already_define_the_var(install):
    """Candidates = live sibling profile dirs whose .env DEFINES the var.

    gamma (never had it), delta (tombstoned) and the hidden ``.deleted`` dir are excluded;
    the active home is never one of its own siblings.
    """
    found = credential_lifecycle.sibling_profiles_with_env_var(KEY)
    assert [name for name, _home in found] == ["alpha", "beta"]

    active_name = install / "profiles" / "alpha"
    found = credential_lifecycle.sibling_profiles_with_env_var(KEY, active_home=active_name)
    assert [name for name, _home in found] == ["beta"]


def test_dedup_by_resolved_path(install):
    """A second path resolving onto an already-seen home is reported once."""
    (install / "profiles" / "alpha-alias").symlink_to(install / "profiles" / "alpha")

    found = credential_lifecycle.sibling_profiles_with_env_var(KEY)
    assert [name for name, _home in found] == ["alpha", "beta"]


def test_no_profiles_dir_is_not_an_error(tmp_path, monkeypatch):
    home = tmp_path / "bare"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    _invalidate()
    assert credential_lifecycle.sibling_profiles_with_env_var(KEY) == []


# ---------------------------------------------------------------------------
# Core: dry run
# ---------------------------------------------------------------------------


def test_dry_run_touches_nothing(install):
    before = _snapshot(install)

    plan = credential_lifecycle.propagate_env_credential_to_siblings(
        KEY, NEW_KEY, old_value=OLD_KEY, apply=False
    )

    assert _snapshot(install) == before, "dry run must not write, chmod, or create anything"
    assert plan["dry_run"] is True
    assert plan["updated"] == 0 and plan["failed"] == 0
    assert [(r["profile"], r["status"], r["stale"]) for r in plan["results"]] == [
        ("alpha", "pending", True),
        ("beta", "pending", True),
    ]
    assert plan["pending"] == 2


# ---------------------------------------------------------------------------
# Core: apply
# ---------------------------------------------------------------------------


def test_apply_writes_env_and_scrubs_the_mirror(install):
    result = credential_lifecycle.propagate_env_credential_to_siblings(
        KEY, NEW_KEY, old_value=OLD_KEY, apply=True
    )

    assert result["updated"] == 2, result
    assert result["failed"] == 0, result
    assert {r["profile"]: r["status"] for r in result["results"]} == {
        "alpha": "updated",
        "beta": "updated",
    }

    for name in ("alpha", "beta"):
        env_text = _read(install / "profiles" / name / ".env")
        assert f"{KEY}={NEW_KEY}" in env_text
        assert OLD_KEY not in env_text

    # alpha's config.yaml mirror of the OLD value is scrubbed, and reported as such.
    alpha_cfg = _read(install / "profiles" / "alpha" / "config.yaml")
    assert OLD_KEY not in alpha_cfg, "stale mirror still shadows the rotated key"
    assert NEW_KEY in alpha_cfg
    alpha_row = next(r for r in result["results"] if r["profile"] == "alpha")
    assert "model.api_key" in alpha_row["config_updates"]

    # gamma never had the key: it must not be provisioned.
    assert KEY not in _read(install / "profiles" / "gamma" / ".env")
    assert "gamma" not in {r["profile"] for r in result["results"]}
    # The tombstoned profile is left alone.
    assert OLD_KEY in _read(install / "profiles" / "delta" / ".env")

    # The active profile was not the fan-out's business.
    assert OLD_KEY in _read(install / ".env")


def test_already_current_sibling_is_skipped(install):
    _write(install / "profiles" / "beta" / ".env", f"{KEY}={NEW_KEY}\n")

    result = credential_lifecycle.propagate_env_credential_to_siblings(
        KEY, NEW_KEY, old_value=OLD_KEY, apply=True
    )

    assert result["updated"] == 1 and result["skipped"] == 1
    row = next(r for r in result["results"] if r["profile"] == "beta")
    assert row["status"] == "skipped"
    assert "already holds the new value" in row["reason"]


def test_write_failure_is_reported_and_does_not_abort_the_others(install, monkeypatch):
    """alpha's write blows up; beta must still be updated and alpha reported as failed."""
    real = credential_lifecycle.save_provider_env_credential
    failing = (install / "profiles" / "alpha").resolve()

    def flaky(env_var, value):
        from hermes_constants import get_hermes_home

        if get_hermes_home().resolve() == failing:
            raise OSError("no space left on device")
        return real(env_var, value)

    monkeypatch.setattr(credential_lifecycle, "save_provider_env_credential", flaky)

    result = credential_lifecycle.propagate_env_credential_to_siblings(
        KEY, NEW_KEY, old_value=OLD_KEY, apply=True
    )

    rows = {r["profile"]: r for r in result["results"]}
    assert rows["alpha"]["status"] == "failed"
    assert "OSError" in rows["alpha"]["reason"]
    assert rows["beta"]["status"] == "updated"
    assert result == {**result, "failed": 1, "updated": 1}
    # The failure did not partially write alpha.
    assert OLD_KEY in _read(install / "profiles" / "alpha" / ".env")
    # ... and beta really was written.
    assert NEW_KEY in _read(install / "profiles" / "beta" / ".env")


def test_refused_write_is_reported_as_failed(install, monkeypatch):
    """A silently refused save (managed/administrator lock) must not be reported as updated."""
    monkeypatch.setattr(
        credential_lifecycle, "save_provider_env_credential",
        lambda env_var, value: {"ok": True, "key": env_var, "config_updates": []},
    )
    result = credential_lifecycle.propagate_env_credential_to_siblings(
        KEY, NEW_KEY, old_value=OLD_KEY, apply=True
    )
    assert result["updated"] == 0
    assert result["failed"] == 2
    for row in result["results"]:
        assert row["status"] == "failed"


# ---------------------------------------------------------------------------
# API: PUT /api/env
# ---------------------------------------------------------------------------


def test_put_without_the_flag_is_unchanged(install):
    """Flag absent = today's behaviour exactly: one .env write, the same response fields."""
    resp = client.put(
        "/api/env", json={"key": KEY, "value": NEW_KEY}, headers=HEADERS
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"ok": True, "key": KEY, "config_updates": ["model.api_key"]}
    assert "propagation" not in resp.text

    # Primary write landed; siblings untouched.
    assert NEW_KEY in _read(install / ".env")
    for name in ("alpha", "beta"):
        assert OLD_KEY in _read(install / "profiles" / name / ".env")


def test_put_dry_run_is_read_only_and_lists_siblings(install):
    before = _snapshot(install)

    resp = client.put(
        "/api/env",
        json={"key": KEY, "value": NEW_KEY, "propagate_to_profiles": "dry_run"},
        headers=HEADERS,
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["dry_run"] is True
    propagation = body["propagation"]
    assert propagation["dry_run"] is True
    assert propagation["pending"] == 2
    assert [r["profile"] for r in propagation["results"]] == ["alpha", "beta"]
    assert _snapshot(install) == before, "a preview must not write the active .env either"


def test_put_apply_fans_out_with_per_profile_results(install):
    resp = client.put(
        "/api/env",
        json={"key": KEY, "value": NEW_KEY, "propagate_to_profiles": True},
        headers=HEADERS,
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True and body["key"] == KEY
    propagation = body["propagation"]
    assert propagation["dry_run"] is False
    assert propagation["updated"] == 2
    assert {r["profile"]: r["status"] for r in propagation["results"]} == {
        "alpha": "updated",
        "beta": "updated",
    }

    assert NEW_KEY in _read(install / ".env")
    for name in ("alpha", "beta"):
        assert NEW_KEY in _read(install / "profiles" / name / ".env")
    assert OLD_KEY not in _read(install / "profiles" / "alpha" / "config.yaml")
    assert KEY not in _read(install / "profiles" / "gamma" / ".env")


def test_put_flag_absence_and_false_are_identical(install):
    absent = client.put(
        "/api/env", json={"key": KEY, "value": NEW_KEY}, headers=HEADERS
    )
    explicit_false = client.put(
        "/api/env",
        json={"key": KEY, "value": OTHER_KEY, "propagate_to_profiles": False},
        headers=HEADERS,
    )
    assert absent.status_code == explicit_false.status_code == 200
    assert set(explicit_false.json()) == set(absent.json())


# Runs last: a profile-scoped request flips this process to multi-profile hosting.
def test_put_with_profile_scope_keeps_the_primary_write_and_excludes_it(install):
    resp = client.put(
        f"/api/env?profile=alpha",
        json={"key": KEY, "value": NEW_KEY, "propagate_to_profiles": "apply"},
        headers=HEADERS,
    )

    assert resp.status_code == 200, resp.text
    propagation = resp.json()["propagation"]
    assert [r["profile"] for r in propagation["results"]] == ["beta"]

    assert NEW_KEY in _read(install / "profiles" / "alpha" / ".env")
    assert NEW_KEY in _read(install / "profiles" / "beta" / ".env")
    # The dashboard's own profile was NOT the target of a profile-scoped write.
    assert OLD_KEY in _read(install / ".env")

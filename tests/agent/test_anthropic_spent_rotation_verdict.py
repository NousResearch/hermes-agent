"""A rotation that was consumed but never committed must stay unusable.

``_refresh_oauth_token()`` already refuses to return an access token whose
refresh half was lost to a failed write.  That verdict was local: the caller
above it (``resolve_anthropic_token()``) simply continued to the next source,
and source 5 (``_resolve_anthropic_pool_token``) enumerates read-only
(``clear_expired=False, refresh=False``) over a pool that ``load_pool()`` has
just re-seeded from the *unchanged* singleton file.  So the very pair whose
refresh token the POST had already spent came back as a healthy token, and
``_refresh_provider_credentials("anthropic")`` reported the refresh as a
success and evicted its cached clients.

That is the same silent-transition failure the fail-closed path exists to
prevent, one layer up: no ``invalid_grant`` is raised until the *next* refresh,
by which point the provenance of the failure is gone.

These tests take the full resolver path, not just the writer: successful POST +
failed commit must make ``resolve_anthropic_token()`` return ``None`` (or a
genuinely independent credential), must make
``_refresh_provider_credentials("anthropic")`` return ``False`` when the spent
family is the only credential, and must keep the spent fingerprint out of every
lease.

Companion: ``test_anthropic_credential_persist_failure.py`` covers the writers
and the pool quarantine; this file covers what resolution does afterwards.
"""

from __future__ import annotations
from hermes_cli.config_credentials import credential_pool_environment as _phase6_auth_environment

import auth.providers.anthropic as _auth_auth_providers_anthropic


import json
import os
import time

import pytest

import auth.providers.anthropic as AA
from agent.auxiliary_client import _refresh_provider_credentials
from auth.credential_pool import AUTH_TYPE_OAUTH

_EXPIRED_MS = 1_000

_STALE_ACCESS = "sk-ant-oat01-spent-stale"
_STALE_REFRESH = "sk-ant-ort01-spent-stale"
_ROTATED_ACCESS = "sk-ant-oat01-spent-rotated"
_ROTATED_REFRESH = "sk-ant-ort01-spent-rotated"
_INDEPENDENT_ACCESS = "sk-ant-oat01-independent"
_INDEPENDENT_REFRESH = "sk-ant-ort01-independent"

_SINGLETON_FILENAMES = frozenset({".credentials.json", ".anthropic_oauth.json"})


@pytest.fixture(autouse=True)
def _clean_spent_registry():
    """The consumed-rotation registry is process-global; isolate each test."""
    _auth_auth_providers_anthropic._SPENT_ROTATION_FINGERPRINTS.clear()
    yield
    _auth_auth_providers_anthropic._SPENT_ROTATION_FINGERPRINTS.clear()


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    (home / "auth.json").write_text(
        json.dumps({"version": 1, "providers": {}}), encoding="utf-8"
    )
    monkeypatch.setattr(
        "hermes_cli.auth.is_provider_explicitly_configured", lambda pid: True
    )
    return home


@pytest.fixture
def claude_credentials(tmp_path, monkeypatch):
    cred_path = tmp_path / "claude" / ".credentials.json"
    cred_path.parent.mkdir(parents=True, exist_ok=True)
    cred_path.write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": _STALE_ACCESS,
                    "refreshToken": _STALE_REFRESH,
                    "expiresAt": _EXPIRED_MS,
                    "scopes": ["user:inference", "user:profile"],
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(_auth_auth_providers_anthropic, "claude_code_credentials_path", lambda: cred_path)
    monkeypatch.setattr(_auth_auth_providers_anthropic, "_read_claude_code_credentials_from_keychain", lambda: None)
    return cred_path


def _rotating_refresh(*_a, **_kw):
    """Stand-in for the token endpoint: the POST always succeeds and rotates."""
    return {
        "access_token": _ROTATED_ACCESS,
        "refresh_token": _ROTATED_REFRESH,
        "expires_at_ms": int(time.time() * 1000) + 3_600_000,
    }


def _break_durable_write(monkeypatch):
    """Only the authoritative singletons fail; auth.json must stay writable."""
    real_replace = os.replace

    def _failing_replace(src, dst):
        if os.path.basename(os.fspath(dst)) in _SINGLETON_FILENAMES:
            raise OSError(13, "Permission denied")
        return real_replace(src, dst)

    monkeypatch.setattr(AA.os, "replace", _failing_replace)


def _add_independent_pool_entry(home):
    """Persist a second, unrelated Anthropic OAuth credential in the pool."""
    path = home / "auth.json"
    store = json.loads(path.read_text(encoding="utf-8"))
    pool = store.setdefault("credential_pool", {})
    pool.setdefault("anthropic", []).append(
        {
            "id": "anthropic-independent",
            "label": "second subscription",
            "auth_type": AUTH_TYPE_OAUTH,
            "priority": 10,
            "source": "manual",
            "access_token": _INDEPENDENT_ACCESS,
            "refresh_token": _INDEPENDENT_REFRESH,
            "expires_at_ms": int(time.time() * 1000) + 3_600_000,
        }
    )
    path.write_text(json.dumps(store), encoding="utf-8")


# ---------------------------------------------------------------------------
# The registry itself
# ---------------------------------------------------------------------------


def test_registry_matches_only_the_recorded_secret():
    _auth_auth_providers_anthropic.mark_rotation_consumed_uncommitted(_STALE_REFRESH, "", None)

    assert _auth_auth_providers_anthropic.is_rotation_consumed_uncommitted(_STALE_REFRESH)
    assert not _auth_auth_providers_anthropic.is_rotation_consumed_uncommitted(_INDEPENDENT_REFRESH)
    assert not _auth_auth_providers_anthropic.is_rotation_consumed_uncommitted("")
    assert not _auth_auth_providers_anthropic.is_rotation_consumed_uncommitted(None)


def test_registry_stays_bounded():
    for i in range(_auth_auth_providers_anthropic._SPENT_ROTATION_MAX_TRACKED * 2):
        _auth_auth_providers_anthropic.mark_rotation_consumed_uncommitted(f"sk-ant-ort01-{i}")

    assert len(_auth_auth_providers_anthropic._SPENT_ROTATION_FINGERPRINTS) == _auth_auth_providers_anthropic._SPENT_ROTATION_MAX_TRACKED
    assert _auth_auth_providers_anthropic.is_rotation_consumed_uncommitted(
        f"sk-ant-ort01-{_auth_auth_providers_anthropic._SPENT_ROTATION_MAX_TRACKED * 2 - 1}"
    ), "the most recent rotation must survive eviction"


# ---------------------------------------------------------------------------
# Full resolution
# ---------------------------------------------------------------------------




def test_resolve_returns_none_when_the_rotation_could_not_commit(
    hermes_home, claude_credentials, monkeypatch
):
    """Full resolver: source 5 must not hand back the pair source 4 refused.

    ``load_pool()`` re-seeds the claude_code row straight from the unchanged
    credentials file, so without the consumed-rotation verdict this returns the
    already-spent access token and the caller sees a success.
    """
    monkeypatch.setattr(_auth_auth_providers_anthropic, "refresh_anthropic_oauth_pure", _rotating_refresh)
    _break_durable_write(monkeypatch)

    assert _auth_auth_providers_anthropic.resolve_anthropic_token(environment=_phase6_auth_environment()) is None, (
        "a consumed-but-uncommitted rotation must not resolve to a usable token"
    )




def test_auxiliary_refresh_reports_failure_for_a_lost_commit(
    hermes_home, claude_credentials, monkeypatch
):
    """``_refresh_provider_credentials`` must fail when this is the only credential.

    Returning True here evicts the cached clients and tells the retry loop the
    provider recovered, which is the point at which the failure stops being
    visible anywhere.
    """
    monkeypatch.setattr(_auth_auth_providers_anthropic, "refresh_anthropic_oauth_pure", _rotating_refresh)
    _break_durable_write(monkeypatch)

    assert _refresh_provider_credentials("anthropic", failed_api_key=_STALE_ACCESS) is False


def test_independent_pool_credential_stays_eligible(
    hermes_home, claude_credentials, monkeypatch
):
    """Failing closed is scoped to the spent family, not to Anthropic as a whole."""
    _add_independent_pool_entry(hermes_home)
    monkeypatch.setattr(_auth_auth_providers_anthropic, "refresh_anthropic_oauth_pure", _rotating_refresh)
    _break_durable_write(monkeypatch)

    resolved = _auth_auth_providers_anthropic.resolve_anthropic_token(environment=_phase6_auth_environment())

    assert resolved == _INDEPENDENT_ACCESS, (
        "an unrelated credential must still be selectable after the quarantine"
    )


def test_successful_commit_leaves_the_credential_usable(
    hermes_home, claude_credentials, monkeypatch
):
    """Control: nothing is quarantined when the commit actually lands."""
    monkeypatch.setattr(_auth_auth_providers_anthropic, "refresh_anthropic_oauth_pure", _rotating_refresh)

    assert _auth_auth_providers_anthropic.resolve_anthropic_token(environment=_phase6_auth_environment()) == _ROTATED_ACCESS
    assert _auth_auth_providers_anthropic._SPENT_ROTATION_FINGERPRINTS == {}


# ---------------------------------------------------------------------------
# Cross-process durability of the verdict (sidecar registry)
# ---------------------------------------------------------------------------


def test_failed_commit_persists_the_verdict_to_the_sidecar(
    hermes_home, claude_credentials, monkeypatch
):
    """The verdict must outlive this process: it lands in the sidecar file."""
    monkeypatch.setattr(_auth_auth_providers_anthropic, "refresh_anthropic_oauth_pure", _rotating_refresh)
    _break_durable_write(monkeypatch)

    assert _auth_auth_providers_anthropic._refresh_oauth_token(_auth_auth_providers_anthropic.read_claude_code_credentials(environment=_phase6_auth_environment()), environment=_phase6_auth_environment()) is None

    sidecar = _auth_auth_providers_anthropic._spent_rotation_sidecar_path(claude_credentials)
    assert sidecar.exists(), "the terminal verdict must be durably persisted"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    fingerprints = set(payload["fingerprints"])
    from auth.persistence import fingerprint_secret_value

    assert fingerprint_secret_value(_STALE_REFRESH) in fingerprints
    assert fingerprint_secret_value(_STALE_ACCESS) in fingerprints
    # Non-secret invariant: no raw token material may reach the sidecar.
    raw = sidecar.read_text(encoding="utf-8")
    for secret in (_STALE_ACCESS, _STALE_REFRESH, _ROTATED_ACCESS, _ROTATED_REFRESH):
        assert secret not in raw


_SECOND_PROCESS_WITNESS = '\nfrom hermes_cli.config_credentials import credential_pool_environment as _phase6_auth_environment\n\nimport json\nimport sys\n\ncred_path_str, sidecar_dir = sys.argv[1], sys.argv[2]\n\nfrom pathlib import Path\n\nimport auth.providers.anthropic as AA\n\ncred_path = Path(cred_path_str)\nAA.claude_code_credentials_path = lambda: cred_path\nAA._read_claude_code_credentials_from_keychain = lambda *a, **k: None\n\nposted = []\n\n\ndef _must_not_post(refresh_token, *a, **kw):\n    posted.append(refresh_token)\n    raise AssertionError("process B replayed a spent refresh token")\n\n\nAA.refresh_anthropic_oauth_pure = _must_not_post\n\ncreds = AA.read_claude_code_credentials(environment=_phase6_auth_environment())\nresult = {\n    "registry_empty": len(AA._SPENT_ROTATION_FINGERPRINTS) == 0,\n    "sidecar_verdict_access": AA.is_rotation_consumed_uncommitted(\n        creds["accessToken"], source_path=cred_path\n    ),\n    "sidecar_verdict_refresh": AA.is_rotation_consumed_uncommitted(\n        creds["refreshToken"], source_path=cred_path\n    ),\n    "resolved": AA._resolve_claude_code_token_from_credentials(creds, environment=_phase6_auth_environment()),\n    "posted": posted,\n}\nprint(json.dumps(result))\n'


def test_second_process_adopts_the_terminal_verdict(
    hermes_home, claude_credentials, monkeypatch, tmp_path
):
    """Two-process witness: A rotates and loses the commit; B fails closed.

    Process B runs in a fresh interpreter whose process-local registry is
    empty, sharing only the credential file (and its sidecar). B must neither
    lease the stale access token nor POST the spent refresh token — the exact
    cross-process gap the process-local OrderedDict could not cover.
    """
    import subprocess
    import sys

    # Process A: successful POST, failed durable commit.
    monkeypatch.setattr(_auth_auth_providers_anthropic, "refresh_anthropic_oauth_pure", _rotating_refresh)
    _break_durable_write(monkeypatch)
    assert _auth_auth_providers_anthropic._refresh_oauth_token(_auth_auth_providers_anthropic.read_claude_code_credentials(environment=_phase6_auth_environment()), environment=_phase6_auth_environment()) is None
    assert _auth_auth_providers_anthropic._spent_rotation_sidecar_path(claude_credentials).exists()

    # Process B: fresh interpreter, same shared credential source.
    import agent as _agent_pkg

    repo_root = str(
        __import__("pathlib").Path(_agent_pkg.__file__).resolve().parents[1]
    )
    env = dict(os.environ)
    env["HERMES_HOME"] = str(hermes_home)
    env["PYTHONPATH"] = repo_root
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        env.pop(var, None)

    proc = subprocess.run(
        [sys.executable, "-c", _SECOND_PROCESS_WITNESS, str(claude_credentials), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        stdin=subprocess.DEVNULL,
    )
    assert proc.returncode == 0, f"witness failed:\n{proc.stdout}\n{proc.stderr}"
    verdict = json.loads(proc.stdout.strip().splitlines()[-1])

    assert verdict["registry_empty"], "precondition: B must start with no local verdict"
    assert verdict["sidecar_verdict_access"], "B must see A's verdict via the sidecar"
    assert verdict["sidecar_verdict_refresh"]
    assert verdict["resolved"] is None, "B must not lease the stale access token"
    assert verdict["posted"] == [], "B must not POST the spent refresh token"


def test_control_second_process_without_sidecar_still_resolves(
    hermes_home, claude_credentials, monkeypatch
):
    """Independent-credential control: no verdict, no quarantine.

    With no failed rotation recorded anywhere, the shared file's (valid)
    credential resolves normally in this process — proving the sidecar gate
    only fires on a recorded verdict, not on every read.
    """
    fresh = dict(
        json.loads(claude_credentials.read_text(encoding="utf-8"))
    )
    fresh["claudeAiOauth"]["expiresAt"] = int(time.time() * 1000) + 3_600_000
    claude_credentials.write_text(json.dumps(fresh), encoding="utf-8")

    assert not _auth_auth_providers_anthropic._spent_rotation_sidecar_path(claude_credentials).exists()
    creds = _auth_auth_providers_anthropic.read_claude_code_credentials(environment=_phase6_auth_environment())
    assert _auth_auth_providers_anthropic._resolve_claude_code_token_from_credentials(creds, environment=_phase6_auth_environment()) == _STALE_ACCESS

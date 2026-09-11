"""Regression tests: off-scope credential authority must never rewrite a secret.

Under ``gateway.multiplex_profiles`` a background thread has no profile secret
scope, so ``get_secret`` fails closed with ``UnscopedSecretError``. Tolerating
that for *availability* (the daemon must still start) is not the same as
granting that context *authority over the credential*.

The embedded-daemon starter reconciles the persisted profile ``.env`` against a
freshly built expectation, and rewrites the file on any difference. If the
off-scope key resolution degrades to ``""``, a perfectly valid persisted secret
compares as drift, and the rewrite truncates the file and replaces the real key
with an empty one — destroying the very value the fallback claims to defer to.

The boundary these tests pin down:

  * secret-bearing materialization happens where the exact profile secret scope
    is present, or with an explicit caller-supplied credential snapshot;
  * an off-scope reconcile may update non-credential settings but must leave the
    persisted credential byte-for-byte intact.
"""

import os
import threading
from pathlib import Path

import pytest

from agent.secret_scope import set_multiplex_active
from plugins.memory.hindsight import (
    _embedded_profile_env_path,
    _materialize_embedded_profile_env,
)
from plugins.memory.hindsight.embedded import _load_simple_env

_PROFILE_SECRET = "profile-secret"
_KEY = "HINDSIGHT_API_LLM_API_KEY"

_CONFIG = {
    "profile": "hermes",
    "llm_provider": "openai",
    "llm_model": "gpt-4o-mini",
}


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Keep every write inside tmp_path — never the real ~/.hindsight."""
    isolated_home = tmp_path / "user-home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: isolated_home))
    return isolated_home


@pytest.fixture
def multiplex_on():
    set_multiplex_active(True)
    try:
        yield
    finally:
        set_multiplex_active(False)


@pytest.fixture
def seeded_profile_env():
    """An already-materialized profile env holding a valid credential."""
    profile_env = _embedded_profile_env_path(_CONFIG)
    profile_env.parent.mkdir(parents=True, exist_ok=True)
    profile_env.write_text(
        "HINDSIGHT_API_LLM_PROVIDER=openai\n"
        f"{_KEY}={_PROFILE_SECRET}\n"
        "HINDSIGHT_API_LLM_MODEL=gpt-4o-mini\n"
        "HINDSIGHT_API_LOG_LEVEL=info\n",
        encoding="utf-8",
    )
    os.chmod(profile_env, 0o600)
    return profile_env


def _run_off_scope(fn):
    """Run ``fn`` on a bare thread — no profile secret scope installed."""
    result = {}

    def target():
        try:
            result["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - asserted on by type
            result["error"] = exc

    t = threading.Thread(target=target)
    t.start()
    t.join(timeout=30)
    assert not t.is_alive(), "off-scope thread hung"
    return result


# ---------------------------------------------------------------------------
# The adversarial case: off-scope startup reconciliation must not erase a key.
# ---------------------------------------------------------------------------


def test_off_scope_reconcile_preserves_persisted_credential_bytes(
    multiplex_on, seeded_profile_env
):
    """Off-scope daemon startup must leave the credential byte-identical.

    This is the destructive path: multiplex active, daemon worker with no secret
    scope, reconciliation runs. The file must not be rewritten with an empty key.
    """
    before = seeded_profile_env.read_bytes()

    from plugins.memory.hindsight import _reconcile_embedded_profile_env

    result = _run_off_scope(lambda: _reconcile_embedded_profile_env(_CONFIG))
    assert "error" not in result, f"reconcile must not raise: {result.get('error')!r}"

    after = seeded_profile_env.read_bytes()
    assert _load_simple_env(seeded_profile_env)[_KEY] == _PROFILE_SECRET, (
        "off-scope reconciliation replaced a valid profile credential with an "
        "empty value; unknown credential authority must preserve the secret"
    )
    assert after == before, "off-scope reconciliation rewrote the profile env file"


def test_off_scope_reconcile_reports_no_credential_drift(
    multiplex_on, seeded_profile_env
):
    """An unresolvable off-scope key is *unknown*, not *changed*.

    A missing secret scope must not be reported as drift, or the caller will
    restart a healthy daemon on every startup.
    """
    from plugins.memory.hindsight import _reconcile_embedded_profile_env

    result = _run_off_scope(lambda: _reconcile_embedded_profile_env(_CONFIG))
    assert result.get("value") is False, (
        "off-scope reconcile claimed the profile env changed, which would "
        "trigger a spurious daemon restart loop"
    )


def test_on_scope_materialization_still_writes_the_credential(seeded_profile_env):
    """The on-scope boundary keeps full authority: an explicit snapshot wins."""
    _materialize_embedded_profile_env(_CONFIG, llm_api_key="sk-rotated")

    assert _load_simple_env(seeded_profile_env)[_KEY] == "sk-rotated"


def test_off_scope_reconcile_updates_non_credential_settings(
    multiplex_on, seeded_profile_env
):
    """Non-credential drift is still reconciled off-scope, credential preserved.

    Preserving the secret must not freeze the whole file: a genuinely changed
    model/provider still reconciles, while the credential is carried forward
    from the persisted value rather than overwritten with an empty string.
    """
    from plugins.memory.hindsight import _reconcile_embedded_profile_env

    changed = dict(_CONFIG, llm_model="gpt-4o")
    result = _run_off_scope(lambda: _reconcile_embedded_profile_env(changed))
    assert "error" not in result, f"reconcile must not raise: {result.get('error')!r}"
    assert result.get("value") is True, "real non-credential drift should reconcile"

    values = _load_simple_env(seeded_profile_env)
    assert values["HINDSIGHT_API_LLM_MODEL"] == "gpt-4o"
    assert values[_KEY] == _PROFILE_SECRET, (
        "reconciling a non-credential setting must carry the persisted "
        "credential forward, not blank it"
    )

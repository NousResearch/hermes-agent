"""Tests for the Vertex AI adapter (agent/vertex_adapter.py).

Vertex uses OAuth2 (short-lived access tokens from a service-account JSON or
ADC), NOT a static API key. These tests mock google-auth entirely — no network
calls — and cover token minting, the config.yaml→env precedence bridge, the
global vs regional base-URL shapes, and the ADC→service-account fallback.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest


def _install_fake_google_auth(monkeypatch, *, adc_ok=True, adc_project="adc-project",
                              sa_project="sa-project", token="ya29.FAKE"):
    """Register a fake google-auth tree in sys.modules and return the module set."""
    ga = types.ModuleType("google.auth")
    gt = types.ModuleType("google.auth.transport")
    gtr = types.ModuleType("google.auth.transport.requests")
    go = types.ModuleType("google.oauth2")
    gsa = types.ModuleType("google.oauth2.service_account")
    gp = types.ModuleType("google")

    gtr.Request = type("Request", (), {})

    class _Creds:
        def __init__(self):
            self.token = None
            self.expiry = None
            self.expired = False

        def refresh(self, req):
            self.token = token

    def _default(scopes=None):
        if not adc_ok:
            raise RuntimeError("Could not automatically determine credentials")
        return _Creds(), adc_project

    ga.default = _default
    ga.transport = gt
    gt.requests = gtr

    class _SA:
        @staticmethod
        def from_service_account_file(path, scopes=None):
            c = _Creds()
            c.project_id = sa_project
            return c

        @staticmethod
        def from_service_account_info(info, scopes=None):
            c = _Creds()
            c.project_id = sa_project
            return c

    gsa.Credentials = _SA
    go.service_account = gsa
    gp.auth = ga
    gp.oauth2 = go

    for name, mod in [
        ("google", gp), ("google.auth", ga), ("google.auth.transport", gt),
        ("google.auth.transport.requests", gtr), ("google.oauth2", go),
        ("google.oauth2.service_account", gsa),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)
    return gp


@pytest.fixture
def vertex_adapter(monkeypatch):
    """Fresh vertex_adapter with a fake google-auth and clean caches/env."""
    for var in ("VERTEX_CREDENTIALS_PATH", "GOOGLE_APPLICATION_CREDENTIALS",
                "VERTEX_PROJECT_ID", "VERTEX_REGION", "GOOGLE_CLOUD_PROJECT"):
        monkeypatch.delenv(var, raising=False)
    _install_fake_google_auth(monkeypatch)
    import agent.vertex_adapter as va
    va = importlib.reload(va)
    va._creds_cache.clear()
    # Neutralize config.yaml by default; individual tests re-patch _vertex_config.
    monkeypatch.setattr(va, "_vertex_config", lambda: {})
    return va












def test_rejected_fresh_token_is_reminted_for_main_client(vertex_adapter, monkeypatch):
    from agent.client_lifecycle import ClientLifecycleMixin

    token, base_url = vertex_adapter.get_vertex_config()
    creds = next(iter(vertex_adapter._creds_cache.values()))[0]
    creds.token = "rejected-but-not-expired"
    refreshed = []

    def refresh(request):
        refreshed.append(request)
        creds.token = "replacement-token"

    monkeypatch.setattr(creds, "refresh", refresh)
    client = ClientLifecycleMixin()
    client.provider = "vertex"
    client.api_mode = "chat_completions"
    client.api_key = creds.token
    client.base_url = base_url
    client._client_kwargs = {"api_key": creds.token, "base_url": base_url}
    rebuilt = []
    client._replace_primary_openai_client = lambda **kw: rebuilt.append(dict(client._client_kwargs)) or True

    assert client._try_refresh_vertex_client_credentials() is True
    assert client.api_key == "replacement-token"
    assert len(refreshed) == 1
    assert rebuilt == [{"api_key": "replacement-token", "base_url": base_url}]
    assert vertex_adapter.get_vertex_config() == ("replacement-token", base_url)
    assert len(refreshed) == 1  # ordinary resolution still reuses a fresh token


@pytest.mark.parametrize("failure", ["raise", "empty"])
def test_failed_forced_refresh_does_not_adopt_credentials(vertex_adapter, monkeypatch, failure):
    from agent.client_lifecycle import ClientLifecycleMixin

    token, base_url = vertex_adapter.get_vertex_config()
    creds = next(iter(vertex_adapter._creds_cache.values()))[0]

    def refresh(request):
        if failure == "raise":
            raise RuntimeError("credential revoked")
        creds.token = None

    monkeypatch.setattr(creds, "refresh", refresh)
    client = ClientLifecycleMixin()
    client.provider, client.api_mode = "vertex", "chat_completions"
    client.api_key, client.base_url = token, base_url
    client._client_kwargs = {"api_key": token, "base_url": base_url}
    client._replace_primary_openai_client = lambda **kw: pytest.fail("invalid credentials adopted")
    assert client._try_refresh_vertex_client_credentials() is False
    assert client._client_kwargs == {"api_key": token, "base_url": base_url}
    assert client.api_key == token


def test_force_refresh_keeps_other_credential_identities(vertex_adapter, monkeypatch, tmp_path):
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for path in paths:
        path.write_text('{"project_id": "test"}')
        vertex_adapter.get_vertex_credentials(str(path))
    entries = dict(vertex_adapter._creds_cache)
    a_key, b_key = entries
    a, b = entries[a_key][0], entries[b_key][0]
    a.token, b.token = "a-old", "b-still-valid"
    monkeypatch.setattr(vertex_adapter, "_vertex_config", lambda: {"project_id": "caller-project"})
    token, url = vertex_adapter.get_vertex_config(str(paths[0]), "europe-west1", force_refresh=True)
    assert token == "ya29.FAKE"
    assert "/projects/caller-project/locations/europe-west1/" in url
    assert vertex_adapter._creds_cache[b_key][0] is b
    assert b.token == "b-still-valid"
    assert vertex_adapter.get_vertex_credentials(str(paths[1])) == ("b-still-valid", "caller-project")
    assert vertex_adapter.get_vertex_credentials(str(paths[0])) == (token, "caller-project")


def test_has_vertex_credentials_via_config_project(vertex_adapter, monkeypatch):
    monkeypatch.setattr(vertex_adapter, "_vertex_config", lambda: {"project_id": "p"})
    assert vertex_adapter.has_vertex_credentials() is True


def test_has_vertex_credentials_false_when_nothing_set(vertex_adapter):
    assert vertex_adapter.has_vertex_credentials() is False




def test_multiplex_scope_takes_precedence_over_raw_environ(vertex_adapter, monkeypatch):
    """In a multiplex gateway, a profile's own secret scope must win over a
    stale value in process os.environ left behind by another profile's
    dotenv load at boot — otherwise Profile B's turn could resolve Profile
    A's Vertex project (or worse, its credentials file path)."""
    from agent import secret_scope

    monkeypatch.setenv("VERTEX_PROJECT_ID", "other-profile-project")

    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"VERTEX_PROJECT_ID": "this-profile-project"})
    try:
        assert vertex_adapter._resolve_project_override() == "this-profile-project"
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(False)


def test_multiplex_unscoped_read_fails_closed(vertex_adapter, monkeypatch):
    """A credential read with no profile scope installed while multiplexing
    is active must raise rather than silently fall back to (possibly another
    profile's) raw os.environ value."""
    from agent import secret_scope

    monkeypatch.setenv("VERTEX_PROJECT_ID", "leaked-project")
    secret_scope.set_multiplex_active(True)
    try:
        with pytest.raises(secret_scope.UnscopedSecretError):
            vertex_adapter._resolve_project_override()
    finally:
        secret_scope.set_multiplex_active(False)


def test_adc_refuses_foreign_profile_google_application_credentials(
    vertex_adapter, monkeypatch, tmp_path
):
    """When this profile's scope defines no Vertex credentials, but os.environ
    still carries a *different* profile's GOOGLE_APPLICATION_CREDENTIALS (left
    there by python-dotenv at gateway boot), ADC must not silently mint a
    token under that foreign service account."""
    from agent import secret_scope

    sa_file = tmp_path / "other_profile_sa.json"
    sa_file.write_text('{"project_id": "other-profile"}')
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(sa_file))

    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({})  # this profile defines nothing
    try:
        assert vertex_adapter.get_vertex_credentials() == (None, None)
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(False)






# --- Pattern D: credential-file rotation invalidates the cache ---


def test_sa_file_rotation_invalidates_creds_cache(vertex_adapter, monkeypatch, tmp_path):
    """Rotating the service-account file on disk must be picked up on the
    next call — the pre-signature cache served tokens minted from the OLD
    identity for the life of the process (Pattern D: stale cache after an
    out-of-band change)."""
    import os as _os

    sa_file = tmp_path / "sa.json"
    sa_file.write_text('{"project_id": "first-identity"}')
    monkeypatch.setattr(
        vertex_adapter, "_resolve_credentials_path", lambda explicit=None: str(sa_file)
    )

    token1, project1 = vertex_adapter.get_vertex_credentials()
    assert token1 == "ya29.FAKE"
    assert len(vertex_adapter._creds_cache) == 1

    # Same file untouched: cache hit (same Credentials object).
    (key1,) = vertex_adapter._creds_cache
    creds_obj_1 = vertex_adapter._creds_cache[key1][0]
    vertex_adapter.get_vertex_credentials()
    (key1b,) = vertex_adapter._creds_cache
    assert key1b == key1
    assert vertex_adapter._creds_cache[key1b][0] is creds_obj_1

    # Rotate: rewrite the file with different content and a bumped mtime.
    sa_file.write_text('{"project_id": "second-identity", "rotated": true}')
    st = _os.stat(sa_file)
    _os.utime(sa_file, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))

    vertex_adapter.get_vertex_credentials()
    # New signature key replaced the old entry — not appended beside it.
    (key2,) = vertex_adapter._creds_cache
    assert key2 != key1
    assert vertex_adapter._creds_cache[key2][0] is not creds_obj_1






def test_adc_failure_retries_with_late_added_sa_file(vertex_adapter, monkeypatch, tmp_path):
    """ADC failure must fall back to a service-account file that appeared
    after startup. The signature-keyed cache turned keys into tuples and the
    old `cache_key == "__adc__"` string comparison silently disabled this
    retry (caught in review); the guard is now `not resolved_path`."""
    sa_file = tmp_path / "late_sa.json"
    sa_file.write_text('{"project_id": "late-identity"}')

    calls = {"n": 0}

    def _resolve(explicit=None):
        # First resolution (entry): nothing configured -> ADC attempt.
        # Second resolution (retry after ADC failure): the file has appeared.
        calls["n"] += 1
        return None if calls["n"] == 1 else str(sa_file)

    monkeypatch.setattr(vertex_adapter, "_resolve_credentials_path", _resolve)
    # Make the ADC attempt itself raise.
    monkeypatch.setattr(
        vertex_adapter.google.auth, "default",
        lambda scopes=None: (_ for _ in ()).throw(RuntimeError("ADC expired")),
    )

    token, project = vertex_adapter.get_vertex_credentials()
    assert token == "ya29.FAKE"
    assert project == "sa-project"
    assert calls["n"] >= 2


def test_metadata_preserving_rotation_invalidates_creds_cache(vertex_adapter, monkeypatch, tmp_path):
    """Atomic replacement that keeps SIZE and MTIME identical (deployment
    tools that restore metadata; equal-length JSON) must still be picked up.
    Review finding on #97701: a (path, mtime_ns, size) stat signature keyed
    the cache, so this replacement served the old private key; the key is
    now a content digest."""
    import os as _os

    sa_file = tmp_path / "sa.json"
    sa_file.write_text('{"project_id": "AAAA-identity"}')
    monkeypatch.setattr(
        vertex_adapter, "_resolve_credentials_path", lambda explicit=None: str(sa_file)
    )

    vertex_adapter.get_vertex_credentials()
    (key1,) = vertex_adapter._creds_cache
    creds_obj_1 = vertex_adapter._creds_cache[key1][0]

    # Same-length content, mtime restored, atomic replace.
    st = _os.stat(sa_file)
    new = tmp_path / "sa.json.new"
    new.write_text('{"project_id": "BBBB-identity"}')  # equal length
    _os.utime(new, ns=(st.st_atime_ns, st.st_mtime_ns))
    _os.replace(new, sa_file)
    st2 = _os.stat(sa_file)
    assert st2.st_size == st.st_size and st2.st_mtime_ns == st.st_mtime_ns

    vertex_adapter.get_vertex_credentials()
    (key2,) = vertex_adapter._creds_cache
    assert key2 != key1, "content change must produce a new cache key"
    assert vertex_adapter._creds_cache[key2][0] is not creds_obj_1

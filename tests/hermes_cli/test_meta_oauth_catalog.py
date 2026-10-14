"""Live-catalog contract for the meta-oauth subscription provider.

The picker must offer the account's live Muse Spark catalog when a stored
subscription credential exists (read-only: a picker never mints or refreshes),
and the curated rows otherwise.
"""

import pytest

from providers.base import ProviderProfile


@pytest.fixture()
def _live_fetch(monkeypatch):
    def fake_fetch_models(_self, **_kwargs):
        return ["muse-spark-1.3", "muse-image-1.0", "muse-spark-1.3-contributor"]

    monkeypatch.setattr(ProviderProfile, "fetch_models", fake_fetch_models)


def test_picker_uses_live_catalog_with_stored_credential(monkeypatch, _live_fetch):
    import hermes_cli.auth as auth

    monkeypatch.setattr(
        auth, "resolve_meta_oauth_runtime_credentials",
        lambda **_kw: {"api_key": "KEY", "base_url": "https://api.meta.ai/v1"})
    monkeypatch.setattr(auth, "_meta_access_token_is_expiring", lambda *_a, **_kw: False)

    from hermes_cli.models import provider_model_ids

    rows = provider_model_ids("meta-oauth", force_refresh=True)
    # Live rows, with the non-chat image model filtered by the profile.
    assert "muse-spark-1.3" in rows
    assert "muse-spark-1.3-contributor" in rows
    assert "muse-image-1.0" not in rows


def test_picker_falls_back_to_curated_rows_without_credential(monkeypatch):
    import hermes_cli.auth as auth

    def _no_creds(**_kw):
        raise RuntimeError("not logged in")

    monkeypatch.setattr(auth, "resolve_meta_oauth_runtime_credentials", _no_creds)

    from hermes_cli.models import provider_model_ids

    rows = provider_model_ids("meta-oauth", force_refresh=True)
    assert "muse-spark-1.3" in rows
    assert "muse-spark-1.3-contributor" in rows

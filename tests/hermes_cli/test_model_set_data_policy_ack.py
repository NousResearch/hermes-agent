"""``POST /api/model/set`` answers the selection confirm from the TARGET profile's config (#102048).

The interactive data-policy acknowledgement (``security.allow_data_training_tiers_interactive``) is
a per-profile setting. The dashboard serves several profiles from one process, so the guard must
resolve it under the profile the request names — not under the launch home — or an acknowledged
profile keeps being asked while an unacknowledged one would silently stop being asked.
"""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

CONTRIBUTOR_MODEL = "muse-spark-1.2-contributor"

PROVIDER_YAML = (
    "custom_providers:\n"
    "  - name: acme\n"
    "    base_url: https://api.acme.test/v1\n"
    "    key_env: ACME_RELAY_KEY\n"
    f"    models: [{CONTRIBUTOR_MODEL}]\n"
)
ACK_YAML = PROVIDER_YAML + "security:\n  allow_data_training_tiers_interactive: true\n"


@pytest.fixture()
def homes(tmp_path, monkeypatch):
    """Launch home without the acknowledgement + a named profile that recorded it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("ACME_RELAY_KEY", "dashboard-home-key")
    from hermes_cli import profiles as profiles_mod
    from hermes_cli.config import invalidate_env_cache

    demo = profiles_mod.get_profile_dir("demo")
    demo.mkdir(parents=True, exist_ok=True)
    (demo / "config.yaml").write_text(ACK_YAML, encoding="utf-8")
    (tmp_path / "config.yaml").write_text(PROVIDER_YAML, encoding="utf-8")
    invalidate_env_cache()
    return tmp_path, demo


@pytest.fixture()
def client(homes):
    from hermes_cli import web_server

    with TestClient(web_server.app, raise_server_exceptions=False) as c:
        c.headers["Authorization"] = f"Bearer {web_server._SESSION_TOKEN}"
        yield c


@pytest.fixture()
def offline_write(monkeypatch):
    """No network: the cost guard is silent and endpoint validation is accepted outright (it has its
    own tests); the contributor-tier data-policy guard is the only thing under test."""
    from hermes_cli import model_cost_guard
    from hermes_cli import models_validate as mv

    monkeypatch.setattr(model_cost_guard, "expensive_model_warning", lambda *a, **k: None)
    monkeypatch.setattr(
        mv, "validate_requested_model",
        lambda model, provider, api_key=None, base_url=None, **kw: {
            "accepted": True, "persist": True, "recognized": True, "message": ""},
    )


def _set_model(client, model, profile=None):
    body = {"scope": "main", "provider": "acme", "model": model}
    if profile is not None:
        body["profile"] = profile
    return client.post("/api/model/set", json=body)


def test_profile_that_acknowledged_is_not_asked_again(client, offline_write):
    """The named profile's setting suppresses the confirm and the switch proceeds."""
    resp = _set_model(client, CONTRIBUTOR_MODEL, profile="demo")
    assert resp.status_code == 200, resp.text
    assert resp.json().get("confirm_required") is not True


def test_unacknowledged_profile_still_confirms_with_a_data_policy_title(client, offline_write):
    """The launch home never acknowledged: it is asked, and the title names the real guard."""
    resp = _set_model(client, CONTRIBUTOR_MODEL)
    body = resp.json()
    assert body.get("confirm_required") is True
    assert body.get("confirm_title") == "Data-Training Tier Warning"
    assert "train" in (body.get("confirm_message") or "").lower()

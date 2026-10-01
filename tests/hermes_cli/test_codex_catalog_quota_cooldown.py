"""A Codex usage-limit cooldown gates inference, not model listing.

The ``/models`` endpoint still answers for an account whose weekly limit is exhausted, so the
picker must list live with the cooling-down pool row's token instead of serving a stale or
hardcoded catalog until the reset — which can be days away.
"""

import base64
import json
import time

from hermes_cli.auth import DEFAULT_CODEX_BASE_URL


def _jwt(offset_seconds: int) -> str:
    def _b64(payload: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).rstrip(b"=").decode("utf-8")

    return f"{_b64({'alg': 'none'})}.{_b64({'exp': int(time.time()) + offset_seconds})}.sig"


def _home(tmp_path, monkeypatch, entry: dict):
    import hermes_cli.codex_models as codex_models

    home = tmp_path / "hermes"
    home.mkdir()
    (home / "auth.json").write_text(
        json.dumps({"version": 1, "credential_pool": {"openai-codex": [entry]}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "no-codex-cli"))
    seen: list = []
    monkeypatch.setattr(codex_models, "_fetch_models_from_api",
                        lambda token, **_kw: seen.append(token) or ["live-only-slug"])
    return seen


def _entry(token: str, **extra) -> dict:
    return {"id": "e0", "label": "device_code", "auth_type": "oauth", "source": "device_code",
            "priority": 0, "request_count": 0, "access_token": token, "refresh_token": "r",
            "base_url": DEFAULT_CODEX_BASE_URL, **extra}


def test_exhausted_pool_row_still_lists_live(tmp_path, monkeypatch):
    from hermes_cli.models import _codex_catalog

    token = _jwt(86400)
    seen = _home(tmp_path, monkeypatch, _entry(
        token, last_status="exhausted", last_error_code=429, last_error_reason="usage_limit_reached",
        last_error_reset_at=time.time() + 3 * 86400))

    models = _codex_catalog("openai-codex", True)

    assert seen == [token]
    assert "live-only-slug" in models


def test_expired_token_in_cooldown_does_not_list_live(tmp_path, monkeypatch):
    from hermes_cli.models import _codex_catalog

    seen = _home(tmp_path, monkeypatch, _entry(
        _jwt(-60), last_status="exhausted", last_error_code=429, last_error_reason="usage_limit_reached",
        last_error_reset_at=time.time() + 3 * 86400))

    models = _codex_catalog("openai-codex", True)

    assert seen == []
    assert models and "live-only-slug" not in models

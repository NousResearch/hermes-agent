"""The activate flow's rescan must fire even when another process owns the local server (#115237).

ensure_local_runtime returns None when the router is supervised elsewhere, so the supervisor
path's rescan_if_unknown never ran — exactly the desktop situation after a download job already
bounced the router once (presets fresh, listing still spawn-only and missing the new model).
"""

from __future__ import annotations

from unittest.mock import patch


def test_activate_rescans_through_state_endpoint_when_supervisor_is_none(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli.web_routers import local_models as lm

    staged = tmp_path / "models"
    staged.mkdir(parents=True)
    (staged / "Qwen3-4B-Q4_K_M.gguf").write_bytes(b"GGUF\x00\x00\x00\x00")

    calls = {"refresh": 0, "known": [{"id": "some-other-model"}]}

    def fake_refresh():
        calls["refresh"] += 1

    def fake_router_request(endpoint, path, *, timeout, payload=None):
        assert path == "/models"
        return {"data": calls["known"]}

    job = lm._job("model-activate", "Qwen3-4B-Q4_K_M", model_id="Qwen3-4B-Q4_K_M")
    with (
        patch.object(lm.bootstrap, "ensure_local_runtime", return_value=None),
        patch.object(
            lm,
            "_state_endpoint",
            return_value={"base_url": "http://127.0.0.1:18434/v1", "api_key": "k"},
        ),
        patch.object(lm.bootstrap, "refresh_local_runtime", fake_refresh),
        patch.object(lm, "_router_request", fake_router_request),
        patch.object(lm, "_set_runtime_enabled", lambda v: {}),
        patch(
            "hermes_cli.web_server_config._apply_model_assignment_sync",
            lambda *a, **k: {"ok": True},
        ),
    ):
        lm._ensure_server(
            job,
            {"local_runtime": {"enabled": True}},
            "Qwen3-4B-Q4_K_M",
            fail_detail="server failed",
            skip_msg="skipped",
        )

    assert calls["refresh"] == 1, (
        "the router must be bounced when its listing lacks the model"
    )


def test_activate_skips_rescan_when_owned_server_already_lists_the_model(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli.web_routers import local_models as lm

    calls = {"refresh": 0}

    class FakeSup:
        def models(self):
            return {"Qwen3-4B-Q4_K_M": "loaded"}

    job = lm._job("model-activate", "Qwen3-4B-Q4_K_M", model_id="Qwen3-4B-Q4_K_M")
    with (
        patch.object(lm.bootstrap, "ensure_local_runtime", return_value=FakeSup()),
        patch.object(
            lm.bootstrap,
            "refresh_local_runtime",
            lambda: calls.__setitem__("refresh", calls["refresh"] + 1),
        ),
    ):
        lm._ensure_server(
            job,
            {"local_runtime": {"enabled": True}},
            "Qwen3-4B-Q4_K_M",
            fail_detail="server failed",
            skip_msg="skipped",
        )

    assert calls["refresh"] == 0, "no bounce when the listing already knows the model"

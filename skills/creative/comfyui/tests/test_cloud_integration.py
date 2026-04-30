"""Integration tests against the live Comfy Cloud API.

These tests are auto-skipped when COMFY_CLOUD_API_KEY is not set.
They never SUBMIT workflows (would need a paid subscription) — they only
verify the read-only endpoints we rely on.
"""

from __future__ import annotations

import pytest

from _common import http_get, parse_model_list, resolve_url


pytestmark = pytest.mark.cloud


class TestCloudEndpointsLive:
    def test_system_stats_reachable(self, cloud_key):
        url = resolve_url("https://cloud.comfy.org", "/system_stats")
        r = http_get(url, headers={"X-API-Key": cloud_key})
        assert r.status == 200
        data = r.json()
        assert "system" in data

    def test_models_endpoint_routed_to_experiment(self, cloud_key):
        # We expect the skill to route /models/checkpoints → /api/experiment/models/checkpoints
        url = resolve_url("https://cloud.comfy.org", "/models/checkpoints")
        assert "/api/experiment/models/checkpoints" in url
        r = http_get(url, headers={"X-API-Key": cloud_key})
        assert r.status == 200

    def test_models_endpoint_returns_dicts(self, cloud_key):
        url = resolve_url("https://cloud.comfy.org", "/models/checkpoints")
        r = http_get(url, headers={"X-API-Key": cloud_key})
        data = r.json()
        assert isinstance(data, list)
        if data:
            # Cloud format: list of dicts with `name`
            assert isinstance(data[0], dict)
            assert "name" in data[0]
            # Our parser normalizes both
            normalized = parse_model_list(data)
            assert len(normalized) == len(data)

    def test_history_renamed_to_v2(self, cloud_key):
        # /history → /api/history_v2 on cloud
        url = resolve_url("https://cloud.comfy.org", "/history/some-fake-id")
        assert "/api/history_v2/some-fake-id" in url

    def test_object_info_paid_tier(self, cloud_key):
        # On free tier, /object_info returns 403 with a recognizable message
        url = resolve_url("https://cloud.comfy.org", "/object_info")
        r = http_get(url, headers={"X-API-Key": cloud_key})
        # Should be either 200 (paid) or 403 (free) — not 404 / 500
        assert r.status in (200, 403)
        if r.status == 403:
            # Body should mention the limitation
            assert "free tier" in r.text().lower() or "subscription" in r.text().lower()


class TestCloudCheckDepsLive:
    def test_check_deps_against_cloud(self, cloud_key, sd15_workflow):
        from check_deps import check_deps
        report = check_deps(sd15_workflow, host="https://cloud.comfy.org", api_key=cloud_key)
        # Either node check passed OR was skipped (free tier)
        assert "missing_models" in report
        assert "is_cloud" in report and report["is_cloud"] is True

    def test_flux_workflow_models_resolved_via_aliases(self, cloud_key, flux_workflow):
        """Flux uses unet/clip folders; cloud has them in diffusion_models/text_encoders.
        With folder aliasing, the check should still find them."""
        from check_deps import check_deps
        report = check_deps(flux_workflow, host="https://cloud.comfy.org", api_key=cloud_key)
        # The exact required Flux files (flux1-dev.safetensors, t5xxl_fp16, clip_l, ae)
        # are present on cloud; with folder aliasing, none should be missing.
        # If this fails, either the cloud removed the model or the aliasing logic broke.
        missing_filenames = {m["value"] for m in report["missing_models"]}
        assert "ae.safetensors" not in missing_filenames, \
            "ae.safetensors should be on cloud's vae folder"
        # t5xxl_fp16 / clip_l should be reachable via the clip → text_encoders alias
        # flux1-dev.safetensors likewise via unet → diffusion_models


class TestHealthCheckLive:
    def test_health_check_passes(self, cloud_key, capsys):
        from health_check import main as health_main
        rc = health_main(["--host", "https://cloud.comfy.org", "--api-key", cloud_key])
        captured = capsys.readouterr()
        # Should produce JSON
        import json
        report = json.loads(captured.out)
        assert report["server"]["reachable"] is True
        assert report["checkpoints"]["queryable"] is True
        assert report["checkpoints"]["count"] > 0

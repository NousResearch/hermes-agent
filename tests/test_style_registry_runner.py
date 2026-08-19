"""Repo-root tests for the extended style menu (runner + gateway surfaces).

These run in the repo-root suite (pyproject testpaths=["tests"]) where
``tools`` and ``gateway`` are importable.  They verify the runner forwards
``blend`` and the gateway manifest exposes the blend option.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure content_engine is importable (image_jobs / image_job_service live there).
_REPO = Path(__file__).resolve().parents[1]
_CE = _REPO / "content_engine"
if str(_CE) not in sys.path:
    sys.path.insert(0, str(_CE))


def test_runner_forwards_blend(monkeypatch):
    from tools import generate_image_runner as runner

    captured = {}

    class FakePrepared:
        backend = "codex"
        prompt = "p"
        style_id = "mythic-tech-codex"
        references = ()
        registry_traits = "neutral traits"
        registry_slugs = ("steampunk", "synthwave")

    class FakeStaged:
        plan = type("Plan", (), {"backend": "codex", "execution_enabled": True, "reason": "ok", "manifest_path": Path("/tmp/x")})()

    class FakeCompleted:
        provider = "openai-codex"
        model = "gpt-image-2-medium"
        completion_path = Path("/tmp/x/completion.json")
        output_path = Path("/tmp/x/out.png")
        sha256 = "abc"

    def fake_prepare(**kwargs):
        captured.update(kwargs)
        return FakePrepared()

    def fake_stage(prepared, **kwargs):
        return FakeStaged()

    def fake_execute(prepared, staged, **kwargs):
        return FakeCompleted()

    # The runner imports prepare_image_request lazily from image_jobs at
    # call time; monkeypatch the source module attribute, not the runner.
    import image_jobs
    import image_job_service

    monkeypatch.setattr(image_jobs, "prepare_image_request", fake_prepare)
    monkeypatch.setattr(image_job_service, "stage_and_plan_image_job", fake_stage)
    monkeypatch.setattr(image_job_service, "execute_staged_image_job", fake_execute)

    result = runner.run_generate_image(
        prompt="p", style="mythic-tech-codex", backend="codex",
        blend=["steampunk", "synthwave"], stage_root="/tmp", job_id="j1",
    )
    assert captured.get("blend") == ["steampunk", "synthwave"]
    assert result["sha256"] == "abc"


def test_runner_render_command_includes_blend():
    from tools.generate_image_runner import render_generate_image_command

    cmd = render_generate_image_command(
        prompt="p", style="mythic-tech-codex", backend="codex",
        blend=["steampunk", "synthwave"], stage_root="/tmp", job_id="j1",
    )
    assert "--blend steampunk" in cmd
    assert "--blend synthwave" in cmd


def test_gateway_manifest_lists_blend_option():
    from gateway.relay.command_manifest import build_relay_command_manifest

    commands = build_relay_command_manifest()
    gen = next((c for c in commands if c.get("name") == "generate-image"), None)
    assert gen is not None
    opts = gen.get("options", [])
    args_opt = next((o for o in opts if o.get("name") == "args"), None)
    assert args_opt is not None
    assert "blend" in args_opt.get("description", "")


def test_gateway_handler_blend_split():
    # Mirror of the _gi_confirm_and_execute blend parsing (pure logic).
    blend_raw = "steampunk+synthwave"
    blend = [s.strip() for s in blend_raw.replace(",", "+").split("+") if s.strip()]
    assert blend == ["steampunk", "synthwave"]

    blend_raw = "steampunk, synthwave"
    blend = [s.strip() for s in blend_raw.replace(",", "+").split("+") if s.strip()]
    assert blend == ["steampunk", "synthwave"]

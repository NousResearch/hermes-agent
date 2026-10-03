"""Video as a first-class multimodal aux task (auxiliary.video config + routing)."""

import pytest


def test_multimodal_task_set_includes_video():
    from agent.auxiliary_client import _MULTIMODAL_AUX_TASKS, _TIMEOUT_NO_RETRY_TASKS

    assert {"vision", "video"} <= _MULTIMODAL_AUX_TASKS
    # Video sits on the same serialized-turn critical path as vision: full-budget
    # timeouts go straight to fallback instead of costing another timeout window.
    assert "video" in _TIMEOUT_NO_RETRY_TASKS


def test_video_task_has_no_concurrency_gate():
    from agent.auxiliary_client import _get_task_max_concurrency

    assert _get_task_max_concurrency("video") is None


def test_resolve_video_task_reads_auxiliary_video_config(monkeypatch):
    """task='video' resolves provider/model from auxiliary.video, not auxiliary.vision."""
    import agent.auxiliary_client as ac

    monkeypatch.setattr(
        ac, "_get_auxiliary_task_config",
        lambda task: {"provider": "custom", "base_url": "http://aux-video.test/v1",
                      "api_key": "k-test", "model": "video-model-x"}
        if task == "video" else {},
    )
    provider, _client, model = ac.resolve_vision_provider_client(task="video")
    assert provider == "custom"
    assert model == "video-model-x"


def test_resolve_vision_task_ignores_auxiliary_video_config(monkeypatch):
    """Default task='vision' must NOT pick up auxiliary.video's model (no cross-talk)."""
    import agent.auxiliary_client as ac

    monkeypatch.setattr(
        ac, "_get_auxiliary_task_config",
        lambda task: {"provider": "custom", "base_url": "http://aux-video.test/v1",
                      "api_key": "k-test", "model": "video-model-x"}
        if task == "video" else {"provider": "openrouter"},
    )
    provider, _client, model = ac.resolve_vision_provider_client()
    assert model != "video-model-x"


def test_vision_requirements_check_accepts_video_only_backend(monkeypatch):
    """check_vision_requirements passes when ONLY auxiliary.video is configured."""
    import tools.vision_tools as vt

    def fake_resolve(**kw):
        return ("custom", object(), "m") if kw.get("task") == "video" else (kw.get("provider", ""), None, None)

    monkeypatch.setattr("agent.auxiliary_client.resolve_vision_provider_client", fake_resolve)

    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("agent.auxiliary_client.aux_probe_mode", lambda: _Ctx())
    assert vt.check_vision_requirements() is True


def test_aux_call_kwargs_video_reads_video_section(monkeypatch):
    """Video calls read auxiliary.video.timeout, not auxiliary.vision.timeout."""
    import tools.vision_tools as vt

    monkeypatch.setattr(
        vt, "_cfg_auxiliary",
        lambda *keys, default=None: {"timeout": 240} if keys == ("video",) else {},
    )
    kwargs = vt._aux_call_kwargs([{"role": "user", "content": "x"}], None, 180.0,
                                 min_timeout=180.0, task="video")
    assert kwargs["task"] == "video"
    assert kwargs["timeout"] == 240


@pytest.mark.asyncio
async def test_video_analyze_tool_sends_video_task(tmp_path, monkeypatch):
    """End-to-end handler→aux wiring: _handle_video_analyze resolves auxiliary.video's
    model and dispatches task='video' (mocked LLM call, real file handling)."""
    import base64

    import tools.vision_tools as vt

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)  # minimal mp4-shaped file

    captured = {}

    async def fake_async_call_llm(**kwargs):
        captured.update(kwargs)

        class _R:
            choices = [type("C", (), {"message": type("M", (), {"content": "desc", "reasoning_content": None})()})()]

        return _R()

    monkeypatch.setattr("agent.auxiliary_client.async_call_llm", fake_async_call_llm)
    monkeypatch.setattr(
        vt, "_configured_aux_model",
        lambda sections, env_vars: "video-model-x" if sections[0] == "video" else None,
    )

    result = await vt._handle_video_analyze({"video_url": str(video), "question": "Describe this"})

    import json as _json
    assert _json.loads(result)["success"] is True
    assert captured["task"] == "video"
    assert captured["model"] == "video-model-x"
    part = captured["messages"][0]["content"][1]
    assert part["type"] == "video_url"
    assert part["video_url"]["url"].startswith("data:video/mp4;base64,")
    base64.b64decode(part["video_url"]["url"].split(",", 1)[1])  # payload decodes

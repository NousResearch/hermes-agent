from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from urllib.error import HTTPError

import pytest


PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "surfsense"


def load_plugin():
    package_name = "surfsense_test_plugin"
    for name in list(sys.modules):
        if name == package_name or name.startswith(f"{package_name}."):
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        package_name,
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


def make_settings(core, tmp_path, **overrides):
    values = {
        "base_url": "http://localhost:8929",
        "frontend_url": "http://localhost:3929",
        "access_token": "",
        "token_file": tmp_path / "token.json",
        "surfsense_root": tmp_path / "SurfSense",
        "timeout": 30,
        "max_stream_chars": 2000,
    }
    values.update(overrides)
    return core.Settings(**values)


def test_register_exposes_tools_and_cli_command():
    plugin = load_plugin()

    class Ctx:
        def __init__(self):
            self.tools = []
            self.commands = []
            self.cli_commands = []

        def register_tool(self, **kwargs):
            self.tools.append(kwargs)

        def register_command(self, *args, **kwargs):
            self.commands.append((args, kwargs))

        def register_cli_command(self, **kwargs):
            self.cli_commands.append(kwargs)

    ctx = Ctx()
    plugin.register(ctx)

    assert {tool["name"] for tool in ctx.tools} == {
        "surfsense_status",
        "surfsense_login",
        "surfsense_searchspaces",
        "surfsense_upload",
        "surfsense_search",
        "surfsense_ask",
        "surfsense_video_plan",
        "surfsense_video_mux",
    }
    assert ctx.commands[0][0][0] == "surfsense"
    assert ctx.cli_commands[0]["name"] == "surfsense"


def test_status_redacts_token_and_reports_docker_compose(tmp_path):
    plugin = load_plugin()
    core = plugin.core
    root = tmp_path / "SurfSense"
    (root / "docker").mkdir(parents=True)
    (root / "docker" / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
    token_file = tmp_path / "token.json"
    token_file.write_text(json.dumps({"access_token": "secret-token"}), encoding="utf-8")

    result = core.status(cfg=make_settings(core, tmp_path, token_file=token_file, surfsense_root=root))

    assert result["ok"] is True
    assert result["access_token_set"] is True
    assert result["access_token"] == "[REDACTED]"
    assert result["docker_compose_file_exists"] is True


def test_login_saves_token_without_returning_secret(tmp_path, monkeypatch):
    plugin = load_plugin()
    core = plugin.core
    saved = {}

    def fake_http_json(method, path, *, cfg, token=None, payload=None, form=None, timeout=None):
        assert method == "POST"
        assert path == "/auth/jwt/login"
        assert form["username"] == "bob@example.test"
        return {"access_token": "new-token", "token_type": "bearer"}

    monkeypatch.setattr(core, "_http_json", fake_http_json)
    monkeypatch.setattr(core, "save_env_value", lambda key, value: saved.setdefault(key, value))

    result = core.login(
        username="bob@example.test",
        password="correct horse battery staple",
        save=True,
        cfg=make_settings(core, tmp_path),
    )

    assert result["ok"] is True
    assert result["access_token"] == "[REDACTED]"
    assert saved["SURFSENSE_ACCESS_TOKEN"] == "new-token"


def test_search_documents_builds_expected_query(tmp_path, monkeypatch):
    plugin = load_plugin()
    core = plugin.core
    calls = []

    def fake_http_json(method, path, *, cfg, token=None, payload=None, form=None, timeout=None):
        calls.append((method, path, token))
        return {"items": [{"id": 10, "title": "Spec.pdf"}], "total": 1}

    monkeypatch.setattr(core, "_http_json", fake_http_json)
    cfg = make_settings(core, tmp_path, access_token="token")

    result = core.search_documents(
        query="vector search",
        search_space_id=7,
        page_size=3,
        cfg=cfg,
    )

    assert result["ok"] is True
    assert calls[0][0] == "GET"
    assert calls[0][1] == "/api/v1/documents/search?search_space_id=7&page_size=3&q=vector+search"
    assert calls[0][2] == "token"


def test_ask_creates_thread_and_collects_bounded_sse(tmp_path, monkeypatch):
    plugin = load_plugin()
    core = plugin.core
    calls = []

    def fake_http_json(method, path, *, cfg, token=None, payload=None, form=None, timeout=None):
        calls.append((method, path, payload))
        if path == "/api/v1/threads":
            return {"id": 42, "title": "Research", "search_space_id": 7}
        raise AssertionError(path)

    def fake_http_text(method, path, *, cfg, token=None, payload=None, timeout=None):
        calls.append((method, path, payload))
        return "data: {\"type\":\"text-delta\",\"text\":\"hello\"}\n\ndata: [DONE]\n"

    monkeypatch.setattr(core, "_http_json", fake_http_json)
    monkeypatch.setattr(core, "_http_text", fake_http_text)

    result = core.ask(
        query="Summarize my sources",
        search_space_id=7,
        title="Research",
        cfg=make_settings(core, tmp_path, access_token="token"),
    )

    assert result["ok"] is True
    assert result["thread_id"] == 42
    assert result["events"][0]["text"] == "hello"
    assert calls[1][0] == "POST"
    assert calls[1][1] == "/api/v1/new_chat"


def test_http_errors_are_redacted(tmp_path, monkeypatch):
    plugin = load_plugin()
    core = plugin.core

    def boom(method, path, *, cfg, token=None, payload=None, form=None, timeout=None):
        raise HTTPError(
            url="http://localhost:8929/api/v1/documents",
            code=401,
            msg="Unauthorized token=abc123abc123abc123abc123abc12312",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(core, "_http_json", boom)
    result = core.handle_search({"query": "x", "search_space_id": 1})

    assert result.startswith("{")
    payload = json.loads(result)
    assert payload["ok"] is False
    assert "abc123abc123abc123abc123abc12312" not in payload["error"]


def test_video_plan_writes_renderer_artifacts(tmp_path):
    plugin = load_plugin()
    core = plugin.core

    result = core.video_plan(
        topic="retrieval augmented research",
        source_text=(
            "SurfSense indexes documents into search spaces. "
            "Hermes can ask cited questions against those sources. "
            "A NotebookLM-style overview should explain the source set, "
            "highlight evidence, and end with next research actions."
        ),
        renderer="all",
        output_dir=str(tmp_path / "video"),
        duration_seconds=90,
        language="ja",
    )

    assert result["ok"] is True
    assert result["renderer"] == "all"
    paths = {Path(path).name for path in result["files"]}
    assert {
        "video_plan.json",
        "script.txt",
        "manim_scene.py",
        "heygen_prompt.txt",
        "hyperframes_index.html",
    }.issubset(paths)
    for path in result["files"]:
        assert Path(path).is_file()
        assert tmp_path in Path(path).resolve().parents


def test_video_plan_writes_integrated_knowledge_layers(tmp_path):
    plugin = load_plugin()
    core = plugin.core

    result = core.video_plan(
        topic="hermes knowledge cycle",
        source_text="SurfSense keeps cited source evidence separate from derived notes.",
        llm_wiki_text="LLM-wiki page: Hermes knowledge cycle links sources to explainers.",
        codegraph_text="codegraph: video_plan calls _build_video_plan and _manim_scene.",
        sleep_text="sleep digest: reconcile yesterday's research notes overnight.",
        memory_text="memory: prefer source-backed claims and mark preferences as memory-derived.",
        evidence_policy="strict",
        renderer="manim",
        output_dir=str(tmp_path / "integrated"),
    )

    assert result["ok"] is True
    assert result["integration_mode"] == "knowledge_cycle"
    paths = {Path(path).name for path in result["files"]}
    assert {
        "knowledge_context.json",
        "llm_wiki_page.md",
        "memory_sleep_packet.json",
        "manim_scene.py",
    }.issubset(paths)

    context = json.loads((tmp_path / "integrated" / "knowledge_context.json").read_text(encoding="utf-8"))
    assert context["evidence_policy"] == "strict"
    assert context["layers"]["source_backed"]["provenance"] == "SurfSense/source_text"
    assert context["layers"]["memory"]["claim_role"] == "preference_or_prior_context"
    assert context["layers"]["sleep_digest"]["claim_role"] == "consolidation_hint"

    manim_scene = (tmp_path / "integrated" / "manim_scene.py").read_text(encoding="utf-8")
    assert "LLM-wiki" in manim_scene
    assert "codegraph" in manim_scene
    assert "sleep digest" in manim_scene
    assert "memory" in manim_scene


def test_video_plan_writes_voice_and_mp4_packet(tmp_path):
    plugin = load_plugin()
    core = plugin.core

    result = core.video_plan(
        topic="source grounded avatar brief",
        source_text=(
            "SurfSense gathered the references. "
            "The presenter should explain the source boundary. "
            "The final video needs narration and an MP4 deliverable."
        ),
        renderer="manim",
        output_dir=str(tmp_path / "voice-video"),
        voice_pipeline="all",
        tts_voice="hakua",
        tts_speed=1.08,
    )

    assert result["ok"] is True
    assert result["voice_pipeline"] == "all"
    paths = {Path(path).name for path in result["files"]}
    assert {
        "voice_script.txt",
        "irodori_tts_request.json",
        "aituber_onair_cue.json",
        "mp4_mux_plan.json",
    }.issubset(paths)

    voice_script = (tmp_path / "voice-video" / "voice_script.txt").read_text(encoding="utf-8")
    assert "source grounded avatar brief" in voice_script

    irodori = json.loads((tmp_path / "voice-video" / "irodori_tts_request.json").read_text(encoding="utf-8"))
    assert irodori["tool"] == "irodori_tts_synthesize"
    assert irodori["arguments"]["voice"] == "hakua"
    assert irodori["arguments"]["format"] == "wav"

    aituber = json.loads((tmp_path / "voice-video" / "aituber_onair_cue.json").read_text(encoding="utf-8"))
    assert aituber["tool"] == "aituber_onair_speak"
    assert aituber["arguments"]["provider"] == "irodori"
    assert aituber["mp4_followup_tool"] == "surfsense_video_mux"

    mux_plan = json.loads((tmp_path / "voice-video" / "mp4_mux_plan.json").read_text(encoding="utf-8"))
    assert mux_plan["output_mp4"].endswith("final_with_voice.mp4")
    assert mux_plan["ffmpeg"]["argv"][0] == "ffmpeg"
    assert "-shortest" in mux_plan["ffmpeg"]["argv"]
    assert result["next_steps"]["mp4_audio"] == "Create voice.wav with irodoriTTS or AITuber OnAir, render the silent video, then run surfsense_video_mux or the ffmpeg argv in mp4_mux_plan.json."


def test_video_mux_dry_run_builds_safe_ffmpeg_command(tmp_path):
    plugin = load_plugin()
    core = plugin.core
    video = tmp_path / "silent.mp4"
    audio = tmp_path / "voice.wav"
    output = tmp_path / "final.mp4"

    result = core.video_mux(
        video_path=str(video),
        audio_path=str(audio),
        output_path=str(output),
        dry_run=True,
    )

    assert result["ok"] is True
    assert result["dry_run"] is True
    assert result["missing_inputs"] == [str(video.resolve()), str(audio.resolve())]
    assert result["ffmpeg"]["argv"][-1] == str(output.resolve())


def test_video_plan_rejects_unsafe_output_dir(tmp_path):
    plugin = load_plugin()
    core = plugin.core

    result = core.video_plan(
        topic="x",
        source_text="short source",
        renderer="manim",
        output_dir=str(tmp_path / ".." / "escape"),
    )

    assert result["ok"] is False
    assert "output_dir" in result["error"]


def test_video_plan_rejects_unbounded_absolute_output_dir(monkeypatch, tmp_path):
    plugin = load_plugin()
    core = plugin.core
    outside = tmp_path / "home" / "Desktop" / "surfsense-video"
    monkeypatch.setattr(core, "get_hermes_home", lambda: str(tmp_path / ".hermes"))
    monkeypatch.setattr(core.Path, "cwd", staticmethod(lambda: tmp_path / "workspace"))
    monkeypatch.setattr(core.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))

    result = core.video_plan(
        topic="x",
        source_text="short source",
        renderer="manim",
        output_dir=str(outside),
    )

    assert result["ok"] is False
    assert "workspace, Hermes home, or temp" in result["error"]

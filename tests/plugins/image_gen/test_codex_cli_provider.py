"""Tests for the bundled Codex CLI image_gen plugin."""

from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess

import pytest

import plugins.image_gen.codex_cli as codex_cli_plugin


_PNG_HEX = (
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
    "ae426082"
)


@pytest.fixture(autouse=True)
def _tmp_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    yield tmp_path


@pytest.fixture
def provider():
    return codex_cli_plugin.CodexCLIImageGenProvider()


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes.fromhex(_PNG_HEX))


def _jsonl(*payloads: dict) -> str:
    return "\n".join(json.dumps(p) for p in payloads)


class TestMetadata:
    def test_name(self, provider):
        assert provider.name == "codex-cli"

    def test_default_model(self, provider):
        assert provider.default_model() == "codex-cli-default"

    def test_list_models(self, provider):
        assert provider.list_models() == [
            {
                "id": "codex-cli-default",
                "display": "Codex CLI Built-in Image Generation",
                "speed": "varies",
                "strengths": "Uses ChatGPT/Codex built-in image tool",
                "price": "included with Codex auth",
            }
        ]


class TestAvailability:
    def test_unavailable_without_codex_binary(self, monkeypatch):
        monkeypatch.setattr(codex_cli_plugin.shutil, "which", lambda _: None)
        assert codex_cli_plugin.CodexCLIImageGenProvider().is_available() is False

    def test_unavailable_when_login_status_fails(self, monkeypatch):
        monkeypatch.setattr(codex_cli_plugin.shutil, "which", lambda _: "/usr/bin/codex")
        monkeypatch.setattr(
            codex_cli_plugin.subprocess,
            "run",
            lambda *a, **k: CompletedProcess(a[0], 1, stdout="", stderr="not logged in"),
        )
        assert codex_cli_plugin.CodexCLIImageGenProvider().is_available() is False

    def test_available_when_codex_logged_in(self, monkeypatch):
        monkeypatch.setattr(codex_cli_plugin.shutil, "which", lambda _: "/usr/bin/codex")
        monkeypatch.setattr(
            codex_cli_plugin.subprocess,
            "run",
            lambda *a, **k: CompletedProcess(a[0], 0, stdout="Logged in using ChatGPT\n", stderr=""),
        )
        assert codex_cli_plugin.CodexCLIImageGenProvider().is_available() is True


class TestConfig:
    def test_codex_home_from_env(self, monkeypatch):
        monkeypatch.setenv("CODEX_HOME", "~/special-codex-home")
        assert codex_cli_plugin._codex_home() == (Path.home() / "special-codex-home")

    def test_codex_home_from_plugin_config(self, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "image_gen:\n  codex_cli:\n    codex_home: ~/.codex-hermes\n"
        )
        assert codex_cli_plugin._codex_home() == (Path.home() / ".codex-hermes")


class TestGenerate:
    def test_empty_prompt_rejected(self, provider):
        result = provider.generate("", aspect_ratio="square")
        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"

    def test_missing_codex_binary(self, monkeypatch, provider):
        monkeypatch.setattr(codex_cli_plugin.shutil, "which", lambda _: None)
        result = provider.generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "missing_dependency"

    def test_missing_generated_image_returns_error(self, monkeypatch, provider):
        monkeypatch.setattr(codex_cli_plugin.shutil, "which", lambda _: "/usr/bin/codex")

        def fake_run(cmd, **kwargs):
            return CompletedProcess(
                cmd,
                0,
                stdout=_jsonl({"type": "thread.started", "thread_id": "thread-123"}),
                stderr="",
            )

        monkeypatch.setattr(codex_cli_plugin.subprocess, "run", fake_run)
        result = provider.generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "empty_response"

    def test_success_uses_thread_dir_file(self, monkeypatch, provider, tmp_path):
        monkeypatch.setattr(codex_cli_plugin.shutil, "which", lambda _: "/usr/bin/codex")
        out = _jsonl(
            {"type": "thread.started", "thread_id": "thread-123"},
            {
                "type": "item.completed",
                "item": {"id": "item_0", "type": "agent_message", "text": "done"},
            },
        )

        def fake_run(cmd, **kwargs):
            image_path = tmp_path / ".codex" / "generated_images" / "thread-123" / "image.png"
            _write_png(image_path)
            return CompletedProcess(cmd, 0, stdout=out, stderr="")

        monkeypatch.setattr(codex_cli_plugin.subprocess, "run", fake_run)
        result = provider.generate("a cat", aspect_ratio="landscape")
        assert result["success"] is True
        assert result["provider"] == "codex-cli"
        assert result["model"] == "codex-cli-default"
        assert result["aspect_ratio"] == "landscape"
        assert Path(result["image"]).exists()
        assert result["thread_id"] == "thread-123"

    def test_fallback_to_new_file_diff_when_no_thread_id(self, monkeypatch, provider, tmp_path):
        monkeypatch.setattr(codex_cli_plugin.shutil, "which", lambda _: "/usr/bin/codex")

        def fake_run(cmd, **kwargs):
            image_path = tmp_path / ".codex" / "generated_images" / "other-thread" / "image.png"
            _write_png(image_path)
            return CompletedProcess(cmd, 0, stdout=_jsonl({"type": "turn.completed"}), stderr="")

        monkeypatch.setattr(codex_cli_plugin.subprocess, "run", fake_run)
        result = provider.generate("a cat")
        assert result["success"] is True
        assert Path(result["image"]).name == "image.png"

    def test_nonzero_exit_returns_error_response(self, monkeypatch, provider):
        monkeypatch.setattr(codex_cli_plugin.shutil, "which", lambda _: "/usr/bin/codex")
        monkeypatch.setattr(
            codex_cli_plugin.subprocess,
            "run",
            lambda cmd, **kwargs: CompletedProcess(cmd, 1, stdout='oops', stderr='boom'),
        )
        result = provider.generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "boom" in result["error"]

    @pytest.mark.parametrize(
        ("aspect_ratio", "expected"),
        [
            ("landscape", "landscape orientation"),
            ("square", "square composition"),
            ("portrait", "portrait orientation"),
        ],
    )
    def test_prompt_includes_aspect_ratio_guidance(self, monkeypatch, provider, aspect_ratio, expected, tmp_path):
        monkeypatch.setattr(codex_cli_plugin.shutil, "which", lambda _: "/usr/bin/codex")
        seen = {}

        def fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            seen["env"] = kwargs.get("env", {})
            image_path = tmp_path / ".codex" / "generated_images" / "thread-xyz" / "image.png"
            _write_png(image_path)
            return CompletedProcess(cmd, 0, stdout=_jsonl({"type": "thread.started", "thread_id": "thread-xyz"}), stderr="")

        monkeypatch.setattr(codex_cli_plugin.subprocess, "run", fake_run)
        provider.generate("a cat", aspect_ratio=aspect_ratio)
        prompt = seen["cmd"][-1]
        assert expected in prompt
        assert "--enable" in seen["cmd"]
        assert "image_generation" in seen["cmd"]
        assert seen["env"]["CODEX_HOME"].endswith(".codex")

    def test_respects_codex_home_in_subprocess_env(self, monkeypatch, provider, tmp_path):
        monkeypatch.setattr(codex_cli_plugin.shutil, "which", lambda _: "/usr/bin/codex")
        monkeypatch.setenv("CODEX_HOME", str(tmp_path / ".codex-hermes"))
        seen = {}

        def fake_run(cmd, **kwargs):
            seen.setdefault("envs", []).append(kwargs.get("env", {}))
            image_path = tmp_path / ".codex-hermes" / "generated_images" / "thread-xyz" / "image.png"
            _write_png(image_path)
            return CompletedProcess(cmd, 0, stdout=_jsonl({"type": "thread.started", "thread_id": "thread-xyz"}), stderr="")

        monkeypatch.setattr(codex_cli_plugin.subprocess, "run", fake_run)
        result = provider.generate("a cat")
        assert result["success"] is True
        assert all(env["CODEX_HOME"] == str(tmp_path / ".codex-hermes") for env in seen["envs"])

    def test_register_wires_provider(self):
        registered = {}

        class Ctx:
            def register_image_gen_provider(self, provider):
                registered["provider"] = provider

        codex_cli_plugin.register(Ctx())
        assert isinstance(registered["provider"], codex_cli_plugin.CodexCLIImageGenProvider)

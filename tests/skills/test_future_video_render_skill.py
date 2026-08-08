import importlib.util
import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT
    / "optional-skills"
    / "creative"
    / "future-video-render"
    / "scripts"
    / "future_video_render.py"
)


@pytest.fixture()
def render_tool():
    spec = importlib.util.spec_from_file_location("future_video_render_helper", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_invalid_request_json_returns_render_tool_error(render_tool, monkeypatch, tmp_path, capsys):
    request_file = tmp_path / "request.json"
    request_file.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("FVS_AGENT_API_KEY", "fvs_live_test")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "future_video_render.py",
            "submit",
            "--request-file",
            str(request_file),
            "--dry-run",
        ],
    )

    assert render_tool.main() == 1

    captured = capsys.readouterr()
    assert "invalid request JSON" in captured.err
    assert "Traceback" not in captured.err


def test_render_url_validation_rejects_untrusted_hosts(render_tool):
    agent_base = "https://app.future.video/api/agent"

    with pytest.raises(render_tool.RenderToolError) as exc:
        render_tool.validate_render_url(
            "https://evil.example/api/agent/renders/proj_api_123",
            agent_base=agent_base,
            kind="status",
        )

    assert "refusing to send an API key" in str(exc.value)


def test_submit_dry_run_validates_payload_and_avoids_network(render_tool, monkeypatch, tmp_path, capsys):
    request_file = tmp_path / "request.json"
    upload_file = tmp_path / "reference.png"
    upload_file.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    request_file.write_text(
        json.dumps(
            {
                "name": "Dry run",
                "project_mode": "scene",
                "screenplay": "Shot 1: A clean test.",
                "assets": [{"filename": upload_file.name, "label": "Reference"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FVS_AGENT_API_KEY", "fvs_live_test")

    def fail_urlopen(*args, **kwargs):
        raise AssertionError("dry-run must not make network calls")

    monkeypatch.setattr(render_tool.urllib.request, "urlopen", fail_urlopen)
    args = render_tool.build_parser().parse_args(
        [
            "submit",
            "--request-file",
            str(request_file),
            "--file",
            str(upload_file),
            "--dry-run",
        ]
    )

    assert args.func(args) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["endpoint"] == "https://app.future.video/api/agent/renders"
    assert output["asset_count"] == 1
    assert output["files"] == [str(upload_file)]


def test_api_error_is_wrapped_as_render_tool_error(render_tool, monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            422,
            "Unprocessable Entity",
            hdrs=None,
            fp=io.BytesIO(b'{"detail":"bad request payload"}'),
        )

    monkeypatch.setattr(render_tool.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(render_tool.RenderToolError) as exc:
        render_tool.request_json_api(
            url="https://app.future.video/api/agent/renders",
            method="POST",
            api_key="fvs_live_test",
            agent_base="https://app.future.video/api/agent",
            body=b"{}",
            content_type="application/json",
        )

    message = str(exc.value)
    assert "HTTP 422" in message
    assert "bad request payload" in message


def test_api_key_cli_argument_is_not_supported(render_tool):
    with pytest.raises(SystemExit):
        render_tool.build_parser().parse_args(
            [
                "status",
                "--api-key",
                "fvs_live_should_not_be_in_argv",
                "--project-id",
                "proj_api_123",
            ]
        )

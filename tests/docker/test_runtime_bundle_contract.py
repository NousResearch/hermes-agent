"""Runtime image contract tests for Hermes Cloud cold-start bundles."""

from pathlib import Path


def _dockerfile_text() -> str:
    return (Path(__file__).parents[2] / "Dockerfile").read_text(encoding="utf-8")


def test_runtime_image_keeps_plugin_tools_gateway_label() -> None:
    df = _dockerfile_text()

    assert 'LABEL hermes.plugin-tools-gateway="true"' in df


def test_runtime_image_bakes_feishu_and_edge_tts_extras() -> None:
    df = _dockerfile_text()

    assert "--extra feishu" in df
    assert "--extra edge-tts" in df


def test_runtime_image_bakes_tirith_binary() -> None:
    df = _dockerfile_text()

    assert "tirith baked into image" in df
    assert "HERMES_HOME=/opt/hermes" in df

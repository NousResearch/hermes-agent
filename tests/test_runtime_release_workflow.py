"""Regression tests for the signed desktop runtime release workflow."""

from pathlib import Path


def _workflow_text() -> str:
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release-runtime.yml"
    return workflow.read_text(encoding="utf-8")


def test_runtime_workflow_freezes_anthropic_sdk_for_minimax_cn():
    """MiniMax-CN uses the Anthropic Messages transport in the desktop runtime.

    The runtime is a frozen PyInstaller executable, so lazy-installing the
    provider SDK at first use cannot work there. The release workflow must
    eagerly install and collect the Anthropic SDK, otherwise desktop users get
    an import failure before MiniMax-CN can make its first request.
    """
    workflow = _workflow_text()

    assert 'pip install -e ".[web,anthropic]"' in workflow
    assert "--collect-submodules anthropic" in workflow
    assert "--copy-metadata anthropic" in workflow
    assert "anthropic-*.dist-info" in workflow

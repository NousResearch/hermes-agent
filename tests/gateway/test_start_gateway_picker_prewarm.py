"""#92244: gateway startup must prewarm the /model picker cache like the TUI."""

from pathlib import Path


def test_start_gateway_prewarms_picker_cache():
    src = Path("gateway/run.py").read_text()
    needle = "prewarm_picker_cache_async"
    assert needle in src, "gateway/run.py must call prewarm_picker_cache_async (#92244)"
    assert src.index(needle) < src.index("success = await runner.start()")


def test_web_server_lifespan_prewarms_picker_cache():
    src = Path("hermes_cli/web_server.py").read_text()
    assert "prewarm_picker_cache_async" in src

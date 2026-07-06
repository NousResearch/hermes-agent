from __future__ import annotations

import os
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(os.environ.get("HERMES_REPO_ROOT", "/Users/ericfreeman/.hermes/hermes-agent"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from torben_morning_brief import WEB_RESEARCH_UNAVAILABLE_MESSAGE, build_web_research_status


def test_web_research_status_reports_unavailable_without_firecrawl_config() -> None:
    status = build_web_research_status({})

    assert status["status"] == "unavailable"
    assert status["configured"] is False
    assert status["message"] == WEB_RESEARCH_UNAVAILABLE_MESSAGE
    assert "web_search" in str(status["agent_instruction"])
    assert "web_extract" in str(status["agent_instruction"])


def test_web_research_status_reports_configured_without_secret_value() -> None:
    status = build_web_research_status({"FIRECRAWL_API_KEY": "secret-value"})

    assert status["status"] == "configured"
    assert status["configured"] is True
    assert status["configured_vars"] == ["FIRECRAWL_API_KEY"]
    assert "secret-value" not in str(status)

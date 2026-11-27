"""Behavior tests: core Gemini-family helpers, Vertex, and child-env security work with no Google plugin loaded."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_core_helpers_vertex_and_env_blocklist_work_without_google_plugin():
    """Run in a fresh interpreter with the google provider package blocked from import."""
    code = r"""
import sys
sys.modules["plugins.model_providers.google"] = None
sys.modules["agent.gemini_native_adapter"] = None
sys.modules["agent.gemini_schema"] = None
sys.modules["agent.gemini_long_decode"] = None

from agent.gemini_urls import is_native_gemini_base_url, is_vertex_express_base_url, normalize_gemini_base_url
from agent.gemini_model_family import GEMINI_DEFAULT_MAX_OUTPUT_TOKENS, effective_gemini_max_output_tokens
from agent.output_budget_timeout import output_budget_stale_multiplier
from agent.reasoning_effort import gemini_thinking_config
from tools.environments.local_env_policy import _build_provider_env_blocklist
import providers

assert normalize_gemini_base_url("https://generativelanguage.googleapis.com") == "https://generativelanguage.googleapis.com/v1beta"
assert is_native_gemini_base_url("https://generativelanguage.googleapis.com/v1beta") is True
assert is_native_gemini_base_url("https://generativelanguage.googleapis.com/v1beta/openai") is False
assert is_vertex_express_base_url("https://aiplatform.googleapis.com/v1beta1") is True
assert is_vertex_express_base_url("https://aiplatform.googleapis.com/v1/projects/p/locations/global/endpoints/openapi") is False
assert effective_gemini_max_output_tokens(4096, {"includeThoughts": True, "thinkingLevel": "high"}) == GEMINI_DEFAULT_MAX_OUTPUT_TOKENS
assert output_budget_stale_multiplier({"extra_body": {"max_output_tokens": 262_144}}) == 8
assert gemini_thinking_config("gemini-4-neon", {"enabled": True, "effort": "xhigh"}) == {
    "includeThoughts": True, "thinkingLevel": "xhigh"
}

vertex = providers.get_provider_profile("vertex")
assert vertex is not None
assert vertex.build_extra_body(model="google/gemini-4-neon", reasoning_config={"enabled": True, "effort": "xhigh"}) == {
    "extra_body": {"google": {"thinking_config": {"include_thoughts": True, "thinking_level": "xhigh"}}}
}

blocked = _build_provider_env_blocklist()
for var in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GEMINI_BASE_URL"):
    assert var in blocked, f"{var} missing from provider env blocklist"
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0 and result.stdout.strip().splitlines()[-1] == "ok", result.stderr[-2000:]

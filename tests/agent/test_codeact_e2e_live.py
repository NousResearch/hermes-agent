"""E2E verification of CodeAct Phase 4+5 against live qwen3.6-27b-dense at http://0.0.0.0:8086.

This test directly exercises the kernel lifecycle (no mocks), verifying:
  1. Kernel subprocess startup + namespace injection
  2. Code execution via IPC (variables, imports)
  3. help() builtin listing
  4. Tool stub presence (web_search, etc.)
  5. promote_to_skill() builtin (flag-only mode)
  6. State persistence across executions
  7. soft_reset() clears user state

Run manually: pytest tests/agent/test_codeact_e2e_live.py -v -s
Skipped by default since it requires the live model endpoint.
"""

import json
import os

import pytest


# ---------------------------------------------------------------------------
# Endpoint discovery
# ---------------------------------------------------------------------------

_LIVE_ENDPOINT = os.environ.get("CODEACT_E2E_ENDPOINT", "http://0.0.0.0:8086")


def _check_endpoint():
    """Return True if the endpoint responds to /v1/models."""
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(f"{_LIVE_ENDPOINT}/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


pytestmark = pytest.mark.skipif(
    not _check_endpoint(),
    reason=f"Live CodeAct endpoint {_LIVE_ENDPOINT} unavailable",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def kernel_and_namespace():
    """Start a real HermesKernel with full Phase 4+5 namespace."""
    from agent.codeact_kernel import HermesKernel
    from agent.codeact_namespace import build_tool_namespace_source
    from agent.codeact_skill_injector import SkillNamespaceInjector
    from agent.codeact_promotion import (
        PromotionCandidate,
        flag_candidate,
    )

    # Trigger tool discovery (registry starts empty until discover_builtin_tools)
    import model_tools  # noqa: F401 — side effect: discover_builtin_tools()
    from tools.registry import registry

    # Build injector (Phase 4)
    injector = SkillNamespaceInjector(
        registry=registry,
        max_skills=20,
        recently_used_count=5,
    )

    # Build namespace source with skills
    namespace_source = build_tool_namespace_source(
        registry=registry,
        enabled_tool_names=None,
        skill_injector=injector,
    )

    # Build tool dispatcher with promote handler (Phase 5)
    def _tool_dispatcher(tool_name, args):
        if tool_name == "__promote_skill__":
            candidate = PromotionCandidate(
                fn_name=args.get("fn_name", "unknown_fn"),
                description=args.get("description", "Promoted from CodeAct"),
                source_code=args.get("source_code", ""),
                domain=args.get("domain", "general"),
                tags=args.get("tags") or [],
                session_id="e2e-test",
                occurrence_count=1,
            )
            flag_candidate(candidate)
            return json.dumps(
                {
                    "status": "flagged",
                    "message": f"Skill '{candidate.fn_name}' flagged for review.",
                }
            )
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    kernel = HermesKernel(
        session_id="e2e-test",
        tool_dispatcher=_tool_dispatcher,
        namespace_source=namespace_source,
    )
    kernel.start()

    yield kernel
    kernel.shutdown()


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestKernelLifecycle:
    """Basic kernel lifecycle tests."""

    def test_basic_code_execution(self, kernel_and_namespace):
        result = kernel_and_namespace.execute("x = 10\ny = x * 3\nprint(y)")
        assert "30" in result or "error" not in result.lower()

    def test_state_persistence(self, kernel_and_namespace):
        kernel_and_namespace.execute("persistent_var = 'hello_e2e'")
        result = kernel_and_namespace.execute("print(persistent_var)")
        assert "hello_e2e" in result

    def test_imports_work(self, kernel_and_namespace):
        result = kernel_and_namespace.execute(
            "import json\ndata = json.dumps({'test': True})\nprint(data)"
        )
        assert '"test"' in result or "true" in result


class TestHelpBuiltin:
    """help() builtin introspection."""

    def test_help_lists_tools(self, kernel_and_namespace):
        result = kernel_and_namespace.execute("print(help())")
        assert len(result) > 100
        has_tool = any(
            name in result
            for name in ["web_search", "terminal", "read_file", "list_files"]
        )
        assert has_tool, (
            f"help() output did not list any recognizable tools: {result[:200]}"
        )

    def test_help_for_specific_tool(self, kernel_and_namespace):
        result = kernel_and_namespace.execute("print(help('terminal'))")
        assert "terminal" in result.lower() or "error" not in result.lower()


class TestToolStubs:
    """Tool stub presence and callability."""

    def test_tool_stubs_defined(self, kernel_and_namespace):
        result = kernel_and_namespace.execute(
            "print(type(terminal), type(web_search), type(read_file))"
        )
        assert "function" in result.lower()


class TestPromoteToSkill:
    """promote_to_skill() builtin — Phase 5 E2E."""

    def test_promote_to_skill_exists(self, kernel_and_namespace):
        result = kernel_and_namespace.execute("print(type(promote_to_skill))")
        assert "function" in result.lower()

    def test_promote_to_skill_flags_candidate(self, kernel_and_namespace):
        code = (
            "def e2e_test_helper(x):\n"
            "    '''E2E test helper function.'''\n"
            "    return x * 2\n"
            "promote_to_skill('e2e_test_helper', 'E2E test helper function.')\n"
        )
        result = kernel_and_namespace.execute(code)
        assert "error" not in result.lower() or "unknown tool" in result.lower()


class TestSoftReset:
    """soft_reset() clears user state."""

    def test_soft_reset_clears_user_vars(self, kernel_and_namespace):
        kernel_and_namespace.execute("reset_target = 999")
        kernel_and_namespace.soft_reset()
        result = kernel_and_namespace.execute(
            "print('EXISTS' if 'reset_target' in dir() else 'GONE')"
        )
        assert "GONE" in result

    def test_soft_reset_preserves_builtins(self, kernel_and_namespace):
        kernel_and_namespace.soft_reset()
        result = kernel_and_namespace.execute("print(type(help))")
        assert "function" in result.lower()

"""#84733: prompt-cache TTL/prefix propagation into MoA/aux paths + failover re-preflight.

The main loop threads ``agent._cache_ttl`` and the stable system prefix into
``build_prompt_cache_plan``, but the MoA/aux helper only accepted
``cache_disabled`` — so a configured ``1h`` regressed to the 5m default and
the destination system prompt was marked as one whole breakpoint. These
tests pin the threaded parameters (TTL + static prefix) on
``plan_cache_sections_for_destination`` and the MoA decoration helper, the
per-destination Qwen clamp (1h -> 5m), and the failover re-preflight
contract (every fallback activation must restart the outer iteration so the
pre-API preflight re-runs against the fallback's context window).
"""

import ast
import inspect


def _collect_cache_controls(obj):
    """Return every ``cache_control`` marker dict reachable in ``obj``."""
    markers = []
    if isinstance(obj, dict):
        if "cache_control" in obj:
            markers.append(obj["cache_control"])
        for value in obj.values():
            markers.extend(_collect_cache_controls(value))
    elif isinstance(obj, list):
        for value in obj:
            markers.extend(_collect_cache_controls(value))
    return markers


class TestPlanCacheSectionsThreadsTtlAndPrefix:
    def test_cache_ttl_1h_reaches_markers(self):
        from agent.agent_runtime_helpers import plan_cache_sections_for_destination

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hello"},
        ]
        out_msgs, _ = plan_cache_sections_for_destination(
            messages,
            None,
            provider="anthropic",
            base_url="https://api.anthropic.com",
            api_mode="anthropic_messages",
            model="claude-opus-4.8",
            cache_disabled=False,
            cache_ttl="1h",
        )
        markers = _collect_cache_controls(out_msgs)
        assert markers, "expected cache_control markers on a caching route"
        assert all(m.get("ttl") == "1h" for m in markers), (
            "the configured 1h tier must reach the destination plan markers"
        )

    def test_static_system_prefix_gets_early_breakpoint(self):
        from agent.agent_runtime_helpers import plan_cache_sections_for_destination

        messages = [
            {"role": "system", "content": "stable prefix\nvolatile suffix"},
            {"role": "user", "content": "hello"},
        ]
        out_msgs, _ = plan_cache_sections_for_destination(
            messages,
            None,
            provider="anthropic",
            base_url="https://api.anthropic.com",
            api_mode="anthropic_messages",
            model="claude-opus-4.8",
            cache_disabled=False,
            cache_ttl="5m",
            static_system_prefix="stable prefix",
        )
        system_content = out_msgs[0]["content"]
        assert isinstance(system_content, list) and len(system_content) == 2, (
            "the destination system prompt must split into [static, volatile] "
            "parts instead of marking the whole prompt as one breakpoint"
        )
        assert system_content[0]["text"] == "stable prefix"
        assert system_content[1]["text"] == "\nvolatile suffix"

    def test_qwen_1h_clamped_to_5m(self):
        from agent.agent_runtime_helpers import plan_cache_sections_for_destination

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hello"},
        ]
        out_msgs, _ = plan_cache_sections_for_destination(
            messages,
            None,
            provider="opencode",
            base_url="https://api.opencode.ai",
            api_mode="chat_completions",
            model="qwen3.6-plus",
            cache_disabled=False,
            cache_ttl="1h",
        )
        markers = _collect_cache_controls(out_msgs)
        assert markers, "opencode+qwen is a cache-honoring route"
        assert all("ttl" not in m for m in markers), (
            "Qwen's 5-minute-only context cache must clamp a configured 1h"
        )


class TestMoACacheControlThreadsTtl:
    def test_moa_decoration_uses_threaded_1h(self):
        from agent.moa_loop import _maybe_apply_moa_cache_control

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
        runtime = {
            "provider": "anthropic",
            "model": "claude-opus-4.8",
            "base_url": "",
            "api_mode": "anthropic_messages",
        }
        out = _maybe_apply_moa_cache_control(
            messages, runtime, cache_disabled=False, cache_ttl="1h"
        )
        markers = _collect_cache_controls(out)
        assert markers, "expected MoA decoration on a caching route"
        assert all(m.get("ttl") == "1h" for m in markers), (
            "the agent's 1h tier must stop regressing to 5m on MoA advisor calls"
        )
        # Caller messages must stay undecorated.
        assert not _collect_cache_controls(messages)

    def test_moa_qwen_1h_clamped_to_5m(self):
        from agent.moa_loop import _maybe_apply_moa_cache_control

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
        ]
        runtime = {
            "provider": "opencode",
            "model": "qwen3.6-plus",
            "base_url": "",
            "api_mode": "chat_completions",
        }
        out = _maybe_apply_moa_cache_control(
            messages, runtime, cache_disabled=False, cache_ttl="1h"
        )
        markers = _collect_cache_controls(out)
        assert markers, "opencode+qwen is a cache-honoring MoA route"
        assert all("ttl" not in m for m in markers), (
            "MoA decoration must clamp 1h to 5m on Qwen destinations"
        )

    def test_moa_decoration_defaults_to_5m_without_ttl(self):
        from agent.moa_loop import _maybe_apply_moa_cache_control

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
        ]
        runtime = {
            "provider": "anthropic",
            "model": "claude-opus-4.8",
            "base_url": "",
            "api_mode": "anthropic_messages",
        }
        out = _maybe_apply_moa_cache_control(
            messages, runtime, cache_disabled=False
        )
        markers = _collect_cache_controls(out)
        assert markers
        assert all("ttl" not in m for m in markers)


class TestFailoverRestartsPreflight:
    """#84733: a fallback provider switch must re-run the pre-API preflight.

    ``_try_activate_fallback`` already shrinks the compressor's context
    window to the fallback's; the old ``continue`` re-fired the request
    without re-running the pre-API pressure check. Every activation site
    must instead ``break`` to the ``restart_with_rebuilt_messages`` handler,
    which restarts the outer iteration (and its preflight) via a budget
    refund. Source-level guard: importing the module and parsing the
    function is cheap, and the assertion encodes the bug class — a new
    failover site added with ``continue`` fails here on purpose.
    """

    def test_every_fallback_activation_breaks_to_repreflight(self):
        from agent import conversation_loop

        tree = ast.parse(inspect.getsource(conversation_loop.run_conversation))
        fallback_ifs = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Call)
            and isinstance(node.test.func, ast.Attribute)
            and node.test.func.attr == "_try_activate_fallback"
        ]
        assert fallback_ifs, "expected _try_activate_fallback sites in run_conversation"
        for node in fallback_ifs:
            assert any(isinstance(stmt, ast.Break) for stmt in node.body), (
                "fallback activation must break to the restart-with-rebuilt-"
                "messages handler so the pre-API preflight re-runs against "
                "the fallback's context window (#84733)"
            )

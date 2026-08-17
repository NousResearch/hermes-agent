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
    window to the fallback's; the pre-API preflight runs at the top of the
    OUTER iteration loop, before the retry loop. So the restart discipline
    is loop-aware:

    - Sites INSIDE the retry loop (``while retry_count < max_retries``)
      must ``break`` out of it with ``restart_with_rebuilt_messages`` set,
      so the handler after the retry loop refunds the budget and
      ``continue``s the outer iteration (which re-runs the preflight).
      A plain ``continue`` there would only re-fire the retry loop and
      skip the preflight — the original bug.
    - Sites DIRECTLY in the outer loop must ``continue`` — the next outer
      iteration re-runs the preflight already. A ``break`` there would
      exit the conversation loop and end the turn without ever calling
      the just-activated fallback.

    Source-level guard: parsing the function is cheap, and the assertion
    encodes the bug class — a new failover site added with the wrong
    restart statement for its loop fails here on purpose.
    """

    def test_every_fallback_activation_restarts_preflight(self):
        from agent import conversation_loop

        tree = ast.parse(inspect.getsource(conversation_loop.run_conversation))

        # Parent map so each site can be bound to its nearest enclosing loop.
        parents = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node

        retry_loops = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.While)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "retry_count"
        ]
        assert retry_loops, "expected the retry loop in run_conversation"
        retry_loop_ids = {id(loop) for loop in retry_loops}

        def _inside_retry_loop(node):
            cur = parents.get(node)
            while cur is not None:
                if id(cur) in retry_loop_ids:
                    return True
                cur = parents.get(cur)
            return False

        fallback_ifs = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Call)
            and isinstance(node.test.func, ast.Attribute)
            and node.test.func.attr == "_try_activate_fallback"
        ]
        assert fallback_ifs, "expected _try_activate_fallback sites in run_conversation"
        # Every reference to _try_activate_fallback must be one of the matched
        # `if agent._try_activate_fallback(...):` sites — a site written as
        # `activated = agent._try_activate_fallback()` would silently escape
        # this guard.
        all_refs = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr == "_try_activate_fallback"
        ]
        assert len(all_refs) == len(fallback_ifs), (
            "every _try_activate_fallback reference must be a direct "
            "`if agent._try_activate_fallback(...):` site so this guard "
            "can bind its restart discipline (#84733)"
        )
        for node in fallback_ifs:
            if _inside_retry_loop(node):
                assert any(isinstance(stmt, ast.Break) for stmt in node.body), (
                    "retry-loop fallback activation must break to the "
                    "restart-with-rebuilt-messages handler so the pre-API "
                    "preflight re-runs against the fallback's context "
                    "window (#84733)"
                )
            else:
                assert any(
                    isinstance(stmt, ast.Continue) for stmt in node.body
                ), (
                    "outer-loop fallback activation must continue the outer "
                    "iteration (which re-runs the preflight); a break here "
                    "would end the turn without calling the fallback (#84733)"
                )
                assert not any(
                    isinstance(stmt, ast.Break) for stmt in node.body
                ), (
                    "outer-loop fallback activation must not break — that "
                    "exits the conversation loop and ends the turn (#84733)"
                )

    def test_restart_handler_clears_preflight_block(self):
        """The single consumer of restart_with_rebuilt_messages must clear
        _preflight_compression_blocked, so every retry-loop failover gets a
        fresh preflight against the fallback's context window (#84733)."""
        from agent import conversation_loop

        tree = ast.parse(inspect.getsource(conversation_loop.run_conversation))
        handlers = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Attribute)
            and node.test.attr == "restart_with_rebuilt_messages"
        ]
        assert handlers, "expected the restart_with_rebuilt_messages handler"
        consumer = [
            node
            for node in handlers
            if any(
                isinstance(stmt, ast.Assign)
                and any(
                    isinstance(t, ast.Attribute)
                    and t.attr == "restart_with_rebuilt_messages"
                    for t in stmt.targets
                )
                for stmt in node.body
            )
        ]
        assert consumer, "expected the flag-consuming handler"
        for node in consumer:
            assert any(
                isinstance(stmt, ast.Assign)
                and any(
                    isinstance(t, ast.Name)
                    and t.id == "_preflight_compression_blocked"
                    for t in stmt.targets
                )
                and isinstance(stmt.value, ast.Constant)
                and stmt.value.value is False
                for stmt in node.body
            ), (
                "the restart handler must clear _preflight_compression_blocked "
                "so the re-run preflight isn't skipped (#84733)"
            )


class TestAuxFallbackReplanThreadsTtl:
    """#84733 follow-up: the auxiliary fallback replan path threads the
    configured tier too — it has no live agent, so it reads the same
    config key agent_init snapshots into ``agent._cache_ttl``."""

    def test_configured_cache_ttl_reads_valid_tiers(self, monkeypatch):
        import agent.agent_runtime_helpers as arh

        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: {"prompt_caching": {"cache_ttl": "1h"}},
        )
        assert arh.configured_cache_ttl() == "1h"
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: {"prompt_caching": {"cache_ttl": "5m"}},
        )
        assert arh.configured_cache_ttl() == "5m"

    def test_configured_cache_ttl_none_for_disabled_or_unknown(self, monkeypatch):
        import agent.agent_runtime_helpers as arh

        for value in ("off", False, None, "2h"):
            monkeypatch.setattr(
                "hermes_cli.config.load_config_readonly",
                lambda value=value: {"prompt_caching": {"cache_ttl": value}},
            )
            assert arh.configured_cache_ttl() is None, value

    def test_replan_threads_configured_ttl_to_markers(self, monkeypatch):
        from agent import auxiliary_client

        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: {"prompt_caching": {"cache_ttl": "1h"}},
        )
        destination = auxiliary_client._FallbackDestination(
            "anthropic",
            "https://api.anthropic.com",
            "anthropic_messages",
            "claude-opus-4.8",
        )
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hello"},
        ]
        out_msgs, _ = auxiliary_client._replan_synchronous_cache_sections(
            messages, None, destination=destination
        )
        markers = _collect_cache_controls(out_msgs)
        assert markers, "expected cache_control markers on a caching route"
        assert all(m.get("ttl") == "1h" for m in markers), (
            "the configured 1h tier must reach auxiliary fallback replans"
        )

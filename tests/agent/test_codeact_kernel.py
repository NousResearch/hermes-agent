"""Tests for agent/codeact_kernel.py and codeact_kernel_process.py.

Integration-level: actually spawns the subprocess and exercises the full
IPC round-trip.  Requires Python 3.10+ and a working Unix domain socket.
"""

import json
import time
import pytest

from agent.codeact_kernel import HermesKernel, KernelNotStartedError, _format_exec_result


# ---------------------------------------------------------------------------
# _format_exec_result (unit, no subprocess)
# ---------------------------------------------------------------------------

class TestFormatExecResult:
    def test_stdout_included(self):
        msg = {"type": "exec_result", "status": "ok", "stdout": "hello\n", "stderr": ""}
        result = _format_exec_result(msg)
        assert "hello" in result

    def test_last_value_included(self):
        msg = {"type": "exec_result", "status": "ok", "stdout": "", "stderr": "",
               "last_value": "42"}
        result = _format_exec_result(msg)
        assert "42" in result

    def test_error_included(self):
        msg = {"type": "exec_result", "status": "error", "stdout": "", "stderr": "",
               "traceback": "Traceback:\n  ZeroDivisionError"}
        result = _format_exec_result(msg)
        assert "ZeroDivisionError" in result

    def test_empty_output(self):
        msg = {"type": "exec_result", "status": "ok", "stdout": "", "stderr": ""}
        result = _format_exec_result(msg)
        assert result == "(no output)"

    def test_truncation(self):
        """Stdout beyond MAX_STDOUT_CHARS should be truncated."""
        from agent.codeact_kernel import _MAX_STDOUT_CHARS
        big = "x" * (_MAX_STDOUT_CHARS + 1000)
        msg = {"type": "exec_result", "status": "ok", "stdout": big, "stderr": ""}
        result = _format_exec_result(msg)
        assert "TRUNCATED" in result
        assert len(result) < len(big)


# ---------------------------------------------------------------------------
# Full kernel integration tests
# ---------------------------------------------------------------------------

def _make_kernel(tool_dispatcher=None) -> HermesKernel:
    """Create and start a HermesKernel with an empty namespace."""
    if tool_dispatcher is None:
        tool_dispatcher = lambda name, args: json.dumps({"result": f"TOOL:{name}"})

    # Minimal namespace source: just the _call_tool bridge placeholder.
    # The real one comes from codeact_namespace; here we use a bare minimum.
    namespace_source = textwrap.dedent("""\
        def help(tool_name=None):
            return 'no tools in test namespace'
        def promote_to_skill(*a, **kw):
            return _call_tool('__promote_skill__', {})
        __protected__ = ['help', 'promote_to_skill', '__protected__']
    """)

    kernel = HermesKernel(
        session_id="test-session",
        tool_dispatcher=tool_dispatcher,
        namespace_source=namespace_source,
    )
    kernel.start()
    return kernel


import textwrap  # noqa: E402 — placed here to not affect the function signature


@pytest.fixture
def kernel():
    k = _make_kernel()
    yield k
    k.shutdown(quiet=True)


class TestHermesKernelLifecycle:
    def test_starts_and_shuts_down(self):
        k = _make_kernel()
        assert k.is_started
        k.shutdown()
        assert not k.is_started

    def test_double_start_is_idempotent(self):
        k = _make_kernel()
        k.start()  # second call should be no-op
        assert k.is_started
        k.shutdown(quiet=True)

    def test_execute_before_start_raises(self):
        k = HermesKernel("x", lambda n, a: "", "")
        with pytest.raises(KernelNotStartedError):
            k.execute("x = 1")


class TestHermesKernelExecution:
    def test_basic_print(self, kernel):
        result = kernel.execute("print('hello world')")
        assert "hello world" in result

    def test_expression_result(self, kernel):
        result = kernel.execute("6 * 7")
        assert "42" in result

    def test_state_persists_across_calls(self, kernel):
        kernel.execute("counter = 10")
        result = kernel.execute("print(counter)")
        assert "10" in result

    def test_variable_accumulation(self, kernel):
        kernel.execute("total = 0")
        kernel.execute("total += 5")
        kernel.execute("total += 3")
        result = kernel.execute("print(total)")
        assert "8" in result

    def test_function_definition_persists(self, kernel):
        kernel.execute("def double(x): return x * 2")
        result = kernel.execute("print(double(21))")
        assert "42" in result

    def test_syntax_error_returns_traceback(self, kernel):
        result = kernel.execute("def broken(:")
        assert "error" in result.lower() or "SyntaxError" in result

    def test_runtime_error_returns_traceback(self, kernel):
        result = kernel.execute("1 / 0")
        assert "ZeroDivisionError" in result

    def test_multiline_code(self, kernel):
        code = "results = []\nfor i in range(3):\n    results.append(i)\nprint(results)"
        result = kernel.execute(code)
        assert "0" in result and "2" in result

    def test_no_output_message(self, kernel):
        result = kernel.execute("x = 42  # just an assignment, no print")
        # Last expression is an assignment, not an expression — should return last_value
        # or "(no output)" since assignments have no value.
        assert result is not None


class TestHermesKernelSoftReset:
    def test_soft_reset_removes_user_vars(self, kernel):
        kernel.execute("my_var = 'hello'")
        kernel.soft_reset()
        result = kernel.execute("print(my_var)")
        assert "NameError" in result or "error" in result.lower()

    def test_soft_reset_preserves_protected(self, kernel):
        """help() should survive a soft reset."""
        kernel.soft_reset()
        result = kernel.execute("print(help())")
        assert "no tools" in result  # from our test namespace_source


class TestHermesKernelToolCalls:
    def test_call_tool_reaches_dispatcher(self):
        """Tool calls from within kernel code should route to the dispatcher."""
        calls = []

        def dispatcher(name, args):
            calls.append((name, args))
            return json.dumps({"result": "mocked"})

        # Namespace source that exposes a tool stub using _call_tool
        namespace_source = textwrap.dedent("""\
            def my_tool(query):
                return _call_tool('my_tool', {'query': query})
            def help(t=None): return ''
            __protected__ = ['my_tool', 'help', '__protected__']
        """)
        k = HermesKernel("test", dispatcher, namespace_source)
        k.start()
        try:
            result = k.execute("print(my_tool(query='test query'))")
            assert calls, "Dispatcher was never called"
            assert calls[0][0] == "my_tool"
            assert calls[0][1]["query"] == "test query"
        finally:
            k.shutdown(quiet=True)

    def test_web_search_json_error_is_promoted_to_clear_codeact_error(self):
        def dispatcher(name, args):
            assert name == "web_search"
            assert args["query"] == "blocked query"
            return json.dumps(
                {
                    "success": False,
                    "error": "SearXNG returned HTTP 429: rate limited",
                }
            )

        namespace_source = textwrap.dedent("""\
            def web_search(query, limit=5):
                return _call_tool('web_search', {'query': query, 'limit': limit})
            def help(t=None): return ''
            __protected__ = ['web_search', 'help', '__protected__']
        """)
        k = HermesKernel("test", dispatcher, namespace_source)
        k.start()
        try:
            result = k.execute("web_search(query='blocked query')")
            assert "web_search failed" in result
            assert "HTTP 429" in result
            assert "research_web" in result
            assert "Scrapling" in result
        finally:
            k.shutdown(quiet=True)

    def test_web_extract_json_error_is_promoted_to_clear_codeact_error(self):
        def dispatcher(name, args):
            assert name == "web_extract"
            assert args["urls"] == ["https://example.com"]
            return json.dumps(
                {
                    "success": False,
                    "error": "Extractor backend unavailable",
                }
            )

        namespace_source = textwrap.dedent("""\
            def web_extract(urls):
                return _call_tool('web_extract', {'urls': urls})
            def help(t=None): return ''
            __protected__ = ['web_extract', 'help', '__protected__']
        """)
        k = HermesKernel("test", dispatcher, namespace_source)
        k.start()
        try:
            result = k.execute("web_extract(urls=['https://example.com'])")
            assert "web_extract failed" in result
            assert "Extractor backend unavailable" in result
            assert "research_web" in result
            assert "Camofox" in result
        finally:
            k.shutdown(quiet=True)

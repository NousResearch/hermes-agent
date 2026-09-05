"""The same-model review fork must advertise the parent's ``tools[]`` verbatim (#103579).

``build_cache_parity_fork`` builds the fork with ``skip_memory=True`` (#5129) so no second
external memory provider is created for the parent's session. That also means
``inject_memory_provider_tools`` early-returns on the fork (no ``_memory_manager``), so the
fork's advertised ``tools[]`` silently loses the parent's memory-provider schemas (holographic
``fact_store`` / ``fact_feedback``, ...) — and with them the byte-exact prefix-cache key the
fork exists to reuse.

The fix copies the parent's advertised tools onto the non-routed fork after construction.
Copying (not appending) matters: the parent's order is base → memory-provider → context-engine,
so a tail append would still diverge. Dispatch stays whitelisted, so the review LLM still
cannot CALL those tools.
"""

import copy
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from agent.background_review import _review_tool_whitelist, build_cache_parity_fork


def _tool(name: str) -> Dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": f"{name} tool",
                                             "parameters": {"type": "object", "properties": {}}}}


# Parent order mirrors real init: base toolset tools, then memory-provider schemas appended by
# ``inject_memory_provider_tools``, then context-engine schemas.
_BASE_TOOLS = [_tool("read_file"), _tool("memory"), _tool("skill_manage")]
_PROVIDER_TOOLS = [_tool("fact_store"), _tool("fact_feedback")]
_CONTEXT_ENGINE_TOOLS = [_tool("context_recall")]
_PARENT_TOOLS = _BASE_TOOLS + _PROVIDER_TOOLS + _CONTEXT_ENGINE_TOOLS


class _FakeMemoryManager:
    """Stand-in for the parent's live external memory provider — never constructed on the fork."""

    def get_all_tool_schemas(self) -> List[Dict[str, Any]]:
        return [tool["function"] for tool in _PROVIDER_TOOLS]


class _FakeParent:
    """Minimal parent surface ``build_cache_parity_fork`` reads."""

    def __init__(self) -> None:
        self.model = "test-model"
        self.provider = "anthropic"
        self.platform = "cli"
        self.session_id = "sess-103579"
        self.max_tokens = 4096
        self.request_overrides: Dict[str, Any] = {}
        self.enabled_toolsets = ["memory", "skills"]
        self.disabled_toolsets: List[str] = []
        self.reasoning_config = {"enabled": True, "effort": "medium"}
        self.ephemeral_system_prompt = None
        self.prefill_messages: List[Dict[str, Any]] = []
        self._credential_pool = None
        self._memory_store = None
        self._memory_enabled = True
        self._user_profile_enabled = False
        self._cached_system_prompt = "PARENT-SYSTEM-PROMPT"
        self.session_start = None
        # The parent HAS a live external memory provider, so its advertised tools carry the
        # provider schemas.
        self._memory_manager = _FakeMemoryManager()
        self.tools = copy.deepcopy(_PARENT_TOOLS)
        self.valid_tool_names = {t["function"]["name"] for t in self.tools}

    def _current_main_runtime(self) -> Dict[str, Any]:
        return {"api_key": "k", "base_url": None, "api_mode": None}


class _ForkRecorder:
    """Stands in for ``AIAgent``: records init kwargs and exposes the tool surface a
    ``skip_memory=True`` fork really gets — base tools only, no provider schemas."""

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = dict(kwargs)
        self.tools = copy.deepcopy(_BASE_TOOLS)
        self.valid_tool_names = {t["function"]["name"] for t in self.tools}
        self._memory_enabled = False
        self._user_profile_enabled = False


def _build(parent: _FakeParent, *, routed: bool = False) -> _ForkRecorder:
    import run_agent

    runtime: Dict[str, Any] = {
        "provider": "openai" if routed else parent.provider,
        "model": "other-model" if routed else parent.model,
        "api_key": "k", "base_url": None, "api_mode": None, "credential_pool": None,
        "request_overrides": {}, "max_tokens": None, "command": None, "args": [],
        "routed": routed,
    }
    with patch.object(run_agent, "AIAgent", _ForkRecorder), \
         patch("agent.background_review._resolve_review_runtime", return_value=runtime):
        fork, _rt, got_routed = build_cache_parity_fork(parent, None, max_iterations=3)
    assert got_routed is routed
    return fork


@pytest.fixture
def parent() -> _FakeParent:
    return _FakeParent()


def test_same_model_fork_advertises_parent_tools_in_order(parent: _FakeParent) -> None:
    """Names AND order must match the parent's, provider tools included.

    Order is the point: the provider schemas sit BETWEEN the base tools and the context-engine
    tools, so appending the missing ones at the tail would still miss the prefix cache.
    """
    fork = _build(parent)

    fork_names = [t["function"]["name"] for t in fork.tools]
    parent_names = [t["function"]["name"] for t in parent.tools]
    assert fork_names == parent_names
    # Sabotage sentinel: on the pre-fix code the fork keeps its own base-only surface.
    assert "fact_store" in fork_names and "fact_feedback" in fork_names


def test_fork_tools_are_deep_copies_not_parent_aliases(parent: _FakeParent) -> None:
    """The fork must not alias the parent's schema dicts — in-place sanitization on the fork
    would otherwise rewrite the parent's own cached request bytes."""
    fork = _build(parent)

    assert fork.tools is not parent.tools
    assert all(f is not p for f, p in zip(fork.tools, parent.tools))


def test_fork_does_not_initialize_a_memory_provider(parent: _FakeParent) -> None:
    """Copying the advertised schemas must not hand the fork a provider instance: the fork is
    still built with ``skip_memory=True`` and grows no ``_memory_manager`` (#5129)."""
    fork = _build(parent)

    assert fork.init_kwargs.get("skip_memory") is True
    assert getattr(fork, "_memory_manager", None) is None


def test_fork_valid_tool_names_include_provider_tools(parent: _FakeParent) -> None:
    """``valid_tool_names`` is rebuilt from the copied schemas, the way normal init does."""
    fork = _build(parent)

    assert {"fact_store", "fact_feedback"} <= fork.valid_tool_names
    assert fork.valid_tool_names == {t["function"]["name"] for t in parent.tools}


def test_routed_fork_keeps_its_own_tools(parent: _FakeParent) -> None:
    """A routed fork runs on a different model — a different cache domain entirely — so it must
    build its own surface rather than inherit the parent's."""
    fork = _build(parent, routed=True)

    fork_names = [t["function"]["name"] for t in fork.tools]
    assert fork_names == [t["function"]["name"] for t in _BASE_TOOLS]
    assert "fact_store" not in fork_names


def test_provider_tools_stay_denied_at_dispatch(parent: _FakeParent) -> None:
    """Advertising is not permission: the dispatch whitelist must still refuse the provider
    tools unless they are named in configured ``extra_tools``."""
    fork = _build(parent)
    fork._memory_enabled = True

    whitelist, configured_extra = _review_tool_whitelist(fork, None)
    assert "fact_store" not in whitelist
    assert "fact_feedback" not in whitelist
    assert configured_extra == set()

    task_cfg: Optional[Dict[str, Any]] = {"extra_tools": ["fact_store"]}
    whitelist, configured_extra = _review_tool_whitelist(fork, task_cfg)
    assert "fact_store" in whitelist
    assert "fact_feedback" not in whitelist


def test_parent_without_memory_provider_leaves_fork_tools_alone(parent: _FakeParent) -> None:
    """No live provider on the parent means nothing extra to mirror — the fork's own
    construction already matches, so we leave it untouched."""
    parent._memory_manager = None
    parent.tools = copy.deepcopy(_BASE_TOOLS)

    fork = _build(parent)

    assert [t["function"]["name"] for t in fork.tools] == [t["function"]["name"] for t in _BASE_TOOLS]

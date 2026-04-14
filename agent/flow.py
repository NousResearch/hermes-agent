"""
Flow Decorator System — 声明式任务图编排

设计参考：CrewAI 的 @start/@listen 装饰器模式

用法：
    from agent.flow import Flow, entry, after

    class MyFlow(Flow):
        @entry        # NO parentheses — entry point, runs first
        def start(self):
            return data

        @after("start")  # parentheses REQUIRED — runs after start
        def next_step(self, result):
            return process(result)

    flow = MyFlow()
    result = flow.run()
"""

from __future__ import annotations

import asyncio
import functools
from collections import defaultdict
from typing import Any, Callable, TypeVar

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

def entry(func: Callable) -> Callable:
    """
    Mark a method as a flow entry point (runs first, concurrently if multiple).

    Usage: @entry   — NO parentheses (like @property)
    """
    @functools.wraps(func)
    def wrapper(self: Flow, *args: Any, **kwargs: Any) -> Any:
        return func(self, *args, **kwargs)
    wrapper.__flow_tags__ = [("__entry__", func.__name__)]
    return wrapper


def after(source: str) -> Callable:
    """
    Mark a method as listening to another method's output.

    The method receives the source's result as its argument.

    Usage: @after("fetch")  — parentheses REQUIRED
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self: Flow, *args: Any, **kwargs: Any) -> Any:
            return func(self, *args, **kwargs)
        prev = getattr(func, "__flow_tags__", [])
        wrapper.__flow_tags__ = prev + [(source, func.__name__)]
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Per-subclass metadata registry
# ---------------------------------------------------------------------------

_FLOW_META: dict[type, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------

class Flow:
    """
    Declarative task graph executor.

    Subclass with @entry and @after("...") decorated methods, then call .run().

    Execution model:
    1. @entry methods run first (concurrently if multiple)
    2. @after("X") methods run after X completes, receiving X's result
    3. Methods in the same layer (no inter-dependencies) run concurrently
    4. DAG built by topological sort: source -> listener edges
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        nodes: dict[str, Callable] = {}
        edges: dict[str, list[str]] = defaultdict(list)
        starts: list[str] = []

        for attr_name, attr_value in vars(cls).items():
            if not callable(attr_value) or attr_name.startswith("_"):
                continue
            tags = getattr(attr_value, "__flow_tags__", [])
            for src, my_name in tags:
                if my_name not in nodes:
                    nodes[my_name] = attr_value
                if src == "__entry__":
                    if my_name not in starts:
                        starts.append(my_name)
                else:
                    if src not in edges[my_name]:
                        edges[my_name].append(src)

        _FLOW_META[cls] = {
            "nodes": nodes,
            "edges": edges,
            "starts": starts,
        }

    def _meta(self) -> dict[str, Any]:
        for cls in type(self).__mro__:
            if cls in _FLOW_META:
                return _FLOW_META[cls]
        return {"nodes": {}, "edges": defaultdict(list), "starts": []}

    def _build_layers(self) -> list[list[str]]:
        """Topological sort into execution layers (Kahn's algorithm)."""
        meta = self._meta()
        nodes = set(meta["nodes"])
        edges = meta["edges"]
        starts = set(meta["starts"])

        if not nodes:
            return []

        in_degree: dict[str, int] = {}
        for n in nodes:
            sources = edges.get(n, [])
            in_degree[n] = 0 if n in starts else len(sources)

        layers, remaining = [], set(nodes)
        while remaining:
            layer = sorted(n for n in remaining if in_degree.get(n, 0) == 0)
            if not layer:
                layer = [next(iter(remaining))]
            layers.append(layer)
            remaining -= set(layer)
            for n in layer:
                for listener, sources in edges.items():
                    if n in sources and listener in remaining:
                        in_degree[listener] = max(0, in_degree.get(listener, 1) - 1)

        return layers

    def _sources_for(self, method_name: str) -> list[str]:
        meta = self._meta()
        return meta["edges"].get(method_name, [])

    async def _run_layer_async(
        self, layer: list[str], results: dict[str, Any]
    ) -> dict[str, Any]:
        meta = self._meta()

        async def run_one(name: str):
            method = meta["nodes"][name]
            srcs = self._sources_for(name)
            args = [results[s] for s in srcs if s in results]

            if asyncio.iscoroutinefunction(method):
                if len(args) == 1:
                    return name, await method(self, args[0])
                return name, await method(self, *args)
            else:
                if len(args) == 1:
                    return name, method(self, args[0])
                return name, method(self, *args)

        tasks = [run_one(n) for n in layer if n in meta["nodes"]]
        if not tasks:
            return results

        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        new_results = dict(results)
        for o in outcomes:
            if isinstance(o, Exception):
                raise RuntimeError(f"Flow error: {o}") from o
            name, val = o
            new_results[name] = val
        return new_results

    async def run_async(self) -> dict[str, Any]:
        """Execute the flow. Returns {method_name: result}."""
        layers = self._build_layers()
        results: dict[str, Any] = {}
        for layer in layers:
            results = await self._run_layer_async(layer, results)
        return results

    def run(self) -> dict[str, Any]:
        """Sync wrapper."""
        return asyncio.run(self.run_async())


# ---------------------------------------------------------------------------
# StatefulFlow
# ---------------------------------------------------------------------------

class StatefulFlow(Flow):
    """
    A Flow that maintains shared state across all steps.

    Each method receives (returns) the state object.
    State is automatically threaded through the flow.

    Usage:
        class CounterFlow(StatefulFlow):
            @entry
            def step1(self, state):
                state["count"] += 1
                return state

            @after("step1")
            def step2(self, state):
                state["count"] *= 2
                return state

        flow = CounterFlow({"count": 0})
        flow.run()
        print(flow.state["count"])  # 2
    """

    def __init__(self, initial_state: T):
        self._state: T = initial_state

    @property
    def state(self) -> T:
        return self._state

    def set_state(self, state: T) -> None:
        self._state = state

    async def run_async(self) -> dict[str, Any]:
        layers = self._build_layers()
        meta = self._meta()
        results: dict[str, Any] = {}
        current = self._state

        for layer in layers:
            async def run_one(name: str):
                nonlocal current
                method = meta["nodes"][name]
                try:
                    result = method(self, current)
                    if asyncio.iscoroutinefunction(method):
                        result = await result
                except TypeError:
                    result = method(self)
                    if asyncio.iscoroutinefunction(method):
                        result = await result
                current = result
                return name, result

            tasks = [run_one(n) for n in layer if n in meta["nodes"]]
            if tasks:
                outcomes = await asyncio.gather(*tasks, return_exceptions=True)
                for o in outcomes:
                    if isinstance(o, Exception):
                        raise RuntimeError(f"StatefulFlow error: {o}") from o
                    name, result = o
                    results[name] = result
                    current = result

        self._state = current
        return results


# ---------------------------------------------------------------------------
# AgentFlow — @entry/@after driving real agent execution
# ---------------------------------------------------------------------------

class AgentFlow(Flow):
    """
    A Flow subclass where @entry/@after methods are agent tasks.

    Subclass defines steps using @entry and @after.
    The coordinator calls step(goal, context) on each method,
    which delegates to the real agent.

    Usage:
        class CodeReviewFlow(AgentFlow):
            @entry
            def start(self, goal):
                return goal  # pass through

            @after("start")
            def review(self, context):
                # Calls delegate_task with the reviewer role
                return self.run_agent("review", context)

            @after("review")
            def synthesize(self, context):
                return self.run_agent("synthesize", context)

    Methods to override:
        run_agent(role, goal, context) -> str
    """

    def run_agent(self, role: str, goal: str, context: str) -> str:
        """
        Override this to plug in actual agent execution.
        Default: raise NotImplementedError.
        """
        raise NotImplementedError(
            "AgentFlow.run_agent() must be overridden to execute real agents. "
            "Use MultiAgentCoordinator for built-in agent execution."
        )

    def _step_result(self, step_name: str) -> str | None:
        """Get result from a completed step (for chaining)."""
        return None  # Override in subclass if steps need to read prior results

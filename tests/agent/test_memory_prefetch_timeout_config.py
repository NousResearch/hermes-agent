"""memory.external_prefetch_timeout must reach MemoryManager.

Guards the config -> runtime chain for the external memory prefetch timeout.
The value is latency dependent (a provider whose recall exceeds it has its
prefetch effectively never applied, because the timed-out worker thread keeps
running and later turns skip while it is alive), so it must stay configurable
and must degrade safely when the configured value is junk.
"""
import pytest

from agent.memory_manager import _EXTERNAL_PREFETCH_TIMEOUT_S, MemoryManager
from hermes_cli.config_defaults import DEFAULT_CONFIG


def resolve(mem_config):
    """Mirror agent_init's resolution of memory.external_prefetch_timeout."""
    try:
        raw = (mem_config or {}).get("external_prefetch_timeout")
        timeout = float(raw) if raw and float(raw) > 0 else None
    except (TypeError, ValueError):
        timeout = None
    return MemoryManager(external_prefetch_timeout=timeout)


def test_default_is_declared_in_config_defaults():
    assert "external_prefetch_timeout" in DEFAULT_CONFIG["memory"]


def test_documented_default_matches_runtime_fallback():
    # A drifting pair would make the documented default a lie.
    assert DEFAULT_CONFIG["memory"]["external_prefetch_timeout"] == pytest.approx(
        _EXTERNAL_PREFETCH_TIMEOUT_S
    )


@pytest.mark.parametrize(
    "configured,expected",
    [
        (30.0, 30.0),
        (45, 45.0),
        ("25", 25.0),   # YAML may hand back a string
        (0.5, 0.5),
    ],
)
def test_configured_value_reaches_manager(configured, expected):
    mgr = resolve({"external_prefetch_timeout": configured})
    assert mgr._external_prefetch_timeout == pytest.approx(expected)


@pytest.mark.parametrize(
    "configured",
    [None, 0, -5, "abc", "", [], {}],
)
def test_invalid_values_fall_back_without_raising(configured):
    # Startup must survive a bad config value rather than crash on it.
    mgr = resolve({"external_prefetch_timeout": configured})
    assert mgr._external_prefetch_timeout == pytest.approx(
        _EXTERNAL_PREFETCH_TIMEOUT_S
    )


@pytest.mark.parametrize("mem_config", [None, {}])
def test_absent_config_uses_default(mem_config):
    mgr = resolve(mem_config)
    assert mgr._external_prefetch_timeout == pytest.approx(
        _EXTERNAL_PREFETCH_TIMEOUT_S
    )


def test_manager_still_rejects_non_positive_when_passed_directly():
    # The guard inside MemoryManager must remain, independent of the caller.
    with pytest.raises(ValueError):
        MemoryManager(external_prefetch_timeout=0)


def test_agent_init_actually_passes_the_timeout():
    """Guard the real call site, not just a mirrored copy of its logic.

    The parametrised tests above exercise resolve(), which reproduces
    agent_init's logic. That cannot catch agent_init reverting to a bare
    MemoryManager() call, so assert on the construction site itself.
    """
    import inspect

    from agent import agent_init

    src = inspect.getsource(agent_init)
    assert "external_prefetch_timeout=_prefetch_timeout" in src, (
        "agent_init must pass the configured timeout into MemoryManager; "
        "a bare MemoryManager() call silently restores the hardcoded default"
    )
    assert '"external_prefetch_timeout"' in src, (
        "agent_init must read memory.external_prefetch_timeout from config"
    )

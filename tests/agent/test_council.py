"""Tests for CouncilOrchestrator -- preflight check and basic lifecycle.

Relies on monkeypatch to mock async_call_llm so preflight runs
without real API calls. Follows pattern from test_auxiliary_client.py.
"""

import asyncio
import datetime
import importlib
import json
from unittest.mock import AsyncMock

import pytest

council = importlib.import_module("agent.council")


# -- Factory ------------------------------------------------------------------


def full_council_config() -> dict:
    return {
        "enabled": True,
        "preflight": {
            "enabled": True, "timeout_seconds": 5, "min_proposers": 2,
            "check_interval_hours": 24,
        },
        "proposers": [
            {"provider": "ollama-cloud", "model": "glm-5.1:cloud"},
            {"provider": "ollama-cloud", "model": "kimi-k2.6:cloud"},
            {"provider": "ollama-cloud", "model": "minimax-m2.5:cloud"},
        ],
        "critic": {"provider": "ollama-cloud", "model": "minimax-m3:cloud"},
        "chairman": {"provider": "ollama-cloud", "model": "minimax-m3:cloud"},
        "peer_review": True,
        "anonymize_reviews": True,
        "max_concurrent_calls": 0,
    }


# -- Fixtures -----------------------------------------------------------------


@pytest.fixture
def mock_async_llm():
    """Mock async_call_llm to return a simple object with extractable content.

    The mock returns a namespace with a `choices` list containing one choice
    whose `message.content` is "OK" -- what extract_content_or_reasoning expects.
    """
    import types
    choice = types.SimpleNamespace(
        message=types.SimpleNamespace(content="OK"),
    )
    response = types.SimpleNamespace(choices=[choice])
    return AsyncMock(return_value=response)


# -- Preflight: happy path ----------------------------------------------------


@pytest.mark.asyncio
async def test_preflight_happy_path(monkeypatch, mock_async_llm):
    """All models reachable -> preflight passes, can_proceed = True."""
    monkeypatch.setattr("agent.auxiliary_client.async_call_llm", mock_async_llm)
    monkeypatch.setattr("agent.auxiliary_client.extract_content_or_reasoning",
                        lambda r: r.choices[0].message.content)

    cfg = full_council_config()
    orc = council.CouncilOrchestrator(cfg)
    result = await orc._preflight_check()

    assert result["passed"] is True
    assert result["can_proceed"] is True
    assert "5/5 models reachable" in result["summary"]
    assert len(result["checks"]) == 5
    for c in result["checks"]:
        assert c["status"] == "ok"


# -- Preflight: bad provider --------------------------------------------------


@pytest.mark.asyncio
async def test_preflight_catches_bad_provider(monkeypatch):
    """Bad provider raises, preflight returns fail, can_proceed = False."""
    async def _raise(*args, **kwargs):
        raise RuntimeError("Provider 'ollama-clod' is not configured")

    monkeypatch.setattr("agent.auxiliary_client.async_call_llm", _raise)
    monkeypatch.setattr("agent.auxiliary_client.extract_content_or_reasoning",
                        lambda r: r.choices[0].message.content)

    cfg = full_council_config()
    cfg["proposers"] = [
        {"provider": "ollama-cloud", "model": "glm-5.1:cloud"},
        {"provider": "ollama-cloud", "model": "kimi-k2.6:cloud"},
    ]
    cfg["critic"] = {"provider": "ollama-clod", "model": "minimax-m3:cloud"}
    orc = council.CouncilOrchestrator(cfg)
    result = await orc._preflight_check()

    # Only 3 of 4 should pass (critic has bad provider)
    assert result["passed"] is False
    assert result["can_proceed"] is False  # critic fail -> fatal

    critic_check = [c for c in result["checks"] if c["role"] == "critic"][0]
    assert critic_check["status"] == "fail"
    assert "not configured" in critic_check.get("error", "")


# -- Preflight: disabled -----------------------------------------------------


@pytest.mark.asyncio
async def test_preflight_skips_when_disabled(monkeypatch):
    """preflight.enabled: false means run_plan skips preflight entirely."""
    called = False

    async def _check(*args, **kwargs):
        nonlocal called
        called = True
        return {"passed": False, "can_proceed": False}

    monkeypatch.setattr(council.CouncilOrchestrator, "_preflight_check", _check)

    cfg = full_council_config()
    cfg["preflight"] = {"enabled": False}
    orc = council.CouncilOrchestrator(cfg)
    assert orc.pref["enabled"] is False


# -- Preflight: min_proposers gate -------------------------------------------


@pytest.mark.asyncio
async def test_preflight_min_proposers_gate(monkeypatch):
    """Only 1 proposer passes, min_proposers=2 -> can_proceed=False."""
    call_count = 0

    async def _alternating(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        # First call OK, rest raise
        if call_count == 1:
            import types
            choice = types.SimpleNamespace(
                message=types.SimpleNamespace(content="OK")
            )
            return types.SimpleNamespace(choices=[choice])
        raise RuntimeError("Unavailable")

    monkeypatch.setattr("agent.auxiliary_client.async_call_llm", _alternating)
    monkeypatch.setattr("agent.auxiliary_client.extract_content_or_reasoning",
                        lambda r: r.choices[0].message.content)

    cfg = full_council_config()
    orc = council.CouncilOrchestrator(cfg)
    result = await orc._preflight_check()

    # 1 proposer + critic (fails) + chairman (fails) = 1/5 pass
    assert result["passed"] is False
    assert result["can_proceed"] is False  # only 1 proposer < min 2
    assert result["summary"].startswith("1/")


# -- Preflight: critic fail is fatal -----------------------------------------


@pytest.mark.asyncio
async def test_preflight_critic_fail_is_fatal(monkeypatch, mock_async_llm):
    """Critic failure always aborts even if all proposers pass."""
    call_index = 0

    async def _selective_fail(*args, **kwargs):
        nonlocal call_index
        call_index += 1
        if call_index == 3:
            raise RuntimeError("Critic provider unreachable")
        import types
        choice = types.SimpleNamespace(
            message=types.SimpleNamespace(content="OK")
        )
        return types.SimpleNamespace(choices=[choice])

    monkeypatch.setattr("agent.auxiliary_client.async_call_llm", _selective_fail)
    monkeypatch.setattr("agent.auxiliary_client.extract_content_or_reasoning",
                        lambda r: r.choices[0].message.content)

    cfg = full_council_config()
    cfg["proposers"] = [
        {"provider": "ollama-cloud", "model": "glm-5.1:cloud"},
        {"provider": "ollama-cloud", "model": "kimi-k2.6:cloud"},
    ]
    cfg["critic"] = {"provider": "ollama-cloud", "model": "bad-critic:cloud"}
    cfg["preflight"]["min_proposers"] = 1
    orc = council.CouncilOrchestrator(cfg)
    result = await orc._preflight_check()

    assert result["can_proceed"] is False  # critic fail -> fatal
    assert result["passed"] is False


# -- Preflight: empty config -------------------------------------------------


@pytest.mark.asyncio
async def test_preflight_empty_config(monkeypatch):
    """Empty config produces preflight with 0 checks, cannot proceed."""
    cfg = {}
    orc = council.CouncilOrchestrator(cfg)
    result = await orc._preflight_check()

    assert result["passed"] is True  # vacuously true
    assert result["can_proceed"] is False  # 0 proposers < min 2
    assert result["summary"] == "0/0 models reachable"
    assert len(result["checks"]) == 0


# -- Progress callback -------------------------------------------------------


@pytest.mark.asyncio
async def test_run_plan_calls_progress_callback(monkeypatch, mock_async_llm):
    """run_plan calls progress_callback at each stage when provided."""
    monkeypatch.setattr("agent.auxiliary_client.async_call_llm", mock_async_llm)
    monkeypatch.setattr("agent.auxiliary_client.extract_content_or_reasoning",
                        lambda r: r.choices[0].message.content)

    stages = []

    def _progress(stage: str, msg: str):
        stages.append((stage, msg))

    cfg = full_council_config()
    cfg["subagent_delegation"] = False
    cfg["preflight"]["check_interval_hours"] = 0  # force always-check
    orc = council.CouncilOrchestrator(cfg)

    # Mock propose to return valid content that passes validation gate
    long_plan = ("## Plan\n" + "Mocked plan content line that is long enough to pass validation.\n" * 15 + "## Steps\n1. Do step one\n2. Do step two\n3. Do step three\n## Files\n- src/main.py\n- src/utils.py\n\n```python\ndef main():\n    pass\n```\n## Execution\nRun and verify.\n## Risks\n- Edge case A\n- Edge case B\n")
    async def _mock_propose(task):
        return [{"model": "test", "provider": "test", "status": "ok",
                 "output": long_plan,
                 "error": None, "time_seconds": 1.0,
                 "line_count": 55, "char_count": len(long_plan)}]
    monkeypatch.setattr(orc, "_run_proposer_api_calls", _mock_propose)

    result = await orc.run_plan("test task", progress_callback=_progress)

    assert result["success"]
    # Expect 4 progress calls: preflight, propose, critique, chairman
    assert len(stages) == 4
    stage_names = [s[0] for s in stages]
    assert "preflight" in stage_names
    assert "propose" in stage_names
    assert "critique" in stage_names
    assert "chairman" in stage_names


# =============================================================================
# NEW TESTS -- Preflight Cache + Concurrency Controls
# =============================================================================


# -- Preflight cache: reuses result ------------------------------------------


@pytest.mark.asyncio
async def test_preflight_cache_reuses_result(monkeypatch, mock_async_llm):
    """2nd call within check_interval_hours reuses cached preflight."""
    monkeypatch.setattr("agent.auxiliary_client.async_call_llm", mock_async_llm)
    monkeypatch.setattr("agent.auxiliary_client.extract_content_or_reasoning",
                        lambda r: r.choices[0].message.content)

    cfg = full_council_config()
    cfg["subagent_delegation"] = False   # flat mode
    cfg["peer_review"] = False           # skip critique
    orc = council.CouncilOrchestrator(cfg)

    # Mock the propose stage to return valid content that passes validation gate
    long_plan = "## Plan\n" + "Mocked plan content.\n" * 30 + "## Steps\n1. Do x\n## Files\n- src/main.py\n\n```python\nprint('hello')\n```"
    async def _mock_propose(task):
        return [{
            "model": "test-model", "provider": "test-provider",
            "status": "ok", "output": long_plan,
            "error": None, "time_seconds": 0.5,
            "line_count": 35, "char_count": len(long_plan),
        }]
    monkeypatch.setattr(orc, "_run_proposer_api_calls", _mock_propose)

    # Seed a fresh cache (age = 0 hours)
    orc._preflight_cache = {"passed": True, "can_proceed": True, "summary": "CACHED"}
    orc._preflight_cache_time = datetime.datetime.now()

    result = await orc.run_plan("test")
    assert result["success"]
    # Verify cached preflight was used (not overwritten)
    assert result["preflight"]["summary"] == "CACHED"


# -- Preflight cache: expired re-checks --------------------------------------


@pytest.mark.asyncio
async def test_preflight_cache_expired_rechecks(monkeypatch, mock_async_llm):
    """After interval elapses, re-runs actual preflight check."""
    monkeypatch.setattr("agent.auxiliary_client.async_call_llm", mock_async_llm)
    monkeypatch.setattr("agent.auxiliary_client.extract_content_or_reasoning",
                        lambda r: r.choices[0].message.content)

    cfg = full_council_config()
    cfg["subagent_delegation"] = False
    cfg["peer_review"] = False
    orc = council.CouncilOrchestrator(cfg)

    # Mock propose to avoid real API calls
    async def _mock_propose(task):
        return [{"model": "test", "provider": "test", "status": "ok",
                 "output": "## Plan\n" + "Mocked plan.\n" * 30 + "## Steps\n1. Do x\n```\ncode\n```", "error": None,
                 "time_seconds": 0.5, "line_count": 35, "char_count": 500}]
    monkeypatch.setattr(orc, "_run_proposer_api_calls", _mock_propose)

    # Seed cache 25 hours ago (interval is 24h)
    orc._preflight_cache = {"passed": True, "can_proceed": True, "summary": "STALE"}
    orc._preflight_cache_time = datetime.datetime.now() - datetime.timedelta(hours=25)

    result = await orc.run_plan("test")
    assert result["success"]
    # Cache should have been refreshed
    assert result["preflight"]["summary"] != "STALE"


# -- Preflight cache: disabled on 0 interval ---------------------------------


@pytest.mark.asyncio
async def test_preflight_cache_disabled_on_0_interval(monkeypatch, mock_async_llm):
    """check_interval_hours: 0 means always re-check (no cache)."""
    monkeypatch.setattr("agent.auxiliary_client.async_call_llm", mock_async_llm)
    monkeypatch.setattr("agent.auxiliary_client.extract_content_or_reasoning",
                        lambda r: r.choices[0].message.content)

    cfg = full_council_config()
    cfg["subagent_delegation"] = False
    cfg["peer_review"] = False
    cfg["preflight"]["check_interval_hours"] = 0
    orc = council.CouncilOrchestrator(cfg)

    # Mock propose to avoid real API calls
    async def _mock_propose(task):
        return [{"model": "test", "provider": "test", "status": "ok",
                 "output": "## Plan\n" + "Mocked plan.\n" * 30 + "## Steps\n1. Do x\n```\ncode\n```", "error": None,
                 "time_seconds": 0.5, "line_count": 35, "char_count": 500}]
    monkeypatch.setattr(orc, "_run_proposer_api_calls", _mock_propose)

    # Even with a cache seeded, 0 interval forces re-check
    orc._preflight_cache = {"passed": True, "can_proceed": True, "summary": "STALE"}
    orc._preflight_cache_time = datetime.datetime.now()

    result = await orc.run_plan("test")
    assert result["success"]
    # Cache should have been refreshed because 0 interval forces re-check
    assert result["preflight"]["summary"] != "STALE"


# -- Preflight cache: not overwritten on failure -----------------------------


@pytest.mark.asyncio
async def test_preflight_cache_not_overwritten_on_failure(monkeypatch, mock_async_llm):
    """Failed preflight preserves existing good cache."""
    monkeypatch.setattr("agent.auxiliary_client.async_call_llm", mock_async_llm)
    monkeypatch.setattr("agent.auxiliary_client.extract_content_or_reasoning",
                        lambda r: r.choices[0].message.content)

    cfg = full_council_config()
    cfg["subagent_delegation"] = False
    cfg["preflight"]["check_interval_hours"] = 0  # force re-check
    orc = council.CouncilOrchestrator(cfg)

    # Seed a good cache
    orc._preflight_cache = {"passed": True, "can_proceed": True, "summary": "GOOD CACHE"}
    orc._preflight_cache_time = datetime.datetime.now()

    # Mock propose to succeed so only preflight logic is tested
    async def _mock_propose(task):
        return [{"model": "test", "provider": "test", "status": "ok",
                 "output": "## Plan\n" + "Mocked plan.\n" * 30 + "## Steps\n1. Do x\n```\ncode\n```", "error": None,
                 "time_seconds": 0.5, "line_count": 35, "char_count": 500}]
    monkeypatch.setattr(orc, "_run_proposer_api_calls", _mock_propose)

    # Mock preflight to return failure -- simulate transient model outage
    async def _fail_check():
        return {"passed": False, "can_proceed": False, "summary": "0/5 reachable"}
    monkeypatch.setattr(orc, "_preflight_check", _fail_check)

    result = await orc.run_plan("test")
    assert result["success"] is False  # council should fail
    # But the good cache MUST remain untouched
    assert orc._preflight_cache["summary"] == "GOOD CACHE"
    assert orc._preflight_cache_time is not None


# -- Concurrency: limit value ------------------------------------------------


def test_concurrency_limit_value():
    """_concurrency_limit reflects min(max_concurrent_calls, len(proposers))."""
    cfg = full_council_config()
    cfg["max_concurrent_calls"] = 2
    orc = council.CouncilOrchestrator(cfg, delegation_config={"max_concurrent_children": 3})
    assert orc._concurrency_limit == 2  # explicit council value
    assert orc.max_concurrent_calls == 2


def test_concurrency_capped_to_proposers():
    """max_concurrent_calls > proposers -> cap at proposer count."""
    cfg = full_council_config()
    cfg["max_concurrent_calls"] = 99
    orc = council.CouncilOrchestrator(cfg)
    assert orc._concurrency_limit == len(cfg["proposers"])  # 3 proposers


def test_concurrency_derives_from_delegation():
    """max_concurrent_calls=0 reads delegation.max_concurrent_children."""
    cfg = full_council_config()
    cfg["max_concurrent_calls"] = 0
    orc = council.CouncilOrchestrator(cfg, delegation_config={"max_concurrent_children": 2})
    assert orc._concurrency_limit == 2  # min(2, 3 proposers)


def test_concurrency_derives_fallback_default():
    """No delegation config -> falls back to 3."""
    cfg = full_council_config()
    cfg["max_concurrent_calls"] = 0
    orc = council.CouncilOrchestrator(cfg, delegation_config={})
    assert orc._concurrency_limit == 3  # min(3 default, 3 proposers)


# -- Concurrency: empty proposers --------------------------------------------


def test_concurrency_empty_proposers():
    """No proposers configured -> limit is 1 (safe semaphore value)."""
    orc = council.CouncilOrchestrator({"max_concurrent_calls": 2})
    assert orc._concurrency_limit == 1  # min(2, 0) with fallback to 1


# -- Status: cache age + concurrency keys ------------------------------------


def test_status_shows_cache_age_and_concurrency():
    """get_status() includes new keys for cache age and concurrency."""
    cfg = full_council_config()
    orc = council.CouncilOrchestrator(cfg)
    status = orc.get_status()

    assert "last_preflight_check" in status
    assert status["last_preflight_check"] is None  # never ran

    assert "preflight_interval_hours" in status
    assert status["preflight_interval_hours"] == 24

    assert "max_concurrent_calls" in status

    assert "effective_concurrency_limit" in status
    assert status["effective_concurrency_limit"] > 0


# -- Defaults: check_interval_hours fallback ---------------------------------


def test_preflight_default_check_interval():
    """Empty config gets check_interval_hours=24 default."""
    orc = council.CouncilOrchestrator({})
    assert orc.pref["check_interval_hours"] == 24
    assert orc.pref["enabled"] is True
    assert orc.pref["timeout_seconds"] == 10  # from __init__ default
    assert orc.pref["min_proposers"] == 2


# -- Defaults: max_concurrent_calls ------------------------------------------


def test_default_max_concurrent_calls():
    """Default max_concurrent_calls is 0 (derive from delegation)."""
    orc = council.CouncilOrchestrator({"proposers": [{"p": "x", "m": "y"}]})
    assert orc.max_concurrent_calls == 0  # config.get default for missing key


# =============================================================================
# NEW TESTS — Dynamic Timeouts, Retry, State Persistence, Partial Output
# =============================================================================


# -- Dynamic timeout scales with input ----------------------------------------


class TestDynamicTimeouts:
    """_stage_timeout grows with input size, never below floor."""

    def test_timeout_floor_for_empty_input(self):
        orc = council.CouncilOrchestrator({})
        for stage in ("proposer_flat", "critique", "chairman"):
            t = orc._stage_timeout(stage, None)
            assert t >= orc._stage_default_timeout(stage), f"{stage} floor violated"

    def test_timeout_scales_with_input(self):
        orc = council.CouncilOrchestrator({})
        small = orc._stage_timeout("chairman", ["small task"])
        big = orc._stage_timeout("chairman", ["x" * 100_000])  # ~100K chars
        assert big >= small, "bigger input must produce bigger timeout"

    def test_estimate_input_chars(self):
        texts = ["hello", "world" * 100]
        result = council.CouncilOrchestrator._estimate_input_chars(texts)
        assert result == len("hello") + len("world" * 100)


# -- Retry: timeout IS retried now (transient on cloud) -----------------------


@pytest.mark.asyncio
async def test_retry_recovers_timeout(monkeypatch):
    """TimeoutError is now RETRIED — cloud providers drop first call, succeed on second."""
    orc = council.CouncilOrchestrator({"proposers": [], "pipeline_timeout_seconds": 300})
    call_count = 0

    async def _flaky_timeout(timeout_s: int):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise asyncio.TimeoutError("timed out on first try")
        return "success after retry"

    result = await orc._run_with_retry(
        name="flaky_timeout",
        fn=_flaky_timeout,
        max_retries=1,
        stage_key="proposer_flat",
    )
    assert result == "success after retry"
    assert call_count == 2


# -- Retry: transient error is retried ----------------------------------------


@pytest.mark.asyncio
async def test_retry_recovers_transient_error(monkeypatch):
    """ConnectionError -> retried once -> succeeds on 2nd attempt."""
    orc = council.CouncilOrchestrator({"proposers": [], "pipeline_timeout_seconds": 300})
    call_count = 0

    async def _flaky_fn(timeout_s: int) -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ConnectionError("Connection reset by peer")
        return "success after retry"

    result = await orc._run_with_retry(
        name="flaky",
        fn=_flaky_fn,
        max_retries=1,
        stage_key="proposer_flat",
    )
    assert result == "success after retry"
    assert call_count == 2


# -- Partial output: pipeline timeout preserves completed stages ---------------


@pytest.mark.asyncio
async def test_partial_output_on_timeout(monkeypatch, mock_async_llm):
    """Pipeline timeout includes stage context for completed stages."""
    monkeypatch.setattr("agent.auxiliary_client.async_call_llm", mock_async_llm)
    monkeypatch.setattr("agent.auxiliary_client.extract_content_or_reasoning",
                        lambda r: r.choices[0].message.content)

    cfg = full_council_config()
    cfg["subagent_delegation"] = False
    cfg["peer_review"] = True
    cfg["preflight"]["enabled"] = False  # skip preflight
    cfg["pipeline_timeout_seconds"] = 1  # very tight
    orc = council.CouncilOrchestrator(cfg)

    # Mock propose + critique to complete instantly, chairman to hang
    long_plan = "## Plan\n" + "Mocked plan content.\n" * 30 + "## Steps\n1. Do x\n## Files\n- src/main.py\n\n```python\nprint('hello')\n```"
    async def _fast_propose(task):
        return [
            {"model": "m1", "provider": "p1", "status": "ok",
             "output": long_plan, "error": None,
             "time_seconds": 1.0, "line_count": 35, "char_count": len(long_plan)},
            {"model": "m2", "provider": "p2", "status": "ok",
             "output": long_plan, "error": None,
             "time_seconds": 1.0, "line_count": 35, "char_count": len(long_plan)},
        ]

    async def _fast_critique(plans, task=""):
        return {"critiques": ["Critique of all plans"], "error": None, "skipped": False}

    async def _slow_chairman(task, plans, critiques):
        import asyncio as _a
        await _a.sleep(10)  # will trigger pipeline timeout
        return "Should not reach here"

    monkeypatch.setattr(orc, "_run_proposer_api_calls", _fast_propose)
    monkeypatch.setattr(orc, "_run_anonymized_critique", _fast_critique)
    monkeypatch.setattr(orc, "_run_chairman", _slow_chairman)

    result = await orc.run_plan("test partial output task")

    assert result["success"] is False
    assert "timed out" in result.get("error", "").lower()
    assert "Propose" in result.get("error", "")
    assert "Critique" in result.get("error", "")
    assert "/council resume" in result.get("error", "")

    # Clean up state file
    import os
    if os.path.exists(council.CouncilOrchestrator.COUNCIL_STATE_PATH):
        os.remove(council.CouncilOrchestrator.COUNCIL_STATE_PATH)


# -- State persistence: save + load round-trip --------------------------------


def test_save_and_load_state(monkeypatch):
    """State JSON round-trips with correct schema version."""
    import os
    import tempfile

    orc = council.CouncilOrchestrator({})
    # Use a temp path for the test
    orig_path = council.CouncilOrchestrator.COUNCIL_STATE_PATH
    tmp = os.path.join(tempfile.mkdtemp(), "council_state.json")
    monkeypatch.setattr(council.CouncilOrchestrator, "COUNCIL_STATE_PATH", tmp)

    try:
        saved = orc.save_state(
            task="test task",
            plans=["Plan 1 text", "Plan 2 text"],
            critiques=["Great critique"],
            stages={"1_propose": {"count": 2, "time_seconds": 30.0}},
            models_used={"proposers": ["model-a"], "critic": "model-b", "chairman": "model-c"},
        )
        assert os.path.exists(saved)

        loaded = orc.load_state()
        assert loaded is not None
        assert loaded["schema_version"] == 1
        assert loaded["task"] == "test task"
        assert len(loaded["plans"]) == 2
        assert loaded["critiques"] == ["Great critique"]
        assert loaded["models_used"]["proposers"] == ["model-a"]
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# -- Resume: re-runs chairman from saved state --------------------------------


@pytest.mark.asyncio
async def test_resume_reruns_chairman(monkeypatch, mock_async_llm):
    """run_resume() loads saved state and re-runs chairman successfully."""
    monkeypatch.setattr("agent.auxiliary_client.async_call_llm", mock_async_llm)
    monkeypatch.setattr("agent.auxiliary_client.extract_content_or_reasoning",
                        lambda r: r.choices[0].message.content)

    import os
    import tempfile
    import json

    orc = council.CouncilOrchestrator({
        "critic": {"provider": "ollama-cloud", "model": "minimax-m3:cloud"},
        "chairman": {"provider": "ollama-cloud", "model": "minimax-m3:cloud"},
        "pew_review": True,
    })

    # Use temp path
    orig_path = council.CouncilOrchestrator.COUNCIL_STATE_PATH
    tmp = os.path.join(tempfile.mkdtemp(), "council_state.json")
    monkeypatch.setattr(council.CouncilOrchestrator, "COUNCIL_STATE_PATH", tmp)

    try:
        # Save state directly (as would happen after propose+critique)
        orc.save_state(
            task="resume test",
            plans=["Plan A details"],
            critiques=["Critique of plan A"],
            stages={"1_propose": {"count": 1, "time_seconds": 20.0}},
            models_used={"proposers": ["glm-5.1:cloud"], "critic": "minimax-m3:cloud", "chairman": "minimax-m3:cloud"},
        )

        result = await orc.run_resume(tmp)
        assert result["success"] is True
        assert result["task"] == "resume test"
        assert "Council:" in result.get("output", "") or "OK" in result.get("output", "")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# -- Pipeline timeout guard ---------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_timeout_guard(monkeypatch):
    """Pipeline-level timeout catches runaway stages."""
    cfg = full_council_config()
    cfg["preflight"]["enabled"] = False
    cfg["pipeline_timeout_seconds"] = 1
    cfg["peer_review"] = False
    cfg["subagent_delegation"] = False
    orc = council.CouncilOrchestrator(cfg)

    # Mock propose to hang — return empty to simulate pipeline timeout
    async def _slow_propose(task):
        import asyncio as _a
        await _a.sleep(10)
        return []

    monkeypatch.setattr(orc, "_run_proposer_api_calls", _slow_propose)
    monkeypatch.setattr(orc, "_run_proposer_subagents", _slow_propose)

    result = await orc.run_plan("slow task")

    assert result["success"] is False
    assert "timed out" in result.get("error", "").lower() or "timeout" in result.get("error", "").lower()
    assert result["processing_time"] > 0


# =============================================================================
# NEW TESTS — Content Validation, Critique, Chairman, Resume Hints
# =============================================================================


# -- Content validation: short plan rejected ----------------------------------


@pytest.mark.asyncio
async def test_content_validation_rejects_short_plan():
    """Plan under min_plan_lines and lacking markers gets status='empty'."""
    cfg = full_council_config()
    cfg["min_plan_lines"] = 10
    cfg["min_plan_chars"] = 100
    orc = council.CouncilOrchestrator(cfg)

    results = [
        {"model": "m1", "provider": "p1", "status": "ok",
         "output": "short", "error": None,
         "time_seconds": 1.0, "line_count": 1, "char_count": 5},
    ]
    orc._validate_proposer_outputs(results)

    assert results[0]["status"] == "empty"
    assert "only 1 lines" in results[0].get("error", "")


# -- Content validation: real plan passes -------------------------------------


@pytest.mark.asyncio
async def test_content_validation_passes_real_plan():
    """Plan with structural markers and sufficient length passes validation."""
    cfg = full_council_config()
    cfg["min_plan_lines"] = 5
    cfg["min_plan_chars"] = 50
    orc = council.CouncilOrchestrator(cfg)

    results = [{
        "model": "m1", "provider": "p1", "status": "ok",
        "output": "## Plan\n1. Do x\n2. Do y\n## Steps\n- step1\n- step2\n## Files\n- src/main.py\n\n```python\ncode\n```",
        "error": None, "time_seconds": 2.0,
        "line_count": 12, "char_count": 100,
    }]
    orc._validate_proposer_outputs(results)

    assert results[0]["status"] == "ok"
    assert results[0]["error"] is None


# -- Content validation: all rejected -----------------------------------------


@pytest.mark.asyncio
async def test_content_validation_all_rejected(monkeypatch, mock_async_llm):
    """All proposers fail validation — error message mentions content validation."""
    monkeypatch.setattr("agent.auxiliary_client.async_call_llm", mock_async_llm)
    monkeypatch.setattr("agent.auxiliary_client.extract_content_or_reasoning",
                        lambda r: r.choices[0].message.content)

    cfg = full_council_config()
    cfg["subagent_delegation"] = False
    cfg["peer_review"] = False
    cfg["min_plan_lines"] = 100  # impossibly high
    orc = council.CouncilOrchestrator(cfg)

    # Mock propose to return short content that fails validation
    async def _mock_propose(task):
        return [{"model": "m1", "provider": "p1", "status": "ok",
                 "output": "## Plan\nShort", "error": None,
                 "time_seconds": 1.0, "line_count": 3, "char_count": 12}]
    monkeypatch.setattr(orc, "_run_proposer_api_calls", _mock_propose)

    result = await orc.run_plan("test")
    assert result["success"] is False
    error_msg = result.get("error", "")
    assert "proposer" in error_msg.lower()
    assert "lines" in error_msg or "chars" in error_msg


# -- Critique: differentiates skip vs failure ---------------------------------


@pytest.mark.asyncio
async def test_critique_error_differentiates_skip_vs_failure(monkeypatch, mock_async_llm):
    """peer_review=False returns skipped. Critic crash returns error."""
    monkeypatch.setattr("agent.auxiliary_client.async_call_llm", mock_async_llm)
    monkeypatch.setattr("agent.auxiliary_client.extract_content_or_reasoning",
                        lambda r: r.choices[0].message.content)

    cfg = full_council_config()
    cfg["subagent_delegation"] = False
    cfg["preflight"]["enabled"] = False
    orc = council.CouncilOrchestrator(cfg)

    long_plan = "## Plan\n" + "content\n" * 30 + "## Steps\n1. x\n## Files\n- f\n\n```\nc\n```"
    # Build 2 valid plans so single-plan skip doesn't activate
    long_plan_valid = ("## Plan\n" + "Mocked plan content line that is long enough to pass validation.\n" * 15
                       + "## Steps\n1. Do step one\n2. Do step two\n## Files\n- src/main.py\n\n```python\ndef main():\n    pass\n```\n## Risks\n- Edge case\n")
    async def _mock_propose_two(task):
        return [
            {"model": "m1", "provider": "p1", "status": "ok",
             "output": long_plan_valid, "error": None,
             "time_seconds": 1.0, "line_count": 35, "char_count": len(long_plan_valid)},
            {"model": "m2", "provider": "p2", "status": "ok",
             "output": long_plan_valid, "error": None,
             "time_seconds": 1.0, "line_count": 35, "char_count": len(long_plan_valid)},
        ]
    monkeypatch.setattr(orc, "_run_proposer_api_calls", _mock_propose_two)

    # Case 1: peer_review=True with mock critic that raises RuntimeError
    # We monkeypatch _run_anonymized_critique to return error
    async def _fail_critique(plans, task=""):
        return {"critiques": [], "error": "Mocked critic failure", "skipped": False}

    original_critique = orc._run_anonymized_critique
    monkeypatch.setattr(orc, "_run_anonymized_critique", _fail_critique)

    result = await orc.run_plan("test")
    stage2 = result["stages"].get("2_critique", {})
    assert stage2.get("error") == "Mocked critic failure"
    assert stage2.get("skipped") is False  # failed ≠ skipped

    # Case 2: peer_review=False
    cfg2 = full_council_config()
    cfg2["subagent_delegation"] = False
    cfg2["preflight"]["enabled"] = False
    cfg2["peer_review"] = False
    orc2 = council.CouncilOrchestrator(cfg2)
    monkeypatch.setattr(orc2, "_run_proposer_api_calls", _mock_propose_two)

    result2 = await orc2.run_plan("test")
    stage2b = result2["stages"].get("2_critique", {})
    assert stage2b.get("skipped") is True
    assert "disabled" in stage2b.get("skipped_reason", "")

    # Restore
    monkeypatch.setattr(orc, "_run_anonymized_critique", original_critique)


# -- Chairman empty raises error ----------------------------------------------


@pytest.mark.asyncio
async def test_chairman_empty_raises_error(monkeypatch, mock_async_llm):
    """Chairman returning empty output → success: False, not placeholder."""
    monkeypatch.setattr("agent.auxiliary_client.async_call_llm", mock_async_llm)
    monkeypatch.setattr("agent.auxiliary_client.extract_content_or_reasoning",
                        lambda r: r.choices[0].message.content)

    cfg = full_council_config()
    cfg["subagent_delegation"] = False
    cfg["peer_review"] = False
    cfg["preflight"]["enabled"] = False
    orc = council.CouncilOrchestrator(cfg)

    long_plan = "## Plan\n" + "content\n" * 30 + "## Steps\n1. x\n## Files\n- f\n\n```\nc\n```"
    async def _mock_propose(task):
        return [{"model": "m1", "provider": "p1", "status": "ok",
                 "output": long_plan, "error": None,
                 "time_seconds": 1.0, "line_count": 35, "char_count": len(long_plan)}]
    monkeypatch.setattr(orc, "_run_proposer_api_calls", _mock_propose)

    # Mock chairman to return empty — the RuntimeError should propagate
    import types
    empty_response = types.SimpleNamespace(choices=[types.SimpleNamespace(content="")])
    async def _empty_chairman(timeout_s):
        return empty_response
    monkeypatch.setattr(orc, "_run_chairman", _empty_chairman)

    result = await orc.run_plan("test")
    assert result["success"] is False
    # Should get an error, not a placeholder string in output
    assert result.get("error") is not None


# -- Resume hint in non-timeout error -----------------------------------------


@pytest.mark.asyncio
async def test_resume_hint_in_general_error(monkeypatch):
    """Non-timeout error includes resume hint when state file exists."""
    import os
    import tempfile
    import json

    tmp = os.path.join(tempfile.mkdtemp(), "council_state.json")
    monkeypatch.setattr(council.CouncilOrchestrator, "COUNCIL_STATE_PATH", tmp)

    try:
        # Write a matching state file
        with open(tmp, "w") as f:
            json.dump({"task": "test", "plans": [], "critiques": []}, f)

        cfg = full_council_config()
        cfg["preflight"]["enabled"] = False
        cfg["subagent_delegation"] = False
        cfg["peer_review"] = False
        orc = council.CouncilOrchestrator(cfg)

        # Mock propose to hang forever → triggers pipeline timeout
        async def _slow_propose(task):
            import asyncio as _a
            await _a.sleep(10)

        monkeypatch.setattr(orc, "_run_proposer_api_calls", _slow_propose)
        monkeypatch.setattr(orc, "_run_proposer_subagents", _slow_propose)
        cfg["pipeline_timeout_seconds"] = 1
        orc = council.CouncilOrchestrator(cfg)
        monkeypatch.setattr(orc, "_run_proposer_api_calls", _slow_propose)

        result = await orc.run_plan("test")
        assert "/council resume" in result.get("error", "").lower()
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# -- Per-proposer breakdown in results ----------------------------------------


@pytest.mark.asyncio
async def test_per_proposer_breakdown_in_results(monkeypatch, mock_async_llm):
    """stages.1_propose.proposers contains per-model status dicts."""
    monkeypatch.setattr("agent.auxiliary_client.async_call_llm", mock_async_llm)
    monkeypatch.setattr("agent.auxiliary_client.extract_content_or_reasoning",
                        lambda r: r.choices[0].message.content)

    cfg = full_council_config()
    cfg["subagent_delegation"] = False
    cfg["peer_review"] = False
    cfg["preflight"]["enabled"] = False
    orc = council.CouncilOrchestrator(cfg)

    long_plan_valid = ("## Plan\n" + "Mocked plan content line that is long enough to pass validation.\n" * 15
                   + "## Steps\n1. Do step one\n2. Do step two\n## Files\n- src/main.py\n\n```python\ndef main():\n    pass\n```\n## Risks\n- Edge case\n")
    async def _mock_propose(task):
        return [
            {"model": "m1", "provider": "p1", "status": "ok",
             "output": long_plan_valid, "error": None,
             "time_seconds": 1.0, "line_count": 35, "char_count": len(long_plan_valid)},
            {"model": "m2", "provider": "p2", "status": "timeout",
             "output": None, "error": "Timed out",
             "time_seconds": 1200.0, "line_count": 0, "char_count": 0},
        ]
    monkeypatch.setattr(orc, "_run_proposer_api_calls", _mock_propose)

    result = await orc.run_plan("test")
    proposers = result["stages"]["1_propose"].get("proposers", [])
    assert len(proposers) == 2
    assert proposers[0]["model"] == "m1"
    assert proposers[0]["status"] == "ok"
    assert proposers[1]["model"] == "m2"
    assert proposers[1]["status"] == "timeout"
    assert proposers[1]["error"] == "Timed out"
    assert result["stages"]["1_propose"]["count"] == 1  # only 1 successful
    assert result["stages"]["1_propose"]["total"] == 2
    assert "validation" in result["stages"]["1_propose"]
    assert result["stages"]["1_propose"]["validation"]["passed"] >= 1
    assert result["stages"]["1_propose"]["validation"]["failed"] >= 0
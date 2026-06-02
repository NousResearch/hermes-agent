"""
Council Module — Multi-Model Deliberation Orchestrator

Implements a 3-stage council pipeline based on Karpathy's LLM Council
architecture with anonymized peer review, adapted for Hermes with
subagent delegation for deep proposer analysis.

Architecture:
  Stage 1 — Propose: N proposer models generate independent plans.
              In subagent mode, each proposer runs as a Hermes
              delegate_task(role='orchestrator') and can further delegate
              subtasks to cheaper models (deepseek-v4-flash, gemma4:31b).
  Stage 2 — Critique: An anonymized reviewer evaluates all plans for
              correctness, completeness, and efficiency.
  Stage 3 — Chairman: Synthesizes the best approach into a structured
              final plan with file manifest, risk register, and vote matrix.

Config (council.* in config.yaml):
  enabled: bool
  subagent_delegation: bool   — flat API calls vs full subagent delegation
  proposers: list[{provider, model}]
  subagents: {fast: {provider, model}, lightweight: {provider, model}}
  critic: {provider, model}
  chairman: {provider, model}
  peer_review: bool
  anonymize_reviews: bool
  max_concurrent_calls: int   — parallel proposer limit; 0 = derive from delegation
  preflight.check_interval_hours: int — hours before re-checking (0 = always check)

Usage:
  orc = CouncilOrchestrator(config, parent_agent)
  result = await orc.run_plan("build auth microservice")
  # result = {"success": True, "output": "...", ...}
"""

import asyncio
import datetime
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

PROPOSER_PROMPT = """You are a senior engineer tasked with producing a detailed implementation plan.

Your goal: decompose the task into subtasks and leverage Hermes' subagent
delegation tool (delegate_task) to research, audit, or draft each subtask.

**Strategy for subagent allocation:**
- **deepseek-v4-flash** — code analysis, file reading, code generation, testing
- **gemma4:31b** — quick lookups, formatting, validation, trivial subtasks

Delegate AT LEAST 2-3 subtasks. Your plan is only as strong as the evidence
your subagents gather. After all subtask results are collected, synthesize
them into a structured plan.

Task: {task}

Output format:
```
## Plan
1. Approach — what architecture/pattern
2. Files — files to create/modify with exact paths
3. Steps — numbered, ordered
4. Risks — edge cases, failure modes, security concerns
5. Testing strategy
```

Be specific. Use file paths. Include code snippets where helpful.

**Lightweight self-critique (REQUIRED):** After your plan, add exactly 2 lines:
- `✅ Evidence check: [confirm each file/path referenced was investigated by your subagents]`
- `✅ Backed by: [list which subagent results support each claim]`
If any reference was NOT verified by a subagent, flag it as `⚠️ Unverified: [...]`"""

PROPOSER_PROMPT_FLAT = """You are a senior engineer designing an implementation plan.
Given the task below, produce a detailed plan with:
1. Approach — what architecture/pattern to use
2. Key files — what files to create or modify
3. Steps — numbered, ordered
4. Risks — edge cases, failure modes, security concerns
5. Testing strategy

Task: {task}

Output in markdown. Be specific. Use file paths.

**Lightweight self-critique (REQUIRED):** After your plan, add exactly 2 lines:
- `✅ Evidence check: [confirm each claim is backed by investigation]`
- `✅ Backed by: [list which files/evidence support each claim]`
If any claim is unverified, flag as `⚠️ Unverified: [...]`"""

CRITIC_PROMPT = """You are a technical reviewer. Below are {count} anonymized implementation plans
for the same task. Evaluate each critically.

{formatted_proposals}

Rate each plan on:
- **Correctness (1-10):** Does the approach solve the problem correctly?
- **Completeness (1-10):** Are edge cases, error handling, and security addressed?
- **Efficiency (1-10):** Is this the simplest correct solution?

Then answer:
1. What does each plan get RIGHT that others miss?
2. What does each plan miss that others catch?
3. Which plan is strongest overall, and why?
4. What is the single biggest gap across ALL plans?

Format your response as:
```
## Critique Plan X
Correctness: N/10  Completeness: N/10  Efficiency: N/10
Strengths: ...
Weaknesses: ...

## Critique Plan Y
...
```

Be specific. Reference exact design choices, not generalities."""

CHAIRMAN_PROMPT = """You are a technical lead synthesizing multiple proposals and reviews
into a single authoritative implementation plan.

Task: {task}

## Proposals
{formatted_proposals}

## Reviews
{formatted_critiques}

Produce a unified plan with the following structure:

```
## Council Plan: [Task Title]

### Consensus Score: N/10

### Selected Approach
[Best approach or hybrid — with justification]

### File Manifest
| File | Action | Rationale | Proposed By |
|------|--------|-----------|-------------|

### Execution Steps
1. [Step]
2. [Step]

### Risk Register
| Risk | Likelihood | Mitigation | Flagged By |
|------|:----------:|------------|------------|

### Disagreements Resolved
- [Disagreement] → [Resolution]

### Proposal Breakdown
| Proposer | Approach Summary | Correctness | Completeness | Efficiency | Avg |
|----------|-----------------|:-----------:|:------------:|:----------:|:---:|
```

Fill in all sections. Be specific. Use exact file paths and code references."""


class CouncilOrchestrator:
    """Orchestrates a multi-model council deliberation.

    Two modes:
    - MODE_FLAT — each proposer is a single API call (fast, ~40s)
    - MODE_SUBAGENT — each proposer runs as a delegate_task subagent
      with role='orchestrator', can further delegate subtasks (~2.5 min)
    """

    MODE_FLAT = "flat"
    MODE_SUBAGENT = "subagent"

    def __init__(self, config: dict, parent_agent: Any = None,
                 delegation_config: dict | None = None):
        """Initialize from council config section.

        Args:
            config: The council.* dict from config.yaml (or DEFAULT_CONFIG fallback)
            parent_agent: Hermes AIAgent instance for delegate_task callbacks
            delegation_config: The delegation.* section for concurrency derivation.
        """
        self.enabled = config.get("enabled", True)
        self.mode = (
            self.MODE_SUBAGENT if config.get("subagent_delegation", False)
            else self.MODE_FLAT
        )
        self.proposers = config.get("proposers", [])
        self.subagents = config.get("subagents", {})
        self.critic = config.get("critic", {})
        self.chairman = config.get("chairman", {})
        self.peer_review = config.get("peer_review", True)
        self.anonymize = config.get("anonymize_reviews", True)
        self.parent_agent = parent_agent
        self.delegation_config = delegation_config or {}

        # Preflight config
        _pref = config.get("preflight", {})
        self.pref = {
            "enabled": _pref.get("enabled", True),
            "timeout_seconds": _pref.get("timeout_seconds", 10),
            "min_proposers": _pref.get("min_proposers", 2),
            "check_interval_hours": _pref.get("check_interval_hours", 24),
        }

        # Pipeline timeout
        self.pipeline_timeout = config.get("pipeline_timeout_seconds", 2400)

        # Preflight cache (fresh per session — survives only one /council lifespan)
        self._preflight_cache: dict | None = None
        self._preflight_cache_time: datetime | None = None

        # Parallel concurrency
        self.max_concurrent_calls = config.get("max_concurrent_calls", 0)
        self._concurrency_limit = self._resolve_max_concurrent()

        # Content validation thresholds
        self.min_plan_lines = config.get("min_plan_lines", 15)
        self.min_plan_chars = config.get("min_plan_chars", 500)

        # Retry tolerance
        self.max_retries = config.get("max_retries", 2)

        if not self.proposers:
            logger.warning("Council: no proposers configured")
        if not self.critic:
            logger.warning("Council: no critic configured")
        if not self.chairman:
            logger.warning("Council: no chairman configured")

    # ── Concurrency resolution ─────────────────────────────────────────────

    def _resolve_max_concurrent(self) -> int:
        """Derive effective concurrency limit from config.

        Priority:
          1. council.max_concurrent_calls (explicit override)
          2. delegation.max_concurrent_children (central config)
          3. 3 (hard fallback)
        Capped to the number of proposers.
        """
        if self.max_concurrent_calls > 0:
            limit = self.max_concurrent_calls
        else:
            limit = self.delegation_config.get("max_concurrent_children", 3)
        return min(limit, len(self.proposers)) if self.proposers else 1

    # ── Dynamic timeout estimation ───────────────────────────────────

    @staticmethod
    def _estimate_input_chars(texts: list[str]) -> int:
        """Count total characters across all input texts."""
        return sum(len(t or "") for t in texts)

    def _stage_default_timeout(self, stage: str) -> int:
        """Conservative floors per stage (even for tiny inputs)."""
        return {
            "proposer_flat": 1200,
            "proposer_subagent": 1200,
            "critique": 1200,
            "chairman": 300,
        }.get(stage, 120)

    def _stage_timeout(self, stage: str, input_texts: list[str] | None = None) -> int:
        """Dynamic timeout per stage based on input size.

        Larger inputs -> more time. ~4 chars/token, ~50 tok/s avg inference.
        Timeout should almost never fire — it's sized generously from actual payload.
        """
        if not input_texts:
            return self._stage_default_timeout(stage)
        input_chars = self._estimate_input_chars(input_texts)
        input_tokens = input_chars / 4
        estimated_output_tokens = input_tokens * 0.3  # output ~30% of input
        inference_time_s = estimated_output_tokens / 50  # 50 tok/s avg
        network_overhead = 30
        return max(
            self._stage_default_timeout(stage),
            int(inference_time_s * 1.5 + network_overhead),
        )

    # ── Retry wrapper (all failures retried — timeouts are transient on cloud) ───

    async def _run_with_retry(
        self, name: str, fn, max_retries: int = 1,
        input_texts: list[str] | None = None,
        stage_key: str = "proposer_flat",
    ) -> str:
        """Run an async call with retry on ALL failures including timeout.

        Timeouts ARE retried — cloud providers routinely drop the first call
        during peak load but succeed on the second. Dynamic timeout sizing
        means a timeout indicates a slow model, but the next attempt may
        return within budget.

        Also catches empty results (RuntimeError("Empty content returned"))
        and retries them — empty responses are often transient rate-limit
        or load-shedding from the provider.

        Retry delay scales: 5s, 15s, 30s (geometric 5 × attempt²).
        After all retries exhausted, raises the last error.
        """
        timeout = self._stage_timeout(stage_key, input_texts)
        last_error = None
        for attempt in range(1 + max_retries):
            try:
                return await asyncio.wait_for(fn(timeout), timeout=timeout)
            except asyncio.TimeoutError as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        "Council: %s attempt %d/%d timed out after %ds (retrying)",
                        name, attempt + 1, 1 + max_retries, timeout,
                    )
                    await asyncio.sleep(5 * (attempt + 1) ** 2)  # 5s, 20s, 45s
                else:
                    logger.warning(
                        "Council: %s exhausted %d retries — last error: timeout after %ds",
                        name, max_retries, timeout,
                    )
                    raise  # re-raise TimeoutError on last attempt
            except RuntimeError as e:
                emsg = str(e)
                # Empty content is retryable (transient provider blip)
                if "Empty content" in emsg and attempt < max_retries:
                    last_error = e
                    logger.warning(
                        "Council: %s attempt %d/%d returned empty content (retrying)",
                        name, attempt + 1, 1 + max_retries,
                    )
                    await asyncio.sleep(5 * (attempt + 1) ** 2)
                else:
                    raise
            except (ConnectionError, OSError) as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        "Council: %s attempt %d/%d failed transiently: %s (retrying)",
                        name, attempt + 1, 1 + max_retries, e,
                    )
                    await asyncio.sleep(5 * (attempt + 1))
                else:
                    raise
            except Exception as e:
                # Generic exception retry once — could be rate limit, provider blip
                if attempt < max_retries:
                    logger.warning(
                        "Council: %s attempt %d/%d failed: %s (retrying)",
                        name, attempt + 1, 1 + max_retries, e,
                    )
                    last_error = e
                    await asyncio.sleep(5 * (attempt + 1))
                else:
                    raise
        raise RuntimeError(f"{name} failed after {max_retries + 1} attempts") from last_error

    # ── State persistence ────────────────────────────────────────────

    COUNCIL_STATE_PATH = os.path.join(get_hermes_home(), "council_state.json")
    STATE_SCHEMA_VERSION = 1

    def save_state(
        self, task: str, plans: list, critiques: list,
        stages: dict, models_used: dict,
    ) -> str:
        """Save council state to JSON file for resume. Returns path."""
        # Strip output from per-proposer stage entries (already in plans[])
        clean_stages = {}
        for k, v in stages.items():
            if isinstance(v, dict) and v.get("time_seconds"):
                d = dict(v)
                if "proposers" in d:
                    d["proposers"] = [
                        {kk: vv for kk, vv in p.items() if kk != "output"}
                        for p in d["proposers"]
                    ]
                clean_stages[k] = d

        state = {
            "schema_version": self.STATE_SCHEMA_VERSION,
            "task": task,
            "plans": [self._extract_text(p) for p in plans if p is not None],
            "critiques": [self._extract_text(c) for c in critiques],
            "stages": clean_stages,
            "models_used": models_used,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        d = os.path.dirname(self.COUNCIL_STATE_PATH)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
        with open(self.COUNCIL_STATE_PATH, "w") as f:
            json.dump(state, f, indent=2)
        logger.info("Council: state saved to %s", self.COUNCIL_STATE_PATH)
        return self.COUNCIL_STATE_PATH

    @classmethod
    def load_state(cls, path: str | None = None) -> dict | None:
        """Load saved council state from JSON."""
        p = path or cls.COUNCIL_STATE_PATH
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
        return None

    async def run_resume(self, state_path: str = "") -> dict:
        """Resume council from saved state (re-run chairman with saved plans+critiques)."""
        state = self.load_state(state_path or None)
        if not state:
            return {"success": False, "error": f"No council state found at {state_path or self.COUNCIL_STATE_PATH}"}

        plans_text = state.get("plans", [])
        critiques = state.get("critiques", [])
        task = state.get("task", "")

        if not task:
            return {"success": False, "error": "Saved state has no task description"}

        logger.info("Council: resuming — re-running chairman for task: %.100s", task)
        try:
            final_output = await self._run_chairman(task, plans_text, critiques)
            return {
                "success": True,
                "output": final_output,
                "task": task,
                "models_used": state.get("models_used", {}),
                "processing_time": 0.0,
                "stages": state.get("stages", {}),
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Chairman synthesis failed on resume: {e}",
                "task": task,
            }

    def _cache_age_hours(self) -> float:
        """Hours since last preflight cache write. Returns inf if never cached."""
        if self._preflight_cache_time is None:
            return float("inf")
        delta = datetime.datetime.now() - self._preflight_cache_time
        return delta.total_seconds() / 3600

    # ── Preflight ───────────────────────────────────────────────────────

    async def _preflight_check(self) -> dict:
        """Validate all council providers+models before running.

        Makes a lightweight test call per unique (provider, model) pair.
        Critic/chairman failures are fatal. Proposer failures are fatal
        only if fewer than min_proposers succeed.

        Returns:
            {"passed": bool, "checks": list, "summary": str, "can_proceed": bool}
        """
        import time
        from agent.auxiliary_client import async_call_llm, extract_content_or_reasoning

        pairs = []
        for i, p in enumerate(self.proposers):
            pairs.append(("proposer", i, p.get("provider"), p.get("model")))
        if self.critic:
            pairs.append(("critic", 0, self.critic.get("provider"), self.critic.get("model")))
        if self.chairman:
            pairs.append(("chairman", 0, self.chairman.get("provider"), self.chairman.get("model")))

        timeout = self.pref["timeout_seconds"]

        async def _check(role, idx, provider, model):
            start = time.time()
            try:
                resp = await async_call_llm(
                    provider=provider,
                    model=model,
                    messages=[{"role": "user", "content": "OK"}],
                    max_tokens=5,
                    timeout=timeout,
                    temperature=0.1,
                )
                content = extract_content_or_reasoning(resp)
                ok = bool(content)
                return {
                    "role": role, "index": idx,
                    "provider": provider, "model": model,
                    "status": "ok" if ok else "empty",
                    "latency_ms": int((time.time() - start) * 1000),
                }
            except Exception as e:
                return {
                    "role": role, "index": idx,
                    "provider": provider, "model": model,
                    "status": "fail",
                    "error": str(e)[:200],
                    "latency_ms": int((time.time() - start) * 1000),
                }

        checks = await asyncio.gather(*[_check(*p) for p in pairs])
        passed = sum(1 for c in checks if c["status"] == "ok")
        total = len(checks)
        min_ok = self.pref.get("min_proposers", 2)

        critical_fails = [
            c for c in checks
            if c["role"] in ("critic", "chairman") and c["status"] != "ok"
        ]
        proposer_ok = sum(
            1 for c in checks if c["role"] == "proposer" and c["status"] == "ok"
        )

        can_proceed = proposer_ok >= min_ok and not critical_fails

        return {
            "passed": passed == total,
            "checks": checks,
            "summary": f"{passed}/{total} models reachable",
            "can_proceed": can_proceed,
        }

    # ── Public entry point ────────────────────────────────────────────────

    async def run_plan(
        self, task: str,
        progress_callback: Optional[callable] = None,
    ) -> dict:
        """Execute full 3-stage council pipeline.

        Args:
            task: The task description to plan
            progress_callback: Optional callable(stage: str, msg: str) called
                after each stage completes for CLI/gateway progress display.

        Returns:
            dict with keys: success, output (markdown string),
            models_used, processing_time, stages (detailed per-stage results)
        """
        start_time = datetime.datetime.now()
        logger.info("Council: starting plan for task: %.100s", task)
        logger.info("Council: mode=%s proposers=%d pipeline_timeout=%d",
                     self.mode, len(self.proposers), self.pipeline_timeout)
        for i, p in enumerate(self.proposers):
            logger.info("Council: proposer[%d] provider=%s model=%s",
                         i, p.get("provider"), p.get("model"))
        if self.critic:
            logger.info("Council: critic provider=%s model=%s",
                         self.critic.get("provider"), self.critic.get("model"))
        if self.chairman:
            logger.info("Council: chairman provider=%s model=%s",
                         self.chairman.get("provider"), self.chairman.get("model"))

        result = {
            "success": False,
            "output": "",
            "models_used": {},
            "processing_time": 0.0,
            "stages": {},
            "error": None,
        }

        async def _pipeline():
            """Inner coroutine that runs all 3 stages.

            Wrapped by asyncio.wait_for for the pipeline-level timeout.
            Stage timeouts are sized dynamically from input — they should
            almost never fire. The pipeline timeout is a last-resort guard.
            """
            nonlocal result
            # ── Preflight ──────────────────────────────────────────
            if self.pref["enabled"]:
                age_hours = self._cache_age_hours()
                check_interval = self.pref.get("check_interval_hours", 24)
                if (self._preflight_cache is not None
                        and check_interval > 0
                        and age_hours < check_interval):
                    logger.info(
                        "Council: using cached preflight (age=%.1f h, interval=%d h)",
                        age_hours, check_interval,
                    )
                    preflight = self._preflight_cache
                else:
                    if self._preflight_cache is None:
                        logger.info("Council: preflight check running (no cache)...")
                    else:
                        logger.info(
                            "Council: preflight cache expired (age=%.1f h, interval=%d h), re-checking...",
                            age_hours, check_interval,
                        )
                    preflight = await self._preflight_check()
                    if preflight["can_proceed"]:
                        self._preflight_cache = preflight
                        self._preflight_cache_time = datetime.datetime.now()
                result["preflight"] = preflight
                if not preflight["can_proceed"]:
                    fail_msgs = " | ".join(
                        f"{c['role']} {c['model'] or '(no model)'}: {c.get('error', c['status'])}"
                        for c in preflight["checks"] if c["status"] != "ok"
                    )
                    raise RuntimeError(
                        f"Preflight failed: {preflight['summary']}. "
                        f"Minimum {self.pref['min_proposers']} proposers required; "
                        f"critic/chairman failures are fatal. Details: {fail_msgs}"
                    )
                logger.info("Council: preflight passed — %s", preflight["summary"])
                if progress_callback:
                    progress_callback("preflight", preflight["summary"])

            # ── Stage 1: Propose ──────────────────────────────────────
            logger.info("Council: Stage 1 — Proposing (mode=%s)", self.mode)
            stage1_start = datetime.datetime.now()

            if self.mode == self.MODE_SUBAGENT:
                proposer_results = await self._run_proposer_subagents(task)
            else:
                proposer_results = await self._run_proposer_api_calls(task)

            stage1_time = (datetime.datetime.now() - stage1_start).total_seconds()

            # ── Content validation gate ────────────────────────────
            proposer_results = self._validate_proposer_outputs(proposer_results)

            # Store per-proposer breakdown
            successful_proposers = [p for p in proposer_results if p.get("status") == "ok"]
            failed_proposers = [p for p in proposer_results if p.get("status") != "ok"]

            result["stages"]["1_propose"] = {
                "count": len(successful_proposers),
                "total": len(proposer_results),
                "time_seconds": stage1_time,
                "proposers": proposer_results,
                "validation": {
                    "passed": len(successful_proposers),
                    "failed": len(failed_proposers),
                },
            }

            fail_summary = "; ".join(
                f"{p.get('model','?')}: {p.get('error','unknown')}"
                for p in failed_proposers
            ) if failed_proposers else ""
            ok_count = len(successful_proposers)
            total_count = len(proposer_results)

            logger.info(
                "Council: Stage 1 done — %d/%d plans succeeded in %.1fs. Failures: %s",
                ok_count, total_count, stage1_time, fail_summary or "none",
            )
            if progress_callback:
                msg = f"{ok_count}/{total_count} plans in {stage1_time:.1f}s"
                if fail_summary:
                    msg += f" — failures: {fail_summary}"
                progress_callback("propose", msg)

            if not successful_proposers:
                fail_details = []
                for p in proposer_results:
                    _m = p.get("model", "?")
                    _err = p.get("error", "unknown")
                    fail_details.append(f"{_m}: {_err}")
                raise RuntimeError(
                    f"All proposers failed — cannot proceed without plans. "
                    f"Proposers: {', '.join(fail_details)}"
                )

            # Extract text outputs for downstream stages (critique, chairman, state)
            plans = [p.get("output") for p in successful_proposers]

            # ── Stage 2: Critique ─────────────────────────────────────
            critiques = []
            if self.peer_review:
                stage2_start = datetime.datetime.now()
                # Single plan → skip critique (no peer comparison possible)
                if len(plans) == 1:
                    logger.info("Council: single proposer — skipping critique (no peer comparison)")
                    critique_result = {"critiques": [], "error": None, "skipped": True,
                                       "skipped_reason": "Single proposer — no peer comparison needed"}
                else:
                    logger.info("Council: Stage 2 — Critiquing (anonymized=%s)", self.anonymize)
                    critique_result = await self._run_anonymized_critique(plans, task=task)
                stage2_time = (datetime.datetime.now() - stage2_start).total_seconds()
                critiques = critique_result.get("critiques", [])
                result["stages"]["2_critique"] = {
                    "count": len(critiques),
                    "time_seconds": stage2_time,
                    "error": critique_result.get("error"),
                    "skipped": critique_result.get("skipped", False),
                    "skipped_reason": critique_result.get("skipped_reason"),
                }
                logger.info(
                    "Council: Stage 2 done — %d critiques in %.1fs%s%s",
                    len(critiques), stage2_time,
                    f" — error: {critique_result.get('error')}" if critique_result.get("error") else "",
                    f" — skipped: {critique_result.get('skipped_reason')}" if critique_result.get("skipped") else "",
                )
                if progress_callback:
                    msg = f"{len(critiques)} critiques in {stage2_time:.1f}s"
                    if critique_result.get("error"):
                        msg += f" — error: {critique_result['error']}"
                    elif critique_result.get("skipped"):
                        msg += f" — skipped"
                    progress_callback("critique", msg)
            else:
                result["stages"]["2_critique"] = {"count": 0, "time_seconds": 0, "skipped": True, "skipped_reason": "peer_review disabled"}

            models_used = {
                "proposers": [r.get("model") for r in proposer_results if r.get("status") == "ok"],
                "proposer_attempts": [
                    {"model": r.get("model"), "provider": r.get("provider"), "status": r.get("status")}
                    for r in proposer_results
                ],
                "critic": self.critic.get("model", ""),
                "chairman": self.chairman.get("model", ""),
            }
            result["models_used"] = models_used

            # ── Save state for resume (before chairman) ─────────────
            self.save_state(task, plans, critiques, result["stages"], models_used)

            # ── Stage 3: Chairman synthesis ───────────────────────────
            logger.info("Council: Stage 3 — Synthesizing")
            stage3_start = datetime.datetime.now()
            final_output = await self._run_chairman(task, plans, critiques)
            stage3_time = (datetime.datetime.now() - stage3_start).total_seconds()
            result["stages"]["3_chairman"] = {"time_seconds": stage3_time}
            logger.info("Council: Stage 3 done in %.1fs", stage3_time)
            if progress_callback:
                progress_callback("chairman", f"synthesis in {stage3_time:.1f}s")

            # ── Assemble result ───────────────────────────────────────
            total_time = (datetime.datetime.now() - start_time).total_seconds()

            result["success"] = True
            result["output"] = final_output
            result["processing_time"] = round(total_time, 1)

            logger.info(
                "Council: complete — %d plans, %d critiques, %.1fs total",
                len(plans), len(critiques), total_time,
            )

            # Clean up state file on successful completion
            if os.path.exists(self.COUNCIL_STATE_PATH):
                try:
                    os.remove(self.COUNCIL_STATE_PATH)
                except OSError:
                    pass

        # ── Execute pipeline with timeout ─────────────────────────────
        try:
            await asyncio.wait_for(_pipeline(), timeout=self.pipeline_timeout)
        except asyncio.TimeoutError:
            elapsed = (datetime.datetime.now() - start_time).total_seconds()
            stage_info = []
            s1 = result["stages"].get("1_propose", {})
            if s1.get("count", 0) > 0:
                stage_info.append(f"Propose ({s1['count']} plans, {s1['time_seconds']:.0f}s)")
            s2 = result["stages"].get("2_critique", {})
            if s2.get("count", 0) > 0:
                stage_info.append(f"Critique ({s2['count']} reviews, {s2['time_seconds']:.0f}s)")

            # Build partial output from completed stages
            if result.get("models_used"):
                partial = {}
                if s1.get("count", 0) > 0:
                    state = self.load_state()
                    if state:
                        partial = {
                            "state_file": self.COUNCIL_STATE_PATH,
                            "task": task,
                            "plans": state.get("plans", []),
                            "critiques": state.get("critiques", []),
                        }
                result["partial_output"] = partial

            resume_hint = ""
            if os.path.exists(self.COUNCIL_STATE_PATH):
                state = self.load_state()
                if state and state.get("task") == task:
                    resume_hint = (
                        f" Plans and critiques saved to {self.COUNCIL_STATE_PATH}. "
                        f"Use /council resume to retry chairman synthesis."
                    )

            error_msg = (
                f"Council pipeline timed out after {elapsed:.0f}s "
                f"(limit: {self.pipeline_timeout}s). "
                f"Completed stages: {', '.join(stage_info) or 'none'}.{resume_hint}"
            )
            logger.error("Council: %s", error_msg)
            result["error"] = error_msg
            result["processing_time"] = round(elapsed, 1)
            result["success"] = False

        except Exception as e:
            elapsed = (datetime.datetime.now() - start_time).total_seconds()
            stage_info = []
            s1 = result["stages"].get("1_propose", {})
            if s1.get("count", 0) > 0:
                stage_info.append(f"Propose ({s1['count']} plans, {s1['time_seconds']:.0f}s)")
            s2 = result["stages"].get("2_critique", {})
            if s2.get("count", 0) > 0:
                stage_info.append(f"Critique ({s2['count']} reviews, {s2['time_seconds']:.0f}s)")

            resume_hint = ""
            if os.path.exists(self.COUNCIL_STATE_PATH):
                state = self.load_state()
                if state and state.get("task") == task:
                    resume_hint = (
                        f" Plans and critiques saved to {self.COUNCIL_STATE_PATH}. "
                        f"Use /council resume to retry chairman synthesis."
                    )

            error_msg = (
                f"Council failed after {elapsed:.0f}s. "
                f"Completed stages: {', '.join(stage_info) or 'none'}. "
                f"Error: {e}{resume_hint}"
            )
            logger.error(error_msg, exc_info=True)
            result["error"] = error_msg
            result["processing_time"] = round(elapsed, 1)

        return result

    # ── Stage 1: Propose ────────────────────────────────────────────────

    async def _run_proposer_api_calls(self, task: str) -> list[dict]:
        """Flat mode — each proposer is a single API call via auxiliary client.

        Concurrency limited by self._concurrency_limit (asyncio.Semaphore).
        Uses dynamic timeout + retry wrapper for all failure types.
        Returns list of dicts with per-proposer metadata including status and error.
        """
        from agent.auxiliary_client import async_call_llm, extract_content_or_reasoning

        sem = asyncio.Semaphore(self._concurrency_limit)
        task_text = [task]  # wrap for _stage_timeout estimation

        async def _call_one(provider: str, model: str) -> dict:
            async with sem:
                import time
                start = time.time()
                try:
                    prompt = PROPOSER_PROMPT_FLAT.format(task=task)

                    async def _do_call(timeout_s: int) -> str:
                        resp = await async_call_llm(
                            provider=provider,
                            model=model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.6,
                            max_tokens=16000,
                            timeout=timeout_s,
                        )
                        content = extract_content_or_reasoning(resp)
                        if not content:
                            raise RuntimeError("Empty content returned")
                        return content

                    try:
                        output = await self._run_with_retry(
                            name=f"proposer {model}",
                            fn=_do_call,
                            input_texts=task_text,
                            stage_key="proposer_flat",
                            max_retries=self.max_retries,
                        )
                        elapsed = time.time() - start
                        lines = output.count('\n') + 1
                        return {
                            "model": model,
                            "provider": provider,
                            "status": "ok",
                            "output": output,
                            "error": None,
                            "time_seconds": round(elapsed, 1),
                            "line_count": lines,
                            "char_count": len(output),
                        }
                    except asyncio.TimeoutError:
                        elapsed = time.time() - start
                        logger.warning("Council: %s timed out after %.1fs", model, elapsed)
                        return {
                            "model": model, "provider": provider,
                            "status": "timeout", "output": None,
                            "error": f"Timed out after {elapsed:.0f}s",
                            "time_seconds": round(elapsed, 1),
                            "line_count": 0, "char_count": 0,
                        }
                    except Exception as e:
                        elapsed = time.time() - start
                        logger.warning("Council: %s failed: %s", model, e)
                        emsg = str(e)
                        status = "empty" if "Empty content" in emsg else "error"
                        return {
                            "model": model, "provider": provider,
                            "status": status, "output": None,
                            "error": emsg[:200],
                            "time_seconds": round(elapsed, 1),
                            "line_count": 0, "char_count": 0,
                        }
                except Exception as e:
                    elapsed = time.time() - start
                    logger.warning("Council: %s failed unexpectedly: %s", model, e)
                    return {
                        "model": model, "provider": provider,
                        "status": "error", "output": None,
                        "error": str(e)[:200],
                        "time_seconds": round(elapsed, 1),
                        "line_count": 0, "char_count": 0,
                    }

        tasks = [
            _call_one(p["provider"], p["model"])
            for p in self.proposers
        ]
        results = await asyncio.gather(*tasks)
        return results

    def _run_single_proposer(self, p: dict, task: str) -> dict:
        """Run a single proposer as a synchronous delegate_task.

        delegate_task is synchronous (not async), so we run each proposer
        sequentially. Parallelism comes from the subagent level (each
        proposer can delegate to its own subtasks concurrently).

        Returns per-proposer result dict matching flat mode format.
        """
        import time
        start = time.time()
        from tools.delegate_tool import delegate_task

        try:
            prompt = PROPOSER_PROMPT.format(task=task)
            raw = delegate_task(
                goal=prompt,
                context=(
                    f"Task: {task[:200]}\n"
                    f"Use deepseek-v4-flash:cloud for code subtasks, "
                    f"gemma4:31b:cloud for quick lookups and formatting.\n"
                    f"You MUST delegate at least 2 subtasks."
                ),
                toolsets=["terminal", "file", "delegation", "web"],
                provider=p["provider"],
                model=p["model"],
                role="orchestrator",
                parent_agent=self.parent_agent,
            )
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            elapsed = time.time() - start
            output_str = self._extract_text(parsed)
            lines = output_str.count('\n') + 1
            return {
                "model": p.get("model"),
                "provider": p.get("provider"),
                "status": "ok",
                "output": output_str,
                "error": None,
                "time_seconds": round(elapsed, 1),
                "line_count": lines,
                "char_count": len(output_str),
            }
        except Exception as e:
            elapsed = time.time() - start
            logger.warning("Council: %s subagent failed: %s", p.get("model"), e)
            return {
                "model": p.get("model"), "provider": p.get("provider"),
                "status": "error", "output": None,
                "error": str(e)[:200],
                "time_seconds": round(elapsed, 1),
                "line_count": 0, "char_count": 0,
            }

    async def _run_proposer_subagents(self, task: str) -> list[dict]:
        """Subagent mode — each proposer runs as a Hermes delegate_task.

        Each proposer receives the PROPOSER_PROMPT which instructs them
        to decompose the task and delegate subtasks. The role='orchestrator'
        ensures they retain the delegate_task tool.
        delegate_task is synchronous — proposers run via run_in_executor
        to avoid blocking the async event loop.

        Concurrency limited by self._concurrency_limit (asyncio.Semaphore).

        Returns list of result dicts (one per successful proposer).
        """
        sem = asyncio.Semaphore(self._concurrency_limit)
        loop = asyncio.get_running_loop()

        async def _run_one(p):
            async with sem:
                timeout = self._stage_timeout("proposer_subagent", [task])
                try:
                    return await asyncio.wait_for(
                        loop.run_in_executor(None, self._run_single_proposer, p, task),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    logger.warning("Council: subagent %s timed out after %ds (wait_for)",
                                   p.get("model"), timeout)
                    return {
                        "model": p.get("model"), "provider": p.get("provider"),
                        "status": "timeout", "output": None,
                        "error": f"Subagent wait_for timed out after {timeout}s",
                        "time_seconds": round(timeout, 1),
                        "line_count": 0, "char_count": 0,
                    }

        results = await asyncio.gather(*[_run_one(p) for p in self.proposers])
        return results

    # ── Stage 2: Critique ───────────────────────────────────────────────

    async def _run_anonymized_critique(self, plans: list, task: str = "") -> dict:
        """Run anonymized peer review on all plans in a single call.

        Each plan is labeled as "Plan A", "Plan B" etc. The critic evaluates
        ALL plans in one call (the CRITIC_PROMPT template is designed for
        multi-plan evaluation). This avoids N× parallel copies of the same
        payload that cause provider timeouts.

        Returns dict with: critiques (list[str]), error (str|None), skipped (bool).
        """
        from agent.auxiliary_client import async_call_llm, extract_content_or_reasoning

        if not plans:
            return {"critiques": [], "error": None, "skipped": True, "skipped_reason": "No plans to critique"}

        # Anonymize: assign letter labels
        if self.anonymize:
            labels = {chr(65 + i): p for i, p in enumerate(plans)}
        else:
            labels = {f"Proposer {i+1}": p for i, p in enumerate(plans)}

        plan_texts = [self._extract_text(p) for p in plans]
        formatted = "\n\n".join(
            f"### {label}\n{text}"
            for label, text in labels.items()
        )

        prompt = CRITIC_PROMPT.format(
            count=len(plans),
            formatted_proposals=formatted,
        )

        try:
            async def _do_critique(timeout_s: int) -> str:
                resp = await async_call_llm(
                    provider=self.critic.get("provider"),
                    model=self.critic.get("model"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4,
                    max_tokens=8000,
                    timeout=timeout_s,
                )
                content = extract_content_or_reasoning(resp)
                if content:
                    return content
                logger.warning("Council: critic returned empty content")
                raise RuntimeError("Empty content returned")

            result = await self._run_with_retry(
                name="critique",
                fn=_do_critique,
                input_texts=[task] + plan_texts,
                stage_key="critique",
                max_retries=self.max_retries,
            )
            return {"critiques": [result], "error": None, "skipped": False} if result else \
                   {"critiques": [], "error": "Critic returned empty content after retries", "skipped": False}
        except Exception as e:
            logger.error(
                "Council: critic %s/%s failed: %s",
                self.critic.get("provider"), self.critic.get("model"), e,
                exc_info=True,
            )
            return {"critiques": [], "error": str(e)[:200], "skipped": False}

    # ── Stage 3: Chairman Synthesis ─────────────────────────────────────

    async def _run_chairman(
        self, task: str, plans: list, critiques: list
    ) -> str:
        """Chairman synthesizes plans + critiques into structured final output.

        Returns structured markdown string.
        """
        from agent.auxiliary_client import async_call_llm, extract_content_or_reasoning

        plan_texts = [self._extract_text(p) for p in plans]
        formatted_plans = "\n\n".join(
            f"### Proposal {chr(65 + i)}\n{text}"
            for i, text in enumerate(plan_texts)
        )

        critique_texts = [self._extract_text(c) for c in critiques]
        formatted_critiques = "\n\n".join(critique_texts) if critique_texts else "(No critiques — peer review disabled)"

        prompt = CHAIRMAN_PROMPT.format(
            task=task,
            formatted_proposals=formatted_plans,
            formatted_critiques=formatted_critiques,
        )

        async def _do_chairman(timeout_s: int) -> str:
            resp = await async_call_llm(
                provider=self.chairman.get("provider"),
                model=self.chairman.get("model"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=16000,
                timeout=timeout_s,
            )
            content = extract_content_or_reasoning(resp)
            if not content:
                raise RuntimeError("Chairman returned empty output")
            return content

        return await self._run_with_retry(
            name="chairman",
            fn=_do_chairman,
            input_texts=[task] + plan_texts + critique_texts,
            stage_key="chairman",
            max_retries=self.max_retries,
        )

    # ── Content validation markers ──────────────────────────────
    REQUIRED_MARKERS = frozenset({
        "## Plan", "## Approach", "## Steps", "## Files", "## Execution",
        "```",
    })

    # ── Content validation ──────────────────────────────────────────

    def _validate_proposer_outputs(self, proposer_results: list[dict]) -> list[dict]:
        """Validate proposer outputs for real content before passing downstream.

        Checks each proposer result with status="ok":
        - line_count >= min_plan_lines
        - char_count >= min_plan_chars
        - Contains at least one structural marker (## Plan, ## Approach, etc.)

        Proposers that fail validation get status downgraded to "empty"
        with a descriptive error. Returns updated results list in-place.
        """
        for p in proposer_results:
            if p.get("status") != "ok":
                continue
            output = p.get("output") or ""
            lines = p.get("line_count", 0)
            chars = p.get("char_count", 0)

            reasons = []
            if lines < self.min_plan_lines:
                reasons.append(f"only {lines} lines (min {self.min_plan_lines})")
            if chars < self.min_plan_chars:
                reasons.append(f"only {chars} chars (min {self.min_plan_chars})")
            if not reasons and not any(marker in output for marker in CouncilOrchestrator.REQUIRED_MARKERS):
                reasons.append("no structural markers found")

            if reasons:
                p["status"] = "empty"
                p["error"] = "; ".join(reasons)
                p["output"] = None
                p["char_count"] = 0
                logger.warning(
                    "Council: %s/%s rejected by validation: %s",
                    p.get("model"), p.get("provider"), p["error"],
                )

        valid = [p for p in proposer_results if p.get("status") == "ok"]
        rejected = [p for p in proposer_results if p.get("status") == "empty"]
        if rejected:
            logger.info(
                "Council: content validation — %d/%d passed, %d rejected",
                len(valid), len(proposer_results), len(rejected),
            )
        return proposer_results

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _extract_text(plan) -> str:
        """Extract text content from a plan which may be a dict, string, or JSON str."""
        if isinstance(plan, dict):
            return plan.get("response") or plan.get("output") or plan.get("summary") or json.dumps(plan, indent=2)
        if isinstance(plan, str):
            return plan
        return str(plan)

    def get_status(self) -> dict:
        """Return current configuration status for display."""
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "proposers": len(self.proposers),
            "peer_review": self.peer_review,
            "anonymize": self.anonymize,
            "proposer_models": [p.get("model") for p in self.proposers],
            "critic_model": self.critic.get("model"),
            "chairman_model": self.chairman.get("model"),
            "preflight": self.pref,
            "last_preflight_check": (
                self._preflight_cache_time.isoformat() if self._preflight_cache_time else None
            ),
            "preflight_interval_hours": self.pref.get("check_interval_hours", 24),
            "max_concurrent_calls": self.max_concurrent_calls,
            "effective_concurrency_limit": self._concurrency_limit,
        }

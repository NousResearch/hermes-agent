"""Token-budgeted fresh tail (compression.target_ratio support for LCM).

Spec: ~/.hermes/plans/2026-07-01_lcm-target-ratio-token-budget-tail-SPEC.md

The LCM engine historically kept a FIXED message-count fresh tail
(fresh_tail_count=32) and ignored the fleet-standard compression.target_ratio
knob. These tests prove the new token-budgeted tail:

  budget = target_ratio × threshold_tokens, capped at fresh_tail_max_tokens
           and 0.9 × threshold_tokens (I-2 convergence clamp)

- config plumbing (Phase 1): compression.target_ratio sourcing, lcm.* knobs,
  env overrides, fail-safe-ON enable flag, clamp behavior
- the chokepoint helper (Phase 2): budget walk, I-3 floor, I-4 degenerate
  fallbacks, frozen-K boundary consistency (AC-10), the AST no-insert guard
- rotate floor (Phase 3 / D-7, AC-11) and the fail-open target_ratio fix (D-10)
"""
from __future__ import annotations

import ast
import inspect
import os
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from plugins.context_engine.lcm.config import (
    LCMConfig,
    _hermes_compression_float,
    _hermes_lcm_int,
    _lcm_config_bool,
)
from plugins.context_engine.lcm.engine import LCMEngine
from plugins.context_engine.lcm.tokens import count_message_tokens, count_messages_tokens


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_BUDGET_ENV_KEYS = (
    "LCM_FRESH_TAIL_TOKEN_BUDGET_ENABLED",
    "LCM_FRESH_TAIL_TOKEN_BUDGET",
    "LCM_FRESH_TAIL_MAX_TOKENS",
    "LCM_TARGET_RATIO",
    "LCM_FRESH_TAIL_COUNT",
    "LCM_CONTEXT_THRESHOLD",
)


def _clean_env(tmp_path: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in _BUDGET_ENV_KEYS}
    env["HERMES_HOME"] = str(tmp_path)
    return env


def _write_cfg(home: Path, body: str) -> None:
    (home / "config.yaml").write_text(textwrap.dedent(body), encoding="utf-8")


def _engine(tmp_path: Path, cfg: LCMConfig | None = None) -> LCMEngine:
    return LCMEngine(config=cfg or LCMConfig(), hermes_home=str(tmp_path))


def _msg(role: str, chars: int, idx: int) -> dict:
    return {"role": role, "content": f"m{idx:05d} " + ("x" * chars)}


def _mixed_corpus(n: int, small: int = 300, big: int = 1400) -> list[dict]:
    """Alternating assistant/tool-shaped rows matching live density ratio."""
    out = [{"role": "system", "content": "system prompt"}]
    for i in range(n):
        if i % 2 == 0:
            out.append(_msg("assistant", small, i))
        else:
            out.append(_msg("tool", big, i))
    return out


# List-GROWING primitives forbidden inside the frozen-K leaf loop (D-11).
# Wrapper AND primitives (pass-3 RC-1): a wrapper-only blacklist is defeated
# by calling the primitive directly.
_FROZEN_K_FORBIDDEN_CALLS = {
    "_sanitize_active_context_messages",
    "_sanitize_tool_pairs",
    "_assemble_context",
    "insert",
    "append",
    "extend",
}

# Message-list variables whose growth inside the span breaks count-arithmetic.
_FROZEN_K_LIST_VARS = {"working_messages", "pressure_messages", "messages"}


def _frozen_k_span_growth_offenders(func_src: str) -> list[str]:
    """Return names of list-growing calls inside any `while` loop of the
    given function source (the frozen-K span in _compress_lossless)."""
    tree = ast.parse(func_src)
    func = tree.body[0]
    offenders: list[str] = []
    for node in ast.walk(func):
        if isinstance(node, ast.While):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                    if sub.func.attr in _FROZEN_K_FORBIDDEN_CALLS:
                        offenders.append(sub.func.attr)
                # Augmented assignment growth: `working_messages += [...]`
                # (Greptile PR review: AugAssign is not a Call node). Integer
                # counters (`leaf_passes += 1`) are fine — flag only when the
                # target is a message list or the RHS is a list expression.
                elif isinstance(sub, ast.AugAssign) and isinstance(sub.op, ast.Add):
                    target = sub.target
                    name = (
                        target.id
                        if isinstance(target, ast.Name)
                        else getattr(target, "attr", "?")
                    )
                    rhs_is_list = isinstance(sub.value, (ast.List, ast.ListComp))
                    if name in _FROZEN_K_LIST_VARS or rhs_is_list:
                        offenders.append(f"augassign:{name}")
    return offenders


# ---------------------------------------------------------------------------
# Phase 1 — config plumbing
# ---------------------------------------------------------------------------

class TestConfigPlumbing:
    def test_defaults(self, tmp_path: Path) -> None:
        with patch.dict(os.environ, _clean_env(tmp_path), clear=True):
            cfg = LCMConfig.from_env()
        assert cfg.fresh_tail_token_budget_enabled is True
        assert cfg.fresh_tail_token_budget == 0
        assert cfg.fresh_tail_max_tokens == 60_000
        assert cfg.target_ratio == 0.20

    def test_target_ratio_from_compression_key(self, tmp_path: Path) -> None:
        _write_cfg(tmp_path, "compression:\n  target_ratio: 0.25\n")
        with patch.dict(os.environ, _clean_env(tmp_path), clear=True):
            cfg = LCMConfig.from_env()
        assert cfg.target_ratio == 0.25

    def test_lcm_target_ratio_key_is_ignored(self, tmp_path: Path) -> None:
        # D-12: the fleet key is compression.target_ratio; lcm.target_ratio is
        # intentionally not read.
        _write_cfg(tmp_path, "lcm:\n  target_ratio: 0.5\n")
        with patch.dict(os.environ, _clean_env(tmp_path), clear=True):
            cfg = LCMConfig.from_env()
        assert cfg.target_ratio == 0.20

    @pytest.mark.parametrize("bad", ["0", "-1", "1.5", "abc"])
    def test_target_ratio_clamped_to_default(self, tmp_path: Path, bad: str) -> None:
        _write_cfg(tmp_path, f"compression:\n  target_ratio: {bad}\n")
        with patch.dict(os.environ, _clean_env(tmp_path), clear=True):
            cfg = LCMConfig.from_env()
        assert cfg.target_ratio == 0.20

    def test_lcm_namespace_knobs(self, tmp_path: Path) -> None:
        _write_cfg(
            tmp_path,
            """\
            lcm:
              fresh_tail_token_budget_enabled: false
              fresh_tail_token_budget: 12345
              fresh_tail_max_tokens: 99000
            """,
        )
        with patch.dict(os.environ, _clean_env(tmp_path), clear=True):
            cfg = LCMConfig.from_env()
        assert cfg.fresh_tail_token_budget_enabled is False
        assert cfg.fresh_tail_token_budget == 12345
        assert cfg.fresh_tail_max_tokens == 99000

    def test_env_overrides_config(self, tmp_path: Path) -> None:
        _write_cfg(tmp_path, "compression:\n  target_ratio: 0.25\n")
        env = _clean_env(tmp_path)
        env.update(
            LCM_TARGET_RATIO="0.3",
            LCM_FRESH_TAIL_TOKEN_BUDGET="777",
            LCM_FRESH_TAIL_MAX_TOKENS="8000",
            LCM_FRESH_TAIL_TOKEN_BUDGET_ENABLED="0",
        )
        with patch.dict(os.environ, env, clear=True):
            cfg = LCMConfig.from_env()
        assert cfg.target_ratio == 0.3
        assert cfg.fresh_tail_token_budget == 777
        assert cfg.fresh_tail_max_tokens == 8000
        assert cfg.fresh_tail_token_budget_enabled is False

    @pytest.mark.parametrize("garbage", ["banana", "", "2", "None"])
    def test_enable_flag_fail_safe_on(self, tmp_path: Path, garbage: str) -> None:
        # Unrecognized values keep the default (ON) — same doctrine as
        # LCM_IDENTIFIER_FIDELITY: a typo must not silently change behavior.
        env = _clean_env(tmp_path)
        env["LCM_FRESH_TAIL_TOKEN_BUDGET_ENABLED"] = garbage
        with patch.dict(os.environ, env, clear=True):
            cfg = LCMConfig.from_env()
        assert cfg.fresh_tail_token_budget_enabled is True

    def test_negative_explicit_budget_rejected(self, tmp_path: Path) -> None:
        _write_cfg(tmp_path, "lcm:\n  fresh_tail_token_budget: -5\n")
        with patch.dict(os.environ, _clean_env(tmp_path), clear=True):
            cfg = LCMConfig.from_env()
        assert cfg.fresh_tail_token_budget == 0

    def test_helper_units(self, tmp_path: Path) -> None:
        _write_cfg(tmp_path, "lcm:\n  fresh_tail_max_tokens: 4000\ncompression:\n  target_ratio: 0.22\n")
        with patch.dict(os.environ, _clean_env(tmp_path), clear=True):
            assert _hermes_lcm_int("fresh_tail_max_tokens", 60_000) == 4000
            assert _hermes_compression_float("target_ratio", 0.20) == 0.22
            assert _lcm_config_bool("NOPE_ENV", "fresh_tail_token_budget_enabled", True) is True


# ---------------------------------------------------------------------------
# Budget resolution (I-2 / D-4 / I-7 lifecycle)
# ---------------------------------------------------------------------------

class TestBudgetResolution:
    def test_derived_budget(self, tmp_path: Path) -> None:
        cfg = LCMConfig()
        cfg.target_ratio = 0.25
        cfg.context_threshold = 0.75
        cfg.fresh_tail_max_tokens = 60_000
        eng = _engine(tmp_path, cfg)
        eng.update_model(model="test-model", context_length=200_000)
        assert eng.threshold_tokens == 150_000
        assert eng._fresh_tail_token_budget == 37_500

    def test_cap_binds_on_large_window(self, tmp_path: Path) -> None:
        # Apollo's live regime: 1M window → derived 187,500 → capped at 60K.
        cfg = LCMConfig()
        cfg.target_ratio = 0.25
        cfg.context_threshold = 0.75
        cfg.fresh_tail_max_tokens = 60_000
        eng = _engine(tmp_path, cfg)
        eng.update_model(model="test-model", context_length=1_000_000)
        assert eng._fresh_tail_token_budget == 60_000

    def test_explicit_budget_override(self, tmp_path: Path) -> None:
        cfg = LCMConfig()
        cfg.fresh_tail_token_budget = 5_000
        eng = _engine(tmp_path, cfg)
        eng.update_model(model="test-model", context_length=200_000)
        assert eng._fresh_tail_token_budget == 5_000

    def test_convergence_clamp_ratio_one(self, tmp_path: Path) -> None:
        # I-2: even ratio=1.0 (max allowed by the (0,1] clamp) cannot make the
        # budget reach threshold_tokens — the 0.9× clamp guarantees compaction
        # always has ≥10% of the threshold to work with.
        cfg = LCMConfig()
        cfg.target_ratio = 1.0
        cfg.context_threshold = 0.5
        cfg.fresh_tail_max_tokens = 10_000_000
        eng = _engine(tmp_path, cfg)
        eng.update_model(model="test-model", context_length=100_000)
        assert eng.threshold_tokens == 50_000
        assert eng._fresh_tail_token_budget <= int(0.9 * 50_000)

    def test_disabled_flag_zeroes_budget(self, tmp_path: Path) -> None:
        cfg = LCMConfig()
        cfg.fresh_tail_token_budget_enabled = False
        eng = _engine(tmp_path, cfg)
        eng.update_model(model="test-model", context_length=200_000)
        assert eng._fresh_tail_token_budget == 0

    def test_no_context_length_means_legacy(self, tmp_path: Path) -> None:
        eng = _engine(tmp_path)
        # No update_model call: threshold_tokens == 0 → budget stays 0.
        assert eng.threshold_tokens == 0
        assert eng._fresh_tail_token_budget == 0

    def test_budget_follows_model_switch(self, tmp_path: Path) -> None:
        # I-7: recompute happens exactly where threshold_tokens recomputes.
        cfg = LCMConfig()
        cfg.target_ratio = 0.25
        cfg.context_threshold = 0.75
        cfg.fresh_tail_max_tokens = 500_000
        eng = _engine(tmp_path, cfg)
        eng.update_model(model="big", context_length=1_000_000)
        big_budget = eng._fresh_tail_token_budget
        eng.update_model(model="small", context_length=200_000)
        small_budget = eng._fresh_tail_token_budget
        assert big_budget == 187_500
        assert small_budget == 37_500


# ---------------------------------------------------------------------------
# Phase 2 — the chokepoint helper
# ---------------------------------------------------------------------------

class TestDynamicFreshTailCount:
    def _budgeted_engine(self, tmp_path: Path, budget: int, base: int = 32) -> LCMEngine:
        cfg = LCMConfig()
        cfg.fresh_tail_count = base
        eng = _engine(tmp_path, cfg)
        eng._fresh_tail_token_budget = budget
        return eng

    def test_budget_walk_small_messages(self, tmp_path: Path) -> None:
        eng = self._budgeted_engine(tmp_path, budget=2_000)
        msgs = [_msg("assistant", 40, i) for i in range(400)]
        count = eng._dynamic_fresh_tail_count(msgs)
        assert count > 32
        tail = msgs[len(msgs) - count:]
        # Walk keeps accumulating while the floor is unmet, so the tail token
        # mass may exceed the budget only via the floor; here messages are tiny
        # so the tail should be within one message of the budget.
        assert count_messages_tokens(tail) <= 2_000 + count_message_tokens(msgs[0])

    def test_floor_with_huge_messages(self, tmp_path: Path) -> None:
        # I-3: budget yields < base → floor at base.
        eng = self._budgeted_engine(tmp_path, budget=500)
        msgs = [_msg("tool", 4_000, i) for i in range(100)]
        assert eng._dynamic_fresh_tail_count(msgs) == 32

    def test_zero_budget_is_legacy(self, tmp_path: Path) -> None:
        eng = self._budgeted_engine(tmp_path, budget=0)
        msgs = [_msg("assistant", 100, i) for i in range(100)]
        assert eng._dynamic_fresh_tail_count(msgs) == 32

    def test_fewer_messages_than_floor(self, tmp_path: Path) -> None:
        eng = self._budgeted_engine(tmp_path, budget=50_000)
        msgs = [_msg("assistant", 100, i) for i in range(5)]
        # Count can exceed len(messages); the cut clamps at 0.
        assert eng._fresh_tail_start(msgs) == 0

    def test_estimator_exception_falls_back_to_legacy(self, tmp_path: Path) -> None:
        # I-4: no exception may escape; malformed rows → legacy count.
        eng = self._budgeted_engine(tmp_path, budget=2_000)
        with patch(
            "plugins.context_engine.lcm.engine.count_message_tokens",
            side_effect=RuntimeError("boom"),
        ):
            msgs = [_msg("assistant", 40, i) for i in range(100)]
            assert eng._dynamic_fresh_tail_count(msgs) == 32

    def test_mixed_density_adapts(self, tmp_path: Path) -> None:
        # Density-adaptivity: same budget, chattier corpus keeps more messages.
        eng = self._budgeted_engine(tmp_path, budget=5_000)
        chatty = [_msg("assistant", 60, i) for i in range(600)]
        toolheavy = [_msg("tool", 2_000, i) for i in range(600)]
        assert eng._dynamic_fresh_tail_count(chatty) > eng._dynamic_fresh_tail_count(toolheavy)


# ---------------------------------------------------------------------------
# AC-10 — frozen-K boundary consistency + the AST no-insert guard
# ---------------------------------------------------------------------------

class TestFrozenKBoundaryConsistency:
    def test_in_pass_cuts_name_same_physical_rows(self, tmp_path: Path) -> None:
        """Simulate the leaf loop's removal-only mutations: with K frozen,
        `len(current) - K` selects the identical physical tail rows before and
        after front-region removals (scaffold-drop / chunk removal), as long
        as removals stay outside the tail (which the leaf loop guarantees:
        the cut point bounds every removal)."""
        cfg = LCMConfig()
        cfg.fresh_tail_count = 4
        eng = _engine(tmp_path, cfg)
        eng._fresh_tail_token_budget = 800

        msgs = [_msg("assistant", 60, i) for i in range(200)]
        K = eng._dynamic_fresh_tail_count(msgs)
        assert K < len(msgs) - 30  # backlog large enough for the removals below
        tail_before = msgs[len(msgs) - K:]

        # Front-region removal #1: scaffold-drop of 3 leading rows.
        mutated = msgs[3:]
        tail_after_scaffold = mutated[max(0, len(mutated) - K):]
        assert [m["content"] for m in tail_after_scaffold] == [m["content"] for m in tail_before]

        # Front-region removal #2: a compacted chunk of 20 more rows.
        mutated2 = mutated[20:]
        tail_after_chunk = mutated2[max(0, len(mutated2) - K):]
        assert [m["content"] for m in tail_after_chunk] == [m["content"] for m in tail_before]

    def test_compress_records_frozen_count(self, tmp_path: Path) -> None:
        # I-6: after compress(), protect_last_n reflects the dynamic count used.
        cfg = LCMConfig()
        cfg.fresh_tail_count = 8
        cfg.leaf_chunk_tokens = 200
        eng = _engine(tmp_path, cfg)
        eng.update_model(model="test-model", context_length=200_000)
        eng._fresh_tail_token_budget = 1_500
        eng.on_session_start("test_frozen_k")

        msgs = [{"role": "system", "content": "sys"}] + [
            _msg("assistant" if i % 2 == 0 else "user", 80, i) for i in range(200)
        ]
        expected_K = eng._dynamic_fresh_tail_count(msgs)
        assert expected_K > 8
        eng.compress(msgs)
        assert eng._last_fresh_tail_count == expected_K
        assert eng.protect_last_n == expected_K

    def test_no_list_inserting_transform_inside_frozen_k_span(self) -> None:
        """AST guard (AC-10b / D-11, hardened per pass-3 RC-1): the leaf-loop
        span in _compress_lossless must not call any list-GROWING primitive —
        not just the sanitizer wrapper. Count-arithmetic frozen-K is only
        sound while the in-pass span is removal-only.
        """
        from plugins.context_engine.lcm import engine as engine_mod

        src = textwrap.dedent(inspect.getsource(engine_mod.LCMEngine._compress_lossless))
        offenders = _frozen_k_span_growth_offenders(src)
        assert not offenders, (
            "list-growing transform inside the frozen-K leaf loop "
            f"breaks D-11 count-arithmetic: {offenders}"
        )

    def test_ast_guard_catches_planted_violation(self) -> None:
        """Negative control (pass-3 RC-1): prove the guard actually catches the
        class — a fixture leaf loop calling the sanitizer/stub primitives or
        growing the list must be flagged."""
        planted_wrapper = textwrap.dedent(
            """
            def _compress_lossless(self, messages):
                while leaf_passes < max_leaf_passes:
                    working_messages = self._sanitize_active_context_messages(working_messages)
            """
        )
        planted_primitive = textwrap.dedent(
            """
            def _compress_lossless(self, messages):
                while leaf_passes < max_leaf_passes:
                    repaired = self._sanitize_tool_pairs(working_messages)
            """
        )
        planted_growth = textwrap.dedent(
            """
            def _compress_lossless(self, messages):
                while leaf_passes < max_leaf_passes:
                    working_messages.insert(0, stub_row)
            """
        )
        planted_augassign = textwrap.dedent(
            """
            def _compress_lossless(self, messages):
                while leaf_passes < max_leaf_passes:
                    working_messages += [stub_row]
            """
        )
        for fixture in (planted_wrapper, planted_primitive, planted_growth, planted_augassign):
            assert _frozen_k_span_growth_offenders(fixture), (
                "AST guard failed to flag a planted frozen-K span violation:\n"
                + fixture
            )

    def test_short_list_legacy_semantics_and_consumer_reconcile(self, tmp_path: Path) -> None:
        """Pass-3 B-NEW-2, ground-truthed: K > len(messages) is the LEGACY
        semantic (protect_last_n is a constant 32 today regardless of list
        length), and the stats consumer already clamps its search window
        (find_inturn_kept_cut: lo = max(0, ...), candidate 0 = whole list).
        Clamping K to len(messages) would CHANGE legacy behavior on short
        lists (violating I-4), so we prove the consumer handles it instead."""
        from agent.compaction_stats import find_inturn_kept_cut

        cfg = LCMConfig()
        cfg.fresh_tail_count = 32
        eng = _engine(tmp_path, cfg)
        eng._fresh_tail_token_budget = 5_000

        short = [_msg("assistant", 50, i) for i in range(5)]
        K = eng._dynamic_fresh_tail_count(short)
        assert K == 32  # floor holds even though the list is shorter (legacy)
        assert eng._fresh_tail_start(short) == 0  # cut clamps at 0

        # Consumer path: identity sanitize; kept tail == the whole short list.
        cut = find_inturn_kept_cut(short, list(short), lambda ms: ms, K)
        assert cut == 0


# ---------------------------------------------------------------------------
# Phase 3 — rotate floor (D-7 / AC-11) + fail-open target_ratio (D-10 / AC-7)
# ---------------------------------------------------------------------------

class TestRotateAndFailOpen:
    def test_rotate_floor_uses_last_dynamic_count(self, tmp_path: Path) -> None:
        cfg = LCMConfig()
        cfg.fresh_tail_count = 8
        eng = _engine(tmp_path, cfg)
        eng.on_session_start("test_rotate_floor")
        eng._last_fresh_tail_count = 50
        result = eng.rotate_active_session(apply=False)
        assert result.get("fresh_tail_count") == 50

    def test_rotate_floor_survives_window_grow_without_compress(self, tmp_path: Path) -> None:
        # AC-11: update_model grow with no intervening compress() must not
        # shrink the preserved count below max(static, last-used dynamic K).
        cfg = LCMConfig()
        cfg.fresh_tail_count = 8
        eng = _engine(tmp_path, cfg)
        eng.on_session_start("test_rotate_grow")
        eng.update_model(model="small", context_length=200_000)
        eng._last_fresh_tail_count = 40
        eng.update_model(model="big", context_length=1_000_000)
        result = eng.rotate_active_session(apply=False)
        assert result.get("fresh_tail_count", 0) >= 40

    def test_rotate_floor_never_below_static(self, tmp_path: Path) -> None:
        cfg = LCMConfig()
        cfg.fresh_tail_count = 32
        eng = _engine(tmp_path, cfg)
        eng.on_session_start("test_rotate_static")
        eng._last_fresh_tail_count = 0  # pre-first-compaction degenerate
        result = eng.rotate_active_session(apply=False)
        assert result.get("fresh_tail_count") == 32

    def test_fail_open_fallback_uses_configured_target_ratio(self, tmp_path: Path) -> None:
        # AC-7 / D-10: the fallback compressor honors compression.target_ratio.
        cfg = LCMConfig()
        cfg.target_ratio = 0.25
        eng = _engine(tmp_path, cfg)
        eng.update_model(model="claude-test", context_length=200_000)
        fallback = eng._build_fail_open_fallback_compressor()
        assert fallback.summary_target_ratio == 0.25


class TestEnvRangeGuard:
    def test_env_target_ratio_out_of_range_clamped(self, tmp_path: Path) -> None:
        # Greptile PR review: the env override path must apply the same (0,1]
        # guard as the config-file path.
        for bad in ("1.5", "-0.2", "0"):
            env = _clean_env(tmp_path)
            env["LCM_TARGET_RATIO"] = bad
            with patch.dict(os.environ, env, clear=True):
                cfg = LCMConfig.from_env()
            assert cfg.target_ratio == 0.20, f"env {bad!r} defeated the clamp"

    def test_env_target_ratio_valid_applies(self, tmp_path: Path) -> None:
        env = _clean_env(tmp_path)
        env["LCM_TARGET_RATIO"] = "0.35"
        with patch.dict(os.environ, env, clear=True):
            cfg = LCMConfig.from_env()
        assert cfg.target_ratio == 0.35

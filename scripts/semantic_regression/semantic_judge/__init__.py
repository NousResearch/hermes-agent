# Semantic Regression LLM Judge for Hermes Agent
#
# A separate, isolated judge that evaluates whether Hermes Agent's actual
# behaviour during a test run matches the expected correct behaviour defined
# in each test scenario. Produces structured pass/fail verdicts with explicit
# reasoning, and never flags crash-related failures (covered by the existing
# crash-focused test suite) as semantic regressions.
#
# Design (per task t_60b946b5 and LogicHunter probe != oracle decoupling):
#   1. crash_filter.py  -- DETERMINISTIC pre-filter. Classifies a test-run log
#      as crash-related vs semantic before any LLM call. Crash-related failures
#      (stack traces, tool-dispatch errors that return stack traces, process
#      crashes, uncaught exceptions) are routed OUT of the semantic judge and
#      reported as crash_related=true with verdict="skip" -- they are covered by
#      the existing crash suite and must never be counted as semantic regressions.
#   2. judge.py         -- LLM judge. Given (expected_behaviour, execution_logs)
#      returns structured {verdict, confidence, reasoning[]}. Only invoked on
#      runs that survive the crash filter (i.e. ran to completion with output).
#   3. prompt.py        -- The judge's system/user prompt (tuned for accuracy).
#   4. run.py           -- CLI entry point: judge a single expected-behaviour +
#      log pair, or a directory of cases.
#   5. validation/      -- 50-run manually-labelled validation set + harness to
#      measure accuracy against AC (95%+).

# Pre-LLM Turn Reasoning Port

## Goal

Allow a `pre_llm_call` hook to return a `reasoning_config` override that applies only to the current user turn. The configured `agent.reasoning_config` remains the persistent fallback and must never be mutated by a hook.

## Design

The turn prologue in `agent/turn_context.py` owns the override lifecycle. It clears stale override state before invoking hooks, passes the configured reasoning value to callbacks, and copies the last valid `reasoning_config` dictionary returned by a hook. Context injection remains independent, so one hook result may contain both `context` and `reasoning_config`.

`AIAgent` exposes one effective-config resolver: a dictionary-valued turn override wins; otherwise the configured value is returned. Every request path in `agent/chat_completion_helpers.py` uses that resolver, including Anthropic Messages, Codex Responses, provider-profile and legacy chat completions, and iteration-limit summary/retry requests. The outer `run_conversation` boundary clears the override after every exit so exceptions, interruptions, and early returns cannot leak turn state.

Shell hooks accept the same return shape as Python callbacks. Multiple reasoning overrides follow existing plugin discovery order; the last valid dictionary wins.

## Verification

Unit tests cover hook parsing, turn setup/reset, configured fallback, combined context and reasoning results, and exception cleanup. Request-builder tests cover the active override on the current provider paths, while conversation-level coverage proves the first turn uses the override and a later turn falls back to the configured default. Public plugin and hook documentation describes the callback input, return shape, precedence, and per-turn scope.

All work remains local on the fork branch until explicit approval to push.

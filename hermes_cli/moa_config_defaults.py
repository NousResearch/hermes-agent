"""Default MoA configuration, split from the general config defaults table."""

MOA_DEFAULTS = {
    "default_preset": "default",
    "active_preset": "",
    # Write each MoA turn (reference + aggregator exact input/output/usage) as JSONL to
    # <hermes_home>/moa-traces/<session_id>.jsonl (or trace_dir) for auditing.
    "save_traces": False,
    # Daily spend alarm in USD; zero disables. Advisory only and never blocks a turn.
    "daily_budget_usd": 2.0,
    "trace_dir": "",
    # PII/credential redaction of advisor outputs: "" off | "display" (UI reference blocks +
    # traces only; aggregator sees raw) | "full" (also the aggregator prompt).
    # Advisors can echo PII from the conversation (emails, formatted phone numbers) and credential
    # shapes into reference blocks, traces, and the aggregator prompt. Modes ('' = off, the default):
    # "display" — redact user-visible surfaces only (reference blocks shown in the UI + saved MoA trace
    # records); the aggregator still sees raw advisor text. "full"    — additionally redact the advisor
    # text injected into the aggregator prompt (issue #59959).
    "privacy_filter": "",
    "presets": {
        "default": {
            "reference_models": [
                {"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"},
                {"provider": "openrouter", "model": "openai/gpt-5.5"},
            ],
            "aggregator": {
                "provider": "anthropic",
                "model": "claude-sonnet-5-5",
                "max_tokens": 2000,
            },
            # None leaves reference output at provider defaults; the cap applies only to advisors.
            "reference_max_tokens": None,
            "enabled": True,
        }
    },
}

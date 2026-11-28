"""Pure-data factory for the ``agent`` section's reasoning knobs (see DEFAULT_CONFIG["agent"]).

Pure-data leaf module — must not import from hermes_cli.config. Comments are the user-facing
docs of config.yaml.
"""


def _reasoning_defaults():
    """``agent.reasoning_overrides`` / ``reasoning_echo`` / ``adaptive_reasoning``, in config.yaml order.
    A fresh dict per call, so DEFAULT_CONFIG never shares a nested default."""
    return {
        # Model name (any reasonable spelling) -> effort level; overrides agent.reasoning_effort
        # when the current model matches. Edit in config.yaml (no CLI support: dots in keys).
        "reasoning_overrides": {},
        # Stored reasoning is replayed to every provider that can read it by default. This flag
        # only adds the must-echo " " pad on reasoning-less tool-call turns, which DeepSeek,
        # Kimi/Moonshot and Xiaomi MiMo require (auto-detected by provider name/host). Set
        # `reasoning_echo: true` on a `model:` or `fallback_providers:` entry for a custom gateway
        # proxying one of them. Strict providers (Mistral, Groq, Cerebras) never get the field.
        "reasoning_echo": False,
        # Opt-in adaptive reasoning adjustment. When enabled, each user turn is classified by a
        # deterministic local heuristic (no extra LLM call) and the effort is temporarily raised
        # above the configured agent.reasoning_effort baseline — up to max_effort — for turns that
        # warrant it (complex debugging, architecture/security work, consequential infrastructure
        # changes, in-depth research). Setting min_effort: low additionally allows clearly simple
        # turns (casual chatter, short fact lookups, single mechanical steps) to drop below the
        # baseline; without it the feature is escalation-only. Requires an explicit
        # agent.reasoning_effort baseline — with no baseline set, adjusting would toggle thinking
        # presence mid-conversation, so the feature stays inert. On direct Anthropic routes each
        # level change re-keys the message cache once (one cache write per change). The
        # baseline is restored after the turn and a one-line notice is shown whenever the level
        # changes. A level chosen for the session (/reasoning, --reasoning, a model pick's level,
        # a distinct Desktop effort selection) always wins. Delegate subagents inherit the
        # effective level + this policy and reclassify their own goal; delegation.reasoning_effort
        # pins them.
        "adaptive_reasoning": {
            "enabled": False,
            "max_effort": "xhigh",  # adjustment ceiling: high or xhigh
            "min_effort": "",  # optional floor; "low" enables downshift
        },
    }

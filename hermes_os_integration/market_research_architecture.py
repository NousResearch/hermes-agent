"""Generic market research worker architecture definition."""


MARKET_RESEARCH_ARCHITECTURE = {
    "control_plane": "Hermes OS",
    "runtime_worker": "official-hermes-agent",
    "pipeline": [
        "research-question",
        "research-agent",
        "evidence-agent",
        "validation-agent",
        "decision-agent",
        "experiment-tracker",
        "dashboard",
    ],
    "direct_runtime_exceptions": {
        "low-latency-market-fetch": "external-integration",
        "final-state-storage": "Hermes OS",
    },
}

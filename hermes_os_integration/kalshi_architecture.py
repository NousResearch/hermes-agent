"""Kalshi research worker architecture definition."""


KALSHI_RESEARCH_ARCHITECTURE = {
    "control_plane": "Hermes OS",
    "runtime_worker": "official-hermes-agent",
    "pipeline": [
        "bucket",
        "research-agent",
        "evidence-agent",
        "validation-agent",
        "portfolio-agent",
        "experiment-tracker",
        "dashboard",
    ],
    "direct_runtime_exceptions": {
        "low-latency-market-fetch": "deepseek-direct",
        "final-state-storage": "Hermes OS",
    },
}

"""Synthetic conversation dataset for memory benchmark.

Generates 50 pairs of related multi-turn conversations across diverse
domains. Each pair has:
  - Conversation A: Original discussion (5-15 turns)
  - Conversation B: Follow-up referencing A using different vocabulary
  - 3 retrieval queries: easy (keyword overlap), medium (partial), hard (none)

The dataset tests whether a memory system can find conceptually related
content when surface-level word matching fails — the "semantic gap" that
FTS5 keyword search cannot cross.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import List, Tuple

# ── Topic templates ──────────────────────────────────────────────────────────

TOPICS = [
    {
        "domain": "python-async",
        "concept": "Python async/await concurrency patterns",
        "terms_a": ["asyncio", "event loop", "coroutine", "await", "gather", "TaskGroup"],
        "terms_b": ["concurrent", "non-blocking", "cooperative", "scheduler", "async framework"],
    },
    {
        "domain": "database-optimization",
        "concept": "PostgreSQL query optimization and indexing",
        "terms_a": ["PostgreSQL", "EXPLAIN ANALYZE", "B-tree index", "sequential scan", "VACUUM"],
        "terms_b": ["database performance", "query planner", "index strategy", "table scan", "maintenance"],
    },
    {
        "domain": "docker-deployment",
        "concept": "Docker container orchestration and deployment",
        "terms_a": ["Docker", "container", "docker-compose", "image", "registry", "volume mount"],
        "terms_b": ["containerized", "orchestration", "service definition", "snapshot", "artifact repo", "persistent storage"],
    },
    {
        "domain": "network-security",
        "concept": "Network security and firewall configuration",
        "terms_a": ["firewall", "iptables", "port", "SSH", "TLS", "certificate", "VPN"],
        "terms_b": ["network filtering", "access control", "encrypted tunnel", "secure shell", "transport security", "private network"],
    },
    {
        "domain": "ml-training",
        "concept": "Machine learning model training pipelines",
        "terms_a": ["training", "GPU", "gradient descent", "epoch", "batch size", "overfitting"],
        "terms_b": ["optimization", "hardware acceleration", "backpropagation", "iteration", "mini-batch", "regularization"],
    },
    {
        "domain": "git-workflow",
        "concept": "Git branching strategies and code review",
        "terms_a": ["git", "branch", "merge", "pull request", "rebase", "conflict"],
        "terms_b": ["version control", "feature branch", "integration", "code review", "history rewrite", "divergence"],
    },
    {
        "domain": "api-design",
        "concept": "REST API design and best practices",
        "terms_a": ["REST", "endpoint", "HTTP method", "JSON", "status code", "pagination"],
        "terms_b": ["web service", "resource URL", "request verb", "serialization", "response code", "cursor-based"],
    },
    {
        "domain": "linux-admin",
        "concept": "Linux system administration and monitoring",
        "terms_a": ["systemd", "journalctl", "cron", "ps", "top", "df", "nginx"],
        "terms_b": ["service manager", "logging", "scheduled task", "process list", "resource monitor", "disk usage", "web server"],
    },
    {
        "domain": "testing",
        "concept": "Software testing strategies and test automation",
        "terms_a": ["unit test", "mock", "pytest", "coverage", "integration test", "fixture"],
        "terms_b": ["test isolation", "stub", "test runner", "code paths", "end-to-end", "setup/teardown"],
    },
    {
        "domain": "data-engineering",
        "concept": "Data pipeline architecture and ETL processes",
        "terms_a": ["ETL", "pipeline", "Apache Spark", "partition", "Parquet", "data lake"],
        "terms_b": ["extract-transform-load", "workflow", "distributed processing", "sharding", "columnar storage", "data warehouse"],
    },
]


@dataclass
class ConversationPair:
    """One pair of related conversations with retrieval queries."""
    pair_id: int
    domain: str
    conversation_a: str      # Full text of original conversation
    conversation_b: str      # Full text of follow-up conversation
    queries: List[Tuple[str, str]]  # [(query_text, difficulty), ...]
    ground_truth_index: int  # Which conversation A this pair belongs to


def _generate_conversation(topic: dict, variant: str) -> str:
    """Generate a synthetic multi-turn conversation on a topic."""
    terms = topic["terms_a"] if variant == "a" else topic["terms_b"]
    concept = topic["concept"]

    turns = []
    num_turns = random.randint(5, 12)

    system_msg = "You are a helpful technical assistant."
    turns.append(f"System: {system_msg}")

    for i in range(num_turns):
        if i % 2 == 0:
            # User turn
            term_sample = random.sample(terms, min(3, len(terms)))
            if i == 0:
                user_msg = (
                    f"I need help with {concept}. Specifically, I'm trying to "
                    f"understand how {term_sample[0]} works with {term_sample[1]}. "
                    f"Can you explain the relationship?"
                )
            else:
                patterns = [
                    f"What about {term_sample[0]}? How does that fit in?",
                    f"I'm seeing issues with {term_sample[0]} and {term_sample[1]}. Any suggestions?",
                    f"Can you elaborate on {term_sample[0]}? The docs aren't clear.",
                    f"How do {term_sample[0]} and {term_sample[1]} interact in practice?",
                    f"I tried using {term_sample[0]} but got unexpected results. Help?",
                ]
                user_msg = random.choice(patterns)
            turns.append(f"User: {user_msg}")
        else:
            # Assistant turn
            term_sample = random.sample(terms, min(3, len(terms)))
            patterns = [
                f"Great question. {term_sample[0]} works together with {term_sample[1]} "
                f"through a well-defined interface. Here's how: first, you set up "
                f"{term_sample[0]} to handle the initial request, then {term_sample[1]} "
                f"processes it asynchronously. The key benefit is that you get better "
                f"performance and cleaner code organization.",

                f"Let me break this down. {term_sample[0]} is the foundation — it manages "
                f"the core lifecycle. {term_sample[1]} builds on top of that to provide "
                f"additional functionality like {term_sample[2] if len(term_sample) > 2 else 'caching'}. "
                f"The recommended approach is to start with {term_sample[0]} and gradually "
                f"add {term_sample[1]} as your needs grow.",

                f"The relationship between {term_sample[0]} and {term_sample[1]} is "
                f"important to understand. Think of {term_sample[0]} as the engine and "
                f"{term_sample[1]} as the transmission — they need to work together "
                f"smoothly. When configured correctly, you'll see significant improvements "
                f"in both reliability and throughput.",
            ]
            turns.append(f"Assistant: {random.choice(patterns)}")

    return "\n".join(turns)


def generate_dataset(num_pairs: int = 50, seed: int = 42) -> List[ConversationPair]:
    """Generate a dataset of conversation pairs for benchmarking.

    Returns list of ConversationPair objects.
    """
    random.seed(seed)
    pairs = []

    for pair_id in range(num_pairs):
        topic = TOPICS[pair_id % len(TOPICS)]
        conv_a = _generate_conversation(topic, "a")
        conv_b = _generate_conversation(topic, "b")

        # Generate 3 queries per pair: easy, medium, hard
        queries = []
        terms_a = topic["terms_a"]
        terms_b = topic["terms_b"]

        # Easy: uses same keywords as conversation A (FTS5 should find this)
        queries.append((
            f"How do I use {random.choice(terms_a)} with {random.choice(terms_a)}?",
            "easy",
        ))

        # Medium: partial keyword overlap
        queries.append((
            f"I need help with {random.choice(terms_a)}. Things are running slow and "
            f"I'm not sure if my {random.choice(terms_b)} approach is correct.",
            "medium",
        ))

        # Hard: zero vocabulary overlap (only conceptually related)
        queries.append((
            f"What's the best way to handle {random.choice(terms_b)} when building "
            f"a {random.choice(terms_b)} system? I want to make sure it scales well.",
            "hard",
        ))

        pairs.append(ConversationPair(
            pair_id=pair_id,
            domain=topic["domain"],
            conversation_a=conv_a,
            conversation_b=conv_b,
            queries=queries,
            ground_truth_index=pair_id,
        ))

    return pairs


def save_dataset(pairs: List[ConversationPair], path: str) -> None:
    """Save dataset as JSON for reproducibility."""
    data = []
    for p in pairs:
        data.append({
            "pair_id": p.pair_id,
            "domain": p.domain,
            "conversation_a": p.conversation_a,
            "conversation_b": p.conversation_b,
            "queries": [{"text": q[0], "difficulty": q[1]} for q in p.queries],
        })
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_dataset(path: str) -> List[ConversationPair]:
    """Load dataset from JSON."""
    with open(path) as f:
        data = json.load(f)
    pairs = []
    for d in data:
        pairs.append(ConversationPair(
            pair_id=d["pair_id"],
            domain=d["domain"],
            conversation_a=d["conversation_a"],
            conversation_b=d["conversation_b"],
            queries=[(q["text"], q["difficulty"]) for q in d["queries"]],
            ground_truth_index=d["pair_id"],
        ))
    return pairs

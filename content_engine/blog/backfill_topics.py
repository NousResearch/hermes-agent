"""Approved blog back-population topics — 12 per stream, locked with Sahil.

This is the single source of truth for the one-off back-population. Each
topic carries {title, angle, tags?, needs_verification?, claim?}. AI topics
that reference a named current event (acquisition, regulation) have
``needs_verification: True`` so the generator web-grounds them before writing.
"""

TOPICS: dict[str, list[dict]] = {
    "ai": [
        {
            "title": "Why context is the bottleneck, not model size",
            "angle": "the 99x context-reduction lesson",
            "tags": ["context-engineering", "inference"],
        },
        {
            "title": "Cheap-first model routing",
            "angle": "running an agent fleet on pennies",
            "tags": ["model-routing", "cost-optimisation"],
        },
        {
            "title": "Agent memory that works",
            "angle": "episodic vs frozen tables vs markdown stores",
            "tags": ["agent-memory", "architecture"],
        },
        {
            "title": "Evals before vibes",
            "angle": "a minimum viable eval harness",
            "tags": ["evals", "quality"],
        },
        {
            "title": "Prompt caching is a 5-minute window, not magic",
            "angle": "designing around the TTL",
            "tags": ["prompt-caching", "inference"],
        },
        {
            "title": "Local models on consumer hardware",
            "angle": "what runs, what bites",
            "tags": ["local-models", "inference"],
        },
        {
            "title": "The GPU squeeze",
            "angle": "bulk-buying Mac minis and pre-caching GPU purchases",
            "tags": ["compute", "infrastructure"],
        },
        {
            "title": "KV-cache offloading to CPU and distributed memory",
            "angle": "the local-inference workaround",
            "tags": ["kv-cache", "inference"],
        },
        {
            "title": "NVIDIA's chip roadmap vs compute pricing",
            "angle": "and the push toward local hosting",
            "tags": ["compute", "economics"],
        },
        {
            "title": "The data-moat logic behind AI acquisitions",
            "angle": "owning a coding tool feeds a frontier model",
            "tags": ["acquisitions", "economics"],
            "needs_verification": True,
            "claim": "a frontier-model company acquiring an AI coding tool",
        },
        {
            "title": "When AI regulation reaches the model layer",
            "angle": "what scrutiny of frontier labs really means",
            "tags": ["regulation", "policy"],
            "needs_verification": True,
            "claim": "government regulatory action against a frontier AI lab",
        },
        {
            "title": "The economics of local vs API inference in 2026",
            "angle": "when self-hosting finally pays",
            "tags": ["economics", "inference"],
        },
    ],
    "pm": [
        {
            "title": "What chain-of-thought research means for how PMs write requirements",
            "angle": "translating CoT reasoning into structured spec writing",
            "tags": ["chain-of-thought", "requirements"],
        },
        {
            "title": "Context windows and model memory",
            "angle": "designing AI features around what models forget",
            "tags": ["context-window", "memory"],
        },
        {
            "title": "RAG for PMs",
            "angle": "when 'just add retrieval' genuinely solves a user problem",
            "tags": ["rag", "retrieval"],
        },
        {
            "title": "Evals as the new acceptance criteria",
            "angle": "bringing AI research rigour to PM sign-off",
            "tags": ["evals", "quality"],
        },
        {
            "title": "What scaling-law research tells PMs about betting on future model capability",
            "angle": "planning product roadmaps around known scaling curves",
            "tags": ["scaling-laws", "roadmap"],
        },
        {
            "title": "Hallucination research and the product guardrails that must follow",
            "angle": "designing AI products that fail gracefully",
            "tags": ["hallucination", "guardrails"],
        },
        {
            "title": "Human-in-the-loop findings",
            "angle": "designing AI review flows users actually trust",
            "tags": ["human-in-the-loop", "trust"],
        },
        {
            "title": "Prompt-engineering research, turned into reusable PM spec-writing workflows",
            "angle": "applying prompt research to day-to-day PM work",
            "tags": ["prompt-engineering", "workflows"],
        },
        {
            "title": "AI adoption curves",
            "angle": "what diffusion research predicts for rolling out AI features",
            "tags": ["adoption", "strategy"],
        },
        {
            "title": "Agentic AI research and what it implies for workflow-product roadmaps",
            "angle": "translating agent capabilities into product features",
            "tags": ["agents", "roadmap"],
        },
        {
            "title": "Fine-tune vs prompt vs RAG",
            "angle": "a research-grounded build decision guide for PMs",
            "tags": ["fine-tuning", "prompt", "rag"],
        },
        {
            "title": "The skills research says matter most for PMs in an AI-native org",
            "angle": "what the literature says PMs need to learn next",
            "tags": ["skills", "career"],
        },
    ],
    "builder": [
        {
            "title": "What a research paper taught me about agent memory",
            "angle": "the concept, made usable",
            "tags": ["agent-memory", "research"],
        },
        {
            "title": "The GitHub repos quietly solving agent orchestration",
            "angle": "and what they get right",
            "tags": ["github", "orchestration"],
        },
        {
            "title": "Building a content engine",
            "angle": "the problem, the approach, what actually moved quality",
            "tags": ["content-engine", "build"],
        },
        {
            "title": "The truth about autonomous agents",
            "angle": "what the demos hide and what it really takes",
            "tags": ["agents", "reality-check"],
        },
        {
            "title": "Reference-anchored image generation",
            "angle": "the idea, and why prompting alone falls short",
            "tags": ["image-generation", "reference"],
        },
        {
            "title": "Cheap-first model routing",
            "angle": "the concept, the trade-offs, where it quietly breaks",
            "tags": ["model-routing", "cost"],
        },
        {
            "title": "Exposing the AI hype cycle",
            "angle": "what creators oversell vs what actually ships",
            "tags": ["hype", "reality-check"],
        },
        {
            "title": "How I'm making my agent more reliable",
            "angle": "the concepts behind the iteration",
            "tags": ["reliability", "agents"],
        },
        {
            "title": "The reality of local models",
            "angle": "what the threads promise vs what runs on real hardware",
            "tags": ["local-models", "reality-check"],
        },
        {
            "title": "Context engineering in practice",
            "angle": "lessons from the papers, applied to a live system",
            "tags": ["context-engineering", "build"],
        },
        {
            "title": "What production AI agent actually means",
            "angle": "and why most demos aren't one",
            "tags": ["agents", "production"],
        },
        {
            "title": "Building an eval harness for agent reliability",
            "angle": "what I learned from instrumenting my own system",
            "tags": ["evals", "quality"],
        },
    ],
}


def topics_for(stream: str) -> list[dict]:
    """Return the approved topic list for a stream, or empty list."""
    return TOPICS.get(stream, [])


def needs_verification(topic: dict) -> bool:
    """True if this topic references a named event that must be web-verified."""
    return bool(topic.get("needs_verification"))

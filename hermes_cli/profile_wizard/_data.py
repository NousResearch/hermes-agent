from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ROLE_OPTIONS: list[tuple[str, str]] = [
    ("moderator", "Moderator — safety, rules, escalation, community hygiene"),
    ("community_manager", "Community Manager — engagement, events, member support"),
    ("utility_bot", "Utility Bot — commands, reminders, workflows, small automations"),
    ("knowledge_base", "Knowledge Base — answers from docs, FAQs, policies, archives"),
    ("entertainment", "Entertainment — games, banter, interactive fun"),
    ("conversational_ai", "Conversational AI — natural chat, coaching, companion-style replies"),
    ("analytics", "Analytics — metrics, reporting, insights, dashboards"),
    ("devops_admin", "DevOps/Admin — ops, deploys, incidents, systems, admin control"),
]

PERSONALITY_OPTIONS: list[tuple[str, str]] = [
    ("commander", "Commander — decisive, clear, action-first"),
    ("analyst", "Analyst — careful, evidence-led, structured"),
    ("entertainer", "Entertainer — playful, energetic, memorable"),
    ("diplomat", "Diplomat — tactful, balanced, community-safe"),
    ("maverick", "Maverick — unconventional, direct, creative"),
    ("scholar", "Scholar — thoughtful, educational, reference-heavy"),
    ("craftsman", "Craftsman — practical, precise, quality-focused"),
    ("sage", "Sage — calm, concise, high-signal guidance"),
]

TONE_OPTIONS: list[tuple[str, str]] = [
    ("casual", "Casual"),
    ("professional", "Professional"),
    ("humorous", "Humorous"),
    ("technical", "Technical"),
    ("hype", "Hype"),
    ("deadpan", "Deadpan"),
    ("motivational", "Motivational"),
    ("sophisticated", "Sophisticated"),
]

TARGET_PLATFORM_OPTIONS: list[tuple[str, str]] = [
    ("cli", "CLI / terminal"),
    ("tui", "Hermes TUI"),
    ("discord", "Discord bot"),
    ("telegram", "Telegram bot"),
    ("webhook", "Webhook / API"),
    ("local_automation", "Local automation"),
]

RESPONSE_LENGTH_OPTIONS: list[tuple[str, str]] = [
    ("concise", "Concise — short answers unless detail is requested"),
    ("balanced", "Balanced — enough context without over-explaining"),
    ("detailed", "Detailed — fuller explanations and reasoning"),
    ("adaptive", "Adaptive — match depth to task stakes and complexity"),
]

EMOJI_OPTIONS: list[tuple[str, str]] = [
    ("none", "None"),
    ("minimal", "Minimal"),
    ("moderate", "Moderate"),
    ("heavy", "Heavy"),
]

FALLBACK_OPTIONS: list[tuple[str, str]] = [
    ("acknowledge_uncertainty", "Acknowledge uncertainty"),
    ("research_reason", "Research/reason before answering"),
    ("redirect", "Redirect to safer or better next step"),
    ("silent_unless_confident", "Stay silent unless confident"),
]

SKILL_CATEGORIES: dict[str, list[tuple[str, str]]] = {
    "Moderation & Safety": [
        ("rule_enforcement", "Rule enforcement"),
        ("spam_detection", "Spam detection"),
        ("toxicity_filtering", "Toxicity filtering"),
        ("incident_escalation", "Incident escalation"),
        ("audit_logs", "Audit logs"),
    ],
    "Community & Engagement": [
        ("welcome_flows", "Welcome flows"),
        ("faq_answers", "FAQ answers"),
        ("event_announcements", "Event announcements"),
        ("member_support", "Member support"),
        ("polls_surveys", "Polls & surveys"),
    ],
    "AI & Intelligence": [
        ("summarisation", "Summarisation"),
        ("retrieval_qa", "Retrieval Q&A"),
        ("research_assistant", "Research assistant"),
        ("reasoning", "Reasoning"),
        ("content_generation", "Content generation"),
    ],
    "Integrations & APIs": [
        ("webhooks", "Webhooks"),
        ("notion", "Notion"),
        ("google_workspace", "Google Workspace"),
        ("github", "GitHub"),
        ("custom_apis", "Custom APIs"),
    ],
    "Utility & Tools": [
        ("reminders", "Reminders"),
        ("scheduling", "Scheduling"),
        ("file_search", "File search"),
        ("data_lookup", "Data lookup"),
        ("automation", "Automation"),
    ],
    "Entertainment & Games": [
        ("trivia", "Trivia"),
        ("mini_games", "Mini-games"),
        ("memes", "Memes"),
        ("icebreakers", "Icebreakers"),
        ("roleplay", "Roleplay"),
    ],
}

SKILL_BUNDLES: dict[str, dict[str, list[str]]] = {
    "starter": {
        "AI & Intelligence": ["summarisation", "reasoning"],
        "Utility & Tools": ["file_search", "reminders"],
    },
    "research": {
        "AI & Intelligence": ["research_assistant", "retrieval_qa", "summarisation", "reasoning"],
        "Utility & Tools": ["file_search", "data_lookup"],
    },
    "developer": {
        "Integrations & APIs": ["github", "webhooks"],
        "Utility & Tools": ["automation", "file_search"],
        "AI & Intelligence": ["reasoning", "summarisation"],
    },
    "admin_productivity": {
        "Integrations & APIs": ["google_workspace", "notion"],
        "Utility & Tools": ["scheduling", "reminders", "file_search", "automation"],
        "AI & Intelligence": ["summarisation"],
    },
    "content_social": {
        "AI & Intelligence": ["content_generation", "research_assistant", "summarisation"],
        "Community & Engagement": ["event_announcements", "member_support"],
        "Integrations & APIs": ["webhooks"],
    },
    "community": {
        "Moderation & Safety": ["rule_enforcement", "spam_detection", "toxicity_filtering", "incident_escalation"],
        "Community & Engagement": ["welcome_flows", "faq_answers", "member_support", "polls_surveys"],
    },
    "operations": {
        "Utility & Tools": ["automation", "data_lookup", "file_search"],
        "Integrations & APIs": ["webhooks", "github", "custom_apis"],
        "Moderation & Safety": ["audit_logs", "incident_escalation"],
    },
}

SKILL_BUNDLE_OPTIONS: list[tuple[str, str]] = [
    ("starter", "Starter — summarisation, reasoning, files, reminders"),
    ("research", "Research — sources, docs, summaries, evidence"),
    ("developer", "Developer — GitHub, automation, code support"),
    ("admin_productivity", "Admin/Productivity — calendar, reminders, docs, workflows"),
    ("content_social", "Content/Social — research, drafting, announcements"),
    ("community", "Community — welcome, FAQ, support, moderation"),
    ("operations", "Operations — audits, webhooks, automation, incidents"),
]

DOCS_SKILL_CATEGORIES: dict[str, list[tuple[str, str]]] = {
    "Skills Hub — Code & GitHub": [
        ("claude-code", "claude-code — Delegate coding to Claude Code CLI"),
        ("codex", "codex — Delegate coding to OpenAI Codex CLI"),
        ("opencode", "opencode — Delegate coding to OpenCode CLI"),
        ("github-code-review", "github-code-review — Review PRs via gh/REST"),
        ("github-pr-workflow", "github-pr-workflow — Branch, commit, PR, CI, merge"),
        ("github-issues", "github-issues — Create, triage, label GitHub issues"),
    ],
    "Skills Hub — Orchestration & Hermes": [
        ("hermes-agent", "hermes-agent — Configure, extend, or contribute to Hermes"),
        ("kanban-orchestrator", "kanban-orchestrator — Decompose and route work"),
        ("kanban-worker", "kanban-worker — Execute Kanban tasks safely"),
        ("webhook-subscriptions", "webhook-subscriptions — Event-driven agent runs"),
    ],
    "Skills Hub — Research & Knowledge": [
        ("arxiv", "arxiv — Search academic papers"),
        ("market-research", "market-research — Competitive/idea validation"),
        ("obsidian", "obsidian — Read/search/create notes"),
        ("youtube-content", "youtube-content — Transcripts to summaries/content"),
    ],
    "Skills Hub — Productivity & Apps": [
        ("google-workspace", "google-workspace — Gmail, Calendar, Drive, Docs, Sheets"),
        ("notion", "notion — Pages, databases, markdown"),
        ("airtable", "airtable — Records CRUD, filters, upserts"),
        ("linear", "linear — Issues, projects, teams"),
        ("mailbox-agent", "mailbox-agent — Cross-account mailbox digest"),
    ],
    "Skills Hub — Social & Content": [
        ("social-content", "social-content — Create/schedule/optimise social content"),
        ("xurl", "xurl — X/Twitter posting/search/media"),
        ("brand-voices", "brand-voices — Brand/personal voice authoring"),
        ("avoid-ai-writing", "avoid-ai-writing — Remove AI writing patterns"),
    ],
    "Skills Hub — Visual & Creative": [
        ("architecture-diagram", "architecture-diagram — Dark SVG architecture diagrams"),
        ("ascii-art", "ascii-art — pyfiglet/cowsay/image-to-ascii"),
        ("sketch", "sketch — Throwaway HTML mockups"),
        ("popular-web-designs", "popular-web-designs — Real design systems as HTML/CSS"),
    ],
}

USER_TYPE_OPTIONS: list[tuple[str, str]] = [
    ("individual", "Individual user"),
    ("developer", "Developer / technical builder"),
    ("founder", "Founder / operator"),
    ("creator", "Creator / marketer"),
    ("researcher", "Researcher / analyst"),
    ("community", "Community manager"),
    ("team", "Team / organisation"),
    ("other", "Other"),
]

INTEREST_OPTIONS: list[tuple[str, str]] = [
    ("ai_automation", "AI / automation"),
    ("software", "Software development"),
    ("business", "Business / startups"),
    ("research", "Research / analysis"),
    ("writing", "Writing / content"),
    ("social", "Social media"),
    ("design", "Design / product"),
    ("learning", "Education / learning"),
    ("finance", "Finance / markets"),
    ("other", "Other"),
]

USE_CASE_OPTIONS: list[tuple[str, str]] = [
    ("organise", "Organise my life/work"),
    ("research", "Research topics and monitor signals"),
    ("content", "Write or improve content"),
    ("coding", "Build/debug software"),
    ("admin", "Manage emails, calendar, or admin"),
    ("automation", "Automate repeated tasks"),
    ("community", "Support a Discord/Telegram community"),
    ("learning", "Learn a subject"),
    ("analysis", "Analyse documents/data"),
    ("multi_agent", "Create a multi-agent team"),
]

APP_TOOL_OPTIONS: list[tuple[str, str]] = [
    ("gmail", "Gmail"),
    ("outlook", "Outlook"),
    ("google_calendar", "Google Calendar"),
    ("google_drive", "Google Drive"),
    ("notion", "Notion"),
    ("obsidian", "Obsidian"),
    ("github", "GitHub"),
    ("linear", "Linear"),
    ("discord", "Discord"),
    ("telegram", "Telegram"),
    ("x_twitter", "X/Twitter"),
    ("linkedin", "LinkedIn"),
    ("youtube", "YouTube"),
    ("slack", "Slack"),
    ("teams", "Microsoft Teams"),
]

HERMES_SETUP_OPTIONS: list[tuple[str, str]] = [
    ("one_assistant", "One general assistant"),
    ("specialists", "A few specialist assistants"),
    ("multi_agent", "A full multi-agent team"),
    ("automation", "A background automation system"),
    ("recommend", "I'm not sure — recommend one"),
]

AUTOMATION_COMFORT_OPTIONS: list[tuple[str, str]] = [
    ("suggest_only", "Suggest only — I approve everything"),
    ("draft_prepare", "Draft and prepare — I approve final actions"),
    ("safe_local", "Execute safe local tasks automatically"),
    ("background_guardrails", "Run background automations with guardrails"),
    ("cautious", "Not sure — start cautious"),
]

SENSITIVE_AREA_OPTIONS: list[tuple[str, str]] = [
    ("money", "Money / purchases"),
    ("external_messages", "External messages"),
    ("publishing", "Publishing content"),
    ("bookings", "Calendar bookings"),
    ("file_deletion", "File deletion"),
    ("credentials", "Credentials/secrets"),
    ("work_data", "Work/company data"),
    ("private_topics", "Personal/private topics"),
]

DEFAULT_BOUNDARY_OPTIONS: list[tuple[str, str]] = [
    ("no_external_messages_without_approval", "Do not send external messages without approval"),
    ("no_spending_without_approval", "Do not spend money or make commitments without approval"),
    ("no_sensitive_data_disclosure", "Do not disclose secrets, credentials, or private data"),
    ("no_medical_legal_financial_authority", "No authoritative medical/legal/financial advice"),
    ("escalate_policy_edge_cases", "Escalate policy/safety edge cases"),
]


IDEA_CATEGORY_OPTIONS: list[tuple[str, str]] = [
    ("ootb", "Starter profiles — outcome-based defaults for new users"),
    ("examples", "Advanced examples — real profile systems and inspiration"),
    ("leads", "Multi-agent team roles — orchestrators and domain leads"),
    ("workers", "Worker/sub-agent profiles — narrow execution agents"),
    ("back", "Back to profile workspace"),
]


@dataclass
class ProfileIdea:
    id: str
    name: str
    category: str
    description: str
    best_for: list[str]
    primary_role: str
    secondary_roles: list[str]
    personality: str
    tones: list[str]
    suggested_skills: list[str]
    suggested_platforms: list[str]
    suggested_provider: str | None = None
    suggested_model: str | None = None
    guardrails: list[str] = field(default_factory=list)
    source: str = "ootb"
    avatar_emoji: str = "✦"
    tagline: str = "Focused help, clear boundaries."
    use_cases: list[str] = field(default_factory=list)


@dataclass
class AgentProfileSpec:
    profile_name: str
    display_name: str
    tagline: str
    avatar_emoji: str
    primary_role: str
    secondary_roles: list[str] = field(default_factory=list)
    personality: str = "analyst"
    tones: list[str] = field(default_factory=list)
    skill_sets: dict[str, list[str]] = field(default_factory=dict)
    custom_skills: list[str] = field(default_factory=list)
    response_length: str = "balanced"
    emoji_usage: str = "minimal"
    fallback_behavior: str = "acknowledge_uncertainty"
    boundaries: list[str] = field(default_factory=list)
    blocked_topics: list[str] = field(default_factory=list)
    user_context: str = ""
    intended_outcomes: str = ""
    operating_context: str = ""
    target_platforms: list[str] = field(default_factory=list)
    model_provider: str = "openai-codex"
    model: str = "gpt-5.5"
    source_idea: str | None = None
    source_idea_name: str | None = None
    use_cases: list[str] = field(default_factory=list)

    @property
    def total_skills(self) -> int:
        return sum(len(v) for v in self.skill_sets.values()) + len(self.custom_skills)


@dataclass
class OnboardingAnswers:
    user_type: str
    working_context: str
    user_context: str
    interests: list[str]
    use_cases: list[str]
    ideas: str
    setup_style: str
    platforms: list[str]
    apps: list[str]
    other_apps: str
    automation_comfort: str
    sensitive_areas: list[str]


def _ideas() -> list[ProfileIdea]:
    """Browseable profile ideas for users who do not know where to start.

    The example profile distributions were reviewed from:
    - SouthpawIN/nous-girl: warm TTS-first creative/music studio front-agent.
    - SouthpawIN/senter: triage orchestrator/router, explicitly not a doer.
    - SouthpawIN/chizul: hands-on Hermes operator/systems engineer.
    - Klerik (local VPS project): meticulous meta-agent for reviewing
      and surgically correcting other Hermes agent profiles.
    """
    return [
        ProfileIdea("general_assistant", "General Assistant", "ootb", "Everyday Hermes assistant for planning, summaries, decisions, and general help.", ["personal productivity", "first Hermes setup", "mixed daily use"], "utility_bot", ["knowledge_base"], "sage", ["professional", "casual"], ["summarisation", "reasoning", "reminders", "file_search"], ["cli", "tui", "discord"], avatar_emoji="✨", tagline="Start simple. Get useful help fast."),
        ProfileIdea("research_assistant", "Research Assistant", "ootb", "Finds sources, compares options, monitors topics, and turns research into clear recommendations.", ["research briefs", "topic monitoring", "comparisons", "learning"], "analytics", ["knowledge_base"], "analyst", ["professional", "technical"], ["research_assistant", "retrieval_qa", "summarisation", "reasoning", "data_lookup"], ["cli", "tui", "discord"], avatar_emoji="🔎", tagline="Evidence first. Recommendation last."),
        ProfileIdea("coding_assistant", "Coding Assistant", "ootb", "Helps plan, debug, review, and ship software work with safe verification steps.", ["coding projects", "debugging", "PR review", "implementation plans"], "devops_admin", ["analytics"], "craftsman", ["technical", "professional"], ["github", "automation", "reasoning", "audit_logs"], ["cli", "tui"], avatar_emoji="🧩", tagline="Small changes. Clean seams."),
        ProfileIdea("admin_assistant_starter", "Admin Assistant", "ootb", "Supports email, calendar, reminders, logistics, documents, and day-to-day admin coordination.", ["inbox/calendar", "personal admin", "task follow-up", "documents"], "utility_bot", ["community_manager"], "diplomat", ["professional", "casual"], ["scheduling", "reminders", "summarisation", "google_workspace", "notion"], ["cli", "discord"], avatar_emoji="📎", tagline="Loose ends closed."),
        ProfileIdea("content_assistant", "Content Assistant", "ootb", "Drafts, rewrites, repurposes, and reviews content while preserving voice.", ["social posts", "drafting", "brand voice", "content planning"], "community_manager", ["conversational_ai"], "entertainer", ["casual", "sophisticated"], ["content_generation", "research_assistant", "summarisation"], ["cli", "discord"], avatar_emoji="✍️", tagline="Voice intact. Slop removed."),
        ProfileIdea("knowledge_base", "Knowledge Base Assistant", "ootb", "Answers questions from notes, docs, files, FAQs, policies, and structured knowledge.", ["internal docs", "notes", "support knowledge", "team memory"], "knowledge_base", ["analytics"], "scholar", ["professional", "technical"], ["retrieval_qa", "summarisation", "file_search", "data_lookup", "research_assistant"], ["cli", "tui", "discord"], avatar_emoji="📚", tagline="Answers from the source, not vibes."),
        ProfileIdea("automation_assistant", "Automation Assistant", "ootb", "Builds and manages recurring workflows, scripts, webhooks, and scheduled background jobs.", ["task automation", "scheduled jobs", "webhooks", "ops workflows"], "devops_admin", ["utility_bot"], "craftsman", ["technical", "professional"], ["automation", "webhooks", "reminders", "audit_logs", "custom_apis"], ["cli", "tui", "webhook"], guardrails=["no_sensitive_data_disclosure", "no_external_messages_without_approval", "escalate_policy_edge_cases"], avatar_emoji="⚙️", tagline="Automate carefully. Verify directly."),
        ProfileIdea("community_assistant", "Community Assistant", "ootb", "Supports Discord/Telegram communities with FAQ answers, welcomes, moderation support, and digests.", ["Discord communities", "Telegram groups", "support communities"], "community_manager", ["moderator"], "diplomat", ["casual", "professional"], ["welcome_flows", "faq_answers", "member_support", "rule_enforcement", "spam_detection"], ["discord", "telegram"], guardrails=["no_sensitive_data_disclosure", "escalate_policy_edge_cases", "no_external_messages_without_approval"], avatar_emoji="🤝", tagline="Help people feel oriented and heard."),
        ProfileIdea("nous_girl", "Nous Girl-style Creative Companion", "examples", "Warm, TTS-first creative front-agent for capturing spoken ideas, building on them, and handing off execution.", ["voice intake", "creative ideation", "music studio workflows", "enthusiastic companion agents"], "conversational_ai", ["community_manager"], "entertainer", ["casual", "motivational"], ["content_generation", "summarisation", "research_assistant"], ["discord", "telegram", "cli"], suggested_provider="nous", suggested_model="qwen/qwen3.6-flash", avatar_emoji="🎙️", tagline="Yes, and — then capture the idea.", use_cases=["voice intake", "creative exploration", "handoff notes"]),
        ProfileIdea("senter", "Senter-style Triage Orchestrator", "examples", "Router profile that receives ideas, decides scope, decomposes work, and sends tasks to workers instead of doing implementation itself.", ["multi-agent teams", "task triage", "Kanban routing", "scope decisions"], "devops_admin", ["analytics"], "analyst", ["professional", "technical"], ["automation", "audit_logs", "reasoning", "incident_escalation"], ["discord", "cli"], suggested_provider="nous", suggested_model="qwen/qwen3.6-plus", guardrails=["no_external_messages_without_approval", "escalate_policy_edge_cases"], avatar_emoji="🧭", tagline="Decompose, route, summarise."),
        ProfileIdea("chizul", "Chizul-style Hermes Operator", "examples", "Hands-on operator profile for implementing Hermes fixes, making targeted changes, and verifying results.", ["Hermes operations", "systems repair", "implementation work", "technical execution"], "devops_admin", ["utility_bot"], "craftsman", ["technical", "professional"], ["automation", "github", "audit_logs", "incident_escalation"], ["cli", "tui"], suggested_provider="nous", suggested_model="qwen/qwen3.6-plus", guardrails=["no_sensitive_data_disclosure", "escalate_policy_edge_cases"], avatar_emoji="🔧", tagline="Do, verify, report."),
        ProfileIdea("klerik", "Klerik — Meta-Agent Profile Reviewer", "examples", "Meticulous meta-agent that reviews how other Hermes profiles behave, compares output to user expectations, diagnoses root-cause misalignment, and makes surgical profile edits.", ["profile review & correction", "agent behaviour alignment", "SOUL.md/USER.md editing", "session audit & diagnosis"], "analytics", ["devops_admin"], "analyst", ["calm", "technical"], ["session_search", "audit_logs", "reasoning", "hermes-agent-skill-authoring", "systematic-debugging"], ["cli", "tui"], suggested_provider="nous", suggested_model="qwen/qwen3.6-plus", guardrails=["no_sensitive_data_disclosure", "escalate_policy_edge_cases", "no_external_messages_without_approval"], avatar_emoji="📋", tagline="Evidence first. Surgical edits. Preserve the voice.", use_cases=["profile review", "behaviour diagnosis", "agent alignment", "session audit"]),
        ProfileIdea("kensei_orchestrator", "Kensei-style Orchestrator", "leads", "Routes work, makes tradeoffs, enforces standards, and coordinates specialist agents.", ["personal operating layers", "multi-agent systems", "cross-domain orchestration"], "devops_admin", ["analytics", "knowledge_base"], "commander", ["professional", "technical"], ["automation", "audit_logs", "reasoning", "summarisation", "incident_escalation"], ["cli", "discord"], avatar_emoji="⚔️", tagline="See the board. Make the call."),
        ProfileIdea("research_lead", "Research Lead", "leads", "Runs deep research, signal scanning, comparisons, and recommendation briefs.", ["market scans", "technical research", "trend monitoring"], "analytics", ["knowledge_base"], "analyst", ["technical", "professional"], ["research_assistant", "summarisation", "retrieval_qa", "data_lookup"], ["cli", "discord"], avatar_emoji="🔎", tagline="Evidence first, recommendation last."),
        ProfileIdea("coding_lead", "Coding Lead", "leads", "Plans, reviews, debugs, and coordinates implementation work safely.", ["software projects", "PR review", "debugging", "architecture checks"], "devops_admin", ["analytics"], "craftsman", ["technical", "professional"], ["github", "automation", "reasoning", "audit_logs"], ["cli", "tui"], avatar_emoji="🧩", tagline="Small changes. Clean seams."),
        ProfileIdea("qa_lead", "QA Lead", "leads", "Tests releases, finds bugs, verifies fixes, and blocks low-quality output.", ["release gates", "bug bashes", "acceptance testing"], "analytics", ["devops_admin"], "analyst", ["technical", "professional"], ["audit_logs", "reasoning", "data_lookup", "incident_escalation"], ["cli", "tui", "discord"], avatar_emoji="✅", tagline="Trust, but reproduce."),
        ProfileIdea("content_lead", "Content Lead", "leads", "Drafts, reviews, and adapts content across channels while preserving voice.", ["social content", "brand voice", "campaigns"], "community_manager", ["conversational_ai"], "entertainer", ["casual", "sophisticated"], ["content_generation", "summarisation", "event_announcements"], ["cli", "discord"], avatar_emoji="✍️", tagline="Voice intact. Slop removed."),
        ProfileIdea("admin_assistant", "Admin Assistant", "leads", "Handles inbox, calendar, logistics, bookings, reminders, and admin coordination.", ["email/admin", "job hunt support", "calendar workflows"], "utility_bot", ["community_manager"], "diplomat", ["professional", "casual"], ["scheduling", "reminders", "data_lookup", "automation"], ["cli", "discord"], avatar_emoji="📎", tagline="Admin handled, loose ends closed."),
        ProfileIdea("security_ops_lead", "Security/Ops Lead", "leads", "Monitors systems, checks security posture, manages credentials, and escalates risks.", ["VPS ops", "security reviews", "credential hygiene", "service health"], "devops_admin", ["analytics"], "commander", ["technical", "professional"], ["audit_logs", "incident_escalation", "automation", "webhooks"], ["cli", "discord"], guardrails=["no_sensitive_data_disclosure", "escalate_policy_edge_cases", "no_external_messages_without_approval"], avatar_emoji="🛡️", tagline="Secure by default. Escalate early."),
        ProfileIdea("teaching_mentor", "Teaching/Mentor Profile", "leads", "Designs lessons, explains concepts, and supports structured learning.", ["AI/ML lessons", "coaching", "study support"], "conversational_ai", ["knowledge_base"], "scholar", ["motivational", "technical"], ["content_generation", "summarisation", "retrieval_qa", "research_assistant"], ["discord", "cli"], avatar_emoji="🎓", tagline="Teach clearly. Build confidence."),
        ProfileIdea("code_worker", "Code Worker", "workers", "Executes narrow coding tasks from a plan with minimal context drift.", ["implementation tasks", "small fixes", "scaffolded coding work"], "devops_admin", [], "craftsman", ["technical"], ["github", "automation", "reasoning"], ["cli"], avatar_emoji="🛠️", tagline="One task. Clean diff."),
        ProfileIdea("research_worker", "Research Worker", "workers", "Runs focused research, extracts facts, and returns concise evidence-backed summaries.", ["source gathering", "comparison tables", "market scans"], "analytics", ["knowledge_base"], "analyst", ["professional", "technical"], ["research_assistant", "summarisation", "retrieval_qa"], ["cli"], avatar_emoji="📡", tagline="Find sources. Extract signal."),
        ProfileIdea("qa_worker", "QA Worker", "workers", "Runs checks, reproduces bugs, gathers evidence, and reports pass/fail clearly.", ["bug reproduction", "release checks", "test evidence"], "analytics", [], "analyst", ["technical"], ["audit_logs", "data_lookup", "reasoning"], ["cli"], avatar_emoji="🧪", tagline="Reproduce before judging."),
        ProfileIdea("content_worker", "Content Worker", "workers", "Drafts or rewrites content to a defined voice and format.", ["social drafts", "copy variants", "summaries", "rewrites"], "community_manager", ["conversational_ai"], "entertainer", ["casual", "sophisticated"], ["content_generation", "summarisation"], ["cli"], avatar_emoji="📝", tagline="Draft fast. Match voice."),
    ]


def profile_ideas(category: str | None = None) -> list[ProfileIdea]:
    ideas = _ideas()
    if category:
        ideas = [idea for idea in ideas if idea.category == category]
    return ideas


def profile_idea_by_id(profile_id: str) -> ProfileIdea | None:
    return next((idea for idea in _ideas() if idea.id == profile_id), None)

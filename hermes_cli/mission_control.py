"""Privacy-minimized Mission Control snapshot for the Hermes dashboard.

This module turns the external Claude Agent blueprint into a live Hermes
operations cockpit.  It deliberately separates:

- static source coverage (what the guide contains), from
- live readiness evidence (what this Hermes runtime currently exposes).

The dashboard should never need to read local files directly or receive raw
logs, prompts, commands, env values, or chat content.  Keep this module as the
single server-only aggregation boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Iterable

from hermes_cli.config import get_hermes_home, load_config

SOURCE_URL = "https://claude-agent-2.vercel.app/"

BLUEPRINT_STEPS: list[dict[str, Any]] = [
    {
        "id": "step-1",
        "number": "01",
        "title": "Create your Telegram bot",
        "domain": "interface",
        "part": "MVP",
        "summary": "Create a BotFather token, collect the allowed Telegram user ID, and lock the bot to the operator.",
        "route": "/channels",
    },
    {
        "id": "step-2",
        "number": "02",
        "title": "Get an LLM key",
        "domain": "model",
        "part": "MVP",
        "summary": "Choose Anthropic, OpenRouter, Ollama, or another model provider through env/config.",
        "route": "/models",
    },
    {
        "id": "step-3",
        "number": "03",
        "title": "Bootstrap the project",
        "domain": "runtime",
        "part": "MVP",
        "summary": "Create the TypeScript project skeleton, dependencies, folders, and gitignore boundary.",
        "route": "/system",
    },
    {
        "id": "step-4",
        "number": "04",
        "title": ".env template",
        "domain": "configuration",
        "part": "MVP",
        "summary": "Centralize provider, Telegram, identity, database, and optional integration settings.",
        "route": "/env",
    },
    {
        "id": "step-5",
        "number": "05",
        "title": "Write your soul",
        "domain": "identity",
        "part": "MVP",
        "summary": "Define the operator-facing personality and personal context without reusing a public prompt verbatim.",
        "route": "/config",
    },
    {
        "id": "step-6",
        "number": "06",
        "title": "Config loader",
        "domain": "configuration",
        "part": "MVP",
        "summary": "Load typed runtime configuration from env and defaults so deployment changes do not touch code.",
        "route": "/config",
    },
    {
        "id": "step-7",
        "number": "07",
        "title": "Tier 1 + 2 — SQLite memory",
        "domain": "memory",
        "part": "MVP",
        "summary": "Store durable facts plus a rolling conversation buffer with summarisation of older messages.",
        "route": "/sessions",
    },
    {
        "id": "step-8",
        "number": "08",
        "title": "Tier 3 — Pinecone semantic memory",
        "domain": "memory",
        "part": "MVP optional",
        "summary": "Add semantic recall across many past conversations with a scoped vector database key.",
        "route": "/system",
    },
    {
        "id": "step-9",
        "number": "09",
        "title": "LLM layer",
        "domain": "model",
        "part": "MVP",
        "summary": "Wrap model calls behind a provider-agnostic chat interface.",
        "route": "/models",
    },
    {
        "id": "step-10",
        "number": "10",
        "title": "Tools — what makes it an agent",
        "domain": "tools",
        "part": "MVP",
        "summary": "Expose safe, typed tools such as shell, files, web, memory, and integrations.",
        "route": "/system",
    },
    {
        "id": "step-11",
        "number": "11",
        "title": "The agent loop (with tool-calling)",
        "domain": "agent-loop",
        "part": "MVP",
        "summary": "Build the prompt, call the LLM, execute tools, append observations, and continue until a response is ready.",
        "route": "/chat",
    },
    {
        "id": "step-12",
        "number": "12",
        "title": "Telegram bot",
        "domain": "interface",
        "part": "MVP",
        "summary": "Run the Telegram adapter with whitelist auth and text/voice message handling.",
        "route": "/channels",
    },
    {
        "id": "step-13",
        "number": "13",
        "title": "Heartbeat scheduler",
        "domain": "automation",
        "part": "MVP",
        "summary": "Schedule proactive check-ins and recurring tasks.",
        "route": "/cron",
    },
    {
        "id": "step-14",
        "number": "14",
        "title": "Tie it together",
        "domain": "runtime",
        "part": "MVP",
        "summary": "Initialize memory, semantic recall, bot adapters, and background schedulers in one entrypoint.",
        "route": "/system",
    },
    {
        "id": "step-15",
        "number": "15",
        "title": "Run it",
        "domain": "runtime",
        "part": "MVP",
        "summary": "Start the agent, message it, inspect errors, and confirm the feedback loop works.",
        "route": "/system",
    },
    {
        "id": "step-16",
        "number": "16",
        "title": "Stream responses",
        "domain": "interface",
        "part": "Beyond MVP",
        "summary": "Stream or edit live responses so long tool chains do not leave the user blind.",
        "route": "/chat",
    },
    {
        "id": "step-17",
        "number": "17",
        "title": "Reflection",
        "domain": "memory",
        "part": "Beyond MVP",
        "summary": "Run a consolidation pass that distills conversations into long-term memory.",
        "route": "/system",
    },
    {
        "id": "step-18",
        "number": "18",
        "title": "Auto-skill creation",
        "domain": "skills",
        "part": "Beyond MVP",
        "summary": "After complex successful work, capture the procedure as a portable SKILL.md.",
        "route": "/skills",
    },
    {
        "id": "step-19",
        "number": "19",
        "title": "Voice transcription",
        "domain": "voice",
        "part": "Beyond MVP",
        "summary": "Transcribe Telegram voice notes and optionally respond with voice.",
        "route": "/channels",
    },
    {
        "id": "step-20",
        "number": "20",
        "title": "Multi-user mode",
        "domain": "interface",
        "part": "Beyond MVP",
        "summary": "Namespace memory and message state per authorized user or team member.",
        "route": "/pairing",
    },
    {
        "id": "step-21",
        "number": "21",
        "title": "MCP server integration",
        "domain": "tools",
        "part": "Production-grade",
        "summary": "Attach pre-built MCP servers such as Gmail, Notion, Slack, Supabase, Linear, and GitHub.",
        "route": "/mcp",
    },
    {
        "id": "step-22",
        "number": "22",
        "title": "Permission / approval flow",
        "domain": "safety",
        "part": "Production-grade",
        "summary": "Require approval for destructive or expensive tools and keep an auto-accept escape hatch explicit.",
        "route": "/system",
    },
    {
        "id": "step-22-5",
        "number": "22½",
        "title": "Prompt-injection defence",
        "domain": "safety",
        "part": "Production-grade",
        "summary": "Treat tool outputs as untrusted, wrap them in markers, and prevent scraped/email content from issuing commands.",
        "route": "/system",
    },
    {
        "id": "step-23",
        "number": "23",
        "title": "Cost & token tracking",
        "domain": "analytics",
        "part": "Production-grade",
        "summary": "Record per-request token usage and cost so model bills do not surprise the operator.",
        "route": "/analytics",
    },
    {
        "id": "step-24",
        "number": "24",
        "title": "Mission Control dashboard",
        "domain": "dashboard",
        "part": "Production-grade",
        "summary": "Expose memory, task, cost, and runtime state through a Vite/React cockpit.",
        "route": "/mission-control",
    },
    {
        "id": "step-25",
        "number": "25",
        "title": "Hosting in production",
        "domain": "hosting",
        "part": "Production-grade",
        "summary": "Choose Docker on a VPS, Railway, or systemd/home-server hosting with safe networking.",
        "route": "/system",
    },
    {
        "id": "step-26",
        "number": "26",
        "title": "Testing",
        "domain": "quality",
        "part": "Production-grade",
        "summary": "Unit-test deterministic pieces, snapshot-test prompts, and run fake-LLM integration tests.",
        "route": "/system",
    },
]

HERMES_FEATURES: list[dict[str, str]] = [
    {"id": "H1", "title": "4-layer memory", "summary": "MEMORY.md / USER.md / SKILL.md / SQLite+FTS5 separates facts, preferences, procedures, and episodic recall.", "where": "Step 7, 17, 18", "domain": "memory"},
    {"id": "H2", "title": "GEPA reflection", "summary": "Nightly dreaming pass consolidates conversations into core memory.", "where": "Step 17", "domain": "memory"},
    {"id": "H3", "title": "Auto-skill creation", "summary": "After 5+ tool calls, the agent writes SKILL.md so future similar work is faster.", "where": "Step 18", "domain": "skills"},
    {"id": "H4", "title": "15+ messaging gateways", "summary": "Telegram, Discord, Slack, WhatsApp, Signal, Matrix, Email, SMS, iMessage, DingTalk, Feishu, and more.", "where": "Step 12 → extend", "domain": "interface"},
    {"id": "H5", "title": "6 deploy backends", "summary": "Local, Docker, SSH, Daytona, Singularity, and Modal execution backends.", "where": "Step 25", "domain": "hosting"},
    {"id": "H6", "title": "Real-time voice", "summary": "Voice in/out via CLI, Telegram, and Discord.", "where": "Step 19", "domain": "voice"},
    {"id": "H7", "title": "Pluggable memory backends", "summary": "Swap memory engines such as Mem0, Honcho, or Byterover without rewriting the agent.", "where": "Custom adapter", "domain": "memory"},
    {"id": "H8", "title": "Skill trust levels", "summary": "Builtin / Official / Trusted / Community source trust gradient for permissions.", "where": "Step 22", "domain": "safety"},
    {"id": "H9", "title": "Bounded memory budgets", "summary": "Hard caps force durable consolidation instead of prompt bloat.", "where": "Step 7 + 17", "domain": "memory"},
    {"id": "H10", "title": "TokenMix optimisation", "summary": "Reduce redundant reasoning/token paths for faster multi-step work.", "where": "Advanced", "domain": "analytics"},
    {"id": "H11", "title": "agentskills.io standard", "summary": "Skills portable across Hermes, Claude Code, Cursor, and Codex.", "where": "Step 18", "domain": "skills"},
]

OPENCLAW_FEATURES: list[dict[str, str]] = [
    {"id": "O1", "title": "22 messaging channels", "summary": "Every Hermes adapter plus iMessage, Nostr, IRC, WeChat, Twitch, and Google Chat.", "where": "Step 12 → extend", "domain": "interface"},
    {"id": "O2", "title": "Native mobile clients", "summary": "macOS, iOS, and Android clients with voice wake-word.", "where": "Out of scope", "domain": "interface"},
    {"id": "O3", "title": "ClawHub skill registry", "summary": "Distribute and install third-party skills publicly.", "where": "Step 18", "domain": "skills"},
    {"id": "O4", "title": "Multi-agent orchestration", "summary": "Spawn sub-agents in parallel for delegated tasks.", "where": "Custom — fork agent.ts", "domain": "agent-loop"},
    {"id": "O5", "title": "Sandboxed tool execution", "summary": "Run shell commands in Docker / SSH / OpenShell-style isolation.", "where": "Step 22 + 25", "domain": "safety"},
    {"id": "O6", "title": "Open Gateway Protocol", "summary": "Cross-harness federation so agents can talk to other agents.", "where": "Out of scope", "domain": "interface"},
    {"id": "O7", "title": "Per-command approval flow", "summary": "Inline approve/deny flow for destructive tool calls.", "where": "Step 22", "domain": "safety"},
    {"id": "O8", "title": "Auto-approve toggle", "summary": "Trust-level escape hatch when the operator does not want to babysit safe calls.", "where": "Step 22", "domain": "safety"},
    {"id": "O9", "title": "Live Canvas UI", "summary": "Visual editor where the agent edits files in real time.", "where": "Step 24", "domain": "dashboard"},
    {"id": "O10", "title": "Tailscale-recommended self-host", "summary": "Mesh-VPN to a home server with no public ports.", "where": "Step 25", "domain": "hosting"},
]

ARCHITECTURE_PIECES: list[dict[str, str]] = [
    {"id": "agent-loop", "title": "Agent loop", "summary": "Prompt bauen, Modell aufrufen, Tools ausführen, Beobachtungen zurückführen.", "route": "/chat"},
    {"id": "memory", "title": "3-tier memory", "summary": "Profil-/Core-Dateien, SQLite-Episoden und optional semantische Vektoren.", "route": "/sessions"},
    {"id": "tools", "title": "Tool system", "summary": "Typed tools, Toolsets, MCP und kontrollierte Ausführung statt reiner Chatbot.", "route": "/system"},
    {"id": "llm", "title": "LLM layer", "summary": "Provider-/Modellrouting mit Reasoning-, Kosten- und Token-Signalen.", "route": "/models"},
    {"id": "telegram", "title": "Messaging gateway", "summary": "Telegram/weitere Adapter mit Whitelist, Pairing und Home-Channel-Konzept.", "route": "/channels"},
    {"id": "scheduler", "title": "Heartbeat scheduler", "summary": "Cronjobs für proaktive Checks, Reflection und wiederkehrende Workflows.", "route": "/cron"},
    {"id": "mcp", "title": "MCP bridge", "summary": "Externe Toolserver werden namenspaced und sicher in den Toolkatalog aufgenommen.", "route": "/mcp"},
]

PREREQUISITES: list[dict[str, str]] = [
    {"id": "node", "title": "Node 20+", "summary": "Für das React/Vite-Dashboard und Plugin-Frontend.", "route": "/system"},
    {"id": "telegram", "title": "Telegram account", "summary": "Messaging-Interface und optionale Voice Notes.", "route": "/channels"},
    {"id": "llm-key", "title": "LLM API key", "summary": "Anthropic/OpenRouter/OpenAI/Ollama oder lokaler Provider.", "route": "/models"},
    {"id": "pinecone", "title": "Optional Pinecone", "summary": "Semantische Erinnerung nur wenn Key/Index vorhanden sind.", "route": "/system"},
    {"id": "supabase", "title": "Optional Supabase", "summary": "Nur anzeigen wenn konfiguriert; kein Secret-Wert im Snapshot.", "route": "/env"},
]

PREFLIGHT_CHECKS: list[dict[str, str]] = [
    {"id": "env_gitignored", "title": ".env nicht committen", "summary": "Env-Datei muss in .gitignore abgedeckt sein.", "route": "/env"},
    {"id": "rotatable_secrets", "title": "Keys rotierbar", "summary": "Dashboard zeigt nur Präsenz/Familien, keine Werte.", "route": "/env"},
    {"id": "allowed_users", "title": "User whitelist", "summary": "Telegram/Pairing müssen auf erlaubte Nutzer begrenzt sein.", "route": "/channels"},
    {"id": "private_chats", "title": "Group chats rejected", "summary": "Gateway sollte private Chat Boundaries unterstützen.", "route": "/channels"},
    {"id": "sandbox_shell", "title": "Shell sandbox", "summary": "Destruktive Shell/File-Tools brauchen Isolation oder Approval.", "route": "/system"},
    {"id": "file_allowlist", "title": "File allowlist", "summary": "Dateizugriff sollte bewusst scoped sein.", "route": "/system"},
    {"id": "dashboard_auth", "title": "Dashboard auth + bind", "summary": "Mission Control muss token-/cookie-geschützt sein und nicht blind öffentlich exponiert werden.", "route": "/mission-control"},
    {"id": "cost_threshold", "title": "Cost alerts", "summary": "Budgetschwellen schützen vor stiller Modellkosten-Eskalation.", "route": "/analytics"},
    {"id": "pinecone_scoped", "title": "Pinecone key scoped", "summary": "Projekt- statt Account-Key; Verifikation ist meist manuell.", "route": "/system"},
    {"id": "voice_key_separation", "title": "Whisper key separation", "summary": "OpenAI Whisper darf nicht implizit Anthropic/OpenRouter-Keys verwenden.", "route": "/channels"},
    {"id": "approval_flow", "title": "Approval flow", "summary": "Destruktive/teure Tools müssen explizit freigegeben werden.", "route": "/system"},
]

CUSTOMIZATION_CHECKLIST: list[dict[str, str]] = [
    {"id": "soul", "title": "Soul / user profile", "summary": "Persönliche Stimme und stabile Präferenzen vorhanden.", "route": "/config"},
    {"id": "tools", "title": "Tools", "summary": "Nur Toolsets aktivieren, die echten Operator-Nutzen haben.", "route": "/system"},
    {"id": "heartbeats", "title": "Heartbeats", "summary": "Morning kick, Recap oder Review als Cronjobs.", "route": "/cron"},
    {"id": "memory-categories", "title": "Memory categories", "summary": "Core facts, User prefs, Skills und Episoden sauber getrennt.", "route": "/sessions"},
    {"id": "skill-seeds", "title": "Skill seeds", "summary": "Wiederkehrende Workflows als SKILL.md verfügbar.", "route": "/skills"},
    {"id": "model-choice", "title": "Model choice", "summary": "Taste/Reasoning/Bulk/Private bewusst routen.", "route": "/models"},
    {"id": "hosting", "title": "Hosting", "summary": "Docker/VPS/systemd/Railway-Entscheidung und Netzgrenze.", "route": "/system"},
    {"id": "reflection-prompt", "title": "Reflection prompt", "summary": "Consolidation-Shape passt zur Domäne.", "route": "/cron"},
    {"id": "approval-threshold", "title": "Approval threshold", "summary": "Welche Tools manuell bestätigt werden müssen.", "route": "/system"},
]

NEXT_TOOLS: list[dict[str, str]] = [
    {"id": "youtube", "title": "YouTube tool", "summary": "Search, transcript, comments.", "route": "/skills"},
    {"id": "gmail", "title": "Gmail via MCP", "summary": "Read, draft, label.", "route": "/mcp"},
    {"id": "calendar", "title": "Calendar via MCP", "summary": "List, create, suggest times.", "route": "/mcp"},
    {"id": "notion", "title": "Notion via MCP", "summary": "Search, create, update pages.", "route": "/mcp"},
    {"id": "invoice", "title": "Invoice generator", "summary": "Branded PDF generation.", "route": "/skills"},
    {"id": "web-research", "title": "Web research", "summary": "Firecrawl/Perplexity-style research.", "route": "/system"},
    {"id": "bank", "title": "Bank summariser", "summary": "Finance ingestion stays opt-in and approval-gated.", "route": "/mcp"},
    {"id": "meeting", "title": "Meeting transcriber", "summary": "Granola/Otter ingest or local audio pipeline.", "route": "/skills"},
]

GLOSSARY: list[dict[str, str]] = [
    {"id": "agent-loop", "term": "Agent loop", "definition": "Reasoning/action loop that keeps using tools until a final answer is ready."},
    {"id": "core-memory", "term": "Core memory", "definition": "Small durable facts injected into future turns."},
    {"id": "conversation-buffer", "term": "Conversation buffer", "definition": "Recent messages available for continuity and recall."},
    {"id": "semantic-memory", "term": "Semantic memory", "definition": "Vector search over prior conversations or knowledge."},
    {"id": "soul", "term": "Soul / system prompt", "definition": "Operator identity, style, safety and tool-use contract."},
    {"id": "heartbeat", "term": "Heartbeat", "definition": "Scheduled proactive job that can message or act later."},
    {"id": "tool", "term": "Tool", "definition": "Typed external capability the model can call."},
    {"id": "mcp", "term": "MCP", "definition": "Model Context Protocol for attaching external tools."},
    {"id": "reflection", "term": "Reflection", "definition": "Consolidation pass that extracts reusable facts and lessons."},
    {"id": "skill", "term": "Skill", "definition": "Portable procedure file loaded on demand."},
    {"id": "progressive-disclosure", "term": "Progressive disclosure", "definition": "Load summaries first, full skill/reference only when needed."},
]

TROUBLESHOOTING: list[dict[str, str]] = [
    {"id": "not-authorised", "symptom": "Bot replies Not authorised", "cause": "Telegram ID missing from allowed users", "fix": "Update whitelist and restart gateway", "route": "/channels"},
    {"id": "mid-task-stop", "symptom": "Agent stops mid-task", "cause": "Iteration/turn budget or tool spiral", "fix": "Inspect sessions/tool counts and raise cap intentionally", "route": "/sessions"},
    {"id": "message-not-modified", "symptom": "message is not modified", "cause": "Streaming edit repeated identical content", "fix": "Debounce and skip unchanged edits", "route": "/channels"},
    {"id": "telegram-429", "symptom": "Telegram 429", "cause": "Edits faster than rate limits", "fix": "Debounce to roughly one second", "route": "/channels"},
    {"id": "pinecone-empty", "symptom": "Pinecone empty results", "cause": "Index not initialised or wrong index", "fix": "Verify key/index presence and wait for readiness", "route": "/system"},
    {"id": "never-calls-tools", "symptom": "Agent never calls tools", "cause": "Tool descriptions/tool_choice/disabled toolset", "fix": "Inspect enabled toolsets and descriptions", "route": "/system"},
    {"id": "reflection-wipes", "symptom": "Reflection wipes memory", "cause": "Existing facts not injected", "fix": "Reflection prompt must merge, not replace", "route": "/cron"},
    {"id": "voice-empty", "symptom": "Voice transcription empty", "cause": "Opus/ffmpeg/provider mismatch", "fix": "Verify STT provider and conversion path", "route": "/channels"},
    {"id": "heartbeat-time", "symptom": "Heartbeat wrong time", "cause": "Server timezone mismatch", "fix": "Set timezone in job/config", "route": "/cron"},
    {"id": "raw-json-tools", "symptom": "Tool calls return raw JSON", "cause": "Formatter/parsing boundary missing", "fix": "Format tool results consistently as data", "route": "/system"},
]

RESOURCES: list[dict[str, str]] = [
    {"id": "anthropic-tools", "title": "Anthropic Tool Use docs", "url": "https://docs.anthropic.com/"},
    {"id": "openai-functions", "title": "OpenAI Function Calling", "url": "https://platform.openai.com/docs/"},
    {"id": "mcp", "title": "Model Context Protocol", "url": "https://modelcontextprotocol.io/"},
    {"id": "telegraf", "title": "Telegraf", "url": "https://telegraf.js.org/"},
    {"id": "pinecone", "title": "Pinecone Node SDK", "url": "https://docs.pinecone.io/"},
    {"id": "openrouter", "title": "OpenRouter", "url": "https://openrouter.ai/docs"},
]

DATA_FLOW_SURFACES: list[dict[str, str]] = [
    {"id": "telegram", "label": "Telegram", "dataSent": "messages / voice / files", "retention": "Telegram-side retention; not E2E for bot chats"},
    {"id": "llm", "label": "LLM provider", "dataSent": "prompt + selected memory/context", "retention": "provider-specific"},
    {"id": "pinecone", "label": "Pinecone", "dataSent": "embeddings / metadata", "retention": "until index deletion"},
    {"id": "whisper", "label": "OpenAI Whisper", "dataSent": "voice-note audio when STT is enabled", "retention": "provider-specific"},
    {"id": "sqlite", "label": "Local SQLite", "dataSent": "local conversation/session metadata", "retention": "local disk until deleted"},
]

_STATE_WEIGHT = {"active": 100, "partial": 68, "watch": 38, "planned": 18}

KNOWN_PROVIDER_FAMILIES = {
    "anthropic",
    "auto",
    "copilot",
    "deepseek",
    "gemini",
    "github-copilot",
    "google",
    "grok",
    "huggingface",
    "kilocode",
    "kimi",
    "local",
    "mistral",
    "moonshot",
    "nous",
    "ollama",
    "openai",
    "openai-codex",
    "opencode-go",
    "opencode-zen",
    "openrouter",
    "qwen-oauth",
    "xai",
    "zai",
}

KNOWN_VOICE_PROVIDERS = {
    "edge",
    "elevenlabs",
    "groq",
    "local",
    "minimax",
    "mistral",
    "neutts",
    "openai",
    "whisper",
}

KNOWN_GATEWAY_FAMILIES = {
    "api",
    "discord",
    "email",
    "feishu",
    "homeassistant",
    "local",
    "matrix",
    "signal",
    "slack",
    "sms",
    "telegram",
    "web",
    "whatsapp",
    "yuanbao",
}

SAFE_SESSION_END_REASONS = {
    "completed",
    "done",
    "error",
    "interrupted",
    "max_iterations",
    "max_turns",
    "stopped",
    "timeout",
    "tool_error",
}

SAFE_HANDOFF_STATES = {"pending", "ready", "sent", "failed", "cancelled", "completed"}
SAFE_CRON_STATUSES = {"success", "ok", "error", "failed", "skipped", "running", "pending", "timeout"}
SAFE_MODEL_WORDS = {
    "ai",
    "anthropic",
    "audio",
    "base",
    "chat",
    "claude",
    "code",
    "coder",
    "codex",
    "dall",
    "deepseek",
    "e",
    "embedding",
    "embeddings",
    "exp",
    "experimental",
    "fast",
    "flash",
    "gemini",
    "gemma",
    "glm",
    "google",
    "gpt",
    "4o",
    "grok",
    "haiku",
    "hermes",
    "high",
    "instruct",
    "kimi",
    "large",
    "latest",
    "llama",
    "medium",
    "meta",
    "mini",
    "mistral",
    "mixtral",
    "moonshot",
    "nous",
    "o",
    "omni",
    "online",
    "openai",
    "openrouter",
    "opus",
    "oss",
    "preview",
    "pro",
    "qwen",
    "realtime",
    "reasoning",
    "search",
    "small",
    "sonnet",
    "thinking",
    "transcribe",
    "turbo",
    "vision",
    "xai",
    "z",
}


@dataclass(frozen=True)
class CapabilityState:
    state: str
    score: int
    evidence: list[str]
    next: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except Exception:
        return default


def _home() -> Path:
    return Path(get_hermes_home()).expanduser().resolve()


def _compact_path(path: Path | str | None, home: Path | None = None) -> str | None:
    if path is None:
        return None
    home = home or _home()
    raw = str(path)
    try:
        p = Path(raw).expanduser().resolve()
        if p == home:
            return "~/.hermes"
        if p.is_relative_to(home):
            rel = p.relative_to(home)
            return "~/.hermes" if str(rel) == "." else f"~/.hermes/{rel.as_posix()}"
        user_home = Path.home().resolve()
        if p == user_home:
            return "~"
        if p.is_relative_to(user_home):
            return f"~/{p.relative_to(user_home).as_posix()}"
    except Exception:
        pass
    # Do not expose arbitrary absolute paths.  Keep only the terminal label.
    name = Path(raw).name
    return name or "configured"


def _source_label(value: Any) -> str | None:
    """Return a platform/source family without exposing IDs or topic keys."""
    raw = str(value or "").strip().lower()
    if not raw:
        return None
    known = {
        "api",
        "cli",
        "cron",
        "discord",
        "email",
        "feishu",
        "gateway",
        "homeassistant",
        "local",
        "matrix",
        "signal",
        "slack",
        "sms",
        "telegram",
        "web",
        "whatsapp",
        "yuanbao",
    }
    for name in known:
        if raw == name or raw.startswith(f"{name}:") or raw.startswith(f"{name}/"):
            return name
    return "other"


def _safe_model_label(value: Any) -> str:
    """Return a model label without exposing local paths or private deployment names."""
    raw = str(value or "auto").strip()
    if not raw:
        return "auto"
    lower = raw.lower()
    looks_like_path = (
        raw.startswith(("/", "~", "./", "../"))
        or lower.startswith("file://")
        or "\\" in raw
        or (len(raw) > 2 and raw[1:3] == ":\\")
        or lower.endswith((".gguf", ".safetensors", ".bin", ".pt", ".pth", ".onnx"))
    )
    if looks_like_path:
        return "local-model"
    if "://" in lower:
        return "custom-model"
    if lower in {"auto", "local-model"}:
        return lower
    if lower == "local":
        return "local-model"
    if lower.startswith(("custom:", "custom/", "custom-")):
        return "custom-model"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/+:-]*", raw):
        return "custom-model"

    # Exact model names can include provider and version separators.  Keep them
    # only when every token is from a public model/provider vocabulary or a pure
    # version token (e.g. ``gpt-5.5``, ``anthropic/claude-sonnet-4``).  Unknown
    # words often encode private deployments, project names, paths, or clients.
    tokens = [token for token in re.split(r"[^a-z0-9.]+", lower) if token]
    version_token = re.compile(r"^(?:v?\d+(?:\.\d+)*|\d+(?:\.\d+)?[bkmt])$")
    public_family_version_token = re.compile(r"^(?:qwen|llama|gemma|glm|o|r)\d+(?:\.\d+)*$")
    if tokens and all(token in SAFE_MODEL_WORDS or version_token.match(token) or public_family_version_token.match(token) for token in tokens):
        return raw
    return "custom-model"


def _safe_choice_label(value: Any, *, allowed: set[str], default: str = "custom") -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    if not raw:
        return default
    return raw if raw in allowed else default


def _safe_family_label(value: Any, *, known: set[str], default: str = "custom") -> str | None:
    """Collapse user-defined provider/platform labels to stable families."""
    raw = str(value or "").strip()
    if not raw:
        return None
    lower = raw.lower()
    if lower.startswith(("/", "~", "./", "../", "file://")) or "://" in lower:
        return "local" if default == "provider" else "custom"
    if lower.startswith("custom"):
        return "custom"
    if lower in known:
        return lower
    for family in sorted(known, key=len, reverse=True):
        if lower == family or lower.startswith(f"{family}:") or lower.startswith(f"{family}/"):
            return family
    return default


def _safe_provider_label(value: Any) -> str:
    label = _safe_family_label(value, known=KNOWN_PROVIDER_FAMILIES, default="custom")
    return label or "auto"


def _safe_voice_label(value: Any) -> str | None:
    return _safe_family_label(value, known=KNOWN_VOICE_PROVIDERS, default="custom")


def _safe_gateway_label(value: Any) -> str:
    label = _source_label(value)
    if label in KNOWN_GATEWAY_FAMILIES:
        return label
    return "other"


def _safe_service_manager_label(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if "systemd" in raw:
        return "systemd"
    if "launchd" in raw:
        return "launchd"
    if "s6" in raw:
        return "s6"
    if "docker" in raw:
        return "docker"
    if "termux" in raw:
        return "termux"
    if "manual" in raw:
        return "manual"
    return "unknown"


def _safe_service_scope(value: Any) -> str | None:
    raw = str(value or "").strip().lower()
    return raw if raw in {"user", "system", "container", "manual"} else None


def _safe_status(value: Any, *, allowed: set[str], default: str = "other") -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    if not raw:
        return "unknown"
    return raw if raw in allowed else default


def _toolset_bucket(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return "unknown"
    if raw.startswith("mcp") or raw.startswith("mcp-") or raw.startswith("mcp:"):
        return "mcp"
    if raw.startswith("plugin") or raw.startswith("plugin-") or raw.startswith("plugin:"):
        return "plugin"
    builtin = {
        "browser", "code_execution", "cronjob", "delegation", "file", "memory",
        "search", "session_search", "skills", "terminal", "todo", "tts", "vision", "web",
        "image_gen", "video", "messaging", "clarify", "kanban", "spotify",
        "homeassistant", "discord", "discord_admin", "feishu_doc", "feishu_drive", "yuanbao",
    }
    return "builtin" if raw in builtin else "custom"


def _increment(mapping: dict[str, int], key: str, amount: int = 1) -> None:
    mapping[key] = mapping.get(key, 0) + amount


def _parse_timestamp(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return float(raw)
    except Exception:
        pass
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _age_bucket(seconds: int | None) -> str:
    if seconds is None:
        return "never"
    if seconds < 3600:
        return "under_1h"
    if seconds < 24 * 3600:
        return "under_24h"
    if seconds < 7 * 24 * 3600:
        return "under_7d"
    return "over_7d"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _env_families(home: Path) -> dict[str, Any]:
    env_path = home / ".env"
    raw_keys: dict[str, str] = {}
    present_keys = 0
    configured = 0
    if env_path.exists():
        for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"\'')
            if not key:
                continue
            present_keys += 1
            raw_keys[key.upper()] = value
            if value:
                configured += 1

    def present(*names: str) -> bool:
        return any(bool(raw_keys.get(name.upper())) for name in names)

    def count_csv(*names: str) -> int:
        for name in names:
            value = raw_keys.get(name.upper(), "")
            if value:
                return len([p for p in value.replace(";", ",").split(",") if p.strip()])
        return 0

    telegram_allowlist_keys = (
        "ALLOWED_USER_IDS",
        "TELEGRAM_ALLOWED_USERS",
        "TELEGRAM_ALLOWED_USER_IDS",
        "TELEGRAM_GROUP_ALLOWED_USERS",
        "GATEWAY_ALLOWED_USERS",
    )

    family_tests = {
        "telegram": lambda k: "TELEGRAM" in k or k in {"ALLOWED_USER_IDS", "GATEWAY_ALLOWED_USERS"},
        "discord": lambda k: "DISCORD" in k,
        "slack": lambda k: "SLACK" in k,
        "matrix": lambda k: "MATRIX" in k,
        "signal": lambda k: "SIGNAL" in k,
        "email": lambda k: any(s in k for s in ["SMTP", "IMAP", "EMAIL"]),
        "sms": lambda k: any(s in k for s in ["TWILIO", "SMS"]),
        "openrouter": lambda k: "OPENROUTER" in k,
        "anthropic": lambda k: "ANTHROPIC" in k,
        "openai": lambda k: "OPENAI" in k or "WHISPER" in k,
        "ollama": lambda k: "OLLAMA" in k,
        "pinecone": lambda k: "PINECONE" in k,
        "supabase": lambda k: "SUPABASE" in k,
        "gemini": lambda k: "GEMINI" in k or "GOOGLE" in k,
        "tailscale": lambda k: "TAILSCALE" in k or "TS_" in k,
        "homeassistant": lambda k: "HASS" in k or "HOMEASSISTANT" in k,
    }
    families = sorted({family for key, value in raw_keys.items() if value for family, test in family_tests.items() if test(key)})
    llm_families = [f for f in families if f in {"anthropic", "openai", "openrouter", "gemini", "ollama"}]
    required = [
        {"key": "TELEGRAM_BOT_TOKEN", "category": "telegram", "isSet": present("TELEGRAM_BOT_TOKEN")},
        {"key": "ALLOWED_USER_IDS", "category": "telegram", "isSet": present(*telegram_allowlist_keys), "count": count_csv(*telegram_allowlist_keys)},
        {"key": "LLM_API_KEY", "category": "model", "isSet": bool(llm_families)},
        {"key": "DASHBOARD_TOKEN", "category": "dashboard", "isSet": present("DASHBOARD_TOKEN", "HERMES_DASHBOARD_TOKEN")},
    ]
    optional = [
        {"key": "PINECONE_API_KEY", "category": "semantic", "isSet": present("PINECONE_API_KEY")},
        {"key": "PINECONE_INDEX", "category": "semantic", "isSet": present("PINECONE_INDEX", "PINECONE_INDEX_NAME")},
        {"key": "OPENAI_API_KEY", "category": "voice", "isSet": present("OPENAI_API_KEY")},
        {"key": "SUPABASE_URL", "category": "storage", "isSet": present("SUPABASE_URL")},
        {"key": "TAILSCALE_AUTHKEY", "category": "network", "isSet": present("TAILSCALE_AUTHKEY")},
    ]
    return {
        "filePresent": env_path.exists(),
        "path": _compact_path(env_path, home),
        "presentKeys": present_keys,
        "configuredKeys": configured,
        "families": families,
        "llmFamilies": llm_families,
        "requiredKeys": required,
        "optionalKeys": optional,
        "telegram": {
            "tokenPresent": present("TELEGRAM_BOT_TOKEN"),
            "allowedUserCount": count_csv(*telegram_allowlist_keys),
        },
        "dashboard": {"tokenPresent": present("DASHBOARD_TOKEN", "HERMES_DASHBOARD_TOKEN")},
        "semantic": {"pineconeKeyPresent": present("PINECONE_API_KEY"), "pineconeIndexPresent": present("PINECONE_INDEX", "PINECONE_INDEX_NAME")},
        "voice": {"openaiKeyPresent": present("OPENAI_API_KEY")},
        "network": {"tailscaleSignalPresent": present("TAILSCALE_AUTHKEY") or "tailscale" in families},
    }

def _state_db_metrics(home: Path) -> dict[str, Any]:
    path = home / "state.db"
    base: dict[str, Any] = {
        "dbPresent": path.exists(),
        "path": _compact_path(path, home),
        "total": 0,
        "active": 0,
        "archived": 0,
        "messages": 0,
        "activeMessages": 0,
        "toolCalls": 0,
        "inputTokens": 0,
        "outputTokens": 0,
        "cacheReadTokens": 0,
        "cacheWriteTokens": 0,
        "reasoningTokens": 0,
        "apiCalls": 0,
        "estimatedCostUsd": 0.0,
        "actualCostUsd": 0.0,
        "todayEstimatedCostUsd": 0.0,
        "todayActualCostUsd": 0.0,
        "sources": {},
        "recent": [],
        "latestAgeSeconds": None,
        "ftsPresent": False,
        "trigramFtsPresent": False,
        "summaries": 0,
        "stateMetaRows": 0,
        "endReasonCounts": {},
        "childSessionCount": 0,
        "rootSessionCount": 0,
        "complexSessionCount": 0,
        "handoffStateCounts": {},
        "rewindTotal": 0,
        "avgApiCalls": 0,
        "maxApiCalls": 0,
        "roleCounts": {},
        "delegateTaskCalls": 0,
    }
    if not path.exists():
        return base
    con: sqlite3.Connection | None = None
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
        base["ftsPresent"] = any("fts" in name.lower() for name in tables)
        base["trigramFtsPresent"] = any("trigram" in name.lower() for name in tables)
        if "sessions" in tables:
            cols = {r[1] for r in con.execute("PRAGMA table_info(sessions)")}
            base["total"] = _safe_int(con.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
            if "archived" in cols:
                base["archived"] = _safe_int(con.execute("SELECT COUNT(*) FROM sessions WHERE archived=1").fetchone()[0])
                base["active"] = max(base["total"] - base["archived"], 0)
            else:
                base["active"] = base["total"]
            for column, key in [
                ("tool_call_count", "toolCalls"),
                ("input_tokens", "inputTokens"),
                ("output_tokens", "outputTokens"),
                ("cache_read_tokens", "cacheReadTokens"),
                ("cache_write_tokens", "cacheWriteTokens"),
                ("reasoning_tokens", "reasoningTokens"),
                ("api_call_count", "apiCalls"),
            ]:
                if column in cols:
                    base[key] = _safe_int(con.execute(f"SELECT COALESCE(SUM({column}), 0) FROM sessions").fetchone()[0])
            for column, key in [("estimated_cost_usd", "estimatedCostUsd"), ("actual_cost_usd", "actualCostUsd")]:
                if column in cols:
                    base[key] = round(float(con.execute(f"SELECT COALESCE(SUM({column}), 0) FROM sessions").fetchone()[0] or 0), 4)
            if "started_at" in cols:
                day_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
                if "estimated_cost_usd" in cols:
                    base["todayEstimatedCostUsd"] = round(float(con.execute("SELECT COALESCE(SUM(estimated_cost_usd), 0) FROM sessions WHERE started_at >= ?", (day_start,)).fetchone()[0] or 0), 4)
                if "actual_cost_usd" in cols:
                    base["todayActualCostUsd"] = round(float(con.execute("SELECT COALESCE(SUM(actual_cost_usd), 0) FROM sessions WHERE started_at >= ?", (day_start,)).fetchone()[0] or 0), 4)
            if "end_reason" in cols:
                end_counts: dict[str, int] = {}
                for row in con.execute("SELECT end_reason, COUNT(*) FROM sessions WHERE COALESCE(end_reason, '') != '' GROUP BY end_reason"):
                    _increment(end_counts, _safe_status(row[0], allowed=SAFE_SESSION_END_REASONS), _safe_int(row[1]))
                base["endReasonCounts"] = end_counts
            if "parent_session_id" in cols:
                base["childSessionCount"] = _safe_int(con.execute("SELECT COUNT(*) FROM sessions WHERE COALESCE(parent_session_id, '') != ''").fetchone()[0])
                base["rootSessionCount"] = max(base["total"] - base["childSessionCount"], 0)
            else:
                base["rootSessionCount"] = base["total"]
            complexity_clauses = []
            if "message_count" in cols:
                complexity_clauses.append("message_count >= 20")
            if "tool_call_count" in cols:
                complexity_clauses.append("tool_call_count >= 10")
            if "api_call_count" in cols:
                complexity_clauses.append("api_call_count >= 10")
            if complexity_clauses:
                complexity_query = "SELECT COUNT(*) FROM sessions WHERE " + " OR ".join(complexity_clauses)
                base["complexSessionCount"] = _safe_int(con.execute(complexity_query).fetchone()[0])
            if "handoff_state" in cols:
                handoff_counts: dict[str, int] = {}
                for row in con.execute("SELECT handoff_state, COUNT(*) FROM sessions WHERE COALESCE(handoff_state, '') != '' GROUP BY handoff_state"):
                    _increment(handoff_counts, _safe_status(row[0], allowed=SAFE_HANDOFF_STATES), _safe_int(row[1]))
                base["handoffStateCounts"] = handoff_counts
            if "rewind_count" in cols:
                base["rewindTotal"] = _safe_int(con.execute("SELECT COALESCE(SUM(rewind_count), 0) FROM sessions").fetchone()[0])
            if "api_call_count" in cols:
                values = con.execute("SELECT COALESCE(AVG(api_call_count), 0), COALESCE(MAX(api_call_count), 0) FROM sessions").fetchone()
                base["avgApiCalls"] = round(float(values[0] or 0), 2)
                base["maxApiCalls"] = _safe_int(values[1])
            if "source" in cols:
                source_counts: dict[str, int] = {}
                for row in con.execute("SELECT source, COUNT(*) FROM sessions GROUP BY source ORDER BY COUNT(*) DESC LIMIT 50"):
                    label = _source_label(row[0])
                    if label:
                        source_counts[label] = source_counts.get(label, 0) + _safe_int(row[1])
                base["sources"] = dict(sorted(source_counts.items(), key=lambda item: item[1], reverse=True)[:12])
            order_col = "started_at" if "started_at" in cols else "rowid"
            select_cols = [c for c in ["source", "model", "started_at", "message_count", "tool_call_count"] if c in cols]
            if select_cols:
                recent = []
                for row in con.execute(f"SELECT {', '.join(select_cols)} FROM sessions ORDER BY {order_col} DESC LIMIT 6"):
                    raw = dict(row)
                    item: dict[str, Any] = {}
                    if raw.get("source"):
                        item["source"] = _source_label(raw.get("source")) or "other"
                    if raw.get("model"):
                        item["model"] = _safe_model_label(raw.get("model"))
                    if "message_count" in raw:
                        item["messageCount"] = _safe_int(raw.get("message_count"), 0)
                    if "tool_call_count" in raw:
                        item["toolCallCount"] = _safe_int(raw.get("tool_call_count"), 0)
                    started = raw.get("started_at")
                    if isinstance(started, (int, float)):
                        item["startedAgeSeconds"] = max(0, int(time.time() - float(started)))
                        if base["latestAgeSeconds"] is None:
                            base["latestAgeSeconds"] = item["startedAgeSeconds"]
                    recent.append(item)
                base["recent"] = recent
        if "messages" in tables:
            msg_cols = {r[1] for r in con.execute("PRAGMA table_info(messages)")}
            base["messages"] = _safe_int(con.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
            if "active" in msg_cols:
                base["activeMessages"] = _safe_int(con.execute("SELECT COUNT(*) FROM messages WHERE active=1").fetchone()[0])
            else:
                base["activeMessages"] = base["messages"]
            if "role" in msg_cols:
                role_counts: dict[str, int] = {}
                for row in con.execute("SELECT role, COUNT(*) FROM messages GROUP BY role"):
                    role = str(row[0] or "unknown").lower()
                    if role not in {"user", "assistant", "tool", "system", "developer"}:
                        role = "other"
                    _increment(role_counts, role, _safe_int(row[1]))
                base["roleCounts"] = role_counts
            if "tool_name" in msg_cols:
                base["delegateTaskCalls"] = _safe_int(con.execute("SELECT COUNT(*) FROM messages WHERE lower(COALESCE(tool_name, '')) = 'delegate_task'").fetchone()[0])
        if "summaries" in tables:
            base["summaries"] = _safe_int(con.execute("SELECT COUNT(*) FROM summaries").fetchone()[0])
        if "state_meta" in tables:
            base["stateMetaRows"] = _safe_int(con.execute("SELECT COUNT(*) FROM state_meta").fetchone()[0])
    except Exception as exc:
        base["error"] = f"state-db-unavailable:{exc.__class__.__name__}"
    finally:
        if con is not None:
            con.close()
    return base

def _memory_metrics(home: Path, sessions: dict[str, Any]) -> dict[str, Any]:
    memory_dir = home / "memories"
    candidates = {
        "memory": [home / "MEMORY.md", home / "memory.md", memory_dir / "MEMORY.md", memory_dir / "memory.md"],
        "user": [home / "USER.md", home / "user.md", memory_dir / "USER.md", memory_dir / "user.md"],
        "soul": [home / "soul.md", home / "SOUL.md"],
    }
    files: dict[str, dict[str, Any]] = {}
    for label, paths in candidates.items():
        existing = [p for p in paths if p.exists()]
        files[label] = {"present": bool(existing), "bytes": sum(p.stat().st_size for p in existing), "locations": [_compact_path(p, home) for p in existing[:3]]}
    return {
        "files": files,
        "sqlite": {
            "dbPresent": bool(sessions.get("dbPresent")),
            "sessions": _safe_int(sessions.get("total")),
            "messages": _safe_int(sessions.get("messages")),
            "ftsPresent": bool(sessions.get("ftsPresent")),
            "trigramFtsPresent": bool(sessions.get("trigramFtsPresent")),
            "summaries": _safe_int(sessions.get("summaries")),
            "activeMessages": _safe_int(sessions.get("activeMessages")),
            "archivedSessions": _safe_int(sessions.get("archived")),
        },
        "budgets": {
            "memoryLimitChars": 2200,
            "userLimitChars": 1375,
            "configured": True,
        },
        "provider": {"active": "builtin", "configuredProviders": ["builtin"]},
    }


def _semantic_metrics(env_info: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    semantic_cfg = cfg.get("semantic") if isinstance(cfg.get("semantic"), dict) else {}
    vector_cfg = cfg.get("vector") if isinstance(cfg.get("vector"), dict) else {}
    raw_provider = semantic_cfg.get("provider") or vector_cfg.get("provider") or ("pinecone" if "pinecone" in env_info.get("families", []) else "none")
    provider = "pinecone" if str(raw_provider).lower() == "pinecone" else ("none" if str(raw_provider).lower() == "none" else _safe_provider_label(raw_provider))
    env_sem = env_info.get("semantic", {}) if isinstance(env_info.get("semantic"), dict) else {}
    index_configured = bool(env_sem.get("pineconeIndexPresent") or semantic_cfg.get("index") or vector_cfg.get("index"))
    return {
        "provider": provider,
        "configured": provider != "none" or bool(env_sem.get("pineconeKeyPresent")) or index_configured,
        "apiKeyPresent": bool(env_sem.get("pineconeKeyPresent")),
        "indexConfigured": index_configured,
        "projectScopedKeyVerified": "unknown" if env_sem.get("pineconeKeyPresent") else "not_configured",
    }


def _analytics_metrics(sessions: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    dashboard = cfg.get("dashboard") if isinstance(cfg.get("dashboard"), dict) else {}
    threshold = dashboard.get("daily_cost_limit_usd") or dashboard.get("cost_alert_threshold_usd") or None
    try:
        threshold_f = float(threshold) if threshold is not None else None
    except Exception:
        threshold_f = None
    today = float(sessions.get("todayActualCostUsd") or sessions.get("todayEstimatedCostUsd") or 0)
    if threshold_f is None:
        state = "not_configured"
    elif today >= threshold_f:
        state = "exceeded"
    elif today >= threshold_f * 0.8:
        state = "warning"
    else:
        state = "ok"
    return {
        "totals": {
            "inputTokens": _safe_int(sessions.get("inputTokens")),
            "outputTokens": _safe_int(sessions.get("outputTokens")),
            "cacheReadTokens": _safe_int(sessions.get("cacheReadTokens")),
            "cacheWriteTokens": _safe_int(sessions.get("cacheWriteTokens")),
            "reasoningTokens": _safe_int(sessions.get("reasoningTokens")),
            "apiCalls": _safe_int(sessions.get("apiCalls")),
            "estimatedCostUsd": float(sessions.get("estimatedCostUsd") or 0),
            "actualCostUsd": float(sessions.get("actualCostUsd") or 0),
        },
        "daily": {
            "todayEstimatedCostUsd": float(sessions.get("todayEstimatedCostUsd") or 0),
            "todayActualCostUsd": float(sessions.get("todayActualCostUsd") or 0),
            "budgetThresholdUsd": threshold_f,
            "budgetAlertState": state,
        },
    }


def _dashboard_metrics(cfg: dict[str, Any], env_info: dict[str, Any], runtime_state: dict[str, Any] | None = None) -> dict[str, Any]:
    raw_dashboard = cfg.get("dashboard") if isinstance(cfg, dict) else None
    dashboard: dict[str, Any] = raw_dashboard if isinstance(raw_dashboard, dict) else {}
    runtime_state = runtime_state or {}
    bind = str(runtime_state.get("bound_host") or dashboard.get("host") or dashboard.get("bind") or "127.0.0.1")
    if bind in {"127.0.0.1", "localhost", "::1"}:
        exposure = "loopback"
    elif bind.startswith("10.") or bind.startswith("192.168.") or bind.startswith("172."):
        exposure = "private"
    elif bind in {"0.0.0.0", "::"}:
        exposure = "public"
    else:
        exposure = "unknown"
    oauth_required = bool(runtime_state.get("auth_required")) if "auth_required" in runtime_state else False
    token_present = bool((env_info.get("dashboard") or {}).get("tokenPresent")) if isinstance(env_info.get("dashboard"), dict) else False
    auth_gated = bool(oauth_required or token_present or exposure == "loopback")
    auth_mode = "oauth-cookie" if oauth_required else ("session-token" if token_present or exposure == "loopback" else "not-gated")
    return {
        "missionControlRoute": "/mission-control",
        "sourceRoute": SOURCE_URL,
        "authGated": auth_gated,
        "authMode": auth_mode,
        "oauthGateRequired": oauth_required,
        "dashboardTokenPresent": token_present,
        "bindExposure": exposure,
        "runtimeHostKnown": bool(runtime_state.get("bound_host")),
        "panels": {
            "memoryBrowser": True,
            "activityFeed": True,
            "skillLibrary": True,
            "costAnalytics": True,
            "productionReadiness": True,
            "preflight": True,
            "sourceCoverage": True,
        },
    }


def _quality_metrics(home: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    py_test = repo_root / "tests" / "hermes_cli" / "test_mission_control.py"
    web_page = repo_root / "web" / "src" / "pages" / "MissionControlPage.tsx"
    package_json = repo_root / "web" / "package.json"
    build_script = False
    if package_json.exists():
        payload = _read_json(package_json)
        build_script = isinstance(payload, dict) and "build" in (payload.get("scripts") or {})
    return {
        "pythonMissionControlTestsPresent": py_test.exists(),
        "frontendRoutePresent": web_page.exists(),
        "frontendBuildScriptPresent": build_script,
        "lastPythonTestEvidence": None,
        "frontendBuildEvidence": None,
        "browserSmokeEvidence": None,
    }


def _multi_user_metrics(home: Path, env_info: dict[str, Any]) -> dict[str, Any]:
    # Count only, no IDs/names/chat IDs. Pairing storage format has changed over
    # time, so this is intentionally tolerant.
    approved = 0
    pending = 0
    platform_count = 0
    for candidate in [home / "gateway" / "pairings.json", home / "pairings.json", home / "gateway_pairings.json"]:
        payload = _read_json(candidate)
        if not isinstance(payload, (dict, list)):
            continue
        raw_items = payload.values() if isinstance(payload, dict) else payload
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status") or item.get("state") or "").lower()
            if status in {"approved", "active", "paired"}:
                approved += 1
            elif status in {"pending", "requested", "waiting"}:
                pending += 1
            if item.get("platform"):
                platform_count += 1
    telegram = env_info.get("telegram", {}) if isinstance(env_info.get("telegram"), dict) else {}
    if approved == 0 and _safe_int(telegram.get("allowedUserCount")):
        approved = _safe_int(telegram.get("allowedUserCount"))
        platform_count = max(platform_count, 1)
    return {
        "approvedUserCount": approved,
        "pendingUserCount": pending,
        "platformCount": platform_count,
        "memoryNamespaceSupported": True,
        "singleUserMode": approved <= 1 and pending == 0,
    }


def _data_flow(runtime: dict[str, Any]) -> list[dict[str, Any]]:
    env_info = runtime.get("env", {})
    families = set(env_info.get("families", [])) if isinstance(env_info, dict) else set()
    rows: list[dict[str, Any]] = []
    for item in DATA_FLOW_SURFACES:
        surface_id = item["id"]
        configured = False
        provider = None
        if surface_id == "telegram":
            configured = "telegram" in families or bool(runtime.get("gateway", {}).get("configuredCount"))
        elif surface_id == "llm":
            provider = runtime.get("model", {}).get("provider")
            configured = bool(provider and provider != "auto")
        elif surface_id == "pinecone":
            configured = bool(runtime.get("semantic", {}).get("configured"))
        elif surface_id == "whisper":
            configured = bool(runtime.get("voice", {}).get("sttEnabled") or (env_info.get("voice") or {}).get("openaiKeyPresent"))
        elif surface_id == "sqlite":
            configured = bool(runtime.get("sessions", {}).get("dbPresent"))
        rows.append({**item, "configured": configured, "provider": provider, "privacy": "metadata-only in this dashboard"})
    return rows


def _status_row(item: dict[str, str], status: str, evidence: list[str]) -> dict[str, Any]:
    return {**item, "status": status, "evidence": [e for e in evidence if e][:3]}


def _preflight_checks(runtime: dict[str, Any], home: Path) -> list[dict[str, Any]]:
    env_info = runtime["env"]
    safety = runtime["safety"]
    dashboard = runtime["dashboard"]
    semantic = runtime["semantic"]
    voice = runtime["voice"]
    gateway = runtime["gateway"]
    analytics = runtime["analytics"]
    gitignore = home / ".gitignore"
    gitignore_text = gitignore.read_text(encoding="utf-8", errors="ignore") if gitignore.exists() else ""
    checks: dict[str, tuple[str, list[str]]] = {
        "env_gitignored": ("pass" if ".env" in gitignore_text or "*.env" in gitignore_text else "unknown", [f".env file present: {env_info['filePresent']}", f".gitignore present: {gitignore.exists()}"]),
        "rotatable_secrets": ("pass", [f"{env_info['configuredKeys']} configured env value(s) reduced to booleans/counts"]),
        "allowed_users": ("pass" if (env_info.get("telegram") or {}).get("allowedUserCount") else ("unknown" if not (env_info.get("telegram") or {}).get("tokenPresent") else "fail"), [f"allowed user count: {(env_info.get('telegram') or {}).get('allowedUserCount', 0)}"]),
        "private_chats": ("pass" if gateway.get("configuredCount") else "unknown", ["gateway adapters enforce platform-specific auth boundaries"]),
        "sandbox_shell": ("pass" if safety.get("terminalIsolated") else ("warn" if safety.get("approvalFlowConfigured") else "fail"), [f"terminal backend: {safety.get('terminalBackend')}", f"approvals: {safety.get('approvalsMode')}"]),
        "file_allowlist": ("unknown", ["file allowlist is toolset/config specific; no raw paths exposed here"]),
        "dashboard_auth": ("pass" if dashboard.get("authGated") and dashboard.get("bindExposure") != "public" else "warn", [f"auth: {dashboard.get('authMode')}", f"bind: {dashboard.get('bindExposure')}"]),
        "cost_threshold": ("pass" if analytics.get("daily", {}).get("budgetAlertState") in {"ok", "warning", "exceeded"} else "unknown", [f"budget state: {analytics.get('daily', {}).get('budgetAlertState')}"]),
        "pinecone_scoped": ("unknown" if semantic.get("apiKeyPresent") else "not_applicable", [f"pinecone configured: {semantic.get('configured')}", "key scope cannot be proven from local metadata"]),
        "voice_key_separation": ("pass" if (not voice.get("sttEnabled") or (env_info.get("voice") or {}).get("openaiKeyPresent") or voice.get("sttProvider") not in {"openai", "whisper"}) else "warn", [f"STT provider: {voice.get('sttProvider') or 'not enabled'}", f"OpenAI key present: {(env_info.get('voice') or {}).get('openaiKeyPresent', False)}"]),
        "approval_flow": ("pass" if safety.get("approvalFlowConfigured") else "fail", [f"approvals mode: {safety.get('approvalsMode')}"]),
    }
    return [_status_row(item, *checks.get(item["id"], ("unknown", []))) for item in PREFLIGHT_CHECKS]


def _customization_runtime(runtime: dict[str, Any]) -> list[dict[str, Any]]:
    checks: dict[str, tuple[str, list[str]]] = {
        "soul": ("active" if runtime["identity"].get("soulPresent") else "partial", [f"profile files: {runtime['identity'].get('profileFiles', 0)}"]),
        "tools": ("active" if runtime["tools"].get("configuredToolsetCount") else "partial", [f"toolsets: {runtime['tools'].get('configuredToolsetCount', 0)}"]),
        "heartbeats": ("active" if runtime["cron"].get("heartbeatJobs") else ("partial" if runtime["cron"].get("enabled") else "watch"), [f"heartbeat jobs: {runtime['cron'].get('heartbeatJobs', 0)}"]),
        "memory-categories": ("active" if runtime["memory"].get("sqlite", {}).get("dbPresent") else "partial", [f"FTS: {runtime['memory'].get('sqlite', {}).get('ftsPresent')}"]),
        "skill-seeds": ("active" if runtime["skills"].get("total") else "watch", [f"skills: {runtime['skills'].get('total', 0)}"]),
        "model-choice": ("active" if runtime["model"].get("model") != "auto" else "partial", [f"model: {runtime['model'].get('model')}"]),
        "hosting": ("partial" if runtime["hosting"].get("installMethod") != "unknown" else "watch", [f"backend: {runtime['hosting'].get('terminalBackend')}"]),
        "reflection-prompt": ("active" if runtime["reflection"].get("enabled") else "watch", [f"reflection jobs: {runtime['reflection'].get('jobs', 0)}"]),
        "approval-threshold": ("active" if runtime["safety"].get("approvalFlowConfigured") else "watch", [f"mode: {runtime['safety'].get('approvalsMode')}"]),
    }
    return [_status_row(item, *checks.get(item["id"], ("unknown", []))) for item in CUSTOMIZATION_CHECKLIST]


def _production_readiness(runtime: dict[str, Any]) -> dict[str, Any]:
    signals = [
        {"id": "dashboard-auth", "label": "Dashboard auth", "status": "pass" if runtime["dashboard"].get("authGated") else "fail", "detail": runtime["dashboard"].get("authMode")},
        {"id": "approvals", "label": "Approvals", "status": "pass" if runtime["safety"].get("approvalFlowConfigured") else "fail", "detail": runtime["safety"].get("approvalsMode")},
        {"id": "redaction", "label": "Secret redaction", "status": "pass" if runtime["safety"].get("redactSecrets") else "fail", "detail": "values never serialized"},
        {"id": "gateway", "label": "Gateway reachability", "status": "pass" if runtime["gateway"].get("running") else ("warn" if runtime["gateway"].get("configuredCount") else "unknown"), "detail": runtime["gateway"].get("state")},
        {"id": "mcp", "label": "MCP", "status": "pass" if runtime["mcp"].get("configured") else "unknown", "detail": f"{runtime['mcp'].get('configured', 0)} configured"},
        {"id": "cron", "label": "Cron", "status": "pass" if runtime["cron"].get("enabled") else "unknown", "detail": f"{runtime['cron'].get('enabled', 0)} enabled"},
        {"id": "quality", "label": "Quality gates", "status": "pass" if runtime["quality"].get("pythonMissionControlTestsPresent") and runtime["quality"].get("frontendBuildScriptPresent") else "warn", "detail": "test/build hooks present; last run evidence shown after verification"},
        {"id": "hosting", "label": "Hosting posture", "status": "pass" if (runtime["hosting"].get("hardenedContainer") or runtime["hosting"].get("managedService")) else ("warn" if runtime["hosting"].get("containerized") else "unknown"), "detail": runtime["hosting"].get("installMethod")},
    ]
    score_map = {"pass": 100, "warn": 60, "unknown": 35, "fail": 0, "not_applicable": 70}
    score = round(sum(score_map.get(str(s.get("status")), 35) for s in signals) / len(signals)) if signals else 0
    blockers = [s for s in signals if s.get("status") in {"fail", "warn"}]
    return {"score": score, "signals": signals, "blockers": blockers[:5]}


def _hosting_metrics(cfg: dict[str, Any], safety: dict[str, Any], env_info: dict[str, Any], gateway_info: dict[str, Any] | None = None) -> dict[str, Any]:
    gateway_info = gateway_info or {}
    terminal_backend = safety.get("terminalBackend") or "local"
    install_method = "docker" if terminal_backend == "docker" else ("ssh" if terminal_backend == "ssh" else "unknown")
    service_manager = str(gateway_info.get("serviceManager") or "unknown")
    managed_service = bool(gateway_info.get("managedService") or gateway_info.get("serviceInstalled") or gateway_info.get("serviceRunning"))
    if install_method == "unknown" and managed_service and service_manager != "unknown":
        install_method = service_manager
    docker = cfg.get("docker") if isinstance(cfg.get("docker"), dict) else {}
    hardened = None
    if install_method == "docker" or docker:
        hardened = bool(docker.get("read_only") or docker.get("cap_drop") == "ALL" or docker.get("non_root"))
        install_method = "docker"
    return {
        "installMethod": install_method,
        "terminalBackend": terminal_backend,
        "containerized": install_method in {"docker", "s6"},
        "hardenedContainer": hardened,
        "managedService": managed_service,
        "serviceManager": service_manager,
        "serviceRunning": bool(gateway_info.get("serviceRunning")),
        "serviceInstalled": bool(gateway_info.get("serviceInstalled")),
        "meshVpnSignal": bool((env_info.get("network") or {}).get("tailscaleSignalPresent")) if isinstance(env_info.get("network"), dict) else False,
        "publicPortSignal": None,
    }

def _skill_metrics(home: Path) -> dict[str, Any]:
    skill_root = home / "skills"
    files = list(skill_root.rglob("SKILL.md")) if skill_root.exists() else []
    by_category: dict[str, int] = {}
    trust: dict[str, int] = {"builtin": 0, "official": 0, "trusted": 0, "community": 0, "unknown": 0}
    agentskills = 0
    auto_created = 0
    for p in files:
        try:
            rel = p.relative_to(skill_root)
            category = "direct" if len(rel.parts) <= 2 else "grouped"
            by_category[category] = by_category.get(category, 0) + 1
            raw = p.read_text(encoding="utf-8", errors="ignore")[:4000]
            low = raw.lower()
            if "description:" in low and ("name:" in low or "---" in low):
                agentskills += 1
            if "created_by: agent" in low or "author: hermes agent" in low:
                auto_created += 1
            level = "unknown"
            for candidate in trust:
                if f"trust: {candidate}" in low or f"trust_level: {candidate}" in low:
                    level = candidate
                    break
            trust[level] = trust.get(level, 0) + 1
        except Exception:
            continue
    usage = _read_json(skill_root / ".usage.json") if skill_root.exists() else None
    usage_count = len(usage) if isinstance(usage, dict) else 0
    return {
        "rootPresent": skill_root.exists(),
        "path": _compact_path(skill_root, home),
        "total": len(files),
        "usageTracked": usage_count,
        "categories": by_category,
        "trustLevels": trust,
        "agentskillsCompliant": agentskills,
        "autoCreatedCount": auto_created,
        "spotlight": [{"label": f"skill-{idx}", "category": cat} for idx, cat in enumerate(list(by_category.keys())[:6], start=1)],
    }

def _cron_metrics(home: Path) -> dict[str, Any]:
    jobs_path = home / "cron" / "jobs.json"
    payload = _read_json(jobs_path)
    jobs: list[dict[str, Any]] = []
    if isinstance(payload, list):
        jobs = [j for j in payload if isinstance(j, dict)]
    elif isinstance(payload, dict):
        raw_jobs = payload.get("jobs")
        if isinstance(raw_jobs, dict):
            jobs = [j for j in raw_jobs.values() if isinstance(j, dict)]
        elif isinstance(raw_jobs, list):
            jobs = [j for j in raw_jobs if isinstance(j, dict)]
        elif all(isinstance(v, dict) for v in payload.values()):
            jobs = [v for v in payload.values() if isinstance(v, dict)]
    enabled = 0
    cadences: dict[str, int] = {"daily": 0, "weekly": 0, "frequent": 0, "one-shot": 0, "custom": 0}
    deliveries: dict[str, int] = {}
    heartbeat = 0
    reflection = 0
    last_error_present = False
    last_status_counts: dict[str, int] = {}
    overdue_count = 0
    failed_jobs = 0
    next_due_values: list[int] = []
    last_run_ages: list[int] = []
    reflection_run_ages: list[int] = []
    last_run_age_buckets: dict[str, int] = {"never": 0, "under_1h": 0, "under_24h": 0, "under_7d": 0, "over_7d": 0}
    now = time.time()
    for job in jobs:
        disabled = job.get("disabled") or job.get("paused") or job.get("enabled") is False
        if not disabled:
            enabled += 1
        schedule = str(job.get("schedule") or job.get("cron") or "").strip().lower()
        if schedule.startswith("every ") or schedule.endswith("m") or schedule.endswith("h"):
            cadence = "frequent"
        elif schedule.count(" ") >= 4 and (schedule.split()[2] != "*" or schedule.split()[3] != "*"):
            cadence = "weekly"
        elif schedule.count(" ") >= 4:
            cadence = "daily"
        elif "t" in schedule and schedule[:4].isdigit():
            cadence = "one-shot"
        else:
            cadence = "custom" if schedule else "custom"
        cadences[cadence] = cadences.get(cadence, 0) + 1
        deliver = str(job.get("deliver") or job.get("target") or "origin")
        label = deliver.split(":", 1)[0] if ":" in deliver else deliver
        if label not in {"origin", "local", "all", "telegram", "discord", "slack", "sms", "email"}:
            label = "other"
        deliveries[label] = deliveries.get(label, 0) + 1
        descriptor = " ".join(str(job.get(k) or "") for k in ["name", "prompt", "skills", "script"]).lower()
        is_heartbeat = any(token in descriptor for token in ["heartbeat", "check-in", "checkin", "morning", "evening", "weekly review"])
        is_reflection = any(token in descriptor for token in ["reflection", "curator", "dream", "consolidat", "memory review"])
        if is_heartbeat:
            heartbeat += 1
        if is_reflection:
            reflection += 1
        if job.get("last_error") or job.get("error"):
            last_error_present = True
        status = _safe_status(job.get("last_status") or job.get("status"), allowed=SAFE_CRON_STATUSES)
        if status != "unknown":
            _increment(last_status_counts, status)
        if status in {"error", "failed", "timeout"} or job.get("last_error") or job.get("error"):
            failed_jobs += 1
        next_ts = _parse_timestamp(job.get("next_run_at") or job.get("next_run"))
        if next_ts is not None:
            due_in = int(next_ts - now)
            next_due_values.append(due_in)
            if not disabled and due_in < 0:
                overdue_count += 1
        last_ts = _parse_timestamp(job.get("last_run_at") or job.get("last_run") or job.get("last_finished_at"))
        if last_ts is not None:
            age_seconds = max(0, int(now - last_ts))
            last_run_ages.append(age_seconds)
            _increment(last_run_age_buckets, _age_bucket(age_seconds))
            if is_reflection:
                reflection_run_ages.append(age_seconds)
        else:
            _increment(last_run_age_buckets, "never")
    reflection_freshness = "not_configured"
    if reflection:
        if not reflection_run_ages:
            reflection_freshness = "unknown"
        else:
            freshest_reflection = min(reflection_run_ages)
            reflection_freshness = "fresh" if freshest_reflection <= 36 * 3600 else "stale"
    return {
        "filePresent": jobs_path.exists(),
        "path": _compact_path(jobs_path, home),
        "total": len(jobs),
        "enabled": enabled,
        "paused": max(len(jobs) - enabled, 0),
        "cadences": cadences,
        "sampleSchedules": [k for k, v in cadences.items() if v][:6],
        "deliveries": deliveries,
        "heartbeatJobs": heartbeat,
        "reflectionJobs": reflection,
        "genericJobs": max(len(jobs) - heartbeat - reflection, 0),
        "lastErrorPresent": last_error_present,
        "lastStatusCounts": last_status_counts,
        "failedJobs": failed_jobs,
        "overdueCount": overdue_count,
        "nextRunKnown": bool(next_due_values),
        "nextRunDueInSeconds": min(next_due_values) if next_due_values else None,
        "lastRunAgeSeconds": min(last_run_ages) if last_run_ages else None,
        "lastRunAgeBuckets": last_run_age_buckets,
        "reflectionLastRunAgeSeconds": min(reflection_run_ages) if reflection_run_ages else None,
        "reflectionFreshness": reflection_freshness,
        "timezoneConfigured": False,
        "timezoneSource": "unknown",
    }

def _mcp_metrics(cfg: dict[str, Any]) -> dict[str, Any]:
    servers: dict[str, Any] = {}
    root_servers = cfg.get("mcp_servers") if isinstance(cfg, dict) else None
    if isinstance(root_servers, dict):
        servers.update(root_servers)
    # Backwards-compatible fallback for older/experimental config shape.
    mcp = cfg.get("mcp") if isinstance(cfg, dict) else None
    nested_servers = mcp.get("servers") if isinstance(mcp, dict) else None
    if isinstance(nested_servers, dict):
        servers.update(nested_servers)
    enabled = 0
    transport_counts: dict[str, int] = {"stdio": 0, "http": 0, "sse": 0, "unknown": 0}
    status_counts: dict[str, int] = {"enabled": 0, "disabled": 0}
    server_summaries: list[dict[str, Any]] = []
    for idx, key in enumerate(sorted(servers.keys()), start=1):
        value = servers.get(key)
        disabled = isinstance(value, dict) and value.get("enabled") is False
        if not disabled:
            enabled += 1
        _increment(status_counts, "disabled" if disabled else "enabled")
        transport = "unknown"
        if isinstance(value, dict):
            raw_transport = str(value.get("transport") or "").lower()
            if raw_transport in transport_counts:
                transport = raw_transport
            elif value.get("url"):
                url = str(value.get("url") or "").lower()
                transport = "sse" if "sse" in url else "http"
            elif value.get("command"):
                transport = "stdio"
        _increment(transport_counts, transport)
        if idx <= 12:
            server_summaries.append({"label": f"server-{idx}", "enabled": not disabled, "transport": transport})
    return {
        "configured": len(servers),
        "servers": [item["label"] for item in server_summaries],
        "serverDetails": server_summaries,
        "serverNamesRedacted": True,
        "enabled": enabled,
        "disabled": max(len(servers) - enabled, 0),
        "statusCounts": status_counts,
        "transportCounts": transport_counts,
    }


def _model_metrics(cfg: dict[str, Any]) -> dict[str, Any]:
    raw_model = cfg.get("model")
    raw_agent = cfg.get("agent")
    raw_delegation = cfg.get("delegation")
    agent: dict[str, Any] = raw_agent if isinstance(raw_agent, dict) else {}
    delegation: dict[str, Any] = raw_delegation if isinstance(raw_delegation, dict) else {}
    if isinstance(raw_model, dict):
        model_name = _safe_model_label(raw_model.get("default") or raw_model.get("model") or raw_model.get("name") or "auto")
        provider = raw_model.get("provider") or cfg.get("provider")
    else:
        model_name = _safe_model_label(raw_model or "auto")
        provider = cfg.get("provider")
    if not isinstance(provider, str) or not provider.strip():
        provider = model_name.split("/", 1)[0] if "/" in model_name else "auto"
    return {
        "provider": _safe_provider_label(provider),
        "model": model_name,
        "reasoning": _safe_choice_label(agent.get("reasoning_effort") or delegation.get("reasoning_effort") or "default", allowed={"none", "minimal", "low", "medium", "high", "xhigh", "default", "show", "hide"}, default="custom"),
        "delegationProvider": _safe_provider_label(delegation.get("provider")) if delegation.get("provider") else None,
        "maxTurns": _safe_int(agent.get("max_turns"), 0),
    }


def _gateway_metrics(cfg: dict[str, Any], env_info: dict[str, Any]) -> dict[str, Any]:
    configured_labels: list[str] = []
    configured_entries = 0
    for family in env_info.get("families", []):
        if family in KNOWN_GATEWAY_FAMILIES:
            configured_labels.append(str(family))
            configured_entries += 1
    platforms_cfg = ((cfg.get("gateway") or {}).get("platforms") or {}) if isinstance(cfg.get("gateway"), dict) else {}
    if isinstance(platforms_cfg, dict):
        for name, pcfg in platforms_cfg.items():
            if isinstance(pcfg, dict) and pcfg.get("enabled"):
                configured_entries += 1
                configured_labels.append(_safe_gateway_label(name))
    running = False
    pid = None
    pid_present = False
    runtime_state = "unknown"
    service_manager = "unknown"
    service_installed = False
    service_running = False
    service_scope = None
    managed_service = False
    try:
        from gateway.status import get_running_pid, read_runtime_status

        pid = get_running_pid()
        pid_present = bool(pid)
        running = bool(pid)
        status = read_runtime_status()
        if isinstance(status, dict):
            runtime_state = _safe_status(status.get("state") or status.get("status") or ("running" if running else "stopped"), allowed={"running", "stopped", "starting", "ready", "error", "failed", "unknown"})
    except Exception:
        runtime_state = "unavailable"
    try:
        from .gateway import get_gateway_runtime_snapshot

        snapshot = get_gateway_runtime_snapshot()
        service_manager = _safe_service_manager_label(getattr(snapshot, "manager", None))
        service_installed = bool(getattr(snapshot, "service_installed", False))
        service_running = bool(getattr(snapshot, "service_running", False))
        service_scope = _safe_service_scope(getattr(snapshot, "service_scope", None))
        snapshot_pids = tuple(getattr(snapshot, "gateway_pids", ()) or ())
        pid_present = pid_present or bool(snapshot_pids)
        running = running or bool(getattr(snapshot, "running", False))
        managed_service = service_installed or service_running or service_manager in {"systemd", "launchd", "s6"}
        if runtime_state in {"unknown", "stopped"} and running:
            runtime_state = "running"
    except Exception:
        pass
    unique = sorted(set(configured_labels))
    return {
        "running": running,
        "pidPresent": pid_present,
        "state": runtime_state,
        "configuredPlatforms": unique,
        "configuredCount": configured_entries or len(unique),
        "platformNamesRedacted": True,
        "customPlatformCount": sum(1 for label in configured_labels if label == "other"),
        "serviceManager": service_manager,
        "serviceInstalled": service_installed,
        "serviceRunning": service_running,
        "serviceScope": service_scope,
        "managedService": managed_service,
    }


def _tool_metrics(cfg: dict[str, Any]) -> dict[str, Any]:
    toolsets = cfg.get("toolsets") or []
    disabled = ((cfg.get("agent") or {}).get("disabled_toolsets") or []) if isinstance(cfg.get("agent"), dict) else []
    tools_cfg = cfg.get("tools") if isinstance(cfg.get("tools"), dict) else {}
    configured_toolsets = list(toolsets) if isinstance(toolsets, list) else []
    disabled_toolsets = list(disabled) if isinstance(disabled, list) else []
    repo_root = Path(__file__).resolve().parents[1]
    builtin_tool_files = list((repo_root / "tools").glob("*.py")) if (repo_root / "tools").exists() else []
    read_file_present = any(p.name in {"file_operations.py", "file.py"} for p in builtin_tool_files)
    terminal_present = any("terminal" in p.name for p in builtin_tool_files)
    buckets: dict[str, int] = {}
    disabled_buckets: dict[str, int] = {}
    for item in configured_toolsets:
        _increment(buckets, _toolset_bucket(item))
    for item in disabled_toolsets:
        _increment(disabled_buckets, _toolset_bucket(item))
    return {
        "configuredToolsets": [f"toolset-{idx}" for idx, _ in enumerate(configured_toolsets[:12], start=1)],
        "configuredToolsetCount": len(configured_toolsets),
        "configuredToolsetBuckets": buckets,
        "configuredToolsetNamesRedacted": True,
        "disabledToolsets": [f"disabled-toolset-{idx}" for idx, _ in enumerate(disabled_toolsets[:12], start=1)],
        "disabledToolsetCount": len(disabled_toolsets),
        "disabledToolsetBuckets": disabled_buckets,
        "toolSearch": bool(tools_cfg.get("tool_search")),
        "registeredToolCount": len(builtin_tool_files),
        "enabledToolCount": max(len(builtin_tool_files) - len(disabled_toolsets), 0),
        "dangerClasses": {"safe": 0, "destructive": 0, "expensive": 0, "unknown": len(builtin_tool_files)},
        "fileAccess": {
            "readFileToolPresent": read_file_present,
            "allowlistConfigured": bool(tools_cfg.get("file_allowlist") or tools_cfg.get("allowlist")),
            "denylistConfigured": bool(tools_cfg.get("file_denylist") or tools_cfg.get("denylist")),
        },
        "execution": {"terminalToolPresent": terminal_present},
    }

def _safety_metrics(cfg: dict[str, Any]) -> dict[str, Any]:
    raw_approvals = cfg.get("approvals")
    raw_security = cfg.get("security")
    raw_terminal = cfg.get("terminal")
    raw_tools = cfg.get("tools")
    approvals: dict[str, Any] = raw_approvals if isinstance(raw_approvals, dict) else {}
    security: dict[str, Any] = raw_security if isinstance(raw_security, dict) else {}
    terminal: dict[str, Any] = raw_terminal if isinstance(raw_terminal, dict) else {}
    tools_cfg: dict[str, Any] = raw_tools if isinstance(raw_tools, dict) else {}
    raw_mode = str(approvals.get("mode") or "manual")
    raw_cron_mode = str(approvals.get("cron_mode") or approvals.get("mode") or "manual")
    safe_modes = {"manual", "smart", "auto", "auto_approve", "off", "none", "disabled"}
    mode = _safe_choice_label(raw_mode, allowed=safe_modes, default="custom")
    cron_mode = _safe_choice_label(raw_cron_mode, allowed=safe_modes, default="custom")
    raw_terminal_backend = str(terminal.get("backend") or "local")
    terminal_backend = _safe_choice_label(raw_terminal_backend, allowed={"local", "docker", "ssh", "daytona", "singularity", "modal", "kubernetes", "firecracker", "managed_modal"}, default="custom")
    isolated = terminal_backend in {"docker", "ssh", "daytona", "singularity", "modal", "kubernetes", "firecracker", "managed_modal"}
    approval_enabled = mode not in {"off", "none", "disabled"}
    auto_enabled = mode in {"smart", "auto", "auto_approve"}
    repo_root = Path(__file__).resolve().parents[1]
    marker_supported = False
    try:
        for p in repo_root.rglob("*.py"):
            if p.name == "mission_control.py":
                continue
            sample = p.read_text(encoding="utf-8", errors="ignore")
            if "<tool_output" in sample or "attackable" in sample:
                marker_supported = True
                break
    except Exception:
        marker_supported = False
    limit_values: list[int] = []
    for key in ["max_output_chars", "tool_output_limit", "tool_output_limit_chars", "terminal_output_limit", "terminal_output_limit_chars"]:
        value = _safe_int(tools_cfg.get(key), 0)
        if value > 0:
            limit_values.append(value)
    return {
        "approvalsMode": mode,
        "cronApprovalsMode": cron_mode,
        "approvalFlowConfigured": approval_enabled,
        "autoApproveEnabled": auto_enabled,
        "destructiveToolsRequireApproval": approval_enabled,
        "expensiveToolsRequireApproval": approval_enabled,
        "redactSecrets": security.get("redact_secrets") is not False,
        "tirithEnabled": bool(security.get("tirith_enabled")),
        "terminalBackend": terminal_backend,
        "terminalIsolated": isolated,
        "privateUrlsAllowed": bool(security.get("allow_private_urls")),
        "toolOutputLimits": {
            "configured": bool(limit_values),
            "count": len(limit_values),
            "minChars": min(limit_values) if limit_values else None,
            "maxChars": max(limit_values) if limit_values else None,
        },
        "promptInjection": {
            "toolOutputMarkersSupported": marker_supported,
            "toolOutputMarkersEnabled": marker_supported,
            "toolOutputLimitConfigured": bool(limit_values),
            "untrustedExternalContentTracked": marker_supported,
            "attackableRequiresManualApproval": approval_enabled and marker_supported,
            "privateUrlsAllowed": bool(security.get("allow_private_urls")),
        },
    }

def _identity_metrics(home: Path) -> dict[str, Any]:
    soul_candidates = [home / "soul.md", home / "SOUL.md"]
    memory_dir = home / "memories"
    candidates = [
        *soul_candidates,
        home / "MEMORY.md",
        home / "USER.md",
        home / "memory.md",
        home / "user.md",
        memory_dir / "MEMORY.md",
        memory_dir / "USER.md",
        memory_dir / "memory.md",
        memory_dir / "user.md",
    ]
    files = [p for p in candidates if p.exists()]
    return {
        "soulPresent": any(p.exists() for p in soul_candidates),
        "profileFiles": len(files),
        "totalBytes": sum(p.stat().st_size for p in files if p.exists()),
        "files": [_compact_path(p, home) for p in files[:8]],
    }

def _build_runtime(cfg: dict[str, Any], home: Path, dashboard_state: dict[str, Any] | None = None) -> dict[str, Any]:
    env_info = _env_families(home)
    sessions = _state_db_metrics(home)
    safety = _safety_metrics(cfg)
    dashboard = _dashboard_metrics(cfg, env_info, dashboard_state)
    runtime = {
        "generatedAt": _now_iso(),
        "home": "~/.hermes",
        "model": _model_metrics(cfg),
        "env": env_info,
        "identity": _identity_metrics(home),
        "sessions": sessions,
        "skills": _skill_metrics(home),
        "cron": _cron_metrics(home),
        "mcp": _mcp_metrics(cfg),
        "gateway": _gateway_metrics(cfg, env_info),
        "tools": _tool_metrics(cfg),
        "safety": safety,
        "voice": {
            "sttEnabled": bool(((cfg.get("stt") or {}).get("enabled")) if isinstance(cfg.get("stt"), dict) else False),
            "sttProvider": _safe_voice_label(((cfg.get("stt") or {}).get("provider")) if isinstance(cfg.get("stt"), dict) else None),
            "ttsProvider": _safe_voice_label(((cfg.get("tts") or {}).get("provider")) if isinstance(cfg.get("tts"), dict) else None),
        },
        "dashboard": dashboard,
    }
    runtime["memory"] = _memory_metrics(home, sessions)
    runtime["semantic"] = _semantic_metrics(env_info, cfg)
    runtime["analytics"] = _analytics_metrics(sessions, cfg)
    runtime["quality"] = _quality_metrics(home)
    runtime["multiUser"] = _multi_user_metrics(home, env_info)
    runtime["reflection"] = {
        "enabled": runtime["cron"].get("reflectionJobs", 0) > 0,
        "jobs": runtime["cron"].get("reflectionJobs", 0),
        "curatorEnabled": runtime["cron"].get("reflectionJobs", 0) > 0,
        "freshness": runtime["cron"].get("reflectionFreshness"),
        "lastRunAgeSeconds": runtime["cron"].get("reflectionLastRunAgeSeconds"),
    }
    runtime["hosting"] = _hosting_metrics(cfg, safety, env_info, runtime["gateway"])
    runtime["dataFlow"] = _data_flow(runtime)
    runtime["preflight"] = _preflight_checks(runtime, home)
    runtime["customization"] = _customization_runtime(runtime)
    runtime["production"] = _production_readiness(runtime)
    return runtime

def _state(state: str, evidence: Iterable[str], next_action: str) -> CapabilityState:
    return CapabilityState(state=state, score=_STATE_WEIGHT[state], evidence=[e for e in evidence if e], next=next_action)


def _readiness_for_feature(feature_id: str, runtime: dict[str, Any]) -> CapabilityState:
    sessions = runtime["sessions"]
    skills = runtime["skills"]
    cron = runtime["cron"]
    mcp = runtime["mcp"]
    gateway = runtime["gateway"]
    safety = runtime["safety"]
    voice = runtime["voice"]
    tools = runtime["tools"]
    env_info = runtime["env"]
    memory = runtime["memory"]
    semantic = runtime["semantic"]

    if feature_id == "H1":
        full = runtime["identity"]["profileFiles"] and sessions["dbPresent"] and skills["total"]
        return _state("active" if full else "partial", [f"profile files: {runtime['identity']['profileFiles']}", f"SQLite sessions: {sessions['total']}", f"skills: {skills['total']}"], "Keep all four layers healthy: profile, user memory, skills, episodic recall.")
    if feature_id == "H2":
        if cron.get("reflectionJobs"):
            return _state("active", [f"{cron['reflectionJobs']} reflection-like cron job(s) detected"], "Review reflection output periodically.")
        if cron["enabled"]:
            return _state("partial", [f"{cron['enabled']} enabled cron job(s), no dedicated reflection signal"], "Add a dedicated nightly consolidation job.")
        return _state("watch", ["No enabled cron jobs detected"], "Create a self-contained nightly consolidation cron job.")
    if feature_id in {"H3", "H11", "O3"}:
        active = skills["total"] and skills.get("agentskillsCompliant", 0)
        return _state("active" if active else ("partial" if skills["total"] else "watch"), [f"{skills['total']} installed skill(s)", f"agentskills-shaped: {skills.get('agentskillsCompliant', 0)}", f"auto-created: {skills.get('autoCreatedCount', 0)}"], "Keep auto-created/community skills reviewed and portable.")
    if feature_id in {"H4", "O1"}:
        status = "active" if gateway["running"] and gateway["configuredCount"] else ("partial" if gateway["configuredCount"] else "watch")
        return _state(status, [f"{gateway['configuredCount']} configured platform family/families", f"gateway state: {gateway['state']}"], "Enable only platforms that have real operator value.")
    if feature_id in {"H5", "O5"}:
        backend = safety["terminalBackend"]
        status = "active" if safety.get("terminalIsolated") else ("partial" if backend else "watch")
        return _state(status, [f"terminal backend: {backend}", f"isolated: {safety.get('terminalIsolated')}", f"registered tool modules: {tools.get('registeredToolCount', 0)}"], "Prefer isolated backends for untrusted or expensive commands.")
    if feature_id == "H6":
        active = voice["sttEnabled"] or bool(voice["ttsProvider"])
        return _state("active" if active else "watch", [f"STT enabled: {voice['sttEnabled']}", f"TTS provider: {voice['ttsProvider'] or 'not configured'}"], "Wire STT/TTS only where voice actually speeds you up.")
    if feature_id == "H7":
        return _state("partial" if semantic.get("configured") else "planned", [f"memory provider: {memory.get('provider', {}).get('active', 'builtin')}", f"semantic provider: {semantic.get('provider')}"], "Add Honcho/Mem0/Pinecone only if builtin memory becomes limiting.")
    if feature_id == "H8":
        return _state("partial" if skills["total"] else "watch", [f"trust metadata: {skills.get('trustLevels', {})}"], "Keep community skills behind review before enabling broad permissions.")
    if feature_id == "H9":
        return _state("active", [f"memory budget: {memory.get('budgets', {}).get('memoryLimitChars')} chars", f"user budget: {memory.get('budgets', {}).get('userLimitChars')} chars"], "Watch memory saturation and prune stale facts.")
    if feature_id == "H10":
        tokens = runtime["analytics"]["totals"]["inputTokens"] + runtime["analytics"]["totals"]["outputTokens"]
        return _state("partial" if tokens else "planned", [f"tracked tokens: {tokens}", "TokenMix-specific optimisation is not a live runtime signal"], "Treat as advanced optimisation after high-value flows are stable.")
    if feature_id == "O2":
        return _state("planned", ["Dashboard is responsive web; native mobile clients are outside this repository"], "Use the web dashboard on mobile before funding native clients.")
    if feature_id == "O4":
        return _state("active", ["delegate_task subagents and kanban workers are supported", "parallel child cap is configurable"], "Use subagents for independent workstreams, not shared-file conflicts.")
    if feature_id == "O6":
        return _state("planned", ["MCP/tool protocols exist; Open Gateway federation is not configured"], "Keep this as a future interoperability project.")
    if feature_id in {"O7", "O8"}:
        mode = safety["approvalsMode"]
        status = "active" if safety.get("approvalFlowConfigured") else "watch"
        return _state(status, [f"approvals mode: {mode}", f"auto-approve enabled: {safety.get('autoApproveEnabled')}", f"secret redaction: {safety['redactSecrets']}"], "Keep destructive approvals visible; use smart/off only intentionally.")
    if feature_id == "O9":
        return _state("partial", ["Mission Control dashboard is the live cockpit route", "file/canvas edit UI is separate from this snapshot"], "Promote high-value cards into editable Live Canvas widgets later.")
    if feature_id == "O10":
        fams = set(env_info.get("families", []))
        if "tailscale" in fams or runtime["hosting"].get("meshVpnSignal"):
            return _state("partial", ["tailscale-related signal detected"], "Verify mesh access and avoid public ports.")
        return _state("watch", ["No Tailscale-specific signal detected"], "Prefer mesh VPN or SSH tunnel if exposing a home server.")
    return _state("watch", ["No feature-specific readiness rule"], "Inspect runtime evidence manually.")

def _readiness_for_step(step: dict[str, Any], runtime: dict[str, Any]) -> CapabilityState:
    domain = step["domain"]
    route = step.get("route")
    step_id = step["id"]
    env_info = runtime["env"]
    gateway = runtime["gateway"]
    safety = runtime["safety"]

    if step_id in {"step-1", "step-12"}:
        tg = env_info.get("telegram", {}) if isinstance(env_info.get("telegram"), dict) else {}
        has_token = bool(tg.get("tokenPresent"))
        allowed = _safe_int(tg.get("allowedUserCount"))
        status = "active" if has_token and allowed and gateway.get("configuredCount") else ("partial" if has_token or allowed or gateway.get("configuredCount") else "watch")
        return _state(status, [f"telegram token present: {has_token}", f"allowed user count: {allowed}", f"gateway state: {gateway.get('state')}"], "Keep Telegram private-chat and whitelist boundaries tight.")
    if step_id == "step-4":
        req = env_info.get("requiredKeys", [])
        configured_required = sum(1 for r in req if isinstance(r, dict) and r.get("isSet"))
        return _state("active" if env_info.get("filePresent") and configured_required >= 2 else ("partial" if env_info.get("filePresent") else "watch"), [f".env present: {env_info.get('filePresent')}", f"configured values: {env_info.get('configuredKeys')}", f"required signals set: {configured_required}/{len(req)}"], "Use /env to verify presence without exposing values.")
    if step_id == "step-5":
        ident = runtime["identity"]
        return _state("active" if ident.get("soulPresent") and ident.get("profileFiles") else "partial", [f"soul present: {ident.get('soulPresent')}", f"profile files: {ident.get('profileFiles')}"], "Keep public prompt snippets out of the operator soul.")
    if step_id == "step-8":
        semantic = runtime["semantic"]
        if semantic.get("configured") and semantic.get("indexConfigured"):
            status = "active"
        elif semantic.get("configured") or semantic.get("apiKeyPresent"):
            status = "partial"
        else:
            status = "planned"
        return _state(status, [f"provider: {semantic.get('provider')}", f"API key present: {semantic.get('apiKeyPresent')}", f"index configured: {semantic.get('indexConfigured')}"], "Keep semantic memory optional and project-scoped.")
    if step_id == "step-16":
        return _state("partial", ["Dashboard supports live refresh; gateway streaming depends on adapter/runtime", f"gateway running: {gateway.get('running')}"], "Debounce live edits and avoid unchanged Telegram updates.")
    if step_id == "step-17":
        if runtime["reflection"].get("enabled"):
            return _state("active", [f"reflection jobs: {runtime['reflection'].get('jobs')}"], "Review consolidation output for stale or overbroad facts.")
        if runtime["cron"].get("enabled"):
            return _state("partial", [f"enabled cron jobs: {runtime['cron'].get('enabled')}", "no dedicated reflection signal detected"], "Add a nightly reflection/curator job.")
        return _state("watch", ["No enabled cron jobs detected"], "Create a self-contained reflection cron job.")
    if step_id == "step-20":
        mu = runtime["multiUser"]
        status = "active" if mu.get("approvedUserCount", 0) > 1 and mu.get("memoryNamespaceSupported") else ("partial" if mu.get("approvedUserCount") else "watch")
        return _state(status, [f"approved users: {mu.get('approvedUserCount')}", f"pending users: {mu.get('pendingUserCount')}", f"single-user mode: {mu.get('singleUserMode')}"], "Keep Pairing counts only; never expose IDs or names in this snapshot.")
    if step_id == "step-21":
        return _state("active" if runtime["mcp"]["configured"] else "watch", [f"{runtime['mcp']['configured']} MCP server(s) configured", f"enabled: {runtime['mcp'].get('enabled', 0)}"], "Install MCP servers only for workflows you actually use.")
    if step_id == "step-22":
        return _state("active" if safety.get("approvalFlowConfigured") else "watch", [f"approvals mode: {safety['approvalsMode']}", f"auto approve: {safety.get('autoApproveEnabled')}", f"secret redaction: {safety['redactSecrets']}"], "Never run destructive/expensive tools without an intentional approval policy.")
    if step_id == "step-22-5":
        pi = safety.get("promptInjection", {})
        full = pi.get("toolOutputMarkersEnabled") and pi.get("attackableRequiresManualApproval") and not pi.get("privateUrlsAllowed")
        status = "active" if full else ("partial" if safety.get("approvalFlowConfigured") and safety.get("redactSecrets") else "watch")
        return _state(status, [f"tool-output markers: {pi.get('toolOutputMarkersEnabled')}", f"output limit configured: {pi.get('toolOutputLimitConfigured')}", f"attackable requires manual approval: {pi.get('attackableRequiresManualApproval')}", f"private URLs allowed: {pi.get('privateUrlsAllowed')}"], "Treat external content as data; require manual approval when attackable content is involved.")
    if step_id == "step-23":
        totals = runtime["analytics"]["totals"]
        tokens = totals["inputTokens"] + totals["outputTokens"]
        return _state("active" if tokens or totals["estimatedCostUsd"] or totals["actualCostUsd"] else "partial", [f"tracked tokens: {tokens}", f"actual cost USD: {totals['actualCostUsd']}", f"budget state: {runtime['analytics']['daily']['budgetAlertState']}"], "Set a cost alert threshold for production runs.")
    if step_id == "step-24":
        dash = runtime["dashboard"]
        return _state("active", ["/mission-control route exposes this blueprint", f"auth mode: {dash.get('authMode')}", f"bind exposure: {dash.get('bindExposure')}"], "Keep coverage, runtime evidence, and privacy boundaries distinct.")
    if step_id == "step-25":
        host = runtime["hosting"]
        status = "active" if (host.get("hardenedContainer") or host.get("managedService")) else ("partial" if host.get("containerized") or host.get("meshVpnSignal") or safety.get("terminalBackend") != "local" else "watch")
        return _state(status, [f"install method: {host.get('installMethod')}", f"terminal backend: {host.get('terminalBackend')}", f"managed service: {host.get('managedService')}", f"mesh VPN signal: {host.get('meshVpnSignal')}"], "Prefer low-cost VPS/mesh networking over public dashboard exposure.")
    if step_id == "step-26":
        q = runtime["quality"]
        status = "active" if q.get("pythonMissionControlTestsPresent") and q.get("frontendBuildScriptPresent") else "partial"
        return _state(status, [f"python tests present: {q.get('pythonMissionControlTestsPresent')}", f"frontend build script: {q.get('frontendBuildScriptPresent')}", "last run evidence is external to the static snapshot"], "Run Python, build, API and browser smokes before shipping.")

    if domain == "model":
        model = runtime["model"]
        configured = model.get("model") != "auto" or bool(env_info.get("llmFamilies"))
        return _state("active" if configured else "partial", [f"provider: {model['provider']}", f"model: {model['model']}", f"reasoning: {model['reasoning']}"], "Keep model routing visible in Models and Config.")
    if domain == "memory":
        mem = runtime["memory"]
        return _state("active" if mem["sqlite"]["dbPresent"] else "partial", [f"session DB present: {mem['sqlite']['dbPresent']}", f"messages counted: {mem['sqlite']['messages']}", f"FTS present: {mem['sqlite']['ftsPresent']}"], "Continue consolidating important facts into memory/skills.")
    if domain == "skills":
        return _state("active" if runtime["skills"]["total"] else "partial", [f"{runtime['skills']['total']} installed skill(s)", f"agentskills-shaped: {runtime['skills'].get('agentskillsCompliant', 0)}"], "Create skills only from reusable, verified workflows.")
    if domain == "automation":
        return _state("active" if runtime["cron"]["enabled"] else "watch", [f"{runtime['cron']['total']} cron job(s), {runtime['cron']['enabled']} enabled", f"cadences: {runtime['cron'].get('cadences', {})}"], "Schedule recurring work only with self-contained prompts.")
    if domain == "tools":
        return _state("active" if runtime["tools"].get("registeredToolCount") else "partial", [f"registered tool modules: {runtime['tools'].get('registeredToolCount', 0)}", f"configured toolset count: {runtime['tools']['configuredToolsetCount']}"], "Keep powerful tools behind the right platform/toolset scope.")
    if domain == "interface":
        return _state("active" if gateway["configuredCount"] else "partial", [f"configured platform families: {gateway['configuredCount']}", f"gateway running: {gateway['running']}"], "Use Pairing/Channels for multi-user safety.")
    if domain == "voice":
        return _state("active" if runtime["voice"]["sttEnabled"] else "watch", [f"STT provider: {runtime['voice']['sttProvider'] or 'not enabled'}", f"TTS provider: {runtime['voice']['ttsProvider'] or 'not configured'}"], "Enable voice where Telegram/Discord audio saves time.")
    if domain == "safety":
        return _state("active" if safety.get("approvalFlowConfigured") else "watch", [f"approvals mode: {safety['approvalsMode']}", f"secret redaction: {safety['redactSecrets']}"], "Never expose raw tool output/logs in the dashboard.")
    if domain == "analytics":
        tokens = runtime["sessions"]["inputTokens"] + runtime["sessions"]["outputTokens"]
        return _state("active" if tokens else "partial", [f"tracked tokens: {tokens}", f"estimated cost USD: {runtime['sessions']['estimatedCostUsd']}"], "Enable token analytics in dashboard config when useful.")
    if domain == "dashboard":
        return _state("active", ["/mission-control route exposes this blueprint", "server-only snapshot protects local state"], "Keep coverage and readiness distinct.")
    if domain == "hosting":
        host = runtime["hosting"]
        status = "active" if host.get("managedService") else ("partial" if host.get("installMethod") != "unknown" else "watch")
        return _state(status, [f"install method: {host.get('installMethod')}", f"managed service: {host.get('managedService')}", f"terminal backend: {safety['terminalBackend']}"], "Prefer low-cost VPS/mesh networking over low-value complexity.")
    if domain == "quality":
        return _state("partial", ["Mission Control tests/build scripts are discoverable", "latest pass/fail evidence comes from verification runs"], "Run browser smoke on desktop and mobile before shipping.")
    if domain in {"configuration", "identity", "runtime", "agent-loop"}:
        return _state("active", [f"dashboard route: {route}", f"Hermes home exposed as {runtime['home']}"], "Keep config readable while secrets stay server-only.")
    return _state("watch", ["No domain-specific rule"], "Review manually.")

def _coverage(runtime: dict[str, Any]) -> dict[str, Any]:
    step_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    domain_scores: dict[str, list[int]] = {}

    for step in BLUEPRINT_STEPS:
        readiness = _readiness_for_step(step, runtime)
        row = {
            **step,
            "sourceUrl": f"{SOURCE_URL}#{step['id']}",
            "status": readiness.state,
            "readiness": readiness.score,
            "evidence": readiness.evidence,
            "next": readiness.next,
            "missionControl": step.get("route") == "/mission-control",
        }
        step_rows.append(row)
        domain_scores.setdefault(step["domain"], []).append(readiness.score)

    for feature in [*HERMES_FEATURES, *OPENCLAW_FEATURES]:
        readiness = _readiness_for_feature(feature["id"], runtime)
        row = {
            **feature,
            "status": readiness.state,
            "readiness": readiness.score,
            "evidence": readiness.evidence,
            "next": readiness.next,
        }
        feature_rows.append(row)
        domain_scores.setdefault(feature["domain"], []).append(readiness.score)

    all_rows = [*step_rows, *feature_rows]
    counts: dict[str, int] = {"active": 0, "partial": 0, "watch": 0, "planned": 0}
    for row in all_rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    total = len(all_rows)
    readiness = round(sum(row["readiness"] for row in all_rows) / total) if total else 0
    domains = [
        {"name": name, "score": round(sum(scores) / len(scores)), "items": len(scores)}
        for name, scores in sorted(domain_scores.items())
        if scores
    ]
    domains.sort(key=lambda d: (d["score"], d["name"]))
    return {
        "summary": {"total": total, "readiness": readiness, "counts": counts},
        "steps": step_rows,
        "features": feature_rows,
        "domains": domains,
        "weakestDomains": domains[:5],
    }


def _action_queue(runtime: dict[str, Any], coverage: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    def add(tone: str, title: str, reason: str, route: str, category: str = "readiness", effort: str = "M") -> None:
        actions.append({"rank": len(actions) + 1, "tone": tone, "title": title, "reason": reason, "route": route, "category": category, "effort": effort})

    preflight_bad = [c for c in runtime.get("preflight", []) if c.get("status") in {"fail", "warn"}]
    if preflight_bad:
        first = preflight_bad[0]
        add("now", f"Fix pre-flight: {first.get('title')}", "; ".join(first.get("evidence", [])[:2]) or first.get("summary", "Security checklist needs attention."), first.get("route", "/mission-control"), "security", "S")
    if runtime["gateway"]["configuredCount"] == 0:
        add("now", "Configure one gateway", "Messaging is the only useful interface if the agent is not reachable.", "/channels", "interface", "M")
    if runtime["safety"].get("privateUrlsAllowed"):
        add("now", "Review private URL policy", "Private URL fetching is enabled; prompt-injection and SSRF boundaries should be intentional.", "/system", "security", "S")
    weakest = coverage.get("weakestDomains", [])[:4]
    for domain in weakest:
        if len(actions) >= 6:
            break
        add("now" if domain["score"] < 55 else "next", f"Strengthen {domain['name']}", f"Average readiness {domain['score']} across {domain['items']} mapped item(s).", "/mission-control", "coverage", "M")
    if runtime["cron"]["enabled"] == 0:
        add("next", "Add one reflection or heartbeat cron", "The blueprint expects proactive/background consolidation.", "/cron", "automation", "M")
    elif runtime["cron"].get("reflectionJobs", 0) == 0:
        add("next", "Add dedicated reflection signal", "Cron exists, but Mission Control cannot see a consolidation/reflection job.", "/cron", "memory", "M")
    if runtime["skills"]["total"] == 0:
        add("next", "Install or create reusable skills", "Skills are the reusable procedure layer of the agent.", "/skills", "skills", "S")
    if runtime["analytics"].get("daily", {}).get("budgetAlertState") == "not_configured":
        add("watch", "Set a daily cost threshold", "Token/cost totals exist, but budget alert threshold is not configured.", "/analytics", "analytics", "S")
    if not actions:
        add("watch", "Keep monitoring Mission Control", "No urgent gaps detected; refresh after deploys, gateway changes, or cron edits.", "/mission-control", "monitoring", "S")
    for idx, action in enumerate(actions, start=1):
        action["rank"] = idx
    return actions[:8]

def _privacy_boundaries() -> list[dict[str, str]]:
    return [
        {"label": "Session content", "policy": "counts only", "detail": "No chat text, tool outputs, reasoning bodies, session titles, or session IDs leave the server snapshot."},
        {"label": "Secrets", "policy": "never values", "detail": "Env files are reduced to booleans, counts, and provider families only."},
        {"label": "Local paths", "policy": "compacted", "detail": "Hermes-owned paths render as ~/.hermes; arbitrary absolutes collapse to labels."},
        {"label": "Commands/logs", "policy": "metadata", "detail": "The cockpit exposes status/readiness, not shell commands, prompts, cron bodies, or log tails."},
        {"label": "Users/chats", "policy": "count-only", "detail": "Pairing, Telegram, and gateway state expose counts/platform families, never user IDs or chat IDs."},
        {"label": "Source guide", "policy": "static public", "detail": "Blueprint sections are public source metadata and safe to render in full."},
    ]

def build_mission_control_snapshot(dashboard_state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a deterministic, privacy-minimized Mission Control snapshot."""
    home = _home()
    try:
        cfg = load_config()
    except Exception:
        cfg = {}
    if not isinstance(cfg, dict):
        cfg = {}
    runtime = _build_runtime(cfg, home, dashboard_state)
    coverage = _coverage(runtime)
    return {
        "ok": True,
        "source": {
            "url": SOURCE_URL,
            "title": "Claude Agent",
            "lastChecked": _now_iso(),
            "extractedWith": "Hermes Mission Control server snapshot",
            "note": "Source guide says 26 build steps; the page also contains a 22½ prompt-injection defence anchor, tracked here as its own required item.",
        },
        "blueprint": {
            "stepCount": len(BLUEPRINT_STEPS),
            "numberedStepCount": 26,
            "hermesFeatureCount": len(HERMES_FEATURES),
            "openclawFeatureCount": len(OPENCLAW_FEATURES),
            "parts": ["MVP", "Beyond MVP", "Production-grade"],
            "architecturePieces": ARCHITECTURE_PIECES,
            "prerequisites": PREREQUISITES,
            "preflightChecks": PREFLIGHT_CHECKS,
            "customizationChecklist": CUSTOMIZATION_CHECKLIST,
            "nextTools": NEXT_TOOLS,
            "glossary": GLOSSARY,
            "troubleshooting": TROUBLESHOOTING,
            "resources": RESOURCES,
            "dataFlowSurfaces": DATA_FLOW_SURFACES,
        },
        "runtime": runtime,
        "coverage": coverage,
        "actionQueue": _action_queue(runtime, coverage),
        "privacy": _privacy_boundaries(),
        "deviceProof": {
            "principles": [
                "safe-area aware responsive grids",
                "horizontal overflow guarded by min-w-0 and wrapped text",
                "reduced-motion friendly static gradients",
                "touch targets sized for mobile navigation",
                "dense sections become compact cards on small screens",
            ],
            "breakpoints": ["360px mobile", "390px mobile", "768px tablet", "1024px desktop", "1440px wide cockpit"],
            "smokeTargets": [
                {"id": "mobile-360", "label": "360px mobile", "status": "target"},
                {"id": "mobile-390", "label": "390px mobile", "status": "target"},
                {"id": "tablet-768", "label": "768px tablet", "status": "target"},
                {"id": "desktop-1440", "label": "1440px desktop", "status": "target"},
            ],
        },
    }

def mission_control_summary() -> dict[str, Any]:
    """Small summary shape for health checks and future sidebar chips."""
    snapshot = build_mission_control_snapshot()
    return {
        "ok": snapshot["ok"],
        "readiness": snapshot["coverage"]["summary"]["readiness"],
        "totalItems": snapshot["coverage"]["summary"]["total"],
        "counts": snapshot["coverage"]["summary"]["counts"],
        "topAction": snapshot["actionQueue"][0] if snapshot["actionQueue"] else None,
        "runtime": {
            "sessions": snapshot["runtime"]["sessions"]["total"],
            "skills": snapshot["runtime"]["skills"]["total"],
            "cron": snapshot["runtime"]["cron"]["total"],
            "mcp": snapshot["runtime"]["mcp"]["configured"],
        },
    }


__all__ = [
    "BLUEPRINT_STEPS",
    "HERMES_FEATURES",
    "OPENCLAW_FEATURES",
    "build_mission_control_snapshot",
    "mission_control_summary",
]

import type { Session, Step } from "./catalog";

/* ---------------- installed IDEs (open externally) ---------------- */
export const INSTALLED_IDES = [
  { id: "cursor", name: "Cursor", version: "3.2.1", scheme: "cursor://", installed: true, attached: true },
  { id: "vscode", name: "VS Code", version: "1.104.2", scheme: "vscode://", installed: true, attached: true },
  { id: "zed", name: "Zed", version: "0.198", scheme: "zed://", installed: true, attached: false },
  { id: "windsurf", name: "Windsurf", version: "2.4.0", scheme: "windsurf://", installed: true, attached: false },
  { id: "jetbrains", name: "IntelliJ IDEA", version: "2026.1", scheme: "idea://", installed: true, attached: false },
  { id: "neovim", name: "Neovim", version: "0.11", scheme: "nvim", installed: true, attached: false },
  { id: "xcode", name: "Xcode", version: "—", scheme: "xcode://", installed: false, attached: false },
];

/* ---------------- unified general agent persona ---------------- */
export const ASSISTANT = { name: "Aro", glyph: "A", from: "#6EE7B7", to: "#10B981", model: "auto-routed" };

export const AGENT_SESSIONS: Session[] = [
  {
    id: "a1", title: "Prep Q3 roadmap review — research, deck & invites", agentId: "claude-code", branch: "workspace", status: "running", updated: "1m", group: "Today", tokens: 61_200, cost: "$0.92", pinned: true,
    steps: [
      { id: "a1u", type: "user", at: "09:12", text: "Pull the last 30 days of Linear velocity and Sentry error trends, draft a 6-slide roadmap review for Thursday, and book 45 minutes with the platform leads. Flag anything that looks off." },
      { id: "a1k", type: "thinking", ms: 2800, text: "Four sub-tasks with different tools: Linear (data), Sentry (data), docs (writing), calendar (scheduling). Research first so the deck is grounded. Scheduling needs an approval because it emails people." },
      { id: "a1t", type: "text", text: "I'll run the research in parallel, then write, then schedule. You'll get one approval prompt before any invite goes out." },
      { id: "a1m", type: "tool", tool: "mcp", title: "Linear", target: "linear.cycles.list(team=PLAT, last=4)", status: "done", ms: 920, lines: ["cycle 41  committed 38  done 35  (92%)", "cycle 42  committed 41  done 33  (80%)", "cycle 43  committed 44  done 29  (66%)  ◀ trending down", "cycle 44  committed 39  done 31  (79%)"] },
      { id: "a1s", type: "tool", tool: "mcp", title: "Sentry", target: "sentry.issues.trend(project=bridge, days=30)", status: "done", ms: 1310, lines: ["ECONNRESET in attachEditor  +212%  (first seen cycle 43)", "TimeoutError /session/*     +38%", "everything else            flat"] },
      { id: "a1b", type: "tool", tool: "browser", title: "Browser", target: "https://linear.app/platform/cycle/43 — screenshot for slide 3", status: "done", ms: 2200, lines: ["captured 1440×900 · attached as slide-3.png"] },
      { id: "a1w", type: "text", text: "Two things look off and they're linked: velocity dipped in **cycle 43** exactly when `ECONNRESET` errors in the bridge spiked. The deck leads with that correlation.\n\nDraft written to `docs/q3-roadmap-review.md` (6 slides, speaker notes included)." },
      { id: "a1a", type: "approval", command: "calendar.create(title='Q3 roadmap review', attendees=[mira, dev, ola], duration=45m, when='Thu 14:00')", reason: "Sends calendar invites to 3 people", risk: "medium", state: "open" },
    ],
  },
  { id: "a2", title: "Triage inbox and draft replies for vendor threads", agentId: "codex", branch: "workspace", status: "waiting", updated: "24m", group: "Today", tokens: 18_400, cost: "$0.21", steps: [] },
  { id: "a3", title: "Compare 4 vector DB vendors — price, latency, SOC2", agentId: "aro", branch: "workspace", status: "done", updated: "2h", group: "Today", tokens: 88_900, cost: "$1.10", steps: [] },
  { id: "a4", title: "Weekly metrics digest → Slack #platform", agentId: "glm-code", branch: "workspace", status: "done", updated: "1d", group: "Yesterday", tokens: 9_100, cost: "$0.04", steps: [] },
  { id: "a5", title: "Fill in SOC2 vendor questionnaire from policy docs", agentId: "claude-code", branch: "workspace", status: "idle", updated: "2d", group: "Earlier", tokens: 42_000, cost: "$0.66", steps: [] },
];

export function buildAgentResponse(prompt: string): Step[] {
  const id = () => Math.random().toString(36).slice(2, 8);
  const topic = prompt.length > 56 ? prompt.slice(0, 56).trim() + "…" : prompt;
  return [
    { id: id(), type: "thinking", ms: 1900, text: "Decompose into research → produce → deliver. Pick the cheapest capable model per sub-step; escalate only if confidence is low." },
    { id: id(), type: "text", text: `On **${topic}**. I'll gather what I need first, then produce the artefact, and only ask you when something leaves the workspace.` },
    { id: id(), type: "tool", tool: "browser", title: "Browser", target: "search → 3 sources opened, 1 kept", status: "done", ms: 2400, lines: ["opened: docs.example.com/pricing", "opened: github.com/…/README", "kept: 2 relevant excerpts (1.1k tokens)"] },
    { id: id(), type: "tool", tool: "mcp", title: "Files", target: "workspace.write('outputs/summary.md')", status: "done", ms: 300, lines: ["wrote 84 lines"] },
    { id: id(), type: "text", text: "Done — the result is in `outputs/summary.md` and pinned to this session. Want me to turn it into a Slack post, an email, or a task list?" },
  ];
}

/* ---------------- tasks ---------------- */
export type Task = {
  id: string; key: string; title: string; status: "backlog" | "ready" | "running" | "review" | "done";
  priority: "p0" | "p1" | "p2" | "p3"; source: "linear" | "github" | "manual" | "agent";
  assignee?: string; human?: boolean; estimate: string; deps?: string[]; sessionId?: string;
  criteria: string[]; tags: string[]; cost?: string;
};
export const TASKS: Task[] = [
  { id: "k1", key: "PLAT-482", title: "Editor-originated edits must carry provenance through the review queue", status: "running", priority: "p0", source: "linear", assignee: "claude-code", estimate: "2h", sessionId: "s1", criteria: ["Patch envelope has origin field", "Review UI shows editor badge", "Checkpoint on human edits too"], tags: ["bridge", "review"], cost: "$3.41" },
  { id: "k2", key: "PLAT-479", title: "Ledger idempotency keys (payments refactor, part 1)", status: "running", priority: "p1", source: "linear", assignee: "codex", estimate: "4h", criteria: ["All ledger writes idempotent", "48 unit tests green"], tags: ["payments"], cost: "$1.84" },
  { id: "k3", key: "GH-1203", title: "Reconnect test flakes on CI (ws close ordering)", status: "ready", priority: "p1", source: "github", assignee: "aro", estimate: "1h", deps: ["k1"], criteria: ["0 flakes in 50 runs"], tags: ["ci", "bridge"] },
  { id: "k4", key: "PLAT-490", title: "Windsurf + Zed adapter extensions", status: "backlog", priority: "p2", source: "linear", estimate: "6h", deps: ["k1"], criteria: ["Attach works in both", "Capability matrix updated"], tags: ["editors"] },
  { id: "k5", key: "AGT-12", title: "Decompose: document wire protocol v1", status: "backlog", priority: "p2", source: "agent", estimate: "45m", criteria: ["docs/protocol.md", "Reviewed by 1 human"], tags: ["docs"] },
  { id: "k6", key: "GH-1198", title: "Best-of-4 query planner — pick variant and land", status: "review", priority: "p1", source: "github", assignee: "codex", estimate: "30m", criteria: ["p99 ≥ +10%", "No regressions in perf suite"], tags: ["perf"], cost: "$6.40" },
  { id: "k7", key: "PLAT-471", title: "Dead code sweep across packages/*", status: "done", priority: "p3", source: "linear", assignee: "aro", estimate: "1h", criteria: ["Bundle −4%"], tags: ["chore"], cost: "$0.37" },
  { id: "k8", key: "ME-3", title: "Review PR #482 and sign off release notes", status: "ready", priority: "p0", source: "manual", human: true, estimate: "20m", criteria: ["Approve or request changes"], tags: ["human"] },
  { id: "k9", key: "AGT-13", title: "Nightly: triage new Sentry issues and open tasks", status: "done", priority: "p3", source: "agent", assignee: "glm-code", estimate: "—", criteria: ["All P0 issues have a task"], tags: ["automation"], cost: "$0.08" },
];

/* ---------------- git ---------------- */
export const REPOS = [
  { id: "g1", name: "aro/platform", path: "~/code/aro", remote: "github.com/aro/platform", branch: "feat/ide-bridge", ahead: 3, behind: 0, dirty: 2, provider: "github", default: true },
  { id: "g2", name: "aro/editors", path: "~/code/aro-editors", remote: "github.com/aro/editors", branch: "main", ahead: 0, behind: 2, dirty: 0, provider: "github", default: false },
  { id: "g3", name: "internal/payments", path: "~/code/payments", remote: "gitlab.com/acme/payments", branch: "refactor/payments", ahead: 7, behind: 1, dirty: 5, provider: "gitlab", default: false },
];
export const COMMITS = [
  { hash: "a91f3c2", msg: "bridge: resumable SessionChannel with 12ms flush", by: "claude-code", when: "9m", files: 3, agent: true },
  { hash: "77bd0e4", msg: "editor-kit: capability handshake + backoff reconnect", by: "claude-code", when: "22m", files: 2, agent: true },
  { hash: "1c3a9f8", msg: "review: accept editor-originated patches", by: "Dev Vale", when: "1h", files: 1, agent: false },
  { hash: "e02aa41", msg: "chore: vitest batch 2", by: "glm-code", when: "3h", files: 81, agent: true },
  { hash: "5d1c77b", msg: "ci: cache pnpm store", by: "Dev Vale", when: "1d", files: 1, agent: false },
];
export const PRS = [
  { num: 482, title: "IDE bridge: live sync for external editors", branch: "feat/ide-bridge", author: "claude-code", agent: true, status: "open" as const, checks: { pass: 11, fail: 0, pending: 1 }, reviews: "1 approval needed", adds: 418, dels: 96, mergeable: false },
  { num: 480, title: "Payments: ledger idempotency keys", branch: "refactor/ledger", author: "codex", agent: true, status: "open" as const, checks: { pass: 9, fail: 1, pending: 0 }, reviews: "changes requested", adds: 1_204, dels: 388, mergeable: false },
  { num: 478, title: "Dead code sweep across packages/*", branch: "chore/dead-code", author: "aro", agent: true, status: "ready" as const, checks: { pass: 12, fail: 0, pending: 0 }, reviews: "approved · 2", adds: 12, dels: 1_830, mergeable: true },
  { num: 474, title: "Best-of-4 planner — variant B", branch: "perf/query-plan", author: "codex", agent: true, status: "merged" as const, checks: { pass: 12, fail: 0, pending: 0 }, reviews: "approved", adds: 210, dels: 64, mergeable: false },
];
export const CHANGED_FILES = [
  { path: "packages/bridge/src/server.ts", adds: 14, dels: 4, status: "M", origin: "agent" },
  { path: "packages/editor-kit/src/attach.ts", adds: 41, dels: 7, status: "M", origin: "agent" },
  { path: "packages/review/src/reviewer.ts", adds: 9, dels: 3, status: "M", origin: "editor" },
  { path: "packages/bridge/test/reconnect.test.ts", adds: 62, dels: 0, status: "A", origin: "agent" },
  { path: "docs/protocol.md", adds: 30, dels: 0, status: "A", origin: "you" },
];

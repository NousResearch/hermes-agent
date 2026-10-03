/* ============================================================
   CONDUCTOR — data catalog
   ============================================================ */

export type AgentKind = "terminal" | "ide" | "cloud" | "hybrid";
export type AgentStatus = "connected" | "auth-required" | "not-installed" | "degraded";

export type Agent = {
  id: string;
  name: string;
  vendor: string;
  glyph: string;
  from: string;
  to: string;
  kind: AgentKind;
  status: AgentStatus;
  version: string;
  models: { id: string; label: string; ctx: string; price: string }[];
  strengths: string[];
  weakSpots: string[];
  invocations: number;
  successRate: number;
  p50: string;
};

export const AGENTS: Agent[] = [
  {
    id: "claude-code",
    name: "Claude Code",
    vendor: "Anthropic",
    glyph: "CC",
    from: "#c9a7ff",
    to: "#7c6bff",
    kind: "hybrid",
    status: "connected",
    version: "v2.1.239",
    models: [
      { id: "opus-4.6", label: "Opus 4.6", ctx: "200k", price: "$25 / 1M" },
      { id: "sonnet-4.6", label: "Sonnet 4.6", ctx: "200k", price: "$9 / 1M" },
      { id: "haiku-4.6", label: "Haiku 4.6", ctx: "200k", price: "$2.4 / 1M" },
    ],
    strengths: ["Plan mode", "Checkpoints / rewind", "Subagents", "Skills & MCP"],
    weakSpots: ["Verbose output", "Token hungry on /design"],
    invocations: 412,
    successRate: 94,
    p50: "1.4s",
  },
  {
    id: "codex",
    name: "Codex",
    vendor: "OpenAI",
    glyph: "CX",
    from: "#8fe8d2",
    to: "#2bb3a0",
    kind: "hybrid",
    status: "connected",
    version: "v0.94.2",
    models: [
      { id: "gpt-5.4-codex", label: "GPT-5.4-Codex", ctx: "400k", price: "$14 / 1M" },
      { id: "gpt-5.5", label: "GPT-5.5", ctx: "400k", price: "$18 / 1M" },
      { id: "gpt-5.4-mini", label: "GPT-5.4-mini", ctx: "272k", price: "$1.6 / 1M" },
    ],
    strengths: ["Long backend refactors", "Test-driven loops", "Cloud delegation"],
    weakSpots: ["No plan mode", "No pre-apply diff preview"],
    invocations: 318,
    successRate: 91,
    p50: "2.1s",
  },
  {
    id: "cursor",
    name: "Cursor Agent",
    vendor: "Cursor Labs",
    glyph: "CU",
    from: "#b9adff",
    to: "#7b6bf0",
    kind: "ide",
    status: "connected",
    version: "3.2.1",
    models: [
      { id: "composer-2.5", label: "Composer 2.5", ctx: "300k", price: "included" },
      { id: "opus-4.6", label: "Opus 4.6", ctx: "200k", price: "$25 / 1M" },
    ],
    strengths: ["Inline edits", "Design mode", "Agent tabs / multitask"],
    weakSpots: ["Editor-bound", "Rules drift"],
    invocations: 264,
    successRate: 89,
    p50: "0.9s",
  },
  {
    id: "aro",
    name: "Aro Agent",
    vendor: "samjuniors",
    glyph: "AR",
    from: "#6EE7B7",
    to: "#10B981",
    kind: "hybrid",
    status: "connected",
    version: "v1.5.0",
    models: [
      { id: "aro-4-405b", label: "Aro 4 405B", ctx: "131k", price: "free" },
      { id: "aro-4-70b", label: "Aro 4 70B", ctx: "131k", price: "free" },
      { id: "glm-4.6", label: "GLM 4.6 (routed)", ctx: "200k", price: "$0 / sub" },
    ],
    strengths: ["Self-improving loop", "39 model providers", "Runs anywhere"],
    weakSpots: ["No inline editor yet"],
    invocations: 96,
    successRate: 86,
    p50: "1.6s",
  },
  {
    id: "glm-code",
    name: "GLM Code",
    vendor: "Zhipu AI",
    glyph: "GL",
    from: "#a5f0d0",
    to: "#2fb98a",
    kind: "terminal",
    status: "connected",
    version: "v1.7.4",
    models: [
      { id: "glm-5", label: "GLM-5", ctx: "200k", price: "$1.4 / 1M" },
      { id: "glm-5-air", label: "GLM-5 Air", ctx: "128k", price: "$0.4 / 1M" },
    ],
    strengths: ["Cheapest per task", "Great for bulk edits"],
    weakSpots: ["Weaker on ambiguity"],
    invocations: 141,
    successRate: 86,
    p50: "1.8s",
  },
  {
    id: "darwin",
    name: "Darwin",
    vendor: "Darwin Labs",
    glyph: "DW",
    from: "#ffd89b",
    to: "#e2913d",
    kind: "cloud",
    status: "connected",
    version: "v0.12.0",
    models: [{ id: "darwin-evo", label: "Darwin Evo", ctx: "1M", price: "$22 / 1M" }],
    strengths: ["Self-evolving prompts", "Overnight sweeps"],
    weakSpots: ["Needs tight guardrails"],
    invocations: 58,
    successRate: 78,
    p50: "5.2s",
  },
  {
    id: "zed",
    name: "Zed Agent",
    vendor: "Zed Industries",
    glyph: "ZD",
    from: "#ffe0a3",
    to: "#f0b429",
    kind: "ide",
    status: "connected",
    version: "0.198",
    models: [
      { id: "zed-fast", label: "Zed Fast Edit", ctx: "128k", price: "included" },
      { id: "sonnet-4.6", label: "Sonnet 4.6", ctx: "200k", price: "$9 / 1M" },
    ],
    strengths: ["Instant latency", "Native multi-buffer"],
    weakSpots: ["Fewer MCP servers"],
    invocations: 73,
    successRate: 88,
    p50: "0.4s",
  },
  {
    id: "opencode",
    name: "OpenCode",
    vendor: "SST",
    glyph: "OC",
    from: "#ffb3bd",
    to: "#e05364",
    kind: "terminal",
    status: "degraded",
    version: "v0.6.9",
    models: [{ id: "any", label: "75+ providers", ctx: "—", price: "varies" }],
    strengths: ["Provider agnostic", "Local-first", "Shareable sessions"],
    weakSpots: ["Config churn"],
    invocations: 64,
    successRate: 81,
    p50: "2.6s",
  },
  {
    id: "aider",
    name: "Aider",
    vendor: "Paul Gauthier",
    glyph: "AD",
    from: "#ffe9a8",
    to: "#c9a227",
    kind: "terminal",
    status: "not-installed",
    version: "—",
    models: [{ id: "any", label: "Any LLM", ctx: "—", price: "varies" }],
    strengths: ["Git-native commits", "Repo map"],
    weakSpots: ["No subagents"],
    invocations: 0,
    successRate: 0,
    p50: "—",
  },
  {
    id: "gemini-cli",
    name: "Gemini CLI",
    vendor: "Google",
    glyph: "GM",
    from: "#a8c8ff",
    to: "#4d7ce8",
    kind: "terminal",
    status: "auth-required",
    version: "v0.11.1",
    models: [{ id: "gemini-3-pro", label: "Gemini 3 Pro", ctx: "2M", price: "$12 / 1M" }],
    strengths: ["2M context", "Free tier", "Multimodal"],
    weakSpots: ["Agentic loop shorter"],
    invocations: 22,
    successRate: 74,
    p50: "3.1s",
  },
];

export const agentById = (id: string) => AGENTS.find((a) => a.id === id) ?? AGENTS[0];

/* ========================================================================
   MODES
   ======================================================================== */
export type ModeId = "plan" | "agent" | "readonly" | "full";
export const MODES: {
  id: ModeId;
  label: string;
  hint: string;
  tone: "iris" | "cyan" | "amber" | "rose";
  shortcut: string;
  rules: string[];
}[] = [
  {
    id: "plan",
    label: "Plan",
    hint: "Reads the repo, asks questions, produces a plan. Zero writes.",
    tone: "iris",
    shortcut: "⇧⌘P",
    rules: ["No file writes", "No shell exec", "May run read-only search", "Must end with an editable plan"],
  },
  {
    id: "agent",
    label: "Agent",
    hint: "Edits files and runs local commands. Approvals for anything destructive.",
    tone: "cyan",
    shortcut: "⇧⌘A",
    rules: ["File writes allowed", "Non-destructive shell allowed", "Network blocked", "Approvals: rm, git push, migration"],
  },
  {
    id: "readonly",
    label: "Read only",
    hint: "Answer questions and explain code. Nothing touches disk.",
    tone: "amber",
    shortcut: "⇧⌘R",
    rules: ["No writes", "No exec", "No network", "Explanation only"],
  },
  {
    id: "full",
    label: "Full access",
    hint: "Autonomous: edits, shell, network, installs. Checkpoint before each write.",
    tone: "rose",
    shortcut: "⇧⌘F",
    rules: ["Everything allowed", "Auto-checkpoint on write", "Sandbox + spend cap enforced"],
  },
];

/* ========================================================================
   TRANSCRIPT
   ======================================================================== */
export type DiffLine = { t: "ctx" | "add" | "del" | "hunk"; text: string; o?: number; n?: number };
export type DiffFile = {
  path: string;
  adds: number;
  dels: number;
  lang: string;
  lines: DiffLine[];
};

export type Step =
  | {
      id: string;
      type: "user";
      text: string;
      at: string;
      attachments?: string[];
    }
  | { id: string; type: "text"; text: string; agentId?: string }
  | { id: string; type: "thinking"; text: string; ms: number }
  | {
      id: string;
      type: "tool";
      tool: "read" | "edit" | "write" | "bash" | "grep" | "mcp" | "browser" | "test" | "plan";
      title: string;
      target?: string;
      status: "done" | "running" | "error" | "pending";
      ms?: number;
      lines?: string[];
      output?: string[];
      diff?: DiffFile;
    }
  | { id: string; type: "approval"; command: string; reason: string; risk: "low" | "medium" | "high"; state: "open" | "allowed" | "denied" }
  | { id: string; type: "checkpoint"; label: string; files: number; at: string; hash: string }
  | { id: string; type: "notice"; tone: "info" | "warn" | "error"; text: string };

export type Session = {
  id: string;
  title: string;
  agentId: string;
  branch: string;
  status: "running" | "idle" | "waiting" | "done" | "failed";
  updated: string;
  group: "Today" | "Yesterday" | "Earlier";
  tokens: number;
  cost: string;
  pinned?: boolean;
  projectId?: string;
  archived?: boolean;
  favorite?: boolean;
  steps: Step[];
};

export const SESSIONS: Session[] = [
  {
    id: "s1",
    title: "IDE bridge + live sync for external editors",
    agentId: "aro",
    branch: "feat/ide-bridge",
    status: "running",
    updated: "2m",
    group: "Today",
    tokens: 148_200,
    cost: "$3.41",
    pinned: true,
    steps: [
      {
        id: "m1",
        type: "user",
        at: "09:41",
        attachments: ["bridge/README.md", "packages/bridge/src/server.ts"],
        text: "Our agents still can't be driven from a real editor. Build the IDE bridge: a local daemon that exposes the session over LSP-ish JSON-RPC so Cursor, VS Code, Windsurf and Zed can attach to a running session and stream edits back. Keep it framework-agnostic, add reconnection, and make sure the diff review panel still works when the edit came from the editor side.",
      },
      {
        id: "m2",
        type: "thinking",
        ms: 4200,
        text: "Three surfaces already exist: CLI, daemon, web. The gap is that editors poll for state instead of subscribing. I should introduce a broadcast channel per session, version the frames, and let the editor be a dumb renderer so the agent stays the source of truth. Need to be careful: Cursor and VS Code have very different extension APIs, so the contract must be transport-only.",
      },
      {
        id: "m3",
        type: "text",
        text: "I mapped the repo first. `packages/bridge` is the right home — it already has the socket plumbing and no editor-specific imports. I'll add a `aro.bridge` capability so editors can declare what they support (inline diff, terminal, worktree) and degrade gracefully.",
      },
      {
        id: "m4",
        type: "tool",
        tool: "read",
        title: "Read",
        target: "packages/bridge/src/server.ts",
        status: "done",
        ms: 180,
        lines: [
          "1   import { createServer } from './net'",
          "2   import { SessionStore } from './store'",
          "3   ",
          "4   export async function startBridge(opts: BridgeOptions) {",
          "5     const store = new SessionStore(opts.root)",
          "6     const server = createServer({ port: opts.port })",
          "7     server.on('connection', (socket) => {",
          "8       socket.send(store.snapshot())",
          "9     })",
          "10    return server",
          "11  }",
        ],
      },
      {
        id: "m5",
        type: "tool",
        tool: "grep",
        title: "Search",
        target: "grep -r \"onDidSave\" packages/ --glob '*.ts'",
        status: "done",
        ms: 640,
        lines: [
          "packages/bridge/src/server.ts:4  // no editor lifecycle hooks",
          "packages/daemon/src/session.ts:88  emit('edit', patch)",
          "packages/editor-kit/src/attach.ts:12  attachEditor(session)",
        ],
      },
      {
        id: "m6",
        type: "tool",
        tool: "plan",
        title: "Plan created",
        status: "done",
        ms: 900,
        lines: [
          "1. Add `SessionChannel` — broadcast frames, 12ms flush, resumable cursor",
          "2. `editor-kit/attach.ts` — capability handshake + reconnect w/ backoff",
          "3. Route editor-originated edits through the same diff reviewer",
          "4. Ship adapters: vscode, cursor, windsurf, zed",
          "5. Integration test: two editors on one session, no lost writes",
        ],
      },
      {
        id: "m7",
        type: "approval",
        command: "pnpm add -D ws @types/ws --filter @aro/bridge",
        reason: "Installs a new dependency in the bridge package",
        risk: "low",
        state: "allowed",
      },
      {
        id: "m8",
        type: "tool",
        tool: "edit",
        title: "Edit",
        target: "packages/bridge/src/server.ts",
        status: "done",
        ms: 320,
        diff: {
          path: "packages/bridge/src/server.ts",
          adds: 14,
          dels: 4,
          lang: "ts",
          lines: [
            { t: "hunk", text: "@@ -4,9 +4,19 @@ export async function startBridge", o: 4, n: 4 },
            { t: "ctx", text: "  const store = new SessionStore(opts.root)", o: 5, n: 5 },
            { t: "del", text: "  const server = createServer({ port: opts.port })", o: 6 },
            { t: "add", text: "  const channel = new SessionChannel(store, {", n: 6 },
            { t: "add", text: "    flushEveryMs: 12,", n: 7 },
            { t: "add", text: "    resume: 'cursor',", n: 8 },
            { t: "add", text: "  })", n: 9 },
            { t: "add", text: "  const server = createServer({ port: opts.port, channel })", n: 10 },
            { t: "ctx", text: " ", o: 7, n: 11 },
            { t: "ctx", text: "  server.on('connection', (socket) => {", o: 8, n: 12 },
            { t: "del", text: "    socket.send(store.snapshot())", o: 9 },
            { t: "add", text: "    const client = attachEditor(socket, channel, {", n: 13 },
            { t: "add", text: "      capabilities: socket.requested ?? [],", n: 14 },
            { t: "add", text: "      onEdit: (patch) => reviewer.enqueue(patch),", n: 15 },
            { t: "add", text: "    })", n: 16 },
            { t: "add", text: "    socket.on('close', client.dispose)", n: 17 },
            { t: "ctx", text: "  })", o: 10, n: 18 },
          ],
        },
      },
      {
        id: "m9",
        type: "checkpoint",
        label: "Auto-checkpoint before bridge rewrite",
        files: 6,
        at: "09:47",
        hash: "a91f3c2",
      },
      {
        id: "m10",
        type: "tool",
        tool: "test",
        title: "Test",
        target: "pnpm --filter @aro/bridge test",
        status: "running",
        ms: 0,
        output: [
          "RUN  v3.2.4 /Users/dev/aro/packages/bridge",
          "",
          " ✓ session-channel.test.ts (14 tests) 284ms",
          " ✓ attach-editor.test.ts (9 tests) 121ms",
          " • reconnect.test.ts (12 tests)  running...",
        ],
      },
    ],
  },
  {
    id: "s2",
    title: "Parallelise codex + claude on the payments refactor",
    agentId: "codex",
    branch: "refactor/payments",
    status: "waiting",
    updated: "18m",
    group: "Today",
    tokens: 96_400,
    cost: "$2.08",
    steps: [
      {
        id: "n1",
        type: "user",
        at: "08:55",
        text: "Split the payments module refactor across two agents in separate worktrees. Codex takes the ledger, Claude takes the webhooks. Reconcile at the end.",
      },
      { id: "n2", type: "text", text: "Two worktrees provisioned. I'll hold reconciliation until both land — the webhook contract depends on the ledger's new event shape." },
      { id: "n3", type: "tool", tool: "bash", title: "Shell", target: "git worktree add ../cnd-ledger -b refactor/ledger", status: "done", ms: 410, output: ["Preparing worktree (checking out 'refactor/ledger')", "HEAD is now at 4f0ba12"] },
      { id: "n4", type: "approval", command: "gh pr create --fill --base main", reason: "Publishes to the shared repository", risk: "high", state: "open" },
    ],
  },
  {
    id: "s3",
    title: "Aro: sweep dead code across packages/*",
    agentId: "aro",
    branch: "chore/dead-code",
    status: "done",
    updated: "1h",
    group: "Today",
    tokens: 41_900,
    cost: "$0.37",
    steps: [],
  },
  {
    id: "s4",
    title: "GLM: migrate 240 test files to vitest",
    agentId: "glm-code",
    branch: "chore/vitest",
    status: "done",
    updated: "3h",
    group: "Yesterday",
    tokens: 210_300,
    cost: "$1.12",
    steps: [],
  },
  {
    id: "s5",
    title: "Cursor: design-mode pass on the review panel",
    agentId: "cursor",
    branch: "ui/review-panel",
    status: "idle",
    updated: "6h",
    group: "Yesterday",
    tokens: 58_100,
    cost: "$0.94",
    steps: [],
  },
  {
    id: "s6",
    title: "Darwin: overnight sweep of flaky e2e specs",
    agentId: "darwin",
    branch: "test/flake-sweep",
    status: "failed",
    updated: "22h",
    group: "Yesterday",
    tokens: 512_700,
    cost: "$11.62",
    steps: [],
  },
  {
    id: "s7",
    title: "Zed: inline completion tuning for TSX",
    agentId: "zed",
    branch: "chore/zed-tuning",
    status: "done",
    updated: "1d",
    group: "Earlier",
    tokens: 12_400,
    cost: "$0.05",
    steps: [],
  },
  {
    id: "s8",
    title: "Codex cloud: best-of-4 on the query planner",
    agentId: "codex",
    branch: "perf/query-plan",
    status: "done",
    updated: "2d",
    group: "Earlier",
    tokens: 780_400,
    cost: "$18.20",
    steps: [],
  },
];

/* ========================================================================
   PLAN
   ======================================================================== */
export type PlanTask = {
  id: string;
  title: string;
  state: "done" | "active" | "todo" | "blocked";
  agentId?: string;
  files: string[];
  note?: string;
};
export const PLAN_TASKS: PlanTask[] = [
  { id: "p1", title: "SessionChannel with resumable cursor + 12ms flush", state: "done", agentId: "claude-code", files: ["packages/bridge/src/channel.ts"] },
  { id: "p2", title: "Capability handshake in attachEditor()", state: "done", agentId: "claude-code", files: ["packages/editor-kit/src/attach.ts"] },
  { id: "p3", title: "Route editor edits through DiffReviewer", state: "active", agentId: "claude-code", files: ["packages/review/src/reviewer.ts", "packages/bridge/src/server.ts"], note: "Blocker: reviewer assumes agent-originated patches" },
  { id: "p4", title: "VS Code + Cursor adapter extensions", state: "todo", agentId: "codex", files: ["editors/vscode/", "editors/cursor/"] },
  { id: "p5", title: "Windsurf + Zed adapter extensions", state: "todo", agentId: "zed", files: ["editors/windsurf/", "editors/zed/"] },
  { id: "p6", title: "Two-editor integration test, zero lost writes", state: "todo", agentId: "aro", files: ["packages/bridge/test/reconnect.test.ts"] },
  { id: "p7", title: "Document the wire protocol (v1)", state: "blocked", files: ["docs/protocol.md"], note: "Needs decision: binary frames or JSONL" },
];

/* ========================================================================
   RUNS (parallel / background)
   ======================================================================== */
export type Run = {
  id: string;
  title: string;
  agentId: string;
  worktree: string;
  progress: number;
  status: "running" | "queued" | "review" | "done" | "failed";
  cost: string;
  tokens: number;
  eta: string;
  variant?: string;
  steps: string[];
};
export const RUNS: Run[] = [
  {
    id: "r1",
    title: "Ledger normalisation + idempotency keys",
    agentId: "codex",
    worktree: "../cnd-ledger",
    progress: 62,
    status: "running",
    cost: "$1.84",
    tokens: 74_200,
    eta: "4m",
    steps: ["Reading ledger/*", "Writing idempotency keys", "Running 48 unit tests"],
  },
  {
    id: "r2",
    title: "Webhook retry semantics",
    agentId: "claude-code",
    worktree: "../cnd-webhooks",
    progress: 38,
    status: "running",
    cost: "$2.11",
    tokens: 61_800,
    eta: "9m",
    steps: ["Mapped retry table", "Editing retry.ts", "Simulating 5xx bursts"],
  },
  {
    id: "r3",
    title: "Best-of-4 · query planner cost model",
    agentId: "codex",
    worktree: "cloud://run-8fa2",
    progress: 91,
    status: "review",
    cost: "$6.40",
    tokens: 318_000,
    eta: "ready",
    variant: "Variant B · +14% p99 win",
    steps: ["Variant A  -3%", "Variant B  +14%", "Variant C  +2%", "Variant D  +11%"],
  },
  {
    id: "r4",
    title: "Dead code sweep across packages/*",
    agentId: "aro",
    worktree: "../cnd-sweep",
    progress: 100,
    status: "done",
    cost: "$0.37",
    tokens: 41_900,
    eta: "done",
    steps: ["Scanned 1,284 files", "Deleted 61 exports", "Updated 22 imports"],
  },
  {
    id: "r5",
    title: "Overnight flake sweep (e2e)",
    agentId: "darwin",
    worktree: "cloud://run-7c1d",
    progress: 44,
    status: "failed",
    cost: "$11.62",
    tokens: 512_700,
    eta: "halted",
    steps: ["Spend cap reached", "3 specs quarantined"],
  },
  {
    id: "r6",
    title: "Vitest migration batch 3",
    agentId: "glm-code",
    worktree: "../cnd-vitest",
    progress: 0,
    status: "queued",
    cost: "—",
    tokens: 0,
    eta: "queued",
    steps: ["Waiting for worktree lock"],
  },
];

/* ========================================================================
   EXTERNAL EDITOR BRIDGES
   ======================================================================== */
export type Bridge = {
  id: string;
  ide: string;
  version: string;
  status: "live" | "ready" | "offline" | "update";
  latency: string;
  capabilities: string[];
  activeSession?: string;
  note: string;
};
export const BRIDGES: Bridge[] = [
  {
    id: "b1",
    ide: "vscode",
    version: "1.104.2",
    status: "live",
    latency: "11ms",
    capabilities: ["Inline diff", "Terminal", "Worktrees", "Selection context"],
    activeSession: "IDE bridge + live sync",
    note: "Bidirectional. Editor edits stream into the review queue.",
  },
  {
    id: "b2",
    ide: "cursor",
    version: "3.2.1",
    status: "live",
    latency: "9ms",
    capabilities: ["Inline diff", "Agent tabs", "Design mode", "Rules sync"],
    activeSession: "Design-mode pass on the review panel",
    note: "Two-way rules sync enabled — aro rules win on conflict.",
  },
  {
    id: "b3",
    ide: "zed",
    version: "0.198",
    status: "ready",
    latency: "—",
    capabilities: ["Multi-buffer", "Inline diff"],
    note: "Extension installed. Attach a session to go live.",
  },
  {
    id: "b4",
    ide: "windsurf",
    version: "2.4.0",
    status: "update",
    latency: "18ms",
    capabilities: ["Inline diff", "Terminal"],
    note: "Extension 2.3.1 detected — update to enable worktree handoff.",
  },
  {
    id: "b5",
    ide: "jetbrains",
    version: "—",
    status: "offline",
    latency: "—",
    capabilities: [],
    note: "No native extension. Use the JetBrains AI plugin as a passthrough.",
  },
  {
    id: "b6",
    ide: "neovim",
    version: "0.11",
    status: "ready",
    latency: "4ms",
    capabilities: ["Terminal", "Lua RPC"],
    note: "Attach via :lua require('aro').attach()",
  },
];

/* ========================================================================
   CONTEXT
   ======================================================================== */
export const PINNED_FILES = [
  { path: "packages/bridge/src/server.ts", tokens: 3_420, tone: "iris" as const },
  { path: "packages/editor-kit/src/attach.ts", tokens: 2_180, tone: "cyan" as const },
  { path: "packages/review/src/reviewer.ts", tokens: 4_910, tone: "amber" as const },
  { path: "docs/protocol.md", tokens: 1_120, tone: "neutral" as const },
];

export const MCP_SERVERS = [
  { name: "github", tools: 18, status: "live", tone: "mint" as const },
  { name: "sentry", tools: 7, status: "live", tone: "mint" as const },
  { name: "linear", tools: 12, status: "live", tone: "mint" as const },
  { name: "postgres", tools: 9, status: "degraded", tone: "amber" as const },
  { name: "figma", tools: 4, status: "live", tone: "mint" as const },
];

export const RULES = [
  { id: "ru1", text: "Never touch generated/ — regenerate instead.", scope: "**", on: true },
  { id: "ru2", text: "All bridge protocol changes need a test in reconnect.test.ts.", scope: "packages/bridge/**", on: true },
  { id: "ru3", text: "Prefer pnpm; never invoke npm install directly.", scope: "**", on: true },
  { id: "ru4", text: "Always run biome format before declaring a task complete.", scope: "**", on: true },
  { id: "ru5", text: "Ask before adding a new dependency.", scope: "**", on: false },
];

export const SLASH_COMMANDS = [
  { cmd: "/plan", desc: "Draft an editable plan from the current brief", hint: "⇧⌘P" },
  { cmd: "/agent", desc: "Switch this session to another coding agent", hint: "⇧⌘A" },
  { cmd: "/bridge", desc: "Attach or detach an external editor", hint: "⌘B" },
  { cmd: "/review", desc: "Open the review queue for this session", hint: "⌘⌥B" },
  { cmd: "/rewind", desc: "Restore the last checkpoint", hint: "esc esc" },
  { cmd: "/fork", desc: "Fork this session into a parallel run", hint: "⌘⇧D" },
  { cmd: "/rules", desc: "Edit scoped rules for this repo", hint: "" },
  { cmd: "/mcp", desc: "Inspect connected MCP servers and tools", hint: "" },
  { cmd: "/design", desc: "Generate editable UI artboards before building", hint: "" },
  { cmd: "/compact", desc: "Summarise the transcript to reclaim context", hint: "" },
];

export const TOKEN_SERIES = [22, 31, 28, 44, 51, 47, 63, 72, 68, 84, 79, 96];
export const COST_SERIES = [4, 6, 5, 9, 11, 10, 14, 16, 15, 19, 18, 23];

/* ========================================================================
   DESIGN SYSTEM SPEC
   ======================================================================== */
export const DS_COLORS = [
  { group: "Surface", swatches: [
    { name: "void", hex: "#06070A", usage: "App chrome, gutters" },
    { name: "sunken", hex: "#090B10", usage: "Inputs, wells, code" },
    { name: "base", hex: "#0C0E14", usage: "Panels, modals" },
    { name: "raise", hex: "#12151D", usage: "Cards, bars, rows" },
    { name: "hover", hex: "#1A1E29", usage: "Hover target" },
    { name: "line", hex: "#1F232F", usage: "Hairline borders" },
  ]},
  { group: "Ink", swatches: [
    { name: "ink", hex: "#EAECF3", usage: "Primary text" },
    { name: "ink-2", hex: "#A3ACBE", usage: "Secondary text" },
    { name: "ink-3", hex: "#6E7890", usage: "Meta, labels" },
    { name: "ink-4", hex: "#4B5465", usage: "Placeholders, ticks" },
  ]},
  { group: "Signal", swatches: [
    { name: "iris", hex: "#7C6BFF", usage: "Brand, primary action, agent" },
    { name: "cyan", hex: "#35D6C4", usage: "Streaming / active work" },
    { name: "mint", hex: "#46D68F", usage: "Success, additions" },
    { name: "amber", hex: "#F2B33D", usage: "Caution, approvals" },
    { name: "rose", hex: "#FF6B7A", usage: "Destructive, deletions" },
    { name: "sky", hex: "#5BA8FF", usage: "Informational" },
  ]},
];

export const DS_TYPE = [
  { token: "display", cls: "font-display text-[26px] leading-[1.12] font-semibold tracking-[-.028em]", spec: "Space Grotesk · 26/29 · 600" },
  { token: "title", cls: "text-[15px] leading-[1.3] font-semibold tracking-[-.01em]", spec: "15 / 20 · 600 · -1%" },
  { token: "body", cls: "text-[13px] leading-[1.6] font-normal", spec: "13 / 21 · 400" },
  { token: "label", cls: "text-[12px] leading-[1.4] font-medium", spec: "12 / 17 · 500" },
  { token: "meta", cls: "text-[11px] leading-[1.4] font-normal text-ink-3", spec: "11 / 15 · 400" },
  { token: "mono", cls: "font-mono text-[11.5px] leading-[1.55]", spec: "JetBrains Mono 11.5" },
  { token: "overline", cls: "font-mono text-[9.5px] font-semibold tracking-[.14em] uppercase text-ink-4", spec: "9.5 · 600 · +14%" },
];

export const DS_RADII = [
  { token: "xs", px: 3, usage: "kbd, tiny chips" },
  { token: "sm", px: 5, usage: "badges, small buttons" },
  { token: "md", px: 7, usage: "buttons, inputs" },
  { token: "lg", px: 10, usage: "cards, panels" },
  { token: "xl", px: 14, usage: "modals, popovers" },
  { token: "2xl", px: 18, usage: "hero surfaces" },
];

export const DS_MOTION = [
  { token: "rise", curve: "cubic-bezier(.22,1,.36,1)", dur: "320ms", usage: "Rows and cards entering" },
  { token: "pop", curve: "cubic-bezier(.34,1.56,.64,1)", dur: "220ms", usage: "Modals, popovers" },
  { token: "slide-down", curve: "cubic-bezier(.22,1,.36,1)", dur: "200ms", usage: "Menus, disclosures" },
  { token: "shimmer", curve: "linear", dur: "2.2s", usage: "Loading skeletons" },
  { token: "breathe", curve: "ease-in-out", dur: "2.6s", usage: "Live agent pulse" },
  { token: "sweep", curve: "ease-in-out", dur: "1.6s", usage: "Indeterminate progress" },
];

export const DS_SPACING = [2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40];

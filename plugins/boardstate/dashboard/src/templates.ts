// Live-bound board templates for the Hermes dashboard.
//
// A template is just a Boardstate workspace doc. Its data widgets are the dedicated
// data-source builtins (usage / sessions / instances / cron), which self-bind to
// their Hermes REST method on the server (see registerHermesDataRpc) — so a template
// shows REAL Hermes data with zero manual bindings and graceful empty states, never
// an error cell. Applied via the non-operator `dashboard.workspace.replace` RPC.
//
// Everything here is data — Hermes-native copy throughout.

export type BoardTemplate = {
  id: string;
  name: string;
  summary: string;
  doc: unknown;
};

type Widget = {
  id: string;
  kind: string;
  title: string;
  grid: { x: number; y: number; w: number; h: number };
  collapsed: false;
  hidden: false;
  props?: Record<string, unknown>;
  bindings?: Record<string, unknown>;
};

function doc(slug: string, title: string, widgets: Widget[]) {
  return {
    schemaVersion: 1,
    workspaceVersion: 1,
    widgetsRegistry: {},
    prefs: { tabOrder: [slug] },
    tabs: [{ slug, title, icon: "layoutDashboard", hidden: false, createdBy: "system", widgets }],
  };
}

const md = (id: string, title: string, x: number, y: number, w: number, h: number, markdown: string): Widget => ({
  id,
  kind: "builtin:markdown",
  title,
  grid: { x, y, w, h },
  collapsed: false,
  hidden: false,
  props: { markdown },
});

// A data-source builtin that self-binds to its Hermes method — no `bindings` needed.
const data = (id: string, kind: string, title: string, x: number, y: number, w: number, h: number, props: Record<string, unknown> = {}): Widget => ({
  id,
  kind,
  title,
  grid: { x, y, w, h },
  collapsed: false,
  hidden: false,
  props,
});

// A grant-gated action button: one click INVOKES an approved external tool (a mutation only
// PARKS — the operator confirms; a readOnly tool runs directly). `connector`/`tool` reference
// an operator-authored connector; until the operator grants that tool the button is inert.
const actionButton = (
  id: string,
  title: string,
  x: number,
  y: number,
  w: number,
  h: number,
  connector: string,
  tool: string,
  label: string,
  args: Record<string, unknown> | null = null,
): Widget => ({
  id,
  kind: "builtin:action-button",
  title,
  grid: { x, y, w, h },
  collapsed: false,
  hidden: false,
  props: { connector, tool, label, args },
});

export const TEMPLATES: BoardTemplate[] = [
  {
    id: "agent-hq",
    name: "Agent HQ",
    summary: "Live operations overview — usage, sessions, connected instances, and schedules.",
    doc: doc("board", "Agent HQ", [
      md("header", "Overview", 0, 0, 12, 2, "# Agent HQ\nLive operations for this Hermes agent."),
      data("usage", "builtin:usage", "Usage", 0, 2, 4, 3),
      data("instances", "builtin:instances", "Instances", 4, 2, 4, 3),
      data("sessions", "builtin:sessions", "Sessions", 8, 2, 4, 5),
      data("cron", "builtin:cron", "Scheduled jobs", 0, 5, 8, 3),
    ]),
  },
  {
    id: "usage-cost",
    name: "Usage & Cost",
    summary: "Spend and token usage at a glance, with the underlying breakdown.",
    doc: doc("board", "Usage & Cost", [
      md("header", "Overview", 0, 0, 12, 2, "# Usage & Cost\nToday's spend and token consumption."),
      data("cost", "builtin:stat-card", "Cost", 0, 2, 3, 2, { metric: "todayCost", format: "usd", label: "Cost (today)" }),
      data("tokens", "builtin:stat-card", "Tokens", 3, 2, 3, 2, { metric: "todayTokens", format: "int", label: "Tokens (today)" }),
      data("usage", "builtin:usage", "Usage detail", 6, 2, 6, 3),
      data("cron", "builtin:cron", "Scheduled jobs", 0, 5, 12, 3),
    ].map((w) =>
      // The two stat-cards read live usage; the dedicated builtins self-bind.
      w.id === "cost" || w.id === "tokens"
        ? { ...w, bindings: { value: { source: "rpc", method: "usage.status" } } }
        : w,
    )),
  },
  {
    id: "sessions-monitor",
    name: "Sessions Monitor",
    summary: "Watch active sessions and connected instances in real time.",
    doc: doc("board", "Sessions Monitor", [
      md("header", "Overview", 0, 0, 12, 2, "# Sessions Monitor\nActive sessions and connected instances."),
      data("sessions", "builtin:sessions", "Sessions", 0, 2, 7, 5),
      data("instances", "builtin:instances", "Instances", 7, 2, 5, 3),
      data("usage", "builtin:usage", "Usage", 7, 5, 5, 2),
    ]),
  },
  {
    // The M5 OPERATIONAL template (epic #37 / #46). Drives OfficeCLI (`officecli mcp`) through
    // approved tools: a grant-gated action-button generates a document, and the artifacts note
    // explains where the results land. Inert until the operator authors the `officecli`
    // connector and grants the tool — a safe scaffold, never an auto-run.
    id: "office-ops",
    name: "Office Ops",
    summary: "Operate OfficeCLI — generate documents and workbooks through approved tools, artifacts on the board.",
    doc: doc("board", "Office Ops", [
      md(
        "header",
        "Overview",
        0,
        0,
        12,
        2,
        "# Office Ops\nDrive **OfficeCLI** (`officecli mcp`) through operator-approved tools. Author the `officecli` connector in `boardstate.connectors.json`, approve its tools in the approvals panel, then act below.",
      ),
      // The REAL OfficeCLI MCP (v1.0.x) exposes ONE tool, `officecli`, taking a CLI
      // command line in `command` — not per-verb tools. Probed from the live manifest.
      actionButton(
        "generate-report",
        "Quarterly report",
        0,
        2,
        4,
        3,
        "officecli",
        "officecli",
        "Generate quarterly report .docx",
        { command: "create quarterly-report.docx" },
      ),
      // Grants + parked actions live HERE — the template is self-contained: request →
      // approve → run, all on this one board.
      data("approvals", "builtin:approvals", "Approvals", 4, 2, 8, 3),
      md(
        "setup",
        "Setup",
        0,
        5,
        12,
        2,
        "### Setup\n1. Install OfficeCLI (`brew install officecli` or a GitHub release) so `officecli` is on PATH.\n2. Author `boardstate.connectors.json` in the state dir with the `officecli` stdio connector.\n3. Approve the tools you want in the approvals panel — nothing runs until you do.",
      ),
    ]),
  },
];

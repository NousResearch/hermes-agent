/* ============================================================
   ARO WORKBENCH — agent-curated memory
   What Aro learned about the user (user modeling) and the work.
   The graph canvas is 900×560 — every node carries fixed
   hand-placed coordinates so the layout is deterministic.
     · preferences  top-left     · people   top-right
     · projects     center       · skills   bottom-left
     · facts        bottom-right
   ============================================================ */

export type MemoryTone = "iris" | "mint" | "cyan" | "sky" | "amber";

export type MemoryCluster = { id: string; label: string; tone: MemoryTone };

export type MemoryNode = {
  id: string;
  clusterId: string;
  label: string;
  detail: string;
  recalls: number;
  lastRecall: string; // e.g. "2h ago"
  source: string; // session id · title
  confidence: number; // 0–100
  pinned: boolean;
  learnedThisWeek: boolean;
  x: number; // fixed position on the 900×560 canvas
  y: number;
};

export const MEMORY_CLUSTERS: MemoryCluster[] = [
  { id: "preferences", label: "preferences", tone: "iris" },
  { id: "people", label: "people", tone: "mint" },
  { id: "projects", label: "projects", tone: "cyan" },
  { id: "skills", label: "skills", tone: "sky" },
  { id: "facts", label: "facts", tone: "amber" },
];

export const MEMORY_NODES: MemoryNode[] = [
  /* ---------- preferences (top-left) ---------- */
  { id: "n-ts", clusterId: "preferences", label: "TypeScript strict, always",
    detail: "Every new package ships with strict: true — no implicit any, no excuses. Aro adds the flag before writing a line.",
    recalls: 96, lastRecall: "18m ago", source: "s1 · IDE bridge rewrite", confidence: 98, pinned: false, learnedThisWeek: false, x: 110, y: 110 },
  { id: "n-semi", clusterId: "preferences", label: "Hates semicolons",
    detail: "Strips trailing semicolons in TS and JS unless the file already uses them. The prettier config mirrors it.",
    recalls: 61, lastRecall: "3h ago", source: "s5 · design-mode pass", confidence: 91, pinned: false, learnedThisWeek: false, x: 222, y: 66 },
  { id: "n-vitest", clusterId: "preferences", label: "vitest over jest",
    detail: "Converts jest specs on sight — 240 files migrated and counting. Reach for vitest matchers and timers first.",
    recalls: 74, lastRecall: "1d ago", source: "s4 · vitest migration (240 files)", confidence: 95, pinned: false, learnedThisWeek: false, x: 228, y: 168 },
  { id: "n-tables", clusterId: "preferences", label: "Tables over prose",
    detail: "Comparisons land as tables, not paragraphs — picked up during the vector-DB vendor review.",
    recalls: 52, lastRecall: "6h ago", source: "a3 · vector DB comparison", confidence: 87, pinned: false, learnedThisWeek: false, x: 112, y: 214 },

  /* ---------- people (top-right) ---------- */
  { id: "n-maria", clusterId: "people", label: "María owns the gateway",
    detail: "Ping her for gateway deploys, routing changes and anything touching :8642. Merge window is after 16:00 local.",
    recalls: 47, lastRecall: "1h ago", source: "s1 · IDE bridge rewrite", confidence: 93, pinned: true, learnedThisWeek: false, x: 692, y: 106 },
  { id: "n-dev", clusterId: "people", label: "Dev signs off merges",
    detail: "Dev Vale reviews everything that lands on main; prefers small diffs with a checkpoint behind each.",
    recalls: 29, lastRecall: "4h ago", source: "s2 · payments refactor", confidence: 86, pinned: false, learnedThisWeek: false, x: 798, y: 68 },
  { id: "n-ola", clusterId: "people", label: "Ola runs SRE on-call",
    detail: "Escalation path for P0s: Ola first, then #platform. Sentry pages route through her rotation.",
    recalls: 12, lastRecall: "2d ago", source: "s6 · flaky e2e sweep", confidence: 74, pinned: false, learnedThisWeek: false, x: 812, y: 166 },
  { id: "n-mira", clusterId: "people", label: "Mira leads planning",
    detail: "Roadmap decks go to Mira before #platform sees them. Runs cycle planning on Mondays.",
    recalls: 22, lastRecall: "1d ago", source: "a1 · Q3 roadmap prep", confidence: 81, pinned: false, learnedThisWeek: true, x: 688, y: 202 },

  /* ---------- projects (center) ---------- */
  { id: "n-bridge", clusterId: "projects", label: "Bridge rewrite context",
    detail: "feat/ide-bridge swaps the one-shot snapshot for a streaming SessionChannel — 12ms flush, resumable cursor, editor edits go through the same reviewer.",
    recalls: 84, lastRecall: "12m ago", source: "s1 · IDE bridge rewrite", confidence: 97, pinned: true, learnedThisWeek: false, x: 452, y: 276 },
  { id: "n-payments", clusterId: "projects", label: "Ledger idempotency",
    detail: "Payments refactor part 1: every ledger write carries an idempotency key. 48 tests green; part 2 is the webhook contract.",
    recalls: 41, lastRecall: "5h ago", source: "s2 · payments refactor", confidence: 90, pinned: false, learnedThisWeek: false, x: 348, y: 208 },
  { id: "n-planner", clusterId: "projects", label: "Best-of-4 planner",
    detail: "Query-planner sweep ran 4 variants — B won p99 by +12% and landed as PR #474. Re-run the sweep before touching it again.",
    recalls: 33, lastRecall: "1d ago", source: "s8 · best-of-4 on the query planner", confidence: 88, pinned: false, learnedThisWeek: true, x: 566, y: 200 },
  { id: "n-tokens", clusterId: "projects", label: "Design tokens landed",
    detail: "Five themes ride the token layer now. New surfaces use tokens, never raw hex — the prototype proved it out.",
    recalls: 27, lastRecall: "2d ago", source: "s5 · design-mode pass", confidence: 92, pinned: false, learnedThisWeek: true, x: 362, y: 356 },

  /* ---------- skills (bottom-left) ---------- */
  { id: "n-reln", clusterId: "skills", label: "New skill: release-notes",
    detail: "Drafts notes from merged PRs since the last tag. Kept after 42 clean runs — v2.1.0.",
    recalls: 42, lastRecall: "3h ago", source: "s3 · dead code sweep", confidence: 96, pinned: false, learnedThisWeek: true, x: 150, y: 420 },
  { id: "n-digest", clusterId: "skills", label: "weekly-digest, Mondays",
    detail: "Agent-authored: posts velocity, spend and incidents to #platform every Monday at 09:00 local.",
    recalls: 26, lastRecall: "1d ago", source: "a4 · weekly metrics digest", confidence: 91, pinned: false, learnedThisWeek: false, x: 102, y: 496 },
  { id: "n-rollback", clusterId: "skills", label: "Migrations ship rollback",
    detail: "Every migration Aro writes pairs a dry-run with a rollback step — learned the hard way in cycle 43.",
    recalls: 9, lastRecall: "3d ago", source: "s2 · payments refactor", confidence: 78, pinned: false, learnedThisWeek: true, x: 244, y: 486 },

  /* ---------- facts (bottom-right) ---------- */
  { id: "n-tz", clusterId: "facts", label: "Timezone Asia/Calcutta",
    detail: "Working hours 10:00–19:00 IST, standup 10:15. Cron jobs schedule around both.",
    recalls: 58, lastRecall: "1h ago", source: "a1 · Q3 roadmap prep", confidence: 99, pinned: true, learnedThisWeek: false, x: 706, y: 416 },
  { id: "n-pnpm", clusterId: "facts", label: "pnpm, never npm",
    detail: "The workspace runs on pnpm — npm or yarn invocations get rewritten before they run.",
    recalls: 66, lastRecall: "20m ago", source: "s3 · dead code sweep", confidence: 97, pinned: false, learnedThisWeek: false, x: 808, y: 388 },
  { id: "n-cadence", clusterId: "facts", label: "Ships Fridays, reviews Mondays",
    detail: "PRs merge Friday afternoon, review pass Monday 10:00. Release notes land before the Friday cutoff.",
    recalls: 19, lastRecall: "2h ago", source: "s1 · IDE bridge rewrite", confidence: 83, pinned: false, learnedThisWeek: true, x: 762, y: 494 },
];

/* associations between nodes (undirected) */
export const MEMORY_EDGES: [string, string][] = [
  ["n-ts", "n-semi"], // code style pair
  ["n-ts", "n-vitest"], // tooling defaults
  ["n-vitest", "n-pnpm"], // workspace tooling
  ["n-tables", "n-planner"], // comparisons shown as tables
  ["n-cadence", "n-dev"], // who reviews on Monday
  ["n-cadence", "n-reln"], // ship ritual
  ["n-tz", "n-cadence"], // schedule
  ["n-tz", "n-digest"], // Monday 09:00 local
  ["n-mira", "n-digest"], // digest routes to platform planning
  ["n-maria", "n-dev"], // platform leads
  ["n-maria", "n-bridge"], // gateway owner
  ["n-ola", "n-bridge"], // on-call for bridge errors
  ["n-bridge", "n-reln"], // PR #482 notes
  ["n-bridge", "n-tokens"], // built on the token layer
  ["n-payments", "n-rollback"], // migration pattern
];

/* digest for the "learned this week" strip — id matches the node it promotes */
export const LEARNED_THIS_WEEK: { id: string; label: string; detail: string; when: string; session: string }[] = [
  { id: "n-cadence", label: "Ships Fridays, reviews Mondays", detail: "Picked up from the standup ritual — merges Friday pm, review pass Monday 10:00.", when: "Mon 09:12", session: "s1 · IDE bridge rewrite" },
  { id: "n-reln", label: "New skill: release-notes", detail: "Promoted from a one-off script after 42 clean runs — v2.1.0.", when: "Tue 16:40", session: "s3 · dead code sweep" },
  { id: "n-mira", label: "Mira leads planning", detail: "Roadmap decks route to Mira before #platform.", when: "Wed 11:05", session: "a1 · Q3 roadmap prep" },
  { id: "n-tokens", label: "Design tokens landed", detail: "Five themes on the token layer — new surfaces use tokens, never raw hex.", when: "Thu 18:22", session: "s5 · design-mode pass" },
  { id: "n-rollback", label: "Migrations ship with rollback", detail: "Dry-run plus rollback is now the default migration shape.", when: "Fri 15:47", session: "s2 · payments refactor" },
  { id: "n-planner", label: "Query planner: variant B won", detail: "Best-of-4 sweep — B took p99 by +12% and landed as PR #474.", when: "Sat 10:03", session: "s8 · best-of-4 planner" },
];

/* ============================================================
   ARTIFACTS — pinned outputs from agent sessions
   Anything an agent produces (doc, deck, data, diagram, site)
   can be pinned; regenerating is one prompt away with the
   session context intact.
   ============================================================ */

export type ArtifactKind = "doc" | "deck" | "data" | "diagram" | "site";

export type Artifact = {
  id: string;
  kind: ArtifactKind;
  title: string;
  /** Session provenance — agentId must exist in the AGENTS catalog. */
  session: { title: string; agentId: string };
  updated: string;
  size: string;
  pinned: boolean;
  desc: string;
};

export const ARTIFACTS: Artifact[] = [
  { id: "f1", kind: "doc", title: "ADR-014 — streaming channel", session: { title: "IDE bridge + live sync for external editors", agentId: "claude-code" }, updated: "2h ago", size: "14 KB", pinned: true,
    desc: "Decision record for the resumable SessionChannel — framing rules, the 12ms flush cadence, and the SSE approach we rejected." },
  { id: "f2", kind: "deck", title: "Q3 review deck", session: { title: "Q3 roadmap review prep", agentId: "claude-code" }, updated: "1d ago", size: "3.2 MB", pinned: true,
    desc: "12 slides with speaker notes. Leads with the cycle-43 velocity dip and its correlation to the bridge error spike." },
  { id: "f3", kind: "data", title: "bridge throughput.csv", session: { title: "Flush latency benchmarks", agentId: "glm-code" }, updated: "5h ago", size: "220 KB", pinned: false,
    desc: "p50/p99 flush latency across 40 runs — before and after the 12ms cadence change, with env columns for each variant." },
  { id: "f4", kind: "diagram", title: "session-architecture.png", session: { title: "Protocol v1 documentation", agentId: "aro" }, updated: "2d ago", size: "480 KB", pinned: true,
    desc: "Editor → bridge → agent path with the review queue in the middle. Redrawn from the Mermaid source at 2x." },
  { id: "f5", kind: "site", title: "landing preview", session: { title: "Marketing page refresh", agentId: "cursor" }, updated: "3d ago", size: "1.1 MB", pinned: false,
    desc: "Static build of the new landing page — hero, pricing, and the agent console demo loop kept warm as a deploy preview." },
  { id: "f6", kind: "doc", title: "API reference.md", session: { title: "Protocol v1 documentation", agentId: "aro" }, updated: "1w ago", size: "62 KB", pinned: false,
    desc: "Wire-protocol reference generated from the TypeScript types — 31 endpoints with request/response examples." },
  { id: "f7", kind: "data", title: "cost model.xlsx", session: { title: "Provider cost comparison", agentId: "codex" }, updated: "1w ago", size: "96 KB", pinned: false,
    desc: "Per-task cost across 6 agents and 12 models, including the free-tier routing break-even point." },
  { id: "f8", kind: "diagram", title: "onboarding flow.excalidraw", session: { title: "First-run friction teardown", agentId: "glm-code" }, updated: "2w ago", size: "38 KB", pinned: false,
    desc: "Whiteboard of the first-run path — doctor → pick folder → first prompt — annotated with drop-off notes from the test cohort." },
  { id: "f9", kind: "doc", title: "changelog v2.4", session: { title: "Release notes v2.4", agentId: "aro" }, updated: "3w ago", size: "9 KB", pinned: false,
    desc: "Drafted from merged PRs since v2.3 — 47 entries, grouped by package and de-jargoned for the release post." },
];

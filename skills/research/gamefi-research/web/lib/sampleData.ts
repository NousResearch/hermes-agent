// Static sample data for the showcase page. No backend, no API calls, no
// secrets. Wording kept neutral and research-focused.
//
// IMPORTANT: every project below is FICTIONAL and exists only to demonstrate
// the UI. None of these are real repositories, projects, or recommendations.

export type Category = "WATCH" | "TEST" | "CONTACT" | "SKIP";

export const PR_URL =
  "https://github.com/NousResearch/hermes-agent/pull/40136";

export const workflowSteps = [
  "Scan",
  "Review Signals",
  "Score",
  "Classify",
  "Generate Report",
];

export const features = [
  "Hermes workflow skill foundation",
  "Project review template",
  "Daily report template",
  "Lightweight scanner prototype",
  "Research signal scoring",
  "WATCH / TEST / CONTACT / SKIP classification",
  "Generated Markdown report output",
];

export const roadmap = [
  "Reusable Hermes-ready interface",
  "Scheduled workflow",
  "Memory-backed comparisons",
  "Cleaner dashboard",
  "Optional scanner improvements",
];

// Short, reusable line surfaced anywhere sample projects are shown.
export const fictionalDisclaimer =
  "All sample projects below are fictional and shown only to demonstrate the UI. They are not real repositories, endorsements, or recommendations.";

export const categoryStyle: Record<Category, string> = {
  WATCH: "bg-amber-500/15 text-amber-300 ring-1 ring-amber-500/30",
  TEST: "bg-emerald-500/15 text-emerald-300 ring-1 ring-emerald-500/30",
  CONTACT: "bg-sky-500/15 text-sky-300 ring-1 ring-sky-500/30",
  SKIP: "bg-slate-500/15 text-slate-300 ring-1 ring-slate-500/30",
};

// --- Stats / cards section -------------------------------------------------
// Illustrative scan summary. Matches the shape of the scanner's report header
// (projects scanned/scored + WATCH/TEST/CONTACT/SKIP counts).

export const categoryCounts: Record<Category, number> = {
  WATCH: 4,
  TEST: 1,
  CONTACT: 2,
  SKIP: 5,
};

export const stats = {
  projectsScanned: 12,
  projectsScored: 12,
  topRanked: 6,
  publicSignalsOnly: true,
};

// --- Top projects table ----------------------------------------------------
// FICTIONAL rows only. Reasons mirror the tone of the scanner's reason_for().

export type TopProject = {
  name: string;
  score: number;
  category: Category;
  reason: string;
};

export const topProjects: TopProject[] = [
  {
    name: "Example Onchain RPG",
    score: 74,
    category: "TEST",
    reason:
      "Concrete way to try it (testnet + demo signal) with basic documentation.",
  },
  {
    name: "Sample Testnet Arena",
    score: 61,
    category: "CONTACT",
    reason: "Substantial signals and docs, but no open test yet — outreach candidate.",
  },
  {
    name: "Demo Crafting Sandbox",
    score: 58,
    category: "CONTACT",
    reason: "Active repository and clear README; worth reaching out to the team.",
  },
  {
    name: "Placeholder Card Battler",
    score: 47,
    category: "WATCH",
    reason: "Interesting signals; needs more observation before action.",
  },
  {
    name: "Mock Idle Miner",
    score: 39,
    category: "WATCH",
    reason: "Early repository with thin docs; track for updates.",
  },
  {
    name: "Test Fixture Clicker",
    score: 22,
    category: "SKIP",
    reason: "Low Web3/GameFi relevance — no domain keywords found in repo text.",
  },
];

// --- AI Research Signal Score breakdown ------------------------------------
// Component model lifted directly from the scanner's score_project(). The demo
// shows the same maxima so the score is legible, not a black box.

export type ScoreComponent = {
  label: string;
  max: string;
  detail: string;
};

export const scoreBreakdown: ScoreComponent[] = [
  {
    label: "Base",
    max: "+20",
    detail: "Every scored project starts here.",
  },
  {
    label: "Repository activity",
    max: "+0 to +20",
    detail: "Public stars and forks, capped so popularity can't dominate.",
  },
  {
    label: "Freshness",
    max: "+0 to +15",
    detail: "Newer repositories score higher; fades to 0 around 30 days old.",
  },
  {
    label: "Documentation",
    max: "+0 to +25",
    detail: "README present, detailed, and with setup / how-to-run instructions.",
  },
  {
    label: "Testing / demo",
    max: "+0 to +20",
    detail: "Early-access / testnet signals and a playable demo or download.",
  },
  {
    label: "Clarity",
    max: "+0 to +10",
    detail: "A real description plus a relevant implementation language.",
  },
  {
    label: "Relevance & risk",
    max: "penalties",
    detail:
      "Deductions for missing README, unclear purpose, promo-without-substance, and a -25 penalty when no Web3/GameFi terms appear.",
  },
];

export const scoreCeiling = 100;

// --- Relevance filter ------------------------------------------------------

export const relevanceFilter = {
  summary:
    "A lightweight relevance filter reduces false positives so off-topic repositories don't crowd the results.",
  detail:
    "A project is treated as relevant only if its public text — name, description, topics, and README — mentions at least one Web3/GameFi domain term (web3, blockchain, nft, token, crypto, gamefi, onchain, wallet, smart contract, testnet, and similar). Matching uses word boundaries, so generic names like \"gamefinder\" do not match \"gamefi\". Repositories with no domain term are penalized and forced to SKIP.",
  domainTerms: [
    "web3",
    "blockchain",
    "nft",
    "token",
    "crypto",
    "gamefi",
    "onchain",
    "wallet",
    "smart contract",
    "testnet",
  ],
};

// --- Source links ----------------------------------------------------------
// Every signal the demo references is a public source. The repo/README links
// use a fictional path purely to illustrate the report's source layout.

export type SourceLink = {
  label: string;
  href: string;
  note: string;
};

export const sourceLinks: SourceLink[] = [
  {
    label: "Pull Request — hermes-agent #40136",
    href: PR_URL,
    note: "The change set behind this workflow skill.",
  },
  {
    label: "GitHub repository search",
    href: "https://github.com/search?q=gamefi&type=repositories",
    note: "Public search the scanner draws candidates from.",
  },
  {
    label: "Sample project repository (fictional)",
    href: "https://github.com/example-org/example-onchain-rpg",
    note: "Illustrative only — this repository does not exist.",
  },
  {
    label: "Sample project README (fictional)",
    href: "https://github.com/example-org/example-onchain-rpg#readme",
    note: "Where detected documentation signals would be verified.",
  },
];

// Kept for backwards-compatibility with the existing sample card.
export const sampleProject = {
  name: "Example Onchain RPG",
  score: 74,
  classification: "TEST" as Category,
  signals: [
    "active repository",
    "clear documentation",
    "testable demo",
    "recent updates",
  ],
  nextAction: "Test the project flow and track updates.",
};

// Sample generated report — same shape the scanner writes. Neutral wording.
export const reportMarkdown = `# Game Research — Daily Report

**Date:** 2026-06-07

## Summary

Reviewed 12 projects. 1 TEST, 2 CONTACT, 4 WATCH, 5 SKIP.

## Top project

### Example Onchain RPG — 74/100 [TEST]

- **Classification:** TEST
- **Signals:** active repository, clear documentation, testable demo, recent updates
- **Project notes:** Documentation explains the core loop; a testable demo is linked; updated this week.
- **Suggested next action:** Test the project flow and track updates.

## Manual verification

All signals are automated and unverified. Open the repository, read the
documentation, and confirm the detected signals before acting.

---

*Neutral research summary for game communities. Public repository signals only.
Not advice of any kind. All sample projects are fictional. Verify all project
details manually.*
`;

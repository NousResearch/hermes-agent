// Static sample data for the showcase page. No backend, no API calls, no
// secrets. Wording kept neutral and research-focused.

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

export const sampleProject = {
  name: "Example Game Project",
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

export const categoryStyle: Record<Category, string> = {
  WATCH: "bg-amber-500/15 text-amber-300 ring-1 ring-amber-500/30",
  TEST: "bg-emerald-500/15 text-emerald-300 ring-1 ring-emerald-500/30",
  CONTACT: "bg-sky-500/15 text-sky-300 ring-1 ring-sky-500/30",
  SKIP: "bg-slate-500/15 text-slate-300 ring-1 ring-slate-500/30",
};

// Sample generated report — same shape the scanner writes. Neutral wording.
export const reportMarkdown = `# Game Research — Daily Report

**Date:** 2026-06-07

## Summary

Reviewed 12 projects. 1 TEST, 2 CONTACT, 4 WATCH, 5 SKIP.

## Top project

### Example Game Project — 74/100 [TEST]

- **Classification:** TEST
- **Signals:** active repository, clear documentation, testable demo, recent updates
- **Project notes:** Documentation explains the core loop; a testable demo is linked; updated this week.
- **Suggested next action:** Test the project flow and track updates.

## Manual verification

All signals are automated and unverified. Open the repository, read the
documentation, and confirm the detected signals before acting.

---

*Neutral research summary for game communities. Public repository signals only.
Not advice of any kind. Verify all project details manually.*
`;

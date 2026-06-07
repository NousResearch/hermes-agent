// Static sample data for the showcase UI.
// Mirrors the real output format of scripts/gamefi_scan.py (discovery →
// scoring → WATCH/TEST/CONTACT/SKIP → report). No live data, no secrets.

export type Category = "WATCH" | "TEST" | "CONTACT" | "SKIP";

export interface Project {
  rank: number;
  name: string;
  url: string;
  score: number;
  category: Category;
  reason: string;
  stats: string;
  signals: string[];
  risks: string[];
  breakdown: string[];
  sources: { repository: string; readme: string };
}

export const meta = {
  generatedAt: "2026-06-07 13:10 UTC",
  scanWindowDays: 30,
  totalScanned: 16,
  scored: 5,
};

export const counts = {
  WATCH: 0,
  TEST: 2,
  CONTACT: 1,
  SKIP: 2,
  penalized: 1,
};

export const projects: Project[] = [
  {
    rank: 1,
    name: "example-org/example-onchain-game",
    url: "https://github.com/example-org/example-onchain-game",
    score: 93,
    category: "TEST",
    reason:
      "Concrete way to try it (demo/playable signal, early-access/testnet signal) with basic documentation.",
    stats: "stars 20 | forks 4 | TypeScript | created 2026-05-20",
    signals: [
      "README present",
      "detailed README",
      "setup/run instructions",
      "early-access/testnet signal",
      "demo/playable signal",
      "relevant language (TypeScript)",
      "relevance: web3, onchain, wallet, testnet",
    ],
    risks: [],
    breakdown: [
      "base +20",
      "activity +12 (stars 20, forks 4)",
      "freshness +6 (18d old)",
      "docs +25",
      "testing/demo +20",
      "clarity +10",
      "relevance ok (web3, onchain, wallet, testnet)",
    ],
    sources: {
      repository: "https://github.com/example-org/example-onchain-game",
      readme: "https://github.com/example-org/example-onchain-game#readme",
    },
  },
  {
    rank: 2,
    name: "demo-labs/pixel-quest",
    url: "https://github.com/demo-labs/pixel-quest",
    score: 73,
    category: "TEST",
    reason:
      "Concrete way to try it (demo/playable signal) with basic documentation.",
    stats: "stars 9 | forks 1 | C# | created 2026-05-28",
    signals: [
      "README present",
      "setup/run instructions",
      "demo/playable signal",
      "relevant language (C#)",
      "relevance: web3, nft",
    ],
    risks: [],
    breakdown: [
      "base +20",
      "activity +4 (stars 9, forks 1)",
      "freshness +10 (10d old)",
      "docs +19",
      "testing/demo +10",
      "clarity +10",
      "relevance ok (web3, nft)",
    ],
    sources: {
      repository: "https://github.com/demo-labs/pixel-quest",
      readme: "https://github.com/demo-labs/pixel-quest#readme",
    },
  },
  {
    rank: 3,
    name: "guildtools/nexus-arena",
    url: "https://github.com/guildtools/nexus-arena",
    score: 61,
    category: "CONTACT",
    reason:
      "Substantial signals and docs, but no open test yet — outreach candidate.",
    stats: "stars 8 | forks 1 | TypeScript | created 2026-05-24",
    signals: [
      "README present",
      "detailed README",
      "setup/run instructions",
      "relevant language (TypeScript)",
      "relevance: web3, onchain",
    ],
    risks: [],
    breakdown: [
      "base +20",
      "activity +4 (stars 8, forks 1)",
      "freshness +8 (14d old)",
      "docs +19",
      "testing/demo +0",
      "clarity +10",
      "relevance ok (web3, onchain)",
    ],
    sources: {
      repository: "https://github.com/guildtools/nexus-arena",
      readme: "https://github.com/guildtools/nexus-arena#readme",
    },
  },
  {
    rank: 4,
    name: "someuser/gamefinder",
    url: "https://github.com/someuser/gamefinder",
    score: 32,
    category: "SKIP",
    reason: "Low Web3/GameFi relevance — no domain keywords found in repo text.",
    stats: "stars 0 | forks 0 | TypeScript | created 2026-06-01",
    signals: [
      "README present",
      "setup/run instructions",
      "relevant language (TypeScript)",
    ],
    risks: ["low web3/gamefi relevance — no domain keywords found"],
    breakdown: [
      "base +20",
      "activity +0 (stars 0, forks 0)",
      "freshness +12 (6d old)",
      "docs +15",
      "testing/demo +0",
      "clarity +10",
      "relevance -25 (no domain terms)",
    ],
    sources: {
      repository: "https://github.com/someuser/gamefinder",
      readme: "https://github.com/someuser/gamefinder#readme",
    },
  },
  {
    rank: 5,
    name: "acme/whitepaper-only",
    url: "https://github.com/acme/whitepaper-only",
    score: 27,
    category: "SKIP",
    reason: "No README; not enough public information to assess.",
    stats: "stars 1 | forks 0 | unknown | created 2026-05-31",
    signals: ["relevance: onchain"],
    risks: ["no README"],
    breakdown: [
      "base +20",
      "activity +0 (stars 1, forks 0)",
      "freshness +12 (7d old)",
      "docs +0",
      "testing/demo +0",
      "clarity +5",
      "relevance ok (onchain)",
      "risk -10",
    ],
    sources: {
      repository: "https://github.com/acme/whitepaper-only",
      readme: "none found",
    },
  },
];

// Raw Markdown report preview — same shape gamefi_scan.py --report writes.
export const sampleMarkdown = `# Game Research — Scan Report

**Generated:** ${meta.generatedAt}
**Scan window:** last ${meta.scanWindowDays} days
**Projects scanned (unique):** ${meta.totalScanned}
**Projects scored:** ${meta.scored}

## Summary

- WATCH: ${counts.WATCH} | TEST: ${counts.TEST} | CONTACT: ${counts.CONTACT} | SKIP: ${counts.SKIP}
- Penalized by relevance filter (no web3/gamefi domain terms): ${counts.penalized}

## Top ${meta.scored} projects (by Game Research Signal Score)

### 1. example-org/example-onchain-game — 93/100 [TEST]

- **Recommendation:** TEST — Concrete way to try it (demo/playable signal, early-access/testnet signal) with basic documentation.
- **Stats:** stars 20 | forks 4 | language TypeScript | created 2026-05-20T10:00:00Z
- **Detected signals:** README present, detailed README, setup/run instructions, early-access/testnet signal, demo/playable signal, relevant language (TypeScript), relevance: web3, onchain, wallet, testnet
- **Risk notes:** none
- **Score breakdown:** base +20, activity +12 (stars 20, forks 4), freshness +6 (18d old), docs +25, testing/demo +20, clarity +10, relevance ok (web3, onchain, wallet, testnet)
- **Sources:**
  - Repository: https://github.com/example-org/example-onchain-game
  - README: https://github.com/example-org/example-onchain-game#readme

### 4. someuser/gamefinder — 32/100 [SKIP]

- **Recommendation:** SKIP — Low Web3/GameFi relevance — no domain keywords found in repo text.
- **Stats:** stars 0 | forks 0 | language TypeScript | created 2026-06-01T00:00:00Z
- **Detected signals:** README present, setup/run instructions, relevant language (TypeScript)
- **Risk notes:** low web3/gamefi relevance — no domain keywords found
- **Score breakdown:** base +20, activity +0 (stars 0, forks 0), freshness +12 (6d old), docs +15, testing/demo +0, clarity +10, relevance -25 (no domain terms)
- **Sources:**
  - Repository: https://github.com/someuser/gamefinder
  - README: https://github.com/someuser/gamefinder#readme

## Manual verification

All signals above are automated and **unverified**. Before acting on any entry, open the repository, read the README, and confirm the detected signals (testnet / demo / early access, etc.) are real and current.

---

*Game Research Signal Score reflects research signal strength, not financial merit. Neutral research summary — not financial advice. Verify all project claims manually.*
`;

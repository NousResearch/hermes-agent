// Throwaway generator: deterministic fake star-map graphs → real share codes
// (runs the actual encoder, so every string round-trips). Run with `npx tsx`.
import { writeFileSync } from 'node:fs'

import type { StarmapEdge, StarmapGraph, StarmapMemoryCard, StarmapNode } from '../src/types/hermes'

import { decodeShareCode, encodeShareCode } from '../src/app/starmap/share-code'

const DAY = 86_400
const END = Math.floor(Date.UTC(2026, 5, 29) / 1000)

// mulberry32 — tiny seeded PRNG so the output is byte-stable across runs.
const rng = (seed: number) => () => {
  seed |= 0
  seed = (seed + 0x6d2b79f5) | 0
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t

  return ((t ^ (t >>> 14)) >>> 0) / 4_294_967_296
}

const pick = <T>(arr: readonly T[], r: number): T => arr[Math.floor(r * arr.length)]!

const CATEGORIES = ['devops', 'research', 'creative', 'security', 'mlops', 'blockchain', 'email', 'health', 'web-development', 'comms'] as const
const STATES = ['active', 'active', 'active', 'archived', 'draft', 'disabled'] as const
const CREATED = [null, 'agent', 'agent', 'user'] as const

const skill = (id: string, label: string, ts: number, r: () => number): StarmapNode => ({
  category: pick(CATEGORIES, r()),
  createdBy: pick(CREATED, r()),
  id,
  kind: 'skill',
  label,
  pinned: r() > 0.85,
  state: pick(STATES, r()),
  timestamp: ts,
  useCount: Math.floor(r() ** 3 * 120)
})

const memNode = (i: number, source: 'memory' | 'profile', label: string, ts: null | number): StarmapNode => ({
  category: 'memory',
  createdBy: 'memory',
  id: `memory:${source}:${i}`,
  kind: 'memory',
  label,
  memorySource: source,
  pinned: false,
  state: 'active',
  timestamp: ts,
  useCount: 0
})

const card = (source: 'memory' | 'profile', title: string, body: string, ts: null | number): StarmapMemoryCard => ({ body, source, timestamp: ts, title })

// ── 1. Tiny + quirky ──────────────────────────────────────────────────────────
function tiny(): StarmapGraph {
  const r = rng(7)
  const nodes: StarmapNode[] = [
    skill('summon-coffee', 'Summon Coffee', END - 40 * DAY, r),
    skill('rubber-duck', 'Rubber-Duck Debugging', END - 22 * DAY, r),
    skill('git-blame-zen', 'Git Blame Without Rage', END - 9 * DAY, r),
    memNode(0, 'profile', 'Prefers tabs, dies on this hill', END - 30 * DAY),
    memNode(1, 'memory', 'The prod incident of last Tuesday', END - 3 * DAY)
  ]
  const edges: StarmapEdge[] = [
    { source: 'memory:memory:1', target: 'git-blame-zen' },
    { source: 'rubber-duck', target: 'git-blame-zen' }
  ]
  const memory = [
    card('profile', 'Prefers tabs, dies on this hill', 'Tabs over spaces. Non-negotiable.', END - 30 * DAY),
    card('memory', 'The prod incident of last Tuesday', 'Never deploy on a Friday again.', END - 3 * DAY)
  ]

  return { clusters: [], edges, memory, nodes, stats: {} }
}

// ── 2. Mid-size, mixed signal ────────────────────────────────────────────────
function mid(): StarmapGraph {
  const r = rng(42)
  const names = ['Kubernetes Whispering', 'Prompt Surgery', 'Threat Modeling', 'Pixel Pushing', 'Vector Janitor', 'Smart-Contract Audit', 'Inbox Zero Ops', 'Sleep Debt Tracker', 'SSR Hydration', 'Standup Telepathy', 'Flaky-Test Exorcism', 'Cost Spelunking']
  const nodes: StarmapNode[] = names.map((label, i) => skill(`s${i}`, label, END - Math.floor(r() * 200) * DAY, r))
  const memTitles = ['Hates meetings before noon', 'Lives in us-east-1', 'Allergic to YAML', 'Caffeine half-life ~5h', 'Reviews in dark mode']

  memTitles.forEach((title, i) => {
    const ts = END - Math.floor(r() * 120) * DAY
    nodes.push(memNode(i, i % 2 ? 'memory' : 'profile', title, ts))
  })

  const edges: StarmapEdge[] = []

  for (let i = 0; i < 9; i += 1) {
    edges.push({ source: `s${Math.floor(r() * names.length)}`, target: `s${Math.floor(r() * names.length)}` })
  }

  const memory = memTitles.map((title, i) => card(i % 2 ? 'memory' : 'profile', title, `${title}. Logged automatically.`, END - Math.floor(rng(99 + i)() * 120) * DAY))

  return { clusters: [], edges, memory, nodes, stats: {} }
}

// ── 3. Dense web, partly undated (ordinal fallback) ──────────────────────────
function web(): StarmapGraph {
  const r = rng(1337)
  const nodes: StarmapNode[] = Array.from({ length: 22 }, (_, i) =>
    // Half the skills carry no timestamp → exercises the ordinal recency path.
    skill(`w${i}`, `Neuron ${String.fromCharCode(65 + (i % 26))}${i}`, i % 2 ? END - Math.floor(r() * 300) * DAY : (null as unknown as number), r)
  )
  const edges: StarmapEdge[] = []

  for (let i = 0; i < 44; i += 1) {
    edges.push({ source: `w${Math.floor(r() * 22)}`, target: `w${Math.floor(r() * 22)}` })
  }

  return { clusters: [], edges, memory: [], nodes, stats: {} }
}

// ── 4. The beast: ~2 years, hundreds of nodes, bursty timeline ───────────────
function beast(): StarmapGraph {
  const r = rng(2024)
  const start = END - 730 * DAY
  const span = END - start
  const nodes: StarmapNode[] = []
  const memory: StarmapMemoryCard[] = []

  // Bursts → an interesting waveform instead of a flat smear.
  const burstAt = (q: number) => Math.floor(start + (q + (r() - 0.5) * 0.06) * span)

  for (let i = 0; i < 240; i += 1) {
    const burst = Math.floor(r() ** 1.5 * 12) / 12 // cluster toward the recent end
    nodes.push(skill(`b${i}`, `Skill ${i} · ${pick(CATEGORIES, r())}`, burstAt(burst), r))
  }

  for (let i = 0; i < 150; i += 1) {
    const ts = burstAt(Math.floor(r() ** 1.5 * 12) / 12)
    const source = r() > 0.5 ? 'memory' : 'profile'
    nodes.push(memNode(i, source, `Memory ${i}: ${pick(['quirk', 'fact', 'preference', 'incident', 'lesson'], r())}`, ts))
    memory.push(card(source, `Memory ${i}`, `Auto-captured note #${i}.`, ts))
  }

  const edges: StarmapEdge[] = []

  for (let i = 0; i < 380; i += 1) {
    const a = Math.floor(r() * 240)
    const b = Math.floor(r() * 240)

    if (a !== b) {
      edges.push({ source: `b${a}`, target: `b${b}` })
    }
  }

  return { clusters: [], edges, memory, nodes, stats: {} }
}

const graphs: [string, StarmapGraph][] = [
  ['tiny + quirky', tiny()],
  ['mid · mixed signal', mid()],
  ['dense web · half undated', web()],
  ['the beast · ~2 years', beast()]
]

const lines: string[] = []

for (const [name, g] of graphs) {
  const code = encodeShareCode(g)
  const back = decodeShareCode(code) // round-trip assert — throws if invalid
  // v2 is viz-only: nodes + edge topology survive; memory prose is dropped.
  const ok = back.nodes.length === g.nodes.length && back.edges.length <= g.edges.length
  console.log(`${ok ? 'ok ' : 'BAD'}  ${name} — ${g.nodes.length} nodes / ${g.edges.length} edges / ${g.memory.length} cards (${code.length} chars)`)
  lines.push(`# ${name} — ${g.nodes.length} nodes, ${g.edges.length} edges, ${g.memory.length} cards`, code, '')
}

writeFileSync(new URL('share-codes.txt', import.meta.url), lines.join('\n'))

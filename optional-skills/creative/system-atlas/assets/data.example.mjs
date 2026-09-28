// Single source of truth for one atlas. Copy to <atlas home>/data.mjs and edit.
// Build: bun <atlas home>/build.mjs  → writes ../SYSTEM.md and ../atlas.html
// The atlas home is docs/<system>/atlas/ in repos that commit design docs, or a
// git-ignored scratch directory in repos that only commit ADRs + CONTEXT.md.

export const META = {
  title: 'Example Agent',                 // "<title> Atlas" in the tab; "<title> — System Definition" in SYSTEM.md
  artifactUrl: '',                        // fill after first publish; keep it stable across rebuilds
  sourcePath: 'docs/example/atlas/data.mjs',
  buildCmd: 'bun docs/example/atlas/build.mjs',
  stats: [{ k: 'System', v: 'example · v0' }, { k: 'Model roles', v: '2' }], // static top-strip stats; chapter/shown/questions are added automatically
  intro: `_**This file is the living source of truth for the design.** The interactive atlas is built from the same data._`,
  onePara: `One paragraph a newcomer can read in 30 seconds: what the system is, what the loop is, what is not built yet.`,
  costModel: ['Optional: cost assumptions and a table. Leave [] to omit.'],
  deepDive: '',                           // optional: summary + links to research/
  platformGives: 'What the runtime/framework provides for free.',
  weOwn: 'What we have to build ourselves.',
  filesystem: `apps/example/\n  agent/…`,
};

// Decisions render as the SYSTEM.md table and as chapter-10's "Decisions locked" list.
export const DECISIONS = [
  { axis: 'Runtime', decision: 'What and why, in one line', adr: '[0001](./adr/0001-slug.md)' },
];

export const GROUPS = [
  { id: 'loop', title: 'The main loop' },
  { id: 'mem', title: 'Memory' },
  { id: 'sup', title: 'Supporting the loop' },
  { id: 'off', title: 'Not yet switched on' },
];

// Structure = one isometric box. Fields:
//   id/code: 1–2 letters shown on the box · name · short (≤14 chars, canvas label) · group
//   gx,gy: grid position · w,d: footprint · h: height px · kind: box|tall|store|cards|slab|screen|gate|job
//   ghost: true → dashed "designed for, not built"
//   one: one-sentence summary (shown first) · what: plain description · how: implementation (may use <code>, <mark>)
//   steps: [[name, desc], …] → the "go inside" view
//   cond: questions — string (open) | {q, r} (resolved, r = answer + date) | {q, to} (routed to a named next step)
export const NODES = [
  { id: 'U', code: 'U', name: 'Web chat', short: 'WEB CHAT', group: 'loop', gx: 1.5, gy: 7.5, w: 2, d: 2, h: 44, kind: 'screen',
    one: 'The page where you talk to the agent.',
    what: 'Plain-language description for a non-engineer.',
    how: 'Implementation with <code>identifiers</code> and <mark>key phrases</mark>.',
    steps: [['Open', 'Create or continue a session.'], ['Stream', 'Render deltas.']],
    cond: ['Where does the page live?', { q: 'Auth scheme?', r: 'Reuse existing session JWT (2026-01-01).' }] },
  { id: 'I', code: 'I', name: 'Root agent', short: 'AGENT', group: 'loop', gx: 10, gy: 2.5, w: 3, d: 3, h: 64, kind: 'tall',
    one: 'The brain — one durable conversation per person.', what: '…', how: '…', steps: [['turn.started', '…'], ['Model call', '…'], ['Tools', '…'], ['Stream', '…']], cond: [] },
  { id: 'M', code: 'M', name: 'Memory', short: 'MEMORY', group: 'mem', gx: 6.5, gy: 9.5, w: 3, d: 3, h: 24, kind: 'store',
    one: 'What the agent knows about you.', what: '…', how: '…', steps: [['Write', '…'], ['Read', '…']], cond: [{ q: 'Vendor?', to: 'Memory deep dive' }] },
  { id: 'P', code: 'P', name: 'Proactive', short: 'PROACTIVE', group: 'off', ghost: true, gx: 12.5, gy: -1, w: 2, d: 2, h: 34,
    one: 'Later: the agent reaches out on a schedule.', what: '…', how: '…', steps: [['Tick', '…']], cond: ['Notification etiquette.'] },
];

// Flows for the final "whole system" chapter. Hop = [from, to, label, payload, bend('xy'|'yx')]
export const FLOWS = [
  { id: 'turn', name: 'One turn', hops: [
    ['U', 'I', 'user message', { message: 'can you move my 3pm?' }, 'yx'],
    ['I', 'M', 'recall', { query: 'move my 3pm', k: 12 }, 'xy'],
    ['M', 'I', 'memories', { hits: 2 }, 'xy'],
    ['I', 'U', 'message.appended', { delta: 'Sure —' }, 'yx'],
  ] },
];

// Chapters = progressive disclosure. Each reveals a few structures and runs ONE small flow among revealed ones.
// The last chapter has reveal: [] and flow: null → shows everything with a flow picker.
export const CH = [
  { id: 'you', title: 'You and the agent', reveal: ['U', 'I'],
    lede: `Strip everything away and this is the system: you type, it answers.`,
    story: `<p>Two or three sentences. <mark>Highlight</mark> the one idea this chapter adds.</p>`,
    flow: [['U', 'I', 'user message', { message: '…' }], ['I', 'U', 'reply', { delta: '…' }]] },
  { id: 'know', title: 'Knowing you', reveal: ['M'],
    lede: `Before answering, the agent recalls what it knows about you.`,
    story: `<p>…</p>`,
    flow: [['U', 'I', 'user message', { message: '…' }], ['I', 'M', 'recall', { k: 12 }], ['M', 'I', 'memories', { hits: 2 }], ['I', 'U', 'reply', { delta: '…' }]] },
  { id: 'later', title: 'Later', reveal: ['P'],
    lede: `Designed for, not switched on.`, story: `<p>…</p>`,
    flow: [['P', 'I', 'scheduled turn', { kind: 'morning_brief' }]] },
  { id: 'all', title: 'The whole system', reveal: [],
    lede: `Everything at once, for free exploration.`,
    story: `<p>Choose which flow runs (bottom left). Hover anything; click to pin; → goes inside. The <mark>Open questions</mark> tab lists every question by ID.</p>`,
    flow: null },
];

// "How it's built" tab with nothing selected (HTML).
export const HOW_HTML = `<div class="eyebrow">Example · v0</div><h1 class="t">How it's built</h1><div class="sub">the shape and what sits around it</div>
<h3 class="sec">Filesystem</h3><pre>apps/example/…</pre>`;

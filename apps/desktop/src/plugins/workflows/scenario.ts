// The SCENARIO the user authors: step definitions, edges, per-step config, and
// the structured-output contracts. This is the canvas's source of truth for
// topology — an engine reports what happened to each step, it never tells us
// what the graph is.
//
// "Scenario" is the authored artifact; a RUN is one execution of it. That split
// is the industry's, not ours: a wargame scenario is "order-of-battle, game
// length, and victory conditions outlined to define a single playing", and
// Chaosium ships Call of Cthulhu adventures as scenarios. Make.com is the one
// automation product that already pairs the two — scenario, then scenario run.

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
export type StepKind = 'agent' | 'gate' | 'human' | 'wait' | 'trigger'

/** What you can mint — the same five the canvas and the runner share. */
export const STEP_KINDS: readonly {
  kind: StepKind
  title: string
  blurb: string
  doing?: string
}[] = [
  { kind: 'trigger', title: 'Trigger', blurb: 'What starts a run' },
  { kind: 'agent', title: 'Agent', blurb: 'A model runs it', doing: 'Working' },
  { kind: 'gate', title: 'Gate', blurb: 'Branch on verdicts', doing: 'Routing' },
  { kind: 'human', title: 'Human', blurb: 'Park for a person' },
  { kind: 'wait', title: 'Wait', blurb: 'Park for the world', doing: 'Waiting' }
]

export interface StepDef {
  id: string // tasks.current_step_key — what `needs:` and gate predicates name
  /** Two families, five kinds.
      WORKERS spend effort and produce schema'd output:
        agent — a model runs it
        human — a person runs it (same contract; the brain is a person, the
                spend is wall-clock). Smithers' HumanTask/ApprovalGate.
      CONTROL spends ~nothing and routes:
        trigger — what starts a run (manual, cron, webhook, event)
        gate — a decision over data that already exists (branch/join)
        wait — a decision the WORLD makes (timer, poller, external event) */
  kind: StepKind
  title: string
  profile?: string
  model?: string
  /** Kind-mark override — the visual judge is an aperture, not the agent die. */
  icon?: string
  /** The live header verb — the shimmering "Coding" / "Reviewing" line above
      the tool ticker while the step works, the GUI's streaming-title pattern. */
  doing?: string
}

// Editable, per-step config — every field is a knob a Hermes task actually has.
//   agent -> a kanban row dispatched to a profile (assignee + body + overrides)
//   gate  -> an orchestrator that inspects child summaries and re-delegates
//
// Capability wiring is not on this list and never will be. An agent reaches for
// tools, skills, and MCP servers by itself; what it may reach for belongs to
// the profile. The canvas orchestrates, it doesn't provision.
//
// Nor is the output contract. A step knows what it feeds — the graph says so —
// so the backend templates the hand-off prompt from the downstream step's goal.
// Authoring a JSON schema by hand was work the graph already encodes.
export interface StepConfig {
  title: string
  /** The task body — the agent's instruction, the human's question. Control
   *  steps have none: a gate's arms and a wait's condition are each that step's
   *  whole instruction already. */
  goal?: string
  profile?: string // tasks.assignee — the specialist that runs this step
  model?: string // tasks.model_override, else the profile's model
  blind?: boolean // excludes upstream output from the subagent's context
  maxIterations?: number // delegation.max_iterations for this step
  maxRetries?: number // tasks.max_retries
  timeoutMins?: number // tasks.max_runtime_seconds; 0 = no cap
  onFail?: OnFail // what the run does when the step can't finish
  maxLoops?: number // gate: re-delegation cap
  /** gate: its outputs, in order. First match wins. */
  arms?: Arm[]
  /** human: who the run parks on. Empty means whoever is watching. */
  assignee?: string
  /** wait: what the world has to do before the run moves on. */
  until?: WaitUntil
  /** trigger: what starts a run. Mid-run holds stay on `until`. */
  on?: TriggerOn
}

/** Which of those knobs each kind ACTUALLY has — the closed part of the schema.
 *
 *  StepConfig is one flat bag because a graph holds all four kinds and they
 *  round-trip through one payload. That's a serialisation shape, not a
 *  statement that every step has every field, and the difference used to be
 *  re-decided in four places: the inspector gated each control on `isAgent`,
 *  the card only knew how to render an agent's, the tool schema offered every
 *  key for every kind, and defaults handed a wait step a retry budget and a
 *  blind flag. Nothing enforced agreement, so `graph_update_step` could put a
 *  model on a timer and no reader would blink.
 *
 *  This is now the one declaration. Defaults build from it, updates prune to
 *  it, the inspector renders it, and the card reflects it, so a config cannot
 *  hold a knob its kind doesn't have and the canvas can't show one. */
export const KIND_FIELDS = {
  // A model runs it: everything about how hard it may try.
  agent: ['title', 'goal', 'profile', 'model', 'blind', 'maxIterations', 'maxRetries', 'timeoutMins', 'onFail'],
  // A person runs it. Same contract, but the brain is a person: no model, no
  // iteration budget, and no retries — you don't re-dispatch someone. It fails
  // one way, by nobody answering, which is the timeout and what follows it.
  human: ['title', 'goal', 'assignee', 'timeoutMins', 'onFail'],
  // Control. A gate reads verdicts that already exist, so it spends nothing and
  // has no attempt to lose: no budgets, no on-failure. Its arms ARE its
  // instruction, which is why there's no goal beside them to drift from it.
  gate: ['title', 'arms', 'maxLoops'],
  // Control. The world decides, so the step holds no opinion at all beyond what
  // it's holding out for.
  wait: ['title', 'until'],
  // Entry only. Play is always a start; cron/webhook/event reuse Hermes
  // surfaces that already exist rather than growing a second listener stack.
  trigger: ['title', 'on']
} as const satisfies Record<StepKind, readonly (keyof StepConfig)[]>

export type KindField = (typeof KIND_FIELDS)[StepKind][number]

/** What to call a field when telling someone it isn't there. The key is the
 *  wire name; this is the word the panel puts beside the control. */
export const FIELD_LABEL: Record<KindField, string> = {
  title: 'name',
  goal: 'goal',
  profile: 'profile',
  model: 'model',
  blind: 'blind flag',
  maxIterations: 'iteration budget',
  maxRetries: 'retry budget',
  timeoutMins: 'timeout',
  onFail: 'on-failure setting',
  maxLoops: 'take limit',
  arms: 'routing rules',
  assignee: 'assignee',
  until: 'wait condition',
  on: 'start condition'
}

/** Does this kind have this knob? The one question every consumer asks. */
export function hasField(kind: StepKind, field: keyof StepConfig): boolean {
  return (KIND_FIELDS[kind] as readonly string[]).includes(field)
}

/** Drop whatever the kind doesn't have. Used wherever a config is built or
 *  changed, so converting a gate to a wait leaves nothing of the gate behind
 *  and a patch can't smuggle a field in. */
export function pruneConfig(kind: StepKind, config: Partial<StepConfig>): Partial<StepConfig> {
  const out: Record<string, unknown> = {}

  for (const [k, v] of Object.entries(config)) {
    if (hasField(kind, k as keyof StepConfig)) {
      out[k] = v
    }
  }

  return out as Partial<StepConfig>
}

/** A wait is a decision the world makes, and there are only three shapes it
 *  comes in: a clock runs out, something calls us, or we keep asking. The spec
 *  is prose per type ("24h", "github.pull_request.merged", "every 5m") because
 *  the executor that honours it differs per deployment. */
export type WaitKind = 'timer' | 'event' | 'poll'

export interface WaitUntil {
  type: WaitKind
  spec: string
}

export const WAIT_KIND_OPTIONS: { value: WaitKind; label: string; hint: string }[] = [
  { value: 'timer', label: 'Timer', hint: 'e.g. 24h — the run resumes when it elapses' },
  { value: 'event', label: 'Event', hint: 'e.g. github.pull_request.merged' },
  { value: 'poll', label: 'Poll', hint: 'GET a URL until it answers, e.g. every 30s https://…' }
]

/** What starts a run. Distinct from a mid-run wait — a trigger is an entry. */
export type TriggerKind = 'manual' | 'cron' | 'webhook' | 'event'

export interface TriggerOn {
  type: TriggerKind
  spec: string
}

export const TRIGGER_KIND_OPTIONS: { value: TriggerKind; label: string; hint: string }[] = [
  { value: 'manual', label: 'Manual', hint: 'Play on the canvas starts it' },
  { value: 'cron', label: 'Cron', hint: 'e.g. every 2h — Hermes cron fires it' },
  { value: 'webhook', label: 'Webhook', hint: 'A URL you can point GitHub, Stripe, or anything at' },
  { value: 'event', label: 'Event', hint: 'e.g. github.pull_request.merged' }
]

// ---------------------------------------------------------------------------
// Branch conditions
//
// A gate's routing table is its OUTGOING EDGES, in order, each carrying the
// condition that selects it — first match wins, and the last one is normally
// `always`. Putting the condition on the edge rather than in a table inside the
// gate means the graph is the single description of where work can go: draw a
// wire out of a gate and you have made a branch, and there is no second place
// that has to agree about the target.
//
// Three tiers, because gates are not all the same size. Most are one word
// ("did everything pass?"), some want a checkable rule, and the rest are
// judgement calls. The last tier is not a cop-out: a gate IS an orchestrator
// agent reading its children's summaries, so prose is something it can actually
// evaluate — and a condition it can read is worth more than one we can parse.
// ---------------------------------------------------------------------------

/** One comparison against a step that has already reported. */
export interface Check {
  step: string
  field: 'verdict' | 'status'
  op: 'is' | 'is not'
  value: string
}

export type Predicate =
  /** Every step feeding the gate returned PASS. The AND-join. */
  | { mode: 'all-pass' }
  /** Any step feeding the gate returned FAIL — the rework branch. */
  | { mode: 'any-fail' }
  /** The default arm. Always last; nothing after it can be reached. */
  | { mode: 'always' }
  /** Checkable rules over named steps, joined by all-of or any-of. */
  | { mode: 'checks'; join: 'all' | 'any'; checks: Check[] }
  /** Prose the orchestrator reads and rules on. */
  | { mode: 'prose'; source: string }

export type PredicateMode = Predicate['mode']

export const PREDICATE_MODES: { value: PredicateMode; label: string; hint: string }[] = [
  { value: 'all-pass', label: 'All pass', hint: 'Every step feeding this gate returned PASS' },
  { value: 'any-fail', label: 'Any fail', hint: 'At least one returned FAIL' },
  { value: 'checks', label: 'Conditions', hint: 'Checkable comparisons over named steps' },
  { value: 'prose', label: 'Judgement', hint: 'The gate agent reads this and decides' },
  { value: 'always', label: 'Anything else', hint: 'The fallback arm — nothing after it is reachable' }
]

export const JOIN_OPTIONS: { value: 'all' | 'any'; label: string }[] = [
  { value: 'all', label: 'AND' },
  { value: 'any', label: 'OR' }
]

/** One output on a gate: the condition that selects it, and what to call it.
 *
 *  Arms belong to the GATE, not to the wires leaving it. They used to be read
 *  off the edges — the condition lived in `EdgeDef.when` — which made an arm
 *  and a wire the same object and meant a rule could not exist until you had
 *  already decided where it went. You couldn't lay out a routing table and
 *  then wire it, and "add a routing rule" had nowhere to put a rule, so it
 *  invented a step to point at. An output is a property of the thing that has
 *  outputs.
 *
 *  Ordered: first match wins, which is why `always` belongs last. */
export interface Arm {
  /** What a wire names in `sourceHandle` to leave by this output. */
  id: string
  when: Predicate
  /** n8n's Rename Output. A condition makes a precise label and a terrible one
   *  — `implement.verdict is PASS and judge.verdict is PASS` is what the arm
   *  tests, not what it's FOR. Naming the output puts the intent on the canvas
   *  and leaves the condition in the panel where there's room for it. */
  label?: string
}

/** The gate's spare output port. Dragging from it mints an arm of its own,
 *  which is what lets a gate have more than the two the fixed pass/fail pair
 *  allowed. */
export const NEW_BRANCH = 'new'

export const CHECK_FIELDS = ['verdict', 'status'] as const
export const CHECK_OPS = ['is', 'is not'] as const

export const defaultPredicate = (mode: PredicateMode): Predicate => {
  switch (mode) {
    case 'checks':
      return { mode, join: 'all', checks: [] }

    case 'prose':
      return { mode, source: '' }

    default:
      return { mode } as Predicate
  }
}

/** One line of routing, for the card and the inspector. Kept here so the
 *  canvas and the panel can never drift into describing a branch differently. */
export function describePredicate(p: Predicate | undefined): string {
  if (!p) {
    return 'anything else'
  }

  switch (p.mode) {
    case 'all-pass':
      return 'all pass'

    case 'any-fail':
      return 'any fail'

    case 'always':
      return 'anything else'

    case 'prose':
      return p.source.trim() || 'the gate decides'
    case 'checks': {
      if (!p.checks.length) {
        return 'no conditions yet'
      }

      const sep = p.join === 'all' ? ' and ' : ' or '

      return p.checks.map(c => `${c.step}.${c.field} ${c.op} ${c.value}`).join(sep)
    }
  }
}

/** What happens when a step exhausts its retries. */
export type OnFail = 'retry' | 'route' | 'halt'

export const ON_FAIL_OPTIONS: { value: OnFail; label: string }[] = [
  { value: 'retry', label: 'Retry' },
  { value: 'route', label: 'Route' },
  { value: 'halt', label: 'Halt' }
]

// Slugs the first starter canvas invented. They are not in any catalog —
// a card that still has one must inherit the profile / default instead.
export const PLACEHOLDER_MODELS = [
  'claude-opus-4.8',
  'gpt-5.6-sol',
  'gpt-5.3-codex',
  'deepseek-v3.2',
  'kimi-k2-thinking'
] as const

// The profiles on this machine — `hermes profiles list`. A profile is a whole
// Hermes install of its own; picking one is how a step gets a specialist.
export const PROFILES = ['designer', 'reviewer', 'judge', 'shipper', 'orchestrator']

// ---------------------------------------------------------------------------
// Seeds
// ---------------------------------------------------------------------------
type Seed = Partial<StepConfig>

const SEEDS: Record<string, Seed> = {
  implement: {
    goal: 'Implement the Figma design (board: Marketing Site v3) into code. Read frames + tokens, reuse our component library, and write components under src/. Keep diffs small and reviewable.',
    maxIterations: 40,
    maxRetries: 2,
    timeoutMins: 30,
    onFail: 'retry'
  },
  review: {
    goal: 'Review the diff against our engineering rules. Block on naming, inline styles, and a11y. Return PASS/FAIL with notes.',
    maxIterations: 20,
    maxRetries: 1,
    onFail: 'route'
  },
  judge: {
    goal: "Render the running app and compare screenshots to the Figma frames. Judge visuals only — you are given the running app, not the diff or the implementer's reasoning. Return PASS/FAIL with pixel deltas.",
    maxIterations: 20,
    maxRetries: 1,
    onFail: 'route',
    blind: true // judges the artifact, never the implementer's output
  },
  // No goal and no on-failure: a gate's instruction IS its routing rules — one
  // arm per output, each with the condition that picks it — and it has no
  // attempt to lose. Both were here, and both were pruned before anything read
  // them; the seed is where that kind of thing goes to rot unnoticed.
  gate: { maxLoops: 5 },
  approve: {
    goal: "Review the validated diff and the judge's screenshots, then approve or send back. Opening a PR is the one irreversible step, so a person signs it.",
    onFail: 'halt'
  },
  ship: {
    goal: 'Create a branch, commit with a conventional message, push, and open a PR.',
    maxIterations: 15,
    maxRetries: 1,
    onFail: 'halt'
  }
}

/** What a step of each kind starts as, before anything is authored on top.
 *
 *  These have to be complete, because the card and the inspector both render
 *  off the config: a step with no model has nothing to put in its meta row and
 *  draws a line shorter than its neighbours. That's exactly what used to
 *  happen — the defaults were read out of SEEDS by step ID, so the six steps
 *  the starter scenario names got a full config and every step anyone created
 *  got a hollow one. Seeded content is an overlay on these now, not the source
 *  of them. */
/** A config for one kind, holding only that kind's fields. Typed off
 *  KIND_FIELDS so the defaults below can't quietly seed a knob the prune then
 *  throws away — which is how a wait step came to be born with a retry budget
 *  and an on-failure setting that nothing could ever show or use. */
type ConfigOf<K extends StepKind> = Partial<Pick<StepConfig, (typeof KIND_FIELDS)[K][number]>>

const KIND_DEFAULTS: { [K in StepKind]: ConfigOf<K> } = {
  agent: {
    maxIterations: 20,
    maxRetries: 1,
    timeoutMins: 0,
    onFail: 'retry'
  },
  // A person is the brain, so there's no model and no iteration budget. The
  // spend is wall-clock and the only failure mode is nobody answering.
  human: { timeoutMins: 0, onFail: 'halt' },
  // Control steps spend nothing and have no attempt to lose. A gate only caps
  // how many times it may send work back.
  gate: {
    maxLoops: 5,
    // A gate is born branching. Both outputs exist before anything is wired to
    // them, which is the point of arms living here: you get the shape of a
    // gate, then decide where each arm goes.
    arms: [
      { id: 'pass', when: { mode: 'all-pass' } },
      { id: 'loop', when: { mode: 'any-fail' } }
    ]
  },
  wait: { until: { type: 'timer', spec: '' } },
  trigger: { on: { type: 'manual', spec: '' } }
}

export function defaultConfig(def: StepDef): StepConfig {
  // Widened deliberately: the table is typed per kind so it can't hold a knob
  // that kind doesn't have, but reading it here is the one place that wants
  // every field at once, before the prune below cuts it back down.
  const kind: Partial<StepConfig> = KIND_DEFAULTS[def.kind]
  const s = SEEDS[def.id]

  // Built wide, then cut to the kind. Writing it out per kind would be four
  // near-copies of the same precedence rule (seed, then def, then kind), and
  // the cut is the same one every other writer makes.
  const all: StepConfig = {
    title: def.title,
    goal: s?.goal ?? '',
    profile: def.profile,
    // An explicit model on the def wins; otherwise the kind's, which is unset
    // for everything but an agent.
    model: def.model ?? kind.model,
    blind: s?.blind ?? false,
    maxIterations: s?.maxIterations ?? kind.maxIterations,
    maxRetries: s?.maxRetries ?? kind.maxRetries,
    timeoutMins: s?.timeoutMins ?? kind.timeoutMins,
    onFail: s?.onFail ?? kind.onFail,
    maxLoops: s?.maxLoops ?? kind.maxLoops,
    assignee: s?.assignee,
    // Copied so two steps minted from the same defaults can't share a list or
    // an object — KIND_DEFAULTS is one instance per kind.
    arms: (s?.arms ?? kind.arms ?? []).map(a => ({ ...a })),
    until: kind.until && { ...kind.until },
    on: kind.on && { ...kind.on }
  }

  return { ...pruneConfig(def.kind, all), title: all.title } as StepConfig
}

// ---------------------------------------------------------------------------
// Scenario definition — the user's north star:
// implement -> [code_review || visual_judge] -> gate (AND-join)
//   The two validators are ONE group. The gate joins them: ALL must pass to
//   continue. If ANY fails, the group blocks — nothing proceeds to ship — and
//   the gate sends feedback straight back to implement. Only the failed
//   validator re-runs on the next take; one that already passed stays satisfied
//   (no wasted tokens re-reviewing unchanged work). All pass -> ship (PR).
// ---------------------------------------------------------------------------
export const STEP_DEFS: StepDef[] = [
  {
    id: 'start',
    kind: 'trigger',
    title: 'Play'
  },
  {
    id: 'implement',
    kind: 'agent',
    title: 'Implement UI',
    profile: 'designer',
    doing: 'Coding'
  },
  {
    id: 'review',
    kind: 'agent',
    title: 'Code Review',
    profile: 'reviewer',
    doing: 'Reviewing'
  },
  {
    id: 'judge',
    kind: 'agent',
    title: 'Visual Judge',
    profile: 'judge',
    icon: 'eye',
    doing: 'Judging'
  },
  {
    id: 'gate',
    kind: 'gate',
    title: 'Quality Gate',
    profile: 'orchestrator',
    doing: 'Routing'
  },
  {
    id: 'approve',
    kind: 'human',
    title: 'Ship Approval'
    // No profile, no model: the assignee is you. Smithers' ApprovalGate —
    // the run parks here and the elapsed wait is the card's one number.
  },
  {
    id: 'ship',
    kind: 'agent',
    title: 'Commit & PR',
    profile: 'shipper',
    doing: 'Shipping'
  }
]

export interface EdgeDef {
  id: string
  source: string
  target: string
  sourceHandle?: string
  targetHandle?: string
  loop?: boolean
}

export const EDGE_DEFS: EdgeDef[] = [
  { id: 'start->implement', source: 'start', target: 'implement' },
  { id: 'implement->review', source: 'implement', target: 'review' },
  { id: 'implement->judge', source: 'implement', target: 'judge' },
  // both validators report into the join (the group)
  { id: 'review->gate', source: 'review', target: 'gate' },
  { id: 'judge->gate', source: 'judge', target: 'gate' },
  // group passes -> approval parks the run -> ship; group fails -> loop back
  { id: 'gate->approve', source: 'gate', target: 'approve', sourceHandle: 'pass' },
  { id: 'approve->ship', source: 'approve', target: 'ship' },
  {
    id: 'gate->implement',
    source: 'gate',
    target: 'implement',
    sourceHandle: 'loop',
    targetHandle: 'loopback',
    loop: true
  }
]

// ---------------------------------------------------------------------------
// The whole authored artifact, as one value.
//
// Everything above describes pieces; this is the thing you can hand to someone.
// The canvas keeps its working copy in React Flow's nodes/edges because that is
// what the library renders, but that shape carries measured sizes, selection
// and live runtime — none of which is the scenario. A tool that wants to read,
// write, diff or validate a workflow needs the authored subset on its own, so
// it lives here and the canvas converts at the boundary.
// ---------------------------------------------------------------------------

export interface ScenarioStep {
  id: string
  kind: StepKind
  config: StepConfig
  /** Where the author left the card. Absent means "you decide" — which is what
   *  an agent authoring a graph from scratch will send. */
  position?: { x: number; y: number }
  icon?: string
  doing?: string
}

export interface Scenario {
  version: 1
  steps: ScenarioStep[]
  edges: EdgeDef[]
}

const PLACEHOLDER = new Set<string>(PLACEHOLDER_MODELS)

/** Drop invented catalog slugs so a saved starter inherits the real default. */
export function scrubScenario(scenario: Scenario): Scenario {
  let dirty = false
  const steps = scenario.steps.map(step => {
    const model = step.config.model

    if (!model || !PLACEHOLDER.has(model)) {
      return step
    }

    dirty = true
    const { model: _drop, ...config } = step.config

    return { ...step, config }
  })

  return dirty ? { ...scenario, steps } : scenario
}

/** The figma → code → review → PR scenario the plugin ships with, as a value.
 *  Offered on the empty state alongside a blank one, because a graph you can
 *  take apart teaches the schema faster than an empty canvas does. */
export const starterScenario = (): Scenario => ({
  version: 1,
  steps: STEP_DEFS.map(def => ({ id: def.id, kind: def.kind, config: defaultConfig(def), icon: def.icon })),
  edges: EDGE_DEFS
})

/** Nothing on the canvas. The composer is the way in. */
export const blankScenario = (): Scenario => ({
  version: 1,
  steps: [],
  edges: []
})

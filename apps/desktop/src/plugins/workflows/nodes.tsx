import { Button, cn, Codicon, GlyphSpinner, PRIMARY_ICON_FACE, type SpinnerName } from '@hermes/plugin-sdk'
import { Handle, type Node, type NodeProps, Position, type ReactFlowState, useStore, useUpdateNodeInternals } from '@xyflow/react'
import { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react'

import { useFlowDir, usePorts } from './direction'
import { armLabel } from './graph'
import { KindMark, kindMarkOf } from './kind-mark'
import type { FlowDir } from './layout'
import { useRunNow } from './player'
import { isBrokenReply, stepLink, type StepRuntime } from './protocol'
import { type Arm, NEW_BRANCH, type StepConfig, type StepDef, WAIT_KIND_OPTIONS } from './scenario'

// React Flow's own API says "node" — NodeProps, nodeTypes, onNodesChange — so
// the library-facing names stay. What the graph *models* is a step of the
// scenario, and that's the word the cards and the inspector use.
export interface NodeData {
  def: StepDef
  config: StepConfig
  rt: StepRuntime
  selected?: boolean
  /** Set while replaying — freezes elapsed at the viewed event. */
  frozenAt?: number | null
  [key: string]: unknown
}

function fmtElapsed(ms: number) {
  const n = Math.max(0, Math.round(ms))
  if (n < 1000) {
    return `${n}ms`
  }

  return `${(n / 1000).toFixed(1)}s`
}

function useElapsed(rt: StepRuntime, frozenAt?: number | null) {
  const [, tick] = useState(0)
  const ticking = (rt.status === 'running' || rt.status === 'looping' || rt.status === 'waiting') && frozenAt == null
  useEffect(() => {
    if (!ticking) {
      return
    }

    const id = setInterval(() => tick(n => n + 1), 100)

    return () => clearInterval(id)
  }, [ticking])

  if (rt.durationMs != null) {
    return fmtElapsed(rt.durationMs)
  }

  if (rt.startedAt) {
    return fmtElapsed((frozenAt ?? Date.now()) - rt.startedAt)
  }

  return null
}

// One stamp per step kind, sitting in a coloured tile. The tile is the only
// place the card carries its kind as colour — everything else stays neutral,
// which is what keeps a graph readable at twenty nodes instead of turning it
// into a bag of Skittles.

/** What the tile's tooltip calls the kind. */
export const KIND_LABEL: Record<string, string> = {
  agent: 'Agent',
  gate: 'Gate',
  human: 'Human',
  wait: 'Wait',
  trigger: 'Trigger'
}

/** The canvas cards read this, not React Flow's `data` prop. RF's NodeWrapper
 *  bails out when its internal node object is reused, so a type or "starts on"
 *  edit can land in the inspector (which reads `nodes` directly) and leave the
 *  card showing the previous kind. This is that same array. */
export const NodeLive = createContext<Node[]>([])

function useLiveData(id: string, fallback: unknown): NodeData {
  const nodes = useContext(NodeLive)
  const mine = nodes.find(n => n.id === id)

  return (mine?.data as NodeData | undefined) ?? (fallback as NodeData)
}

// React Flow measures a node's handles once and bakes the offsets into every
// edge path, so anything that moves a handle after that leaves the wires behind.
// Two different things move them, and they need two different mechanisms.
//
// The lift is a transform: a raised card translates upward, which no observer
// reports because the box never changes size. That one is driven by rAF for the
// length of the transition (~160ms) plus a beat.
//
// A resize is the other: the card grows when its result line or its numbers
// arrive and shrinks when the progress track goes away. A ResizeObserver is
// exact here — it fires on every frame the height is interpolating and on
// nothing else, so the edges track a growing card the same way they track a
// lifting one, without anybody having to enumerate which state changes resize.
function useCardGeometry(nodeId: string, raised: boolean, kind: string) {
  const update = useUpdateNodeInternals()
  const ref = useRef<HTMLDivElement>(null)
  const first = useRef(true)
  const dir = useFlowDir()

  // Which side a handle sits on is cached in React Flow's node internals, not
  // read from the DOM each render — so flipping the flow re-renders the card
  // with its ports on the other edge while every wire still leaves the old one.
  // Kind is here too: trigger→agent swaps the play button for a body line and
  // the card's height changes without RF remounting the wrapper.
  useEffect(() => update(nodeId), [dir, kind, nodeId, update])

  // eslint-disable-next-line no-restricted-syntax -- `first` is a mount flag, not a mirrored atom.
  useEffect(() => {
    // Skip the initial mount — nothing has moved yet, and re-measuring every
    // node on first paint fights the auto-tidy layout pass.
    if (first.current) {
      first.current = false

      return
    }

    let raf = 0
    const start = performance.now()

    const tick = () => {
      update(nodeId)

      if (performance.now() - start < 260) {
        raf = requestAnimationFrame(tick)
      }
    }

    raf = requestAnimationFrame(tick)

    return () => cancelAnimationFrame(raf)
  }, [nodeId, raised, update])

  useEffect(() => {
    const el = ref.current

    if (!el) {
      return
    }

    let mounted = false

    const ro = new ResizeObserver(() => {
      // The observer's first callback is the initial measurement, not a change.
      if (!mounted) {
        mounted = true

        return
      }

      update(nodeId)
    })

    ro.observe(el)

    return () => ro.disconnect()
  }, [nodeId, update])

  return ref
}

/** The desktop's working ring. Worn while the engine is on this step — not on
 *  selection, which has its own accent ring to answer the click. */
function NodeArc({ status }: { status: string }) {
  if (status !== 'running' && status !== 'looping') {
    return null
  }

  return <span aria-hidden className="arc-border" />
}

const STATUS_LABEL: Record<string, string> = {
  idle: 'idle',
  queued: 'queued',
  running: 'running',
  waiting: 'waiting on you',
  looping: 'reworking',
  done: 'done',
  failed: 'failed'
}

// The card's ONE statement of how the step is doing.
//
// A failed step used to say so four times over: the summary opened with "FAIL",
// the meta row spelled out "Failed", a chip on the right said "FAIL" again, and
// the card was ringed red — four objects, one fact, and three of them competing
// for the space the actual result needed.
//
// It's a glyph now, not a coloured dot. A dot could only speak in hue, which
// made "done" and "failed" the same shape distinguished by green-vs-red — the
// exact reading the rest of the canvas refuses. Settled states are the codicons
// the app's own tool rows end on (check/×), so the card's status speaks the
// product's language. The word stays on the tooltip.
//
// The two MOVING states are braille glyph spinners instead (GlyphSpinner, the
// desktop's `ui/glyph-spinner.tsx`): the app spins unicode frames, not a
// rotating ring, everywhere it reports live work — the composer status row,
// the model pill, trigger lookups — and it takes those from the Ink TUI so all
// three surfaces read the same. A CSS-rotated circle was the one liveness cue
// on this canvas that came from nowhere in the product.
const STATUS_ICON: Record<string, string> = {
  queued: 'circle-outline',
  waiting: 'bell',
  done: 'check',
  failed: 'close'
}

/** Which unicode animation a moving state runs. `braille` is the app's default
 *  (composer, model pill); `orbit` reads as going-around, which is what a
 *  loop-back take is doing. */
const STATUS_SPINNER: Record<string, SpinnerName> = {
  running: 'braille',
  looping: 'orbit'
}

function StatusSeal({ status }: { status: string }) {
  const spinner = STATUS_SPINNER[status]

  return (
    <span className={`seal seal-${status}`} title={STATUS_LABEL[status]}>
      {spinner ? (
        <GlyphSpinner ariaLabel={STATUS_LABEL[status]} spinner={spinner} />
      ) : (
        <Codicon name={STATUS_ICON[status]} />
      )}
    </span>
  )
}

// What this step is spending, in the one place that shows it. Iteration count
// moved to the inspector with the goal: it's a budget you set, not a result you
// scan for, and on the card it was a third number competing with tokens and
// elapsed for the same row.
function fmtTokens(n: number) {
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : `${n}`
}

// Everything the card knows about a step that isn't its name or its result:
// what it's doing, what it runs on, what it spent. One quiet row at the foot,
// dot-separated, all at the same weight — because none of it is the thing you
// scan a graph for. The name is.
//
// Status is NOT here. It moved to a single seal in the head (see NodeHead),
// because the card was stating one outcome in four places at once and this row
// held two of them — the spelled-out status and the verdict chip beside it.
// What's left is only what the row is for: numbers.
//
// The row is always in the DOM and collapses to nothing when it's empty, rather
// than unmounting. A height can only interpolate between two heights, and an
// element React has removed has neither — an unmounted row can only pop.
function NodeMeta({ rt, config, elapsed }: { rt: StepRuntime; config: StepConfig; elapsed: string | null }) {
  const started = rt.status !== 'idle'
  const items: { label: string; title?: string; tabular?: boolean }[] = []
  // The spec alone is the legible half — "24h", "every 5m" — so the card shows
  // that and the type explains it on hover.
  const waitTip = (u: NonNullable<StepConfig['until']>) => WAIT_KIND_OPTIONS.find(o => o.value === u.type)?.hint

  // At rest the row says what the step IS; once it starts it says what the step
  // is DOING. Both at once doesn't fit — the model name is the longest thing
  // here and the run's own numbers are what you're watching, so a running card
  // was rendering "gpt-5…" to make room for them. Configuration is on the card
  // whenever nothing is happening, and in the inspector always.
  if (!started) {
    // One line per kind, saying the thing that distinguishes two steps of it.
    // Only the agent's knobs were ever here, so every gate, human and wait card
    // rendered an empty row and you had to open the panel to tell a 24h timer
    // from a poll — which is the canvas failing to reflect config it holds.
    if (config.model) {
      items.push({ label: config.model })
    }

    if (config.blind) {
      items.push({
        label: 'blind',
        title: 'Does not see upstream output — judges the artifact only'
      })
    }

    if (config.assignee) {
      items.push({ label: config.assignee, title: 'The run parks on them' })
    }

    if (config.until?.spec) {
      items.push({ label: config.until.spec, title: waitTip(config.until) })
    }

    if (config.on) {
      items.push({
        label: config.on.spec.trim() || config.on.type,
        title: config.on.type === 'manual' ? 'Play starts it' : `Starts on ${config.on.type}`
      })
    }

    if (config.maxLoops) {
      items.push({ label: `${config.maxLoops} takes`, title: 'Sends back at most this many times' })
    }
  }

  // Held over from a previous take — not re-run, no tokens spent. The one
  // outcome word left on a card, because nothing else says it: a skipped step
  // wears its predecessor's colour, so the seal can't carry this the way it
  // carries pass and fail.
  if (rt.skipped) {
    items.push({ label: 'skipped', title: rt.skipped })
  }

  // How far, which is only a question while the step is still moving.
  if (rt.status === 'running' || rt.status === 'looping') {
    if (rt.todos.length > 0) {
      const done = rt.todos.filter(t => t.status === 'completed').length
      items.push({ label: `${done}/${rt.todos.length}`, tabular: true })
    }
  }

  if (rt.tokens > 0) {
    items.push({ label: `${fmtTokens(rt.tokens)} tok`, tabular: true })
  }

  if (started && elapsed) {
    items.push({ label: elapsed, tabular: true })
  }

  return (
    <div className={`node-meta${items.length > 0 ? ' open' : ''}`}>
      {items.map(item => (
        <span className={`meta-item${item.tabular ? ' tabular' : ''}`} key={item.label} title={item.title}>
          {item.label}
        </span>
      ))}
    </div>
  )
}

// The head: a kind mark and the name. One line.
//
// There was a type label under the title — "Agent" / "Gate" — which is the one
// thing on the card the tile beside it already says, in colour and in glyph,
// without spending a row. It was there because the reference has it, but the
// reference has no other row on the card at all; here it sat between the name
// and the step's actual state and pushed both apart.
// The status seal rides here, pinned right, once the step has started. It sits
// in the head rather than down in the meta row so it has ONE fixed home on
// every card — the meta row comes and goes with what the step has spent, and a
// state indicator that moves between rows, or vanishes with them, isn't one.
//
// The take count is NOT here. It was a chip beside the title, and a loop-back
// lit it on all four cards in the loop body at once, every one of them reading
// "take 2" — because the iteration belongs to the loop, not to the steps it
// sweeps up.
function NodeHead({
  def,
  config,
  rt,
  play
}: {
  def: StepDef
  config: StepConfig
  rt: StepRuntime
  play?: boolean
}) {
  return (
    <div className="node-head">
      <KindMark kind={kindMarkOf(def)} title={KIND_LABEL[def.kind]} />
      <div className="node-name">{config.title}</div>
      {play ? <ManualPlay /> : rt.status !== 'idle' && <StatusSeal status={rt.status} />}
    </div>
  )
}

function ManualPlay() {
  const run = useRunNow()

  return (
    <Button
      className={cn(PRIMARY_ICON_FACE, 'node-play nodrag nopan')}
      disabled={run.running}
      onClick={e => {
        e.stopPropagation()
        run.start()
      }}
      onMouseDown={e => e.stopPropagation()}
      size="icon-2xs"
      title={run.running ? 'Already running' : 'Run this workflow'}
      type="button"
    >
      <Codicon name="triangle-right" />
    </Button>
  )
}

// There is no outcome chip. It was a filled PASS / FAIL badge pinned to the
// right of the meta row, and on a settled card it was the third thing saying
// what the dot and the ring had already said in colour — the loudest of the
// three, because it was the only one with a fill behind it. Skipped and the
// todo count were the two facts it carried that nothing else did, and both are
// plain items in the meta row now (see NodeMeta).

// A summary line carries the run's own numbers — "diff +468 −0 · 8 files" —
// and the two signed ones are the only part of it a reader scans for. They get
// the app's added/removed pair so a diff on the canvas is coloured the way a
// diff is coloured everywhere else in the product.
//
// Split on the capture so the alternation is positional: with one capturing
// group, String.split puts the matches at every odd index and the text between
// them at every even one. Nothing else in a summary is signed — "#1234",
// "400≠700" and "16≠24px" all have no leading + or −, so none of them match.
const DIFF_STAT = /([+−-]\d+)/

// The engine writes a summary with its verdict on the front — "FAIL · H1
// 400≠700 · pad 16≠24px" from a judge, "group PASS → delegate Commit & PR"
// from a gate. On the card that word is another copy of a fact the seal already
// carries, and it costs the part the reader came for: the gate's line was
// truncating to "group PASS → delegate Commit & …", spending its width saying
// the same thing as the check beside it and eliding the route it chose.
//
// Stripped at render, not at the source: the verdict belongs in the event
// payload, drives the seal and the ring, and the Data tab still prints it whole.
const VERDICT_LEAD = /^(?:group\s+)?(?:PASS|FAIL)\s*(?:·|→)\s*/

function unwrapSummary(text: string) {
  const trimmed = text.replace(VERDICT_LEAD, '').trim()
  return trimmed.length >= 2 && ((trimmed.startsWith('"') && trimmed.endsWith('"')) || (trimmed.startsWith("'") && trimmed.endsWith("'")))
    ? trimmed.slice(1, -1)
    : trimmed
}

function Summary({ text }: { text: string }) {
  return (
    <>
      {unwrapSummary(text)
        .split(DIFF_STAT)
        .map((part, i) =>
          i % 2 === 1 ? (
            <span className={part.startsWith('+') ? 'stat-add' : 'stat-del'} key={i}>
              {part}
            </span>
          ) : (
            part
          )
        )}
    </>
  )
}

// The card's one content line: what this step DID. A live tool call while it
// works, the artifact link or the summary once it settles.
//
// The goal lives in the inspector. The body is what this step DID — a live
// tool call, then the artifact or the summary.
//
// Nothing is reserved here. The block used to hold its line and its progress
// track in every state so a card wouldn't resize mid-run, which traded a resize
// nobody sees for a permanent empty band on every card in the graph.
//
// The condition is "is there a line to print", not "has the step started" —
// queued is a started state with nothing to say yet, and gating on status left
// every queued card with the same empty band the reservation used to cause.
//
// "Nothing reserved" is about painted height, not about the element: the block
// stays mounted at height zero so it has something to grow from. The progress
// track inside it does the same, so a step settling gives up its bar smoothly
// instead of the card jumping 11px the instant the last tool call returns.
function NodeBody({ def, rt }: { def: StepDef; rt: StepRuntime }) {
  const live = rt.status === 'running' || rt.status === 'looping'
  // Parked on a person: no tools, no progress — the summary IS the ask, and
  // the card stays visibly open on it while the run waits.
  const waiting = rt.status === 'waiting'
  const tool = live ? rt.currentTool : null
  // Only once the step has settled — a URL mid-run is a half-written artifact.
  const link = live || waiting ? null : stepLink(rt.output)
  // A live step keeps the block even with nothing printed yet: the progress
  // track is the line in that case.
  const failed = rt.status === 'failed' || isBrokenReply(rt.summary)
  const open = live || waiting || Boolean(tool) || Boolean(rt.summary) || Boolean(link)

  // One line element in all three states, rather than three that swap. It has
  // to be a single persistent box for the same reason the blocks do: the line
  // arriving is a height change, and swapping one element for another gives the
  // browser nothing to interpolate. Collapsed it's 0; open it's a fixed 17px,
  // and because that's a real number the block above it — which is `auto` —
  // grows along with it instead of snapping when its content changes.
  //
  // Live, the block is the GUI's in-flight tool card: a shimmering header verb
  // ("Coding…", the streaming-title treatment from fallback.tsx:311) with the
  // ToolRunTicker under it — the reel of calls, each new one sliding the last
  // up and out of a one-line window (run-ticker.tsx / .tool-ticker). No
  // progress bar: a run doesn't know how done it is; it shows what it's DOING.
  const rows = live ? [...rt.toolCalls, ...(tool && tool !== rt.toolCalls[rt.toolCalls.length - 1] ? [tool] : [])] : []
  const ticking = live && rows.length > 0

  const line = ticking ? (
    <div className="tool-ticker" data-tool-ticker="">
      <div className="tool-ticker__reel" style={{ '--tool-ticker-index': rows.length - 1 } as React.CSSProperties}>
        {rows.map((c, i) => (
          <div className="tool-ticker__row" key={i}>
            <span className="tool-name">{c.name}</span>
            {c.arg && <span className="tool-arg">{c.arg}</span>}
          </div>
        ))}
      </div>
    </div>
  ) : link ? (
    // A step that produced a URL keeps its summary as the label and hangs the
    // href off it: "PR #1234 opened" is what the agent said happened, and it's
    // more readable than the URL it happened at. The artifact is what the run
    // was for — it shouldn't be three clicks into the Data tab.
    <a
      className="node-link"
      href={link.href}
      // The canvas swallows clicks to select/drag; let the anchor win.
      onClick={e => e.stopPropagation()}
      onMouseDown={e => e.stopPropagation()}
      rel="noreferrer"
      target="_blank"
      title={link.href}
    >
      {rt.summary ? <Summary text={rt.summary} /> : link.label}
    </a>
  ) : rt.summary ? (
    <Summary text={rt.summary} />
  ) : null

  return (
    <div className={`node-body${live ? ' live' : ''}${open ? ' open' : ''}`}>
      {/* The shimmering verb — "Coding", "Reviewing" — above the reel, exactly
          the GUI's streaming header + ticker stack. Own collapsible line so
          the card grows/shrinks by whole rows. */}
      <div className={`doing-line${ticking ? ' open' : ''}`}>
        <span className="shimmer">{def.doing ?? 'Working'}</span>
      </div>
      <div
        className={`logline${line ? ' open' : ''}${ticking ? ' toolline' : ''}${failed && line ? ' is-fail' : ''}`}
        title={tool ? `${tool.name} ${tool.arg}` : (rt.summary ?? undefined)}
      >
        {line}
      </div>
    </div>
  )
}

/** Where a rework arm plugs back in. It faces down because the wire it takes
 *  runs against the flow, which is the whole visual signal that the step is
 *  being redone rather than fed.
 *
 *  Every step has one — any step can be the one you go back to — but a second
 *  dot under every card is a lot of canvas spent on a port most of them will
 *  never use. So it shows when it's carrying a wire, when you're over the
 *  card, or the moment a connection starts and it becomes somewhere to drop. */
function BackPort({ id }: { id: string }) {
  const wired = useStore(
    useCallback((s: ReactFlowState) => s.edges.some(e => e.target === id && e.targetHandle === 'loopback'), [id])
  )

  const vertical = useFlowDir() === 'TB'

  // It sits on whichever face the flow ISN'T using, so a rework wire never
  // shares an edge with the forward one.
  return (
    <Handle
      className={`handle-back${wired ? ' is-wired' : ''}`}
      id="loopback"
      position={vertical ? Position.Left : Position.Bottom}
      style={vertical ? { top: '32%' } : { left: '32%' }}
      type="target"
    />
  )
}

/** Everything the card shell's outer div carries. The one step card renders
 *  this, so a hook that belongs on every kind can't be added to only one
 *  branch of the kind switch.
 *
 *  `data-tour` is the step's addressable handle. The tour engine's target
 *  scanner treats `data-tour` as the one selector durable enough to survive a
 *  re-render, and a React Flow card otherwise offers nothing it will accept —
 *  no id, no test id, no unique aria-label — so a tour of the graph could only
 *  point at nodes by `nth-child` path, which stops being the same step the
 *  moment anything reorders. `aria-label` is what makes the scan READABLE:
 *  without it a target is labelled by the card's concatenated text, and the
 *  agent picks its steps out of a wall of run output. */
const cardProps = (id: string, def: StepDef, dir: FlowDir, rt: StepRuntime, selected?: boolean) => ({
  'aria-label': `${def.title} — ${def.kind} step`,
  className: `node ${def.kind} dir-${dir.toLowerCase()} status-${rt.status}${rt.skipped ? ' skipped' : ''}${selected ? ' sel' : ''}`,
  'data-tour': `step:${id}`
})

// ---- Step card ---------------------------------------------------------------
// One component for every kind. React Flow keys wrappers by node id, not type,
// so switching Type in the inspector used to leave the previous card mounted
// (a trigger still showing Play after it became an agent). Kind is data; the
// card branches on it.

/** The gate's outputs, one row per outgoing wire.
 *
 *  The gate is the only step that branches, so it's the only one that
 *  enumerates its outputs. They're rows on the card with the port ON the row,
 *  not a sentence in the footer describing where the wires go — naming an
 *  output is a port's job, and the legend was doing it in prose while the
 *  actual handles sat unlabelled.
 *
 *  The rows are DERIVED from the edges rather than a fixed pass/fail pair: two
 *  hardcoded ports capped every gate at two arms and made the labels a lie the
 *  moment a branch said anything else. Forward arms face right, with the flow;
 *  a rework arm faces down, because it returns against it. The spare port at
 *  the bottom is what you drag to make the next branch.
 *
 *  Subscribed as a joined string so an unrelated edge change can't re-render
 *  the card — the store selector has to return something stable by value. */
function GatePorts({ id, arms }: { id: string; arms: Arm[] }) {
  // The gate's config says which outputs exist; the store only says what's
  // plugged into them. An arm with nothing plugged in still gets its row —
  // that's the port you drag from once you know where the rule should go.
  const wired = useStore(
    useCallback(
      (s: ReactFlowState) =>
        s.edges
          .filter(e => e.source === id)
          .map(e => `${e.sourceHandle ?? 'out'}${(e.data as { loop?: boolean })?.loop ? '!' : ''}`)
          .join(','),
      [id]
    )
  )

  const plugged = new Set(wired ? wired.split(',') : [])
  const ports = usePorts()

  return (
    <div className="node-ports">
      {arms.map(arm => {
        const loop = plugged.has(`${arm.id}!`)
        const open = !loop && !plugged.has(arm.id)

        return (
          <div className={`port-row${loop ? ' is-loop' : ''}${open ? ' is-open' : ''}`} key={arm.id}>
            <span className="port-label">{armLabel(arm)}</span>
            {/* Every output faces right, the rework arm included. It used to
                face Bottom so its wire left downward — but the handle sits on
                its row, mid-card, and edges paint under nodes, so the first
                stretch of the loop was hidden behind the gate and the wire read
                as coming from nowhere. It leaves at its dot now and does the
                U-turn in open canvas. */}
            <Handle id={arm.id} position={ports.source} type="source" {...(loop ? { className: 'handle-loop' } : {})} />
          </div>
        )
      })}
      <div className="port-row is-spare">
        <span className="port-label">+</span>
        <Handle id={NEW_BRANCH} position={ports.source} type="source" />
      </div>
    </div>
  )
}

export function StepNode({ id, data }: NodeProps) {
  const { def, config, rt, selected, frozenAt } = useLiveData(id, data)
  const elapsed = useElapsed(rt, frozenAt)
  const dir = useFlowDir()
  const ports = usePorts()
  const worker = def.kind === 'agent' || def.kind === 'human'
  const play = def.kind === 'trigger' && (config.on?.type ?? 'manual') === 'manual' && rt.status === 'idle'

  // Must match the CSS raised set exactly, or the edges re-measure at the
  // wrong moments. Waiting lifts a worker (the run is parked on someone) and
  // not a gate (a gate doesn't wait on a person).
  const ref = useCardGeometry(
    id,
    !!selected || rt.status === 'running' || rt.status === 'looping' || (worker && rt.status === 'waiting'),
    def.kind
  )

  return (
    <div {...cardProps(id, def, dir, rt, selected)} ref={ref}>
      {/* Every step carries the full set. These used to be conditional on two
          hardcoded seed ids — the first step had no input and the last had no
          output — so the head of the flow could never be fed and nothing could
          ever be wired after the tail. Which step is first or last is a fact
          about the wiring, not about the card, and the card was enforcing it. */}
      <Handle id="in" position={ports.target} type="target" />
      <BackPort id={id} />

      <NodeArc status={rt.status} />
      <NodeHead config={config} def={def} play={play} rt={rt} />
      <NodeBody def={def} rt={rt} />
      <NodeMeta config={config} elapsed={elapsed} rt={rt} />
      {def.kind === 'gate' ? (
        <GatePorts arms={config.arms ?? []} id={id} />
      ) : (
        <Handle id="out" position={ports.source} type="source" />
      )}
    </div>
  )
}

export const nodeTypes = {
  agent: StepNode,
  human: StepNode,
  gate: StepNode,
  wait: StepNode,
  trigger: StepNode,
  note: NoteNode
}

/** React Flow's text node — a label in the graph, not a step. Pans and
 *  drags with the canvas; never saved into the scenario. */
export const CANVAS_NOTE_ID = '__note'

export function canvasNote(label = 'The canvas is listening'): Node {
  return {
    id: CANVAS_NOTE_ID,
    type: 'note',
    position: { x: 0, y: 0 },
    data: { label },
    draggable: true,
    connectable: false,
    selectable: false
  }
}

function NoteNode({ data }: NodeProps<Node<{ label: string }, 'note'>>) {
  return <div className="canvas-note">{data.label}</div>
}

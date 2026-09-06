// Two controls that only look adjacent: a run button that starts the scenario,
// and a replay scrubber over the event log. The playhead is just "how many
// events are applied" — dragging it re-renders the graph as it was. It never
// drives the run: the log is append-only, forward only, no rewind.
//
// Two kinds of mark sit on the track, and they are not the same thing:
//   splits      — every step boundary. Speedrunning's word: the timestamped
//                 markers between segments of a run. Dense, informational.
//   checkpoints — the durable stops the ⏮/⏭ buttons jump between.

import { Button, cn, Codicon, GHOST_ICON_BTN, PRIMARY_ICON_FACE } from '@hermes/plugin-sdk'
import { type PointerEvent, useCallback, useMemo, useRef, useState } from 'react'

import type { Player } from './player'
import type { ProtoEvent, Verdict } from './protocol'

/** A coloured tick on the track: every step start and end, plus loops. */
interface Split {
  kind: 'start' | 'end' | 'fail' | 'loop'
  label: string
}

function buildSplits(events: ProtoEvent[]): (Split | null)[] {
  // NodeFinished fires whether the agent passed or rejected — the verdict
  // arrives on the trace summary just before it, so carry it to the end split.
  const verdicts = new Map<string, Verdict>()

  return events.map(e => {
    switch (e.type) {
      case 'NodeStarted':
        return { kind: 'start', label: `${e.payload.nodeId} started` }

      case 'AgentTraceSummary':
        verdicts.set(e.payload.nodeId, e.payload.verdict ?? null)

        return null
      case 'NodeFinished': {
        const failed = verdicts.get(e.payload.nodeId) === 'FAIL'

        return {
          kind: failed ? 'fail' : 'end',
          label: `${e.payload.nodeId} ${failed ? 'FAIL' : 'finished'}`
        }
      }

      case 'NodeFailed':
        return { kind: 'fail', label: `${e.payload.nodeId} failed` }

      case 'GateEvaluated':
        return e.payload.decision === 'fail'
          ? { kind: 'fail', label: 'gate · blocked' }
          : { kind: 'end', label: 'gate · passed' }

      case 'LoopAdvanced':
        return {
          kind: 'loop',
          label: `take ${e.payload.iteration + 1} → ${e.payload.to}`
        }

      case 'HumanWaiting':
        // The park is a stop worth scrubbing back to — it's where a real run
        // spends most of its wall clock.
        return { kind: 'loop', label: `${e.payload.nodeId} · waiting on you` }

      case 'HumanResponded':
        return e.payload.decision === 'approved'
          ? { kind: 'end', label: `${e.payload.nodeId} · approved` }
          : { kind: 'fail', label: `${e.payload.nodeId} · denied` }

      default:
        return null
    }
  })
}

/** Spotify-style elapsed on a track: `0:07`, `1:23`, `1:02:03`. */
function fmtScrub(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000))
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60

  if (h > 0) {
    return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
  }

  return `${m}:${String(s).padStart(2, '0')}`
}

function elapsedAt(events: ProtoEvent[], head: number): number {
  const origin = events[0]?.ts

  if (origin == null || head <= 0) {
    return 0
  }

  return Math.max(0, (events[Math.min(head, events.length) - 1]?.ts ?? origin) - origin)
}

interface Scrub {
  ratio: number
  head: number
}

export function Timeline({ p }: { p: Player }) {
  const trackRef = useRef<HTMLDivElement>(null)
  const dragging = useRef(false)
  const [scrub, setScrub] = useState<Scrub | null>(null)
  const total = p.events.length

  const atClientX = useCallback(
    (clientX: number): Scrub | null => {
      const el = trackRef.current

      if (!el || total === 0) {
        return null
      }

      const box = el.getBoundingClientRect()
      const ratio = Math.max(0, Math.min(1, (clientX - box.left) / box.width))

      return { ratio, head: Math.round(ratio * total) }
    },
    [total]
  )

  const seekTo = useCallback(
    (head: number) => {
      p.seek(head)
    },
    [p]
  )

  const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    // React Flow owns the pane under this dock; if the press leaks, a scrub
    // becomes a canvas pan and the playhead never moves.
    e.stopPropagation()
    e.preventDefault()
    e.currentTarget.setPointerCapture(e.pointerId)
    dragging.current = true
    const at = atClientX(e.clientX)

    if (at) {
      setScrub(at)
      seekTo(at.head)
    }
  }

  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    const at = atClientX(e.clientX)

    if (!at) {
      return
    }

    setScrub(at)

    if (dragging.current || e.buttons === 1) {
      seekTo(at.head)
    }
  }

  const onPointerUp = (e: PointerEvent<HTMLDivElement>) => {
    dragging.current = false

    try {
      e.currentTarget.releasePointerCapture(e.pointerId)
    } catch {
      // Capture already released.
    }
  }

  const onPointerLeave = () => {
    if (!dragging.current) {
      setScrub(null)
    }
  }

  const splits = useMemo(() => buildSplits(p.events), [p.events])

  const pct = total ? (p.head / total) * 100 : 0
  const armed = total > 0
  const paused = p.pauseState === 'paused'
  const pausing = p.pauseState === 'pausing'
  const tipHead = scrub?.head ?? p.head
  const tipPct = scrub ? scrub.ratio * 100 : pct
  const tipLabel = armed
    ? [fmtScrub(elapsedAt(p.events, tipHead)), splits[Math.max(0, tipHead - 1)]?.label]
        .filter(Boolean)
        .join(' · ')
    : ''

  // Run control, not a view control: play starts or resumes the scenario;
  // pause suspends it when available — in-flight steps finish, the next one
  // doesn't dispatch (LangGraph-interrupt / Prefect-pause semantics).
  // Space is the transport key — named in every state this button can be in.
  // Parked on a person comes first: the run IS stopped, so offering a pause
  // would be offering to stop it twice, and the only thing that moves it again
  // is the answer. So the button reopens the question.
  const runBtn = p.asking
    ? { act: p.reveal, title: 'Waiting on you — reopen the question (Space)', pause: false }
    : !p.running
      ? { act: p.start, title: armed ? 'Run again (Space)' : 'Run scenario (Space)', pause: false }
      : paused
        ? { act: p.resume, title: 'Resume — dispatch the next step (Space)', pause: false }
        : pausing
          ? { act: undefined, title: 'Pausing — letting the step in flight finish', pause: true }
          : {
              act: p.requestPause,
              title: 'Pause when available, before the next step (Space)',
              pause: true
            }

  return (
    <div className="tl">
      <Button
        className={cn(PRIMARY_ICON_FACE, pausing && 'animate-pulse')}
        disabled={!runBtn.act}
        onClick={runBtn.act}
        size="icon-2xs"
        title={runBtn.title}
      >
        {/* `triangle-right` is the only genuinely solid play in the set — both
            `play` and `debug-start` draw an outline around the triangle, which
            on the filled primary circle reads as a disabled ring. This matches
            the solid bars of the `debug-pause` it swaps with. */}
        <Codicon name={runBtn.pause ? 'debug-pause' : 'triangle-right'} />
      </Button>
      <Button
        className={GHOST_ICON_BTN}
        disabled={!armed}
        onClick={p.restart}
        size="icon"
        title="Restart run"
        variant="ghost"
      >
        <Codicon name="debug-restart" size="0.875rem" />
      </Button>
      {/* The debug toolbar's reverse/forward pair. Checkpoints are the durable
          stops in a run, which is the same thing VS Code's continue and reverse
          continue jump between. */}
      <Button
        className={GHOST_ICON_BTN}
        disabled={!armed}
        onClick={() => p.stepCheckpoint(-1)}
        size="icon"
        title="Previous checkpoint"
        variant="ghost"
      >
        <Codicon name="debug-reverse-continue" size="0.875rem" />
      </Button>
      <Button
        className={GHOST_ICON_BTN}
        disabled={!armed}
        onClick={() => p.stepCheckpoint(1)}
        size="icon"
        title="Next checkpoint"
        variant="ghost"
      >
        <Codicon name="debug-continue" size="0.875rem" />
      </Button>

      <div
        className={`tl-track nodrag nopan nowheel${armed ? '' : ' empty'}`}
        onPointerCancel={onPointerUp}
        onPointerDown={armed ? onPointerDown : undefined}
        onPointerLeave={onPointerLeave}
        onPointerMove={armed ? onPointerMove : undefined}
        onPointerUp={onPointerUp}
        ref={trackRef}
      >
        <div className={`tl-fill${p.live ? '' : ' hist'}`} style={{ width: `${pct}%` }} />
        {p.checkpoints.map(c => (
          <span
            className="tl-tick"
            key={c.no}
            style={{ left: `${total ? ((c.at + 1) / total) * 100 : 0}%` }}
          />
        ))}
        {splits.map((s, i) =>
          s ? (
            <span
              className={`tl-mark m-${s.kind}`}
              key={i}
              style={{ left: `${total ? ((i + 1) / total) * 100 : 0}%` }}
            />
          ) : null
        )}
        <span className="tl-headline" style={{ left: `${pct}%` }} />
        {scrub && tipLabel && (
          <span className="tl-tip" style={{ left: `${tipPct}%` }}>
            {tipLabel}
          </span>
        )}
      </div>
    </div>
  )
}

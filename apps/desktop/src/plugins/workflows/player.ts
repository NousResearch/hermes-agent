// The canvas folds a prefix of the gateway event log. Play starts a real run;
// the bus is the only source after a one-shot catch-up. Seeking reapplies
// fewer events — the run itself is not rewound.

import { host } from '@hermes/plugin-sdk'
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'

import { noteRun } from './documents'
import type { RunPlan } from './graph'
import { type Checkpoint, type ProtoEvent, type RunShape, type World } from './protocol'
import { checkpointsOf, reduceEvents } from './protocol-world'
import {
  activeRun,
  askInCanvas,
  cancelRun,
  LIVE,
  pauseRun,
  respondRun,
  resumeRun,
  runEvents,
  startRun
} from './run-rpc'
import type { OnFail } from './scenario'
import { demandComplete } from './validation'

/** Play from a manual trigger card or the inspector — same start as the dock. */
export interface RunNow {
  start: () => void
  fireWebhook: () => Promise<void>
  running: boolean
}

const RunNowCtx = createContext<RunNow>({
  start: () => {},
  fireWebhook: async () => {},
  running: false
})

export const RunNowProvider = RunNowCtx.Provider
export const useRunNow = () => useContext(RunNowCtx)

function sameEvent(a: ProtoEvent, b: ProtoEvent) {
  return a.runId === b.runId && a.seq === b.seq && a.type === b.type && a.ts === b.ts
}

function runIsFinished(events: ProtoEvent[]) {
  return events.some(e => e.type === 'RunFinished')
}

/** none → (pause requested) pausing → (boundary reached) paused → resume. */
export type PauseState = 'none' | 'pausing' | 'paused'

/** A question the run is parked on. The stream stops here and does not move
 *  again until someone answers — which is the difference between a human step
 *  and a sleep. */
export interface Question {
  nodeId: string
  prompt: string
  who: string
  onFail: OnFail
}

/** Structural equality over the JSON-shaped values a StepRuntime holds. Written
 *  generically rather than as a field list so it can't rot the next time a
 *  field lands on StepRuntime. */
function sameValue(a: unknown, b: unknown): boolean {
  if (a === b) {
    return true
  }

  if (a === null || b === null || typeof a !== 'object' || typeof b !== 'object') {
    return false
  }

  if (Array.isArray(a) !== Array.isArray(b)) {
    return false
  }

  const av = a as Record<string, unknown>
  const bv = b as Record<string, unknown>
  const keys = Object.keys(av)

  if (keys.length !== Object.keys(bv).length) {
    return false
  }

  return keys.every(k => sameValue(av[k], bv[k]))
}

export interface Player {
  events: ProtoEvent[]
  world: World
  checkpoints: Checkpoint[]
  /** Events applied to the current view. */
  head: number
  /** True when the view is pinned to the tail of the stream. */
  live: boolean
  /** The scenario is executing (events still arriving or holding at a pause). */
  running: boolean
  pauseState: PauseState
  /** Set while the run is parked on a person. */
  asking: Question | null
  /** The parked question is off screen — hidden, not answered. Lives here
   *  rather than in the view because the run is what's blocked by it: the
   *  transport has to know that "carry on" means "bring the question back". */
  deferred: boolean
  /** Put the question away to go look at the graph. The run stays parked. */
  defer: () => void
  /** Bring a deferred question back. */
  reveal: () => void
  /** Answer the parked question and let the run move again. */
  respond: (decision: 'approved' | 'denied') => void
  /** Event timestamp the view is frozen at, or null while live. */
  frozenAt: number | null
  start: () => void
  fireWebhook: () => Promise<void>
  reset: () => void
  /** Cancel a live run (or a stuck one) and start again. */
  restart: () => void
  /** Suspend at the next safe point — before the next step dispatch. */
  requestPause: () => void
  resume: () => void
  seek: (head: number) => void
  stepCheckpoint: (dir: -1 | 1) => void
  goLive: () => void
}

/** The reducer only needs to know which steps and wires exist. */
const shapeOf = (plan: RunPlan): RunShape => ({
  steps: plan.steps.map(s => s.id),
  edges: plan.edges
})

/** The run is built from the graph at the moment you press play, not once at
 *  mount. That's the honest model — you run what's on the canvas — and it's why
 *  the shape travels with it: seeking replays the log against the graph the run
 *  was built from, so editing mid-replay can't retune the past. */
export function usePlayer(planOf: () => RunPlan): Player {
  const [shape, setShape] = useState<RunShape>(() => shapeOf(planOf()))
  const [events, setEvents] = useState<ProtoEvent[]>([])
  const [head, setHead] = useState<number | null>(null) // null = follow tail
  const [running, setRunning] = useState(false)
  const [pauseState, setPauseState] = useState<PauseState>('none')
  const runIdRef = useRef('run-idle')
  const eventsRef = useRef<ProtoEvent[]>([])
  const headRef = useRef(0)
  const pauseRef = useRef<PauseState>('none')
  const [asking, setAsking] = useState<Question | null>(null)
  const askingRef = useRef<Question | null>(null)
  const [deferred, setDeferred] = useState(false)
  const liveRun = useRef<string | null>(null)
  const planOfRef = useRef(planOf)
  planOfRef.current = planOf

  const setPause = (s: PauseState) => {
    pauseRef.current = s
    setPauseState(s)
  }

  // Every arrival and every clearing goes through here, so a deferral can
  // never outlive the question it was about.
  const ask = (q: Question | null) => {
    askingRef.current = q
    setAsking(q)
    setDeferred(false)
  }

  const adoptLive = useCallback((runId: string, incoming: ProtoEvent[]) => {
    liveRun.current = runId
    runIdRef.current = runId
    setEvents(incoming)

    if (runIsFinished(incoming)) {
      setRunning(false)
      setPause('none')
    }

    const waiting = [...incoming].reverse().find(e => e.type === 'HumanWaiting')

    const answered = waiting
      ? incoming.some(e => e.type === 'HumanResponded' && e.payload.nodeId === waiting.payload.nodeId && e.seq > waiting.seq)
      : false

    ask(!answered && waiting?.type === 'HumanWaiting' ? waiting.payload : null)
  }, [])

  const adoptRun = useCallback(
    async (runId: string, workflowId: string) => {
      liveRun.current = runId
      runIdRef.current = runId
      noteRun(workflowId)

      const snap = await runEvents(runId)

      if (liveRun.current === runId) {
        adoptLive(runId, snap.events ?? [])
      }
    },
    [adoptLive]
  )

  const start = useCallback(() => {
    // Every way to run funnels through here — the transport, Space, a trigger
    // card, the agent's `run_control` — so it's the one place that can tell the
    // inspector the draft was declared finished. See validation.ts.
    demandComplete()
    const plan = planOf()
    setShape(shapeOf(plan))
    setPause('none')
    ask(null)
    setEvents([])
    setHead(null)

    if (!plan.id) {
      setRunning(false)

      return
    }

    setRunning(true)
    void startRun(plan, 'manual')
      .then(res => adoptRun(res.runId, plan.id))
      .catch(() => {
        liveRun.current = null
        setRunning(false)
      })
  }, [adoptRun, planOf])

  const fireWebhook = useCallback(async () => {
    const plan = planOf()

    if (!plan.id) {
      throw new Error('No workflow to fire')
    }

    setShape(shapeOf(plan))
    setPause('none')
    ask(null)
    setEvents([])
    setHead(null)
    setRunning(true)

    try {
      const res = await startRun(plan, 'webhook', { ok: true })

      await adoptRun(res.runId, plan.id)
    } catch (err) {
      liveRun.current = null
      setRunning(false)
      throw err
    }
  }, [adoptRun, planOf])

  const reset = useCallback(() => {
    const id = liveRun.current

    if (id) {
      void cancelRun(id).catch(() => {})
    }

    liveRun.current = null
    setPause('none')
    ask(null)
    setEvents([])
    setHead(null)
    setRunning(false)
  }, [])

  const restart = useCallback(() => {
    const id = liveRun.current
    liveRun.current = null
    setPause('none')
    ask(null)
    setEvents([])
    setHead(null)
    const kick = () => start()

    if (id) {
      void cancelRun(id).then(kick, kick)
    } else {
      kick()
    }
  }, [start])

  const respond = useCallback((decision: 'approved' | 'denied') => {
    const q = askingRef.current

    if (!q || !liveRun.current) {
      return
    }

    void respondRun(liveRun.current, q.nodeId, decision, q.who)
      .then(() => ask(null))
      .catch(() => {})
  }, [])

  const requestPause = useCallback(() => {
    if (askingRef.current || !liveRun.current || pauseRef.current !== 'none') {
      return
    }

    const id = liveRun.current
    setPause('pausing')
    void pauseRun(id)
      .then(res => {
        if (res.status === 'paused') {
          setPause('paused')

          return
        }

        if (!res.status || !LIVE.has(res.status)) {
          setPause('none')
          setRunning(false)
        }
      })
      .catch(() => setPause('none'))
  }, [])

  const resume = useCallback(() => {
    if (pauseRef.current !== 'paused' || !liveRun.current) {
      return
    }

    setHead(null)
    setPause('none')
    void resumeRun(liveRun.current).catch(() => {})
  }, [])

  const seek = useCallback((h: number) => {
    const total = eventsRef.current.length
    const clamped = Math.max(0, Math.min(h, total))

    if (clamped >= total) {
      headRef.current = total
      setHead(null)

      return
    }

    if (liveRun.current && pauseRef.current === 'none') {
      requestPause()
    }

    headRef.current = clamped
    setHead(clamped)
  }, [requestPause])

  const goLive = useCallback(() => {
    setHead(null)
  }, [])

  const checkpoints = useMemo(() => checkpointsOf(events), [events])
  const effHead = head ?? events.length

  eventsRef.current = events
  headRef.current = effHead

  const stepCheckpoint = useCallback(
    (dir: -1 | 1) => {
      const from = headRef.current
      const stops = checkpointsOf(eventsRef.current).map(c => c.at + 1)
      const next = dir === -1 ? [...stops].reverse().find(s => s < from) : stops.find(s => s > from)

      if (next != null) {
        seek(next)
      } else if (dir === 1) {
        goLive()
      }
    },
    [goLive, seek]
  )

  const prevWorld = useRef<World | null>(null)

  const world: World = useMemo(() => {
    const next = reduceEvents(events, shape, effHead)
    const prev = prevWorld.current

    if (prev) {
      for (const id in next.steps) {
        if (sameValue(prev.steps[id], next.steps[id])) {
          next.steps[id] = prev.steps[id]
        }
      }
    }

    prevWorld.current = next

    return next
  }, [events, effHead, shape])

  useEffect(() => {
    return host.onEvent('workflow.run', event => {
      const incoming = event.payload as ProtoEvent | undefined

      if (!incoming?.runId || incoming.runId !== liveRun.current) {
        return
      }

      setEvents(prev => (prev.some(e => sameEvent(e, incoming)) ? prev : [...prev, incoming]))

      if (incoming.type === 'HumanWaiting') {
        ask(incoming.payload)
      }

      if (incoming.type === 'UserAsk') {
        askInCanvas(planOfRef.current().id, incoming.payload.nodeId, incoming.payload.prompt)
      }

      if (incoming.type === 'HumanResponded' && askingRef.current?.nodeId === incoming.payload.nodeId) {
        ask(null)
      }

      if (incoming.type === 'RunPaused') {
        setPause('paused')
      }

      if (incoming.type === 'RunFinished') {
        setRunning(false)
        setPause('none')
      }
    })
  }, [])

  useEffect(() => {
    const plan = planOf()

    if (!plan.id) {
      return
    }

    let cancelled = false
    void activeRun(plan.id)
      .then(res => {
        if (cancelled || !res.runId || !res.run || !LIVE.has(res.run.status)) {
          return
        }

        setShape(shapeOf(plan))
        adoptLive(res.runId, res.events ?? [])
        setRunning(true)
        setHead(null)

        if (res.run.status === 'paused') {
          setPause('paused')
        }
      })
      .catch(() => {})

    return () => {
      cancelled = true
    }
  }, [adoptLive, planOf])

  return {
    events,
    world,
    checkpoints,
    head: effHead,
    live: head == null,
    running,
    pauseState,
    asking,
    deferred,
    defer: useCallback(() => setDeferred(true), []),
    reveal: useCallback(() => setDeferred(false), []),
    respond,
    frozenAt: head == null ? null : (events[effHead - 1]?.ts ?? null),
    start,
    fireWebhook,
    reset,
    restart,
    requestPause,
    resume,
    seek,
    stepCheckpoint,
    goLive
  }
}

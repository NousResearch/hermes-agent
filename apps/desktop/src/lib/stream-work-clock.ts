import { randomUuid } from '@hermes/shared'

type StreamWorkBridge = Required<
  Pick<NonNullable<Window['hermesDesktop']>, 'scheduleStreamWorkWake' | 'onStreamWorkWake'>
>

interface StreamWork {
  dueAt: number
  timer: ReturnType<typeof setTimeout>
  run: () => void
}

const pending = new Set<StreamWork>()
let bridge: StreamWorkBridge | null = null
let unsubscribe: (() => void) | null = null
let nativeWake: { token: string; dueAt: number } | null = null
let contextToken = ''
let generation = 0
let draining = false

function runWork(work: StreamWork): void {
  if (!pending.delete(work)) {
    return
  }

  clearTimeout(work.timer)

  try {
    work.run()
  } finally {
    syncNativeWake()
  }
}

function onNativeWake(token: string): void {
  if (nativeWake?.token !== token) {
    return
  }

  nativeWake = null
  const now = performance.now()
  draining = true

  try {
    for (const work of pending) {
      if (work.dueAt <= now) {
        runWork(work)
      }
    }
  } finally {
    draining = false
    syncNativeWake()
  }
}

function connectBridge(): void {
  const candidate = typeof window === 'undefined' ? undefined : window.hermesDesktop

  if (candidate?.scheduleStreamWorkWake && candidate.onStreamWorkWake) {
    bridge = {
      scheduleStreamWorkWake: candidate.scheduleStreamWorkWake,
      onStreamWorkWake: candidate.onStreamWorkWake
    }
    contextToken ||= randomUuid()
    unsubscribe = bridge.onStreamWorkWake(onNativeWake)
  }
}

function syncNativeWake(): void {
  if (draining) {
    return
  }

  if (pending.size === 0) {
    if (nativeWake && bridge) {
      bridge.scheduleStreamWorkWake(nativeWake.token, null)
    }

    nativeWake = null
    unsubscribe?.()
    unsubscribe = null
    bridge = null

    return
  }

  if (!bridge) {
    connectBridge()
  }

  if (!bridge) {
    return
  }

  let dueAt = Infinity

  for (const work of pending) {
    dueAt = Math.min(dueAt, work.dueAt)
  }

  if (nativeWake?.dueAt === dueAt) {
    return
  }

  nativeWake = { token: `${contextToken}:${++generation}`, dueAt }
  bridge.scheduleStreamWorkWake(nativeWake.token, Math.min(400, Math.max(0, dueAt - performance.now())))
}

/** Pending delta/journal work only: race the normal timer with one native wake.
 * No rAF, idle polling or background-throttling override. Cancellation is final. */
export function scheduleStreamWork(run: () => void, delayMs: number): () => void {
  const work: StreamWork = {
    dueAt: performance.now() + delayMs,
    timer: setTimeout(() => runWork(work), delayMs),
    run
  }

  pending.add(work)
  syncNativeWake()

  return () => {
    if (pending.delete(work)) {
      clearTimeout(work.timer)
      syncNativeWake()
    }
  }
}

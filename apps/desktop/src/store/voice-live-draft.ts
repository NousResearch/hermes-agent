import { atom } from 'nanostores'

/**
 * Words of a recorded GPT-Live call the chat has not saved yet, per stored chat id.
 *
 * Display only. The owner saves a speaker run when the next speaker starts or the
 * call closes, so the run being spoken would otherwise appear only afterwards. The
 * thread shows these runs after the saved transcript and retires each one when the
 * saved row for that run lands: the voice row naming the run's first provider event
 * (or, without ids, the same speaker, known call timing and words).
 * Nothing here is sent to the owner, and none of it decides what is saved or delegated.
 */
export interface VoiceLiveDraftRun {
  /** Submission id of the delegation that claimed exactly this run; while that submission
   *  is pending, its optimistic bubble shows the words. */
  claimedBy?: string
  /** Latest fragment end, call-relative; null once any fragment lacked timing. */
  endMs: null | number
  /** The first fragment's provider event id; the saved row names the same one. */
  firstEventId: null | string
  id: string
  role: 'assistant' | 'user'
  /** Earliest fragment start, call-relative; null once any fragment lacked timing. */
  startMs: null | number
  text: string
}

export interface VoiceLiveDraftTiming {
  endMs: number
  eventId?: string
  startMs: number
  timingMissing?: boolean
}

type Drafts = Readonly<Record<string, readonly VoiceLiveDraftRun[]>>

export const $voiceLiveDrafts = atom<Drafts>({})

// Bounds memory on a very long call while saving lags (a long delegated turn).
const MAX_RUNS = 40
let nextRun = 0

export function newVoiceLiveDraftRunId(): string {
  nextRun += 1

  return `voice-draft-${nextRun}`
}

const widen = (current: null | number, next: null | number, pick: (a: number, b: number) => number) =>
  current === null || next === null ? null : pick(current, next)

/** Grow `runId` (one speaker run in arrival order), or start it. */
export function appendVoiceLiveDraft(
  storedSessionId: string,
  runId: string,
  role: VoiceLiveDraftRun['role'],
  text: string,
  timing: VoiceLiveDraftTiming
) {
  if (!storedSessionId || !text) {
    return
  }

  const startMs = timing.timingMissing ? null : timing.startMs
  const endMs = timing.timingMissing ? null : timing.endMs
  const drafts = $voiceLiveDrafts.get()
  const runs = drafts[storedSessionId] ?? []

  const next = runs.some(run => run.id === runId)
    ? runs.map(run =>
        run.id === runId
          ? {
              ...run,
              endMs: widen(run.endMs, endMs, Math.max),
              startMs: widen(run.startMs, startMs, Math.min),
              text: run.text + text
            }
          : run
      )
    : [...runs, { endMs, firstEventId: timing.eventId || null, id: runId, role, startMs, text }].slice(-MAX_RUNS)

  $voiceLiveDrafts.set({ ...drafts, [storedSessionId]: next })
}

/** The run a delegation claimed, by the claim's first provider event id. */
export function markVoiceLiveDraftClaimed(storedSessionId: string, firstEventId: string, submissionId: string) {
  const drafts = $voiceLiveDrafts.get()
  const runs = drafts[storedSessionId]

  if (!runs?.some(run => run.firstEventId === firstEventId && run.claimedBy !== submissionId)) {
    return
  }

  $voiceLiveDrafts.set({
    ...drafts,
    [storedSessionId]: runs.map(run => (run.firstEventId === firstEventId ? { ...run, claimedBy: submissionId } : run))
  })
}

export function retireVoiceLiveDrafts(storedSessionId: string, ids: ReadonlySet<string>) {
  const drafts = $voiceLiveDrafts.get()
  const runs = drafts[storedSessionId]

  if (!runs) {
    return
  }

  const kept = runs.filter(run => !ids.has(run.id))

  if (kept.length === runs.length) {
    return
  }

  const rest = Object.fromEntries(Object.entries(drafts).filter(([key]) => key !== storedSessionId))
  $voiceLiveDrafts.set(kept.length ? { ...rest, [storedSessionId]: kept } : rest)
}

export const voiceLiveSpeech = (text: string) => text.replace(/\s+/g, ' ').trim()

export interface VoiceLiveDraftMessage {
  role: string
  text: string
  voiceLive?: { endMs: null | number; firstEventId?: string; startMs: null | number }
}

function isRunRow(run: VoiceLiveDraftRun, message: VoiceLiveDraftMessage): boolean {
  const voice = message.voiceLive

  if (!voice || message.role !== run.role) {
    return false
  }

  // A run with an id matches only its own row: an id-less row (saved before rows
  // carried ids) at the same call-relative time may belong to another call.
  if (run.firstEventId) {
    return run.firstEventId === voice.firstEventId
  }

  // Without ids, require known timing too: untimed words repeat.
  return (
    run.startMs !== null &&
    run.endMs !== null &&
    voice.startMs === run.startMs &&
    voice.endMs === run.endMs &&
    voiceLiveSpeech(message.text) === voiceLiveSpeech(run.text)
  )
}

/** Runs whose own saved voice row has landed. A row without voice metadata (Hermes's
 *  reply, an optimistic bubble) or another call's row never matches. */
export function savedVoiceLiveDraftIds(
  runs: readonly VoiceLiveDraftRun[],
  messages: readonly VoiceLiveDraftMessage[]
): Set<string> {
  const saved = new Set<string>()

  for (const run of runs) {
    if (voiceLiveSpeech(run.text) && messages.some(message => isRunRow(run, message))) {
      saved.add(run.id)
    }
  }

  return saved
}

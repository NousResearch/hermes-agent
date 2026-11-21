import { randomUuid } from '@hermes/shared'
import { atom } from 'nanostores'

import { type CronOwnerScope, deleteCronJob, pauseCronJob, resumeCronJob } from '@/hermes'
import { readJson, writeJson } from '@/lib/storage'

// Delete with Undo. A real delete destroys the routine's run history, so Delete only HOLDS it while the
// "deleted" banner is up: the routine is paused on its owner (it cannot fire) and hidden from the list.
// Undo restores it exactly (resumed only if it was on); dismissing the banner, deleting another
// routine, leaving Routines or closing the window makes the delete final.
//
// A held delete is also written down, so a window that closes or crashes mid-hold is finished on the
// next launch. Every step is ordered to fail toward NOT deleting the wrong thing:
// - hold writes the marker before pausing: a crash after it can only complete what was asked;
// - undo drops the marker before resuming: an undone routine is never deleted later;
// - a delete that fails drops the marker and shows the routine again, paused, never hidden and alive.

/** A routine whose delete is being held. */
export interface HeldRoutine {
  id: string
  name: string
  /** Its owner, the agent on its machine: every call (pause, resume, delete) reaches that owner. */
  scope: { connectionId: null | string; profile: string }
  /** It was on, so it was paused for the hold and Undo resumes it. */
  resume: boolean
}

/** The owner calls the hold makes; the routine API in the app, fakes in tests. */
export interface HeldDeleteBackend {
  pause: (id: string, scope: CronOwnerScope) => Promise<unknown>
  resume: (id: string, scope: CronOwnerScope) => Promise<unknown>
  remove: (id: string, scope: CronOwnerScope) => Promise<unknown>
}

export const ROUTINE_BACKEND: HeldDeleteBackend = {
  pause: pauseCronJob,
  remove: deleteCronJob,
  resume: resumeCronJob
}

/** Every window's held deletes on this device, each naming its owner and the window holding it. */
const MARKERS_KEY = 'hermes.desktop.routines.heldDeletes.v1'
/** A holding window re-stamps its marker this often... */
const BEAT_MS = 10_000
/** ...so a marker unstamped this long has no live window left to finish it. */
export const ABANDONED_AFTER_MS = 60_000

interface Marker extends HeldRoutine {
  holder: string
  beat: number
}

let windowId: null | string = null

/** This window's holder id, drawn on first use. */
function thisWindow(): string {
  windowId ??= randomUuid()

  return windowId
}

function readMarkers(): Marker[] {
  const stored = readJson<unknown>(MARKERS_KEY)

  return Array.isArray(stored) ? stored.filter(isMarker) : []
}

function isMarker(value: unknown): value is Marker {
  const marker = value as Partial<Marker> | null

  return Boolean(
    marker &&
      typeof marker.id === 'string' &&
      typeof marker.holder === 'string' &&
      typeof marker.beat === 'number' &&
      typeof marker.scope?.profile === 'string'
  )
}

const sameRoutine = (a: HeldRoutine, b: HeldRoutine) =>
  a.id === b.id && a.scope.profile === b.scope.profile && (a.scope.connectionId ?? '') === (b.scope.connectionId ?? '')

function writeMarkers(markers: Marker[]) {
  writeJson(MARKERS_KEY, markers.length ? markers : null)
}

function dropMarker(routine: HeldRoutine) {
  writeMarkers(readMarkers().filter(marker => !sameRoutine(marker, routine)))
}

function stampMarker(routine: HeldRoutine, nowMs: number) {
  const others = readMarkers().filter(marker => !sameRoutine(marker, routine))
  writeMarkers([...others, { ...routine, beat: nowMs, holder: thisWindow() }])
}

/** The routine this window holds a delete for: hidden from every list while it is set. */
export const $heldRoutine = atom<HeldRoutine | null>(null)

let beat: null | ReturnType<typeof setInterval> = null

function stopBeat() {
  if (beat !== null) {
    clearInterval(beat)
    beat = null
  }
}

function startBeat(routine: HeldRoutine) {
  stopBeat()
  beat = setInterval(() => stampMarker(routine, Date.now()), BEAT_MS)
}

const isGone = (error: unknown) => /\bjob not found\b/i.test(error instanceof Error ? error.message : String(error))

/** Hold `routine`'s delete. Any delete already held is made final first (one banner at a time); if
 *  that fails, its error is thrown and nothing new is held. A failed pause puts everything back. */
export async function holdRoutineDelete(routine: HeldRoutine, backend: HeldDeleteBackend): Promise<void> {
  await commitHeldRoutine(backend)

  stampMarker(routine, Date.now())
  $heldRoutine.set(routine)

  if (routine.resume) {
    try {
      await backend.pause(routine.id, routine.scope)
    } catch (error) {
      $heldRoutine.set(null)
      dropMarker(routine)

      throw error
    }
  }

  // An Undo or a commit that landed during the pause owns the routine now.
  if ($heldRoutine.get() === routine) {
    startBeat(routine)
  }
}

/** Undo: the routine comes back as it was. Returns it (the page selects it again), or null when
 *  nothing is held. A failed resume is thrown: the routine is back, still paused. */
export async function undoHeldRoutine(backend: HeldDeleteBackend): Promise<HeldRoutine | null> {
  const held = $heldRoutine.get()

  if (!held) {
    return null
  }

  $heldRoutine.set(null)
  stopBeat()
  dropMarker(held)

  if (held.resume) {
    await backend.resume(held.id, held.scope)
  }

  return held
}

/** Make the held delete final. A routine already gone counts as deleted; any other failure is thrown,
 *  with the routine shown again, paused. */
export async function commitHeldRoutine(backend: HeldDeleteBackend): Promise<HeldRoutine | null> {
  const held = $heldRoutine.get()

  if (!held) {
    return null
  }

  $heldRoutine.set(null)
  stopBeat()

  try {
    await backend.remove(held.id, held.scope)
  } catch (error) {
    if (!isGone(error)) {
      dropMarker(held)

      throw error
    }
  }

  dropMarker(held)

  return held
}

/** Finish the deletes a closed or crashed window was holding (its marker stopped beating). Run once
 *  per launch. A delete that fails is dropped rather than retried later: the routine stays, paused,
 *  and a person who sees it again decides, so a stale marker never deletes it behind their back. */
export async function commitAbandonedRoutineDeletes(
  backend: HeldDeleteBackend = ROUTINE_BACKEND,
  nowMs = Date.now()
): Promise<void> {
  const abandoned = readMarkers().filter(
    marker => marker.holder !== thisWindow() && nowMs - marker.beat >= ABANDONED_AFTER_MS
  )

  for (const marker of abandoned) {
    await backend.remove(marker.id, marker.scope).catch(() => undefined)
    dropMarker(marker)
  }
}

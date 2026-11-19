import { useEffect, useRef } from 'react'

import type { CronJob } from '@/hermes'
import { notifyError } from '@/store/notifications'

import {
  $heldRoutine,
  commitAbandonedRoutineDeletes,
  commitHeldRoutine,
  type HeldDeleteBackend,
  holdRoutineDelete,
  ROUTINE_BACKEND,
  undoHeldRoutine
} from './routine-held-delete'
import { routinePaused, routineTitle } from './routine-model'
import type { RoutineSource } from './routine-source'
import { useRoutinesCopy } from './routines-copy'
import { selectRoutine } from './routines-store'

export interface HeldDelete {
  /** Hold `job`'s delete (the banner offers Undo); false when it could not be held. */
  hold: (job: CronJob) => Promise<boolean>
  undo: () => Promise<void>
  /** The banner's ×: the delete becomes final. */
  dismiss: () => Promise<void>
}

/** The Routines page's Delete with Undo, bound to the page's source. Leaving the page or closing the
 *  window makes a held delete final; a launch after a crash finishes it (commitAbandonedRoutineDeletes). */
export function useHeldDelete(source: RoutineSource, backend: HeldDeleteBackend = ROUTINE_BACKEND): HeldDelete {
  const copy = useRoutinesCopy()
  const { connectionId, refresh, scopeFor } = source

  const commit = async () => {
    const held = $heldRoutine.get()

    try {
      await commitHeldRoutine(backend)
    } catch (error) {
      notifyError(error, copy.deleteFailed(held?.name ?? ''))
    }

    void refresh()
  }

  const commitOnLeave = useRef(commit)
  commitOnLeave.current = commit

  useEffect(() => {
    // Opening the page also finishes what a window that went away was holding (the launch does too).
    void commitAbandonedRoutineDeletes(backend)

    // The window closing: start the delete now; the marker finishes it on the next launch if the
    // request does not land first (nothing is left on screen to report to).
    const close = () => void commitHeldRoutine(backend).catch(() => undefined)
    window.addEventListener('pagehide', close)

    return () => {
      window.removeEventListener('pagehide', close)
      // Leaving Routines is leaving the banner: the delete is final, and a failure still says so.
      void commitOnLeave.current()
    }
  }, [backend])

  return {
    dismiss: commit,
    hold: async job => {
      const name = routineTitle(job)
      const scope = scopeFor(job)

      // One banner at a time: the delete already held becomes final first, reporting its own failure.
      if ($heldRoutine.get()) {
        await commit()
      }

      try {
        await holdRoutineDelete(
          {
            id: job.id,
            name,
            resume: !routinePaused(job) && job.state !== 'completed',
            scope: { connectionId: scope.connectionId ?? connectionId, profile: job.profile ?? '' }
          },
          backend
        )

        return true
      } catch (error) {
        notifyError(error, copy.deleteFailed(name))

        return false
      } finally {
        void refresh()
      }
    },
    undo: async () => {
      const held = $heldRoutine.get()

      try {
        const restored = await undoHeldRoutine(backend)

        if (restored) {
          selectRoutine({ id: restored.id, profile: restored.scope.profile })
        }
      } catch (error) {
        if (held) {
          selectRoutine({ id: held.id, profile: held.scope.profile })
        }

        notifyError(error, copy.undoFailed(held?.name ?? ''))
      }

      void refresh()
    }
  }
}

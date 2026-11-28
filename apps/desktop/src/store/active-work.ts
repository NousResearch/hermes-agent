/**
 * Mirror of "which chats are mid-turn" to the main process.
 *
 * The renderer is the only side that knows a turn is in flight, and the main
 * process is the only side that can intercept a quit. This module bridges the
 * two: it publishes a small summary on every membership change, and
 * `electron/quit-guard.ts` turns that into the confirmation dialog.
 *
 * Imported for its side effect from `main.tsx`, alongside `store/translucency`.
 */

import { computed } from 'nanostores'

import type { HermesActiveWork } from '@/global'
import { $dirtyPreviewUrls } from '@/store/preview-edit'
import { $sessions } from '@/store/session'
import { $workingSessionIds } from '@/store/session-states'
import { $voiceStreamPaintOwners } from '@/store/voice-stream-paint'

/** Display name for a dirty preview target: the file name, else the raw url. */
function previewFileName(url: string): string {
  const path = url.split(/[?#]/, 1)[0] ?? url
  const last = path.split(/[\\/]/).filter(Boolean).pop() ?? url

  try {
    return decodeURIComponent(last)
  } catch {
    return last
  }
}

const $activeWork = computed(
  [$workingSessionIds, $sessions, $voiceStreamPaintOwners, $dirtyPreviewUrls],
  (workingIds, sessions, voiceOwners, dirtyUrls): HermesActiveWork => {
    const titleById = new Map(sessions.map(session => [session.id, session.title?.trim() ?? '']))
    const unsavedFiles = Object.keys(dirtyUrls).map(previewFileName)

    return {
      count: workingIds.length,
      titles: workingIds.map(id => titleById.get(id) ?? '').filter(Boolean),
      // Auto-speak waits for completed replies; only local conversations need live text.
      ...(voiceOwners > 0 ? { voiceActive: true } : {}),
      // The quit / update / close guards in the main process name these files.
      ...(unsavedFiles.length > 0 ? { unsavedFiles } : {})
    }
  }
)

if (typeof window !== 'undefined') {
  // `$sessions` republishes on unrelated churn (previews, heartbeats), so only
  // send when the summary itself moved — this crosses a process boundary.
  let lastSent = ''

  $activeWork.subscribe(work => {
    const next = JSON.stringify(work)

    if (next === lastSent) {
      return
    }

    lastSent = next
    window.hermesDesktop?.setActiveWork?.(work)
  })

  // Closing or reloading this window with unsaved Files-pane edits: cancel the
  // unload so the main process's will-prevent-unload can ask first (it lets a
  // confirmed quit or an update hand-off through without asking again).
  window.addEventListener('beforeunload', event => {
    if (Object.keys($dirtyPreviewUrls.get()).length > 0) {
      event.preventDefault()
      event.returnValue = ''
    }
  })
}

/**
 * A Webapp tab moves onto the bundle its gateway serves now.
 *
 * `hermes update` restarts the gateway first and rebuilds the Webapp bundle after it, so a tab
 * reconnects to the new backend while still running the old JavaScript. The page records the
 * bundle it booted (`__HERMES_WEBAPP_BUNDLE__`, injected by `_serve_index`); the gateway reports
 * the bundle a page load gets now (`GET /api/webapp/bundle`). When they differ the tab reloads
 * once, but only onto a CURRENT bundle: a rebuild that failed leaves the old bundle marked stale,
 * its page load is refused, and reloading would replace a working tab with that refusal.
 *
 * Native Desktop ships its renderer inside the app and never reloads for a backend's bundle.
 */

import { ATTACHMENTS_SLOT, composerPlainText, RICH_INPUT_SLOT } from '@/app/chat/composer/rich-editor'
import { appViewForPath, NEW_CHAT_ROUTE } from '@/app/routes'
import { hasStashedAttachments } from '@/store/composer'
import { $dirtyPreviewUrls } from '@/store/preview-edit'
import { $gatewayState } from '@/store/session'
import { $sessionStates, isLiveTurnAwaitingEvents } from '@/store/session-states'
import { $voiceStreamPaintOwners } from '@/store/voice-stream-paint'

interface ServedBundle {
  id: null | string
  current: boolean
}

// Every id this tab reloaded for. A boot that still runs another bundle (a proxy serving a cached
// page, or served ids flipping back and forth) must not reload for an id again: at most one reload
// per served bundle id in this tab.
const RELOADED_FOR_KEY = 'hermes.webapp.reloadedForBundles'
const CHECK_INTERVAL_MS = 60_000

/** The bundle this page booted, or null where no Webapp gateway served it: only the Webapp's
 *  `_serve_index` names one, and native Desktop loads its own renderer, never that page. */
function bootedBundle(): null | string {
  const id = (window as Window & { __HERMES_WEBAPP_BUNDLE__?: unknown }).__HERMES_WEBAPP_BUNDLE__

  return typeof id === 'string' && id ? id : null
}

/** Unsent work on the page a reload would lose: text in ANY mounted composer editor (main, session
 *  tiles, an inline edit; visible or not), an attachment card in any of them, or an attachment
 *  stashed with a session's draft. The saved draft trails the editor by a debounce, and
 *  attachments are never saved across a reload, so the page itself is the record. */
function unsentWork(): boolean {
  const editors = document.querySelectorAll(`[data-slot="${RICH_INPUT_SLOT}"]`)

  return (
    [...editors].some(editor => composerPlainText(editor).trim()) ||
    document.querySelector(`[data-slot="${ATTACHMENTS_SLOT}"]`) !== null ||
    hasStashedAttachments()
  )
}

/** The person is somewhere a reload would cut short: a page other than the chat (settings, a
 *  profile's SOUL, a plugin page each hold their own unsaved forms), an open dialog (the update
 *  overlay included), a file being edited in the preview, or a voice conversation. */
function personElsewhere(): boolean {
  const route = window.location.hash.replace(/^#/, '') || NEW_CHAT_ROUTE

  return (
    appViewForPath(route) !== 'chat' ||
    document.querySelector('[role="dialog"], [role="alertdialog"]') !== null ||
    Object.keys($dirtyPreviewUrls.get()).length > 0 ||
    $voiceStreamPaintOwners.get() > 0
  )
}

/** Nothing on screen a reload would cut off: no turn running, awaiting its reply or waiting on the
 *  user, no unsent work, and the person on the chat with nothing open over it. */
function tabIdle(): boolean {
  const live = Object.values($sessionStates.get()).some(
    state => isLiveTurnAwaitingEvents(state) || Boolean(state?.needsInput)
  )

  return !live && !unsentWork() && !personElsewhere()
}

async function servedBundle(): Promise<ServedBundle | null> {
  try {
    return await window.hermesDesktop.api<ServedBundle>({ path: '/api/webapp/bundle', timeoutMs: 5000 })
  } catch {
    return null
  }
}

let reload: () => void = () => window.location.reload()

function reloadOnto(id: string): void {
  try {
    const reloaded: unknown = JSON.parse(sessionStorage.getItem(RELOADED_FOR_KEY) ?? '[]')
    const ids = Array.isArray(reloaded) ? reloaded : []

    if (ids.includes(id)) {
      return
    }

    sessionStorage.setItem(RELOADED_FOR_KEY, JSON.stringify([...ids, id]))
  } catch {
    // Without readable storage there is no loop guard: never reload blind.
    return
  }

  reload()
}

/** Reload this tab when its gateway now serves a different, current bundle. A busy tab is left
 *  alone; the next check asks again. Two checks at once (a reconnect during the minute check)
 *  reload at most once: the guard is read and written synchronously before the reload. */
export async function reloadOntoServedBundle(): Promise<void> {
  const booted = bootedBundle()

  if (!booted) {
    return
  }

  const served = await servedBundle()

  if (served?.current && served.id && served.id !== booted && tabIdle()) {
    reloadOnto(served.id)
  }
}

/** Check on every (re)connect and once a minute: the rebuild finishes after the gateway is back,
 *  so the reconnect alone comes too early, and a tab that was busy is asked again. Returns the
 *  stop function. */
export function startServedBundleWatch(): () => void {
  if (!bootedBundle()) {
    return () => undefined
  }

  const timer = window.setInterval(() => void reloadOntoServedBundle(), CHECK_INTERVAL_MS)

  const unsub = $gatewayState.listen(state => {
    if (state === 'open') {
      void reloadOntoServedBundle()
    }
  })

  return () => {
    window.clearInterval(timer)
    unsub()
  }
}

/** Tests: replace the navigation. */
export function setBundleReloadForTests(next: (() => void) | null): void {
  reload = next ?? (() => window.location.reload())
}

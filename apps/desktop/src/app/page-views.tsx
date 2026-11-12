/**
 * The code-split page views, declared once so every mount site (workspace routes, route tiles,
 * overlays) shares one chunk promise, and so the shell can warm them before they are clicked.
 */
import { lazyView } from '@/lib/lazy-view'

export const AgentsView = lazyView(async () => (await import('./agents')).AgentsView)
export const ArtifactsView = lazyView(async () => (await import('./artifacts')).ArtifactsView)
export const CapabilitiesView = lazyView(async () => (await import('./capabilities')).CapabilitiesView)
export const CommandCenterView = lazyView(async () => (await import('./command-center')).CommandCenterView)
export const CronView = lazyView(async () => (await import('./cron')).CronView)
export const MessagingView = lazyView(async () => (await import('./messaging')).MessagingView)
export const ProfilesView = lazyView(async () => (await import('./profiles')).ProfilesView)
export const SettingsView = lazyView(async () => (await import('./agent-settings/entry')).SettingsView)
export const StarmapView = lazyView(async () => (await import('./starmap')).StarmapView)
export const WebhooksView = lazyView(async () => (await import('./webhooks')).WebhooksView)

/** The global rail's destinations (and the Agents page's nested chunks), in likely-click order. */
const RAIL_PRELOADS: (() => Promise<unknown>)[] = [
  ProfilesView.preload,
  async () => (await import('./profiles')).preloadAgentDirectory(),
  SettingsView.preload,
  async () => (await import('./agent-settings/entry')).preloadAgentSettingsView(),
  CapabilitiesView.preload,
  ArtifactsView.preload,
  CronView.preload
]

/**
 * Warm the rail destinations one idle slice at a time, so a click paints its page in the next
 * frame instead of waiting on a chunk fetch, its evaluation and React's Suspense reveal delay.
 * Returns a canceller.
 */
export function preloadRailPagesWhenIdle(): () => void {
  const queue = [...RAIL_PRELOADS]
  let cancelled = false

  let cancelSlice = () => {}

  const next = () => {
    const run = queue.shift()

    if (!run || cancelled) {
      return
    }

    // A failed warm-up is not an error: the click that needs the page loads it again.
    cancelSlice = whenIdle(() => void run().then(next, next))
  }

  next()

  return () => {
    cancelled = true
    cancelSlice()
  }
}

// Safari (the Webapp) has no requestIdleCallback.
function whenIdle(task: () => void): () => void {
  if (typeof requestIdleCallback === 'function') {
    const id = requestIdleCallback(task, { timeout: 2_000 })

    return () => cancelIdleCallback(id)
  }

  const id = setTimeout(task, 200)

  return () => clearTimeout(id)
}

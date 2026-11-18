/**
 * The code-split page views, declared once so every mount site (workspace routes, route tiles,
 * overlays) shares one chunk promise, and so the shell can warm them before they are clicked.
 */
import { lazyView } from '@/lib/lazy-view'
import { $portalShell } from '@/store/shell-variant'

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

/** Each global rail destination's chunks (and the reads its page lays out from), keyed by rail area. */
const RAIL_PAGE_PRELOADS: Record<string, (() => Promise<unknown>)[]> = {
  profiles: [
    ProfilesView.preload,
    async () => (await import('./profiles')).preloadAgentDirectory(),
    SettingsView.preload,
    async () => (await import('./agent-settings/entry')).preloadAgentSettingsView(),
    // The landing agent's Overview reads too, so its page paints with data, not placeholders.
    async () => (await import('./agent-settings/overview-queries')).prefetchLandingAgentOverview()
  ],
  // Capabilities and Library read lists before they can lay out; warm those too.
  capabilities: [
    CapabilitiesView.preload,
    async () => (await import('./capabilities/prefetch')).prefetchCapabilitiesInventory()
  ],
  artifacts: [ArtifactsView.preload, async () => (await import('@/store/item-index')).refreshItemIndex()],
  cron: [CronView.preload],
  settings: [SettingsView.preload]
}

/** In likely-click order. */
const RAIL_PRELOADS = Object.values(RAIL_PAGE_PRELOADS).flat()

/** Warm one rail destination now (the pointer is on its button, or focus is), so a click that lands
 *  before the idle warm-up reached it still paints its page at once. */
export function preloadRailPage(area: string): void {
  for (const run of RAIL_PAGE_PRELOADS[area] ?? []) {
    // A failed warm-up is not an error: the click that needs the page loads it again.
    run().catch(() => undefined)
  }
}

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

/** Warms the rail destinations while this window renders the redesign, whose global rail they are.
 *  Classic has no rail and loads each page (and its reads, e.g. the artifact index's transcripts)
 *  only when it opens, as on main. Returns a canceller. */
export function preloadRailPagesInRedesign(): () => void {
  let cancel: (() => void) | undefined

  const stop = $portalShell.subscribe(portal => {
    cancel?.()
    cancel = portal ? preloadRailPagesWhenIdle() : undefined
  })

  return () => {
    stop()
    cancel?.()
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

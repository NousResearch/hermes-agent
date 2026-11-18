/**
 * The code-split page views, declared once so every mount site (workspace routes, route tiles,
 * overlays) shares one chunk promise.
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

import { useSyncExternalStore } from 'react'

import { GlyphSpinner } from '@/components/ui/glyph-spinner'

import { PageSearchShell } from '../page-search-shell'

import {
  getDashboardPluginComponent,
  getDashboardPluginLoadError,
  onDashboardPluginRegistered
} from './registry'
import { useDashboardPluginAssets, useDashboardPluginManifest } from './use-dashboard-plugins'

interface DashboardPluginViewProps {
  name: string
}

const noopSearchChange = () => undefined

const DESKTOP_PLUGIN_HOST_CSS = `
[data-hermes-dashboard-plugin-host] .hermes-kanban-drawer-shade,
[data-hermes-dashboard-plugin-host] .hermes-kanban-dialog-backdrop {
  position: absolute;
  inset: 0;
  z-index: 30;
}

[data-hermes-dashboard-plugin-host] .hermes-kanban-drawer {
  height: 100%;
  max-height: 100%;
}

[data-hermes-dashboard-plugin-host] .hermes-kanban-dialog {
  max-height: calc(100% - 2rem);
  max-width: calc(100% - 2rem);
}
`

export function DashboardPluginView({ name }: DashboardPluginViewProps) {
  const { error: manifestError, loading: manifestLoading, manifest } = useDashboardPluginManifest(name)
  const { error: assetError, loading: assetLoading } = useDashboardPluginAssets(manifest)

  const Component = useSyncExternalStore(
    onDashboardPluginRegistered,
    () => getDashboardPluginComponent(name) ?? null,
    () => null
  )

  const loadError = useSyncExternalStore(
    onDashboardPluginRegistered,
    () => getDashboardPluginLoadError(name) ?? null,
    () => null
  )

  if (Component) {
    return (
      <PageSearchShell
        onSearchChange={noopSearchChange}
        searchHidden
        searchPlaceholder=""
        searchValue=""
        tabs={[{ id: name, label: manifest?.label || 'Kanban' }]}
      >
        <style>{DESKTOP_PLUGIN_HOST_CSS}</style>
        <div className="relative h-full min-w-0 overflow-hidden px-3 pb-3" data-hermes-dashboard-plugin-host={name}>
          <div className="relative h-full min-w-0 overflow-auto rounded-lg border border-border bg-(--ui-editor-surface-background) p-4 text-foreground">
            <Component />
          </div>
        </div>
      </PageSearchShell>
    )
  }

  if (manifestLoading || assetLoading) {
    return <DashboardPluginMessage description="Loading dashboard plugin…" loading title="Loading Kanban" />
  }

  const error = manifestError ?? assetError

  if (error) {
    return <DashboardPluginMessage description={error.message} title="Kanban failed to load" />
  }

  if (!manifest) {
    return <DashboardPluginMessage description="The kanban dashboard plugin is not installed or is hidden." title="Kanban unavailable" />
  }

  if (loadError) {
    return <DashboardPluginMessage description={formatPluginLoadError(loadError)} title="Kanban failed to register" />
  }

  return <DashboardPluginMessage description="Waiting for the plugin bundle to register its page." loading title="Loading Kanban" />
}

function DashboardPluginMessage({
  description,
  loading = false,
  title
}: {
  description: string
  loading?: boolean
  title: string
}) {
  return (
    <div className="flex h-full min-h-0 items-center justify-center bg-(--ui-editor-surface-background) p-6">
      <div className="max-w-md rounded-lg border border-border bg-card p-5 text-sm text-card-foreground shadow-sm">
        <div className="mb-2 flex items-center gap-2 text-sm font-semibold">
          {loading && <GlyphSpinner className="text-muted-foreground" />}
          <span>{title}</span>
        </div>
        <p className="text-muted-foreground">{description}</p>
      </div>
    </div>
  )
}

function formatPluginLoadError(code: string): string {
  if (code === 'LOAD_FAILED') {return 'The plugin script could not be loaded.'}

  if (code === 'NO_REGISTER') {return 'The plugin script loaded but did not register a page component.'}

  return code
}

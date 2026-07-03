import type * as React from 'react'

export interface DashboardPluginTab {
  hidden?: boolean
  override?: string
  path: string
  position?: string
}

export interface DashboardPluginManifest {
  css?: string | null
  description?: string
  entry: string
  has_api?: boolean
  icon?: string
  label: string
  name: string
  slots?: string[]
  source?: string
  tab: DashboardPluginTab
  version?: string
}

export interface RegisteredDashboardPlugin {
  component: React.ComponentType
  manifest: DashboardPluginManifest
}

export type DashboardPluginRegistryListener = () => void

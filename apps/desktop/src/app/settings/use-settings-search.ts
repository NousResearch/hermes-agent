import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useCallback, useEffect } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { $pluginRecords } from '@/contrib/plugins-store'
import { getEnvVars, getHermesConfigSchema } from '@/hermes'
import { useI18n } from '@/i18n'
import { type IconComponent, Monitor, Package, Settings2, Wrench } from '@/lib/icons'
import { $agentPlugins, isDesktopRelevantPlugin, loadAgentPlugins } from '@/store/agent-plugins'
import { $gatewayState } from '@/store/session'

import { useHermesConfigRecord } from '../hooks/use-config-record'
import { useOnProfileSwitch } from '../hooks/use-on-profile-switch'

import { SECTIONS } from './constants'
import { OTHER_SUBPAGES } from './other-subpages'
import { settingSearchTargets } from './settings-manifest'
import { buildConfigSearchEntries, buildCredentialSearchEntries, type SettingsSearchEntry } from './settings-search'
import { settingsSubpages } from './subpages'
import type { SettingsView } from './types'

/** An installed plugin row, deep-linkable as `/capabilities?tab=plugins&plugin=<id>`. */
export interface PluginSearchEntry {
  context: string
  description?: string
  icon: IconComponent
  id: string
  keywords: string[]
  label: string
  plugin: string
}

/**
 * The granular settings-search catalog (appearance controls, config fields,
 * credentials) for the command palette's Settings page. Page destinations stay
 * on the palette side — it already owns section/page rows — this hook only
 * contributes the deep, schema-driven targets.
 */
export function useSettingsSearchCatalog(enabled: boolean) {
  const { t } = useI18n()
  const configQuery = useHermesConfigRecord()

  const schemaQuery = useQuery({
    queryKey: ['hermes-config-schema'],
    queryFn: () => getHermesConfigSchema(),
    enabled,
    staleTime: 5 * 60 * 1000
  })

  const {
    data: envVars,
    isError: envVarsError,
    isFetching: envVarsFetching,
    refetch: refetchEnvVars
  } = useQuery({
    queryKey: ['desktop-settings-search-env-vars'],
    queryFn: () => getEnvVars(),
    enabled,
    staleTime: 5 * 60 * 1000
  })

  const refreshCatalog = useCallback(() => {
    void refetchEnvVars()
  }, [refetchEnvVars])

  useOnProfileSwitch(refreshCatalog)

  // Plugin rows: desktop plugins are already in their store (discovered at
  // boot); agent plugins ride the gateway, so load them the first time the
  // catalog is wanted — same RPC the Plugins page fires on mount, deduped by
  // the store's own inflight guard.
  const { requestGateway } = useGatewayRequest()
  const gatewayState = useStore($gatewayState)
  const desktopPluginRecords = useStore($pluginRecords)
  const agentPlugins = useStore($agentPlugins)

  useEffect(() => {
    if (enabled && gatewayState === 'open') {
      void loadAgentPlugins(requestGateway)
    }
  }, [enabled, gatewayState, requestGateway])

  // Installed plugin rows (both halves) — they live on Capabilities → Plugins,
  // so each entry carries the `?plugin=` row selector for that page.
  const pluginEntries: PluginSearchEntry[] = [
    ...Object.values(desktopPluginRecords).map(record => ({
      context: t.settings.plugins.title,
      description: record.description,
      icon: Monitor,
      id: `plugin:desktop:${record.id}`,
      keywords: ['plugin', 'extension', 'desktop', record.id],
      label: record.name,
      plugin: record.id
    })),
    ...agentPlugins.filter(isDesktopRelevantPlugin).map(row => ({
      context: t.skills.plugins.agentTitle,
      description: row.description || undefined,
      icon: Package,
      id: `plugin:agent:${row.key ?? row.name}`,
      keywords: ['plugin', 'extension', 'agent', ...(row.key ? [row.key] : [])],
      label: row.name,
      plugin: row.key ?? row.name
    }))
  ]

  // Never expose stale profile-scoped targets while a catalog is refreshing.
  // Field/key results wait for the current profile's data rather than briefly
  // pointing into the previous one.
  const configEntries =
    configQuery.isFetching || schemaQuery.isFetching || configQuery.isError || schemaQuery.isError
      ? []
      : buildConfigSearchEntries(schemaQuery.data?.fields, configQuery.data, {
          fieldDescriptions: t.settings.fieldDescriptions,
          fieldLabels: t.settings.fieldLabels,
          sections: t.settings.sections
        })

  const credentialEntries = buildCredentialSearchEntries(
    envVarsFetching || envVarsError ? null : envVars,
    {
      settings: t.settings.nav.keysSettings,
      tools: t.settings.nav.keysTools
    },
    { settings: Settings2, tools: Wrench }
  )

  const pageLabels: Record<string, string> = t.settings.nav

  // The pages that own rows: config sections and the standalone views.
  const parents = [
    ...SECTIONS.map(section => ({
      view: `config:${section.id}` as SettingsView,
      label: t.settings.sections[section.id] ?? section.label,
      icon: section.icon
    })),
    ...Object.keys(OTHER_SUBPAGES).map(view => ({
      view: view as SettingsView,
      label: pageLabels[view],
      icon: Settings2
    }))
  ]

  const parentOf = (view: SettingsView) => parents.find(parent => parent.view === view)

  // Every hand-built settings row, straight from the manifest that also
  // routes and ids them — the palette cannot drift from the pages.
  const settingEntries: SettingsSearchEntry[] = settingSearchTargets(t).map(({ id, view, ...entry }) => {
    const parent = parentOf(view)

    return {
      ...entry,
      context: parent?.label ?? view,
      icon: parent?.icon ?? Settings2,
      id: `setting:${id}`,
      target: { setting: id, view }
    }
  })

  // A page named after its one setting (Appearance › Theme) would show up as
  // two identical rows; the setting wins because it lands on the row itself.
  const settingLabels = new Set(settingEntries.map(entry => `${entry.context}\u0000${entry.label}`))

  const subpageEntries: SettingsSearchEntry[] = parents
    .flatMap(parent =>
      settingsSubpages(parent.view).map(page => ({
        context: parent.label,
        icon: parent.icon,
        id: `settings-page:${parent.view}:${page.id}`,
        keywords: [parent.label, page.id],
        label: t.settings.subpages[page.labelKey],
        target: { view: parent.view, subpage: page.id }
      }))
    )
    .filter(entry => !settingLabels.has(`${entry.context}\u0000${entry.label}`))

  return {
    subpageEntries,
    settingEntries,
    configEntries,
    credentialEntries,
    pluginEntries
  }
}

import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useCallback, useEffect } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { $pluginRecords } from '@/contrib/plugins-store'
import { getEnvVars, getHermesConfigSchema, profileScopeKey } from '@/hermes'
import { useI18n } from '@/i18n'
import { type IconComponent, Monitor, Package, Palette, Settings2, Wrench } from '@/lib/icons'
import { $agentPlugins, isDesktopRelevantPlugin, loadAgentPlugins } from '@/store/agent-plugins'
import { $gatewayState } from '@/store/session'
import { $settingsOwner } from '@/store/settings-scope'

import { useHermesConfigRecord } from '../hooks/use-config-record'
import { useOnProfileSwitch } from '../hooks/use-on-profile-switch'

import { appearanceSearchTargets } from './appearance-subpages'
import { SECTIONS } from './constants'
import { OTHER_SUBPAGES } from './other-subpages'
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
  const owner = useStore($settingsOwner)
  const scopeKey = owner ? profileScopeKey(owner) : 'unavailable'
  const configQuery = useHermesConfigRecord(owner ?? undefined, enabled && Boolean(owner))

  const schemaQuery = useQuery({
    queryKey: ['hermes-config-schema', scopeKey],
    queryFn: () => getHermesConfigSchema(owner!),
    enabled: enabled && Boolean(owner),
    staleTime: 5 * 60 * 1000
  })

  const {
    data: envVars,
    isError: envVarsError,
    isFetching: envVarsFetching,
    refetch: refetchEnvVars
  } = useQuery({
    queryKey: ['desktop-settings-search-env-vars', scopeKey],
    queryFn: () => getEnvVars(owner!),
    enabled: enabled && Boolean(owner),
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

  const appearanceContext = t.settings.sections.appearance

  // Every hand-built Appearance row, straight from the manifest that also
  // routes and ids them — the palette cannot drift from the page.
  const appearanceEntries: SettingsSearchEntry[] = appearanceSearchTargets(t).map(({ id, ...entry }) => ({
    ...entry,
    context: appearanceContext,
    icon: Palette,
    id: `setting:${id}`,
    target: { setting: id, view: 'config:appearance' }
  }))

  const credentialEntries = buildCredentialSearchEntries(
    envVarsFetching || envVarsError ? null : envVars,
    {
      settings: t.settings.nav.keysSettings,
      tools: t.settings.nav.keysTools
    },
    { settings: Settings2, tools: Wrench }
  )

  const pageLabels: Record<string, string> = {
    ...t.settings.nav,
    sessions: t.settings.nav.archivedChats
  }

  // A page named after its one setting (Appearance › Theme) would show up as
  // two identical rows; the setting wins because it lands on the row itself.
  const settingLabels = new Set(appearanceEntries.map(entry => `${entry.context}\u0000${entry.label}`))

  const subpageEntries: SettingsSearchEntry[] = [
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
    subpageEntries: [
      ...subpageEntries,
      ...(window.hermesDesktop?.hudModifier
        ? [
            {
              context: t.keybinds.title,
              icon: Settings2,
              id: 'setting:hud-modifier',
              keywords: ['HUD', 'summon', 'modifier', 'tap', 'Ctrl', 'Alt', 'Command', 'Option'],
              label: t.settings.hudModifier.title,
              description: t.settings.hudModifier.description,
              target: { view: 'keybinds' as const, subpage: 'hud-gesture', setting: 'hud-modifier' }
            }
          ]
        : [])
    ],
    appearanceEntries,
    configEntries,
    credentialEntries,
    pluginEntries
  }
}

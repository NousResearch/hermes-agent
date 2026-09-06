import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useCallback, useEffect } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { $pluginRecords } from '@/contrib/plugins-store'
import { getEnvVars, getHermesConfigSchema } from '@/hermes'
import { useI18n } from '@/i18n'
import { Package, Palette, Settings2, Wrench } from '@/lib/icons'
import { $agentPlugins, isDesktopRelevantPlugin, loadAgentPlugins } from '@/store/agent-plugins'
import { $gatewayState } from '@/store/session'
import { TRANSLUCENCY_SUPPORTED } from '@/store/translucency'

import { useHermesConfigRecord } from '../hooks/use-config-record'
import { useOnProfileSwitch } from '../hooks/use-on-profile-switch'

import {
  APPEARANCE_SETTING_IDS,
  buildConfigSearchEntries,
  buildCredentialSearchEntries,
  type SettingsSearchEntry
} from './settings-search'

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

  const pluginContext = t.settings.nav.plugins

  const pluginEntries: SettingsSearchEntry[] = [
    ...Object.values(desktopPluginRecords).map(record => ({
      context: pluginContext,
      description: record.description,
      icon: Package,
      id: `plugin:desktop:${record.id}`,
      keywords: ['plugin', 'extension', record.id],
      label: record.name,
      target: { plugin: record.id, view: 'plugins' as const }
    })),
    ...agentPlugins.filter(isDesktopRelevantPlugin).map(row => ({
      context: pluginContext,
      description: row.description || undefined,
      icon: Package,
      id: `plugin:agent:${row.key ?? row.name}`,
      keywords: ['plugin', 'extension', ...(row.key ? [row.key] : [])],
      label: row.name,
      target: { plugin: row.key ?? row.name, view: 'plugins' as const }
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
  const appearance = t.settings.appearance

  const appearanceEntries: SettingsSearchEntry[] = [
    {
      context: appearanceContext,
      description: t.language.description,
      icon: Palette,
      id: `setting:${APPEARANCE_SETTING_IDS.language}`,
      keywords: ['locale'],
      label: t.language.label,
      target: { setting: APPEARANCE_SETTING_IDS.language, view: 'config:appearance' }
    },
    {
      context: appearanceContext,
      description: appearance.themeDesc,
      icon: Palette,
      id: `setting:${APPEARANCE_SETTING_IDS.theme}`,
      keywords: ['color mode', 'skin'],
      label: appearance.themeTitle,
      target: { setting: APPEARANCE_SETTING_IDS.theme, view: 'config:appearance' }
    },
    {
      context: appearanceContext,
      icon: Palette,
      id: `setting:${APPEARANCE_SETTING_IDS.uiScale}`,
      keywords: ['zoom', 'size'],
      label: appearance.uiScaleTitle,
      target: { setting: APPEARANCE_SETTING_IDS.uiScale, view: 'config:appearance' }
    },
    // Linux has no translucency row to land on, and a palette hit that scrolls
    // to nothing is worse than no hit.
    ...(TRANSLUCENCY_SUPPORTED
      ? [
          {
            context: appearanceContext,
            description: appearance.translucencyDesc,
            icon: Palette,
            id: `setting:${APPEARANCE_SETTING_IDS.translucency}`,
            keywords: ['opacity', 'transparent'],
            label: appearance.translucencyTitle,
            target: { setting: APPEARANCE_SETTING_IDS.translucency, view: 'config:appearance' as const }
          }
        ]
      : []),
    {
      context: appearanceContext,
      description: appearance.userBubbleDesc,
      icon: Palette,
      id: `setting:${APPEARANCE_SETTING_IDS.userBubble}`,
      keywords: ['opacity', 'transparent', 'message', 'bubble'],
      label: appearance.userBubbleTitle,
      target: { setting: APPEARANCE_SETTING_IDS.userBubble, view: 'config:appearance' }
    },
    {
      context: appearanceContext,
      description: appearance.backdropDesc,
      icon: Palette,
      id: `setting:${APPEARANCE_SETTING_IDS.backdrop}`,
      keywords: ['background', 'blur'],
      label: appearance.backdropTitle,
      target: { setting: APPEARANCE_SETTING_IDS.backdrop, view: 'config:appearance' }
    },
    {
      context: appearanceContext,
      description: appearance.introSplashDesc,
      icon: Palette,
      id: `setting:${APPEARANCE_SETTING_IDS.introSplash}`,
      keywords: ['splash', 'wordmark', 'empty chat', 'new chat'],
      label: appearance.introSplashTitle,
      target: { setting: APPEARANCE_SETTING_IDS.introSplash, view: 'config:appearance' }
    },
    {
      context: appearanceContext,
      description: appearance.toolViewDesc,
      icon: Palette,
      id: `setting:${APPEARANCE_SETTING_IDS.toolView}`,
      keywords: ['tool display', 'technical'],
      label: appearance.toolViewTitle,
      target: { setting: APPEARANCE_SETTING_IDS.toolView, view: 'config:appearance' }
    },
    {
      context: appearanceContext,
      description: appearance.embedsDesc,
      icon: Palette,
      id: `setting:${APPEARANCE_SETTING_IDS.embeds}`,
      keywords: ['external content', 'privacy'],
      label: appearance.embedsTitle,
      target: { setting: APPEARANCE_SETTING_IDS.embeds, view: 'config:appearance' }
    }
  ]

  const credentialEntries = buildCredentialSearchEntries(
    envVarsFetching || envVarsError ? null : envVars,
    {
      settings: t.settings.nav.keysSettings,
      tools: t.settings.nav.keysTools
    },
    { settings: Settings2, tools: Wrench }
  )

  return {
    appearanceEntries,
    configEntries,
    credentialEntries,
    pluginEntries
  }
}

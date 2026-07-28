import { useQuery } from '@tanstack/react-query'
import { useCallback } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'

import { getEnvVars, getHermesConfigSchema } from '@/hermes'
import { useI18n } from '@/i18n'
import { Palette, Settings2, Wrench } from '@/lib/icons'
import { normalize } from '@/lib/text'

import { useHermesConfigRecord } from '../hooks/use-config-record'
import { useOnProfileSwitch } from '../hooks/use-on-profile-switch'
import type { OverlayNavGroup } from '../overlays/overlay-split-layout'

import {
  APPEARANCE_SETTING_IDS,
  buildConfigSearchEntries,
  buildCredentialSearchEntries,
  filterSettingsSearchEntries,
  type SettingsSearchEntry,
  type SettingsSearchTarget
} from './settings-search'
import type { SettingsView as SettingsViewId } from './types'

// Bespoke pages do not expose config-schema fields, so aliases make their
// important controls discoverable without pretending they have field targets.
const SETTINGS_PAGE_ALIASES: Record<string, string[]> = {
  'pview:accounts': ['oauth', 'login', 'sign in', 'subscription'],
  'pview:custom-endpoints': ['local model', 'ollama', 'vllm', 'openai compatible'],
  'pview:keys': ['api key', 'token', 'secret'],
  about: ['version', 'update'],
  billing: ['balance', 'plan', 'credits', 'subscription'],
  gateway: ['connection', 'remote', 'cloud'],
  keybinds: ['keyboard', 'shortcut', 'hotkey'],
  notifications: ['alerts', 'sound'],
  plugins: ['extension', 'addon'],
  sessions: ['archive', 'history']
}

function navSearchTarget(id: string): SettingsSearchTarget {
  if (id.startsWith('pview:')) {
    return { providerView: id.slice('pview:'.length) as SettingsSearchTarget['providerView'], view: 'providers' }
  }

  if (id.startsWith('kview:')) {
    return { keysView: id.slice('kview:'.length) as SettingsSearchTarget['keysView'], view: 'keys' }
  }

  return { view: id as SettingsViewId }
}

export function useSettingsSearch({ groups, onSelect, query }: UseSettingsSearchOptions) {
  const { t } = useI18n()
  const navigate = useNavigate()
  const { hash, pathname, search } = useLocation()
  const configQuery = useHermesConfigRecord()

  const schemaQuery = useQuery({
    queryKey: ['hermes-config-schema'],
    queryFn: getHermesConfigSchema,
    staleTime: 5 * 60 * 1000
  })

  const {
    data: envVars,
    isError: envVarsError,
    isFetching: envVarsFetching,
    refetch: refetchEnvVars
  } = useQuery({
    queryKey: ['desktop-settings-search-env-vars'],
    queryFn: getEnvVars,
    staleTime: 5 * 60 * 1000
  })

  const refreshCatalog = useCallback(() => {
    void refetchEnvVars()
  }, [refetchEnvVars])

  useOnProfileSwitch(refreshCatalog)

  const pageEntries: SettingsSearchEntry[] = groups.flatMap(group => [
    {
      context: t.settings.search.root,
      icon: group.icon,
      id: `page:${group.id}`,
      keywords: SETTINGS_PAGE_ALIASES[group.id] ?? [],
      label: group.label,
      target: navSearchTarget(group.id)
    },
    ...(group.children ?? []).map(child => ({
      context: group.label,
      icon: child.icon,
      id: `page:${child.id}`,
      keywords: SETTINGS_PAGE_ALIASES[child.id] ?? [],
      label: child.label,
      target: navSearchTarget(child.id)
    }))
  ])

  // Never expose stale profile-scoped targets while a catalog is refreshing.
  // Page destinations remain available, but field/key results wait for the
  // current profile's data rather than briefly pointing into the previous one.
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
    {
      context: appearanceContext,
      description: appearance.translucencyDesc,
      icon: Palette,
      id: `setting:${APPEARANCE_SETTING_IDS.translucency}`,
      keywords: ['opacity', 'transparent'],
      label: appearance.translucencyTitle,
      target: { setting: APPEARANCE_SETTING_IDS.translucency, view: 'config:appearance' }
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

  const normalizedQuery = normalize(query)

  const entries = filterSettingsSearchEntries(
    [...pageEntries, ...appearanceEntries, ...configEntries, ...credentialEntries],
    query
  )

  const loading = Boolean(normalizedQuery) && (configQuery.isFetching || schemaQuery.isFetching || envVarsFetching)
  const error = configQuery.isError || schemaQuery.isError || envVarsError

  const retry = () => {
    void configQuery.refetch()
    void schemaQuery.refetch()
    void refetchEnvVars()
  }

  const selectEntry = (entry: SettingsSearchEntry) => {
    const params = new URLSearchParams(search)
    const target = entry.target

    for (const param of ['field', 'key', 'kview', 'pview', 'setting']) {
      params.delete(param)
    }

    params.set('tab', target.view)

    if (target.providerView) {
      params.set('pview', target.providerView)
    }

    if (target.keysView) {
      params.set('kview', target.keysView)
    }

    if (target.field) {
      params.set('field', target.field)
    }

    if (target.setting) {
      params.set('setting', target.setting)
    }

    if (target.key) {
      params.set('key', target.key)
    }

    const qs = params.toString()
    navigate({ hash, pathname, search: qs ? `?${qs}` : '' }, { replace: true })
    onSelect()
  }

  return {
    entries,
    error,
    loading,
    retry,
    selectEntry
  }
}

interface UseSettingsSearchOptions {
  groups: OverlayNavGroup[]
  onSelect: () => void
  query: string
}

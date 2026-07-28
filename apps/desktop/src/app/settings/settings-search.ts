import type { IconComponent } from '@/lib/icons'
import { normalize } from '@/lib/text'
import type { ConfigFieldSchema, EnvVarInfo, HermesConfigRecord } from '@/types/hermes'

import { FIELD_LABELS, SECTIONS } from './constants'
import { credentialRowLabel } from './credential-key-ui'
import { fieldCopyForSchemaKey } from './field-copy'
import { prettyName, sectionFieldEntries, voiceFieldVisible } from './helpers'
import type { DesktopConfigSection, SettingsView } from './types'

export type CredentialSettingsView = 'settings' | 'tools'

export const APPEARANCE_SETTING_IDS = {
  backdrop: 'appearance.backdrop',
  embeds: 'appearance.embeds',
  language: 'appearance.language',
  theme: 'appearance.theme',
  toolView: 'appearance.tool-view',
  translucency: 'appearance.translucency',
  uiScale: 'appearance.ui-scale'
} as const

export interface SettingsSearchTarget {
  field?: string
  key?: string
  keysView?: CredentialSettingsView
  providerView?: 'accounts' | 'custom-endpoints' | 'keys'
  setting?: string
  view: SettingsView
}

export interface SettingsSearchEntry {
  context: string
  description?: string
  icon: IconComponent
  id: string
  keywords: string[]
  label: string
  target: SettingsSearchTarget
}

interface ConfigSearchCopy {
  fieldDescriptions: Record<string, string>
  fieldLabels: Record<string, string>
  sections: Record<string, string>
}

interface CredentialSearchCopy {
  settings: string
  tools: string
}

export function credentialSettingsView(info: EnvVarInfo): CredentialSettingsView | null {
  if (info.channel_managed) {
    return null
  }

  if (info.category === 'tool') {
    return 'tools'
  }

  if (info.category === 'setting' || info.category === 'messaging') {
    return 'settings'
  }

  return null
}

function configFieldLabel(key: string, copy: ConfigSearchCopy): string {
  return (
    fieldCopyForSchemaKey(copy.fieldLabels, key) ??
    fieldCopyForSchemaKey(FIELD_LABELS, key) ??
    prettyName(key.split('.').pop() ?? key)
  )
}

function configFieldDescription(key: string, field: ConfigFieldSchema, copy: ConfigSearchCopy): string {
  return fieldCopyForSchemaKey(copy.fieldDescriptions, key) ?? field.description ?? ''
}

export function buildConfigSearchEntries(
  schema: Record<string, ConfigFieldSchema> | null | undefined,
  config: HermesConfigRecord | null | undefined,
  copy: ConfigSearchCopy,
  sections: DesktopConfigSection[] = SECTIONS
): SettingsSearchEntry[] {
  if (!schema || !config) {
    return []
  }

  const sectionFields = sectionFieldEntries(schema, config)

  return sections.flatMap(section => {
    const context = copy.sections[section.id] ?? section.label
    const fields = sectionFields.get(section.id) ?? []
    const visibleFields = section.id === 'voice' ? fields.filter(([key]) => voiceFieldVisible(key, config)) : fields

    return visibleFields.map(([key, field]) => ({
      context,
      description: configFieldDescription(key, field, copy),
      icon: section.icon,
      id: `config-field:${key}`,
      keywords: ['settings', section.id, section.label, key],
      label: configFieldLabel(key, copy),
      target: {
        field: key,
        view: `config:${section.id}` as SettingsView
      }
    }))
  })
}

export function buildCredentialSearchEntries(
  vars: Record<string, EnvVarInfo> | null | undefined,
  copy: CredentialSearchCopy,
  icons: Record<CredentialSettingsView, IconComponent>
): SettingsSearchEntry[] {
  if (!vars) {
    return []
  }

  return Object.entries(vars)
    .sort(([a], [b]) => a.localeCompare(b))
    .flatMap(([key, info]) => {
      const view = credentialSettingsView(info)

      if (!view) {
        return []
      }

      return [
        {
          context: view === 'tools' ? copy.tools : copy.settings,
          description: info.description || undefined,
          icon: icons[view],
          id: `credential:${key}`,
          keywords: [key, info.url ?? '', ...(Array.isArray(info.tools) ? info.tools : [])],
          label: credentialRowLabel(key, info),
          target: {
            key,
            keysView: view,
            view: 'keys' as const
          }
        }
      ]
    })
}

function searchScore(entry: SettingsSearchEntry, query: string): number {
  const needle = normalize(query)

  if (!needle) {
    return 0
  }

  const label = normalize(entry.label)
  const context = normalize(entry.context)
  const haystack = normalize([entry.label, entry.context, entry.description ?? '', ...entry.keywords].join(' '))
  const terms = needle.split(/\s+/).filter(Boolean)

  if (!terms.every(term => haystack.includes(term))) {
    return 0
  }

  if (label === needle) {
    return 100
  }

  if (label.startsWith(needle)) {
    return 90
  }

  if (label.includes(needle)) {
    return 80
  }

  if (context.includes(needle)) {
    return 70
  }

  if (terms.every(term => label.includes(term) || context.includes(term))) {
    return 60
  }

  return 50
}

export function filterSettingsSearchEntries(entries: SettingsSearchEntry[], query: string): SettingsSearchEntry[] {
  return entries
    .map((entry, index) => ({ entry, index, score: searchScore(entry, query) }))
    .filter(result => result.score > 0)
    .sort((a, b) => b.score - a.score || a.index - b.index)
    .map(result => result.entry)
}

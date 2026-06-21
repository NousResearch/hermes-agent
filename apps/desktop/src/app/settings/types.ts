import type { Dispatch, SetStateAction } from 'react'

import type { LucideIcon } from '@/lib/icons'
import type { DesktopLanguage } from '@/store/language'
import type { EnvVarInfo } from '@/types/hermes'

export type SettingsView = 'gateway' | 'keys' | 'tools' | `config:${string}`
export type SettingsQueryKey = 'config' | 'gateway' | 'keys' | 'tools'
export type EnvPatch = Partial<Pick<EnvVarInfo, 'is_set' | 'redacted_value'>>

export interface SettingsPageProps {
  onClose: () => void
  onConfigSaved?: () => void
}

export interface SearchProps {
  query: string
}

export interface ProviderGroup {
  name: string
  priority: number
  entries: [string, EnvVarInfo][]
  hasAnySet: boolean
}

export interface DesktopConfigSection {
  id: string
  label: string
  icon: LucideIcon
  keys: string[]
}

export interface EnvRowProps {
  varKey: string
  info: EnvVarInfo
  edits: Record<string, string>
  revealed: Record<string, string>
  saving: string | null
  setEdits: Dispatch<SetStateAction<Record<string, string>>>
  onSave: (key: string) => void
  onClear: (key: string) => void
  onReveal: (key: string) => void
  language: DesktopLanguage
  compact?: boolean
}

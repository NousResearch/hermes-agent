import type { Dispatch, SetStateAction } from 'react'

import type { HermesGateway } from '@/hermes'
import type { IconComponent } from '@/lib/icons'
import type { EnvVarInfo } from '@/types/hermes'

export type SettingsView = 'about' | 'gateway' | 'keys' | 'mcp' | 'sessions' | `config:${string}`
export type SettingsQueryKey = 'about' | 'config' | 'gateway' | 'keys' | 'mcp' | 'sessions'
export type EnvPatch = Partial<Pick<EnvVarInfo, 'is_set' | 'redacted_value'>>

export interface SettingsPageProps {
  gateway?: HermesGateway | null
  onClose: () => void
  onConfigSaved?: () => void
  onMainModelChanged?: (provider: string, model: string) => void
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
  icon: IconComponent
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
  compact?: boolean
}

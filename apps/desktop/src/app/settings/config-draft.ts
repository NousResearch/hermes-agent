import { createContext, type RefObject } from 'react'

import type { ProfileScope } from '@/hermes'
import type { HermesConfigRecord } from '@/types/hermes'

export interface ConfigSettingsDraft {
  config: HermesConfigRecord
  baseline: HermesConfigRecord
  writeScope: ProfileScope | undefined
  version: number
  savedVersion: number
  queue: Promise<void>
}

/** The owner route retains one editor draft when its presentation changes. */
export const ConfigSettingsDraftContext = createContext<RefObject<ConfigSettingsDraft | null> | null>(null)

import type { EnvVarInfo } from '@/types/hermes'

/** An unset secret in `category`. The Keys tab and the settings search both
 *  bucket by category and branch on `is_set`, so those are what tests vary. */
export function envVar(category: string, patch: Partial<EnvVarInfo> = {}): EnvVarInfo {
  return {
    advanced: false,
    category,
    description: '',
    is_password: true,
    is_set: false,
    redacted_value: null,
    tools: [],
    url: '',
    ...patch
  }
}

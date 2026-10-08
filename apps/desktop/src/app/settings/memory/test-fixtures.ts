import type { MemoryProviderField } from '@/types/hermes'

export function field(
  overrides: Partial<MemoryProviderField> & Pick<MemoryProviderField, 'key' | 'kind'>
): MemoryProviderField {
  return {
    label: overrides.key,
    value: '',
    description: '',
    placeholder: '',
    is_set: false,
    inline: false,
    group: 'Other',
    options: [],
    ...overrides
  }
}

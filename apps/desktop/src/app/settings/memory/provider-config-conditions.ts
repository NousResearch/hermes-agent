import type { MemoryProviderFieldCondition } from '@/types/hermes'

export function conditionsMatch(
  conditions: MemoryProviderFieldCondition[] | undefined,
  values: Record<string, string>
): boolean {
  return (conditions ?? []).every(condition => {
    const value = values[condition.key] ?? ''

    if (condition.values.length > 0 && !condition.values.includes(value)) {
      return false
    }

    if (condition.pattern) {
      try {
        return new RegExp(condition.pattern, 'i').test(value)
      } catch {
        return false
      }
    }

    return true
  })
}

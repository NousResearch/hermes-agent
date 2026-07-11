import type { HermesConfigRecord } from '@/types/hermes'

export function getNested(config: HermesConfigRecord | Record<string, unknown>, path: string): unknown {
  return path.split('.').reduce<unknown>((acc, key) => {
    if (!acc || typeof acc !== 'object') {
      return undefined
    }

    return (acc as Record<string, unknown>)[key]
  }, config)
}

export function setNested(config: HermesConfigRecord, path: string, value: unknown): HermesConfigRecord {
  const keys = path.split('.')
  const root: Record<string, unknown> = { ...config }
  let cursor: Record<string, unknown> = root

  for (let i = 0; i < keys.length - 1; i += 1) {
    const key = keys[i]
    const next = cursor[key]

    const clone =
      next && typeof next === 'object' && !Array.isArray(next) ? { ...(next as Record<string, unknown>) } : {}

    cursor[key] = clone
    cursor = clone
  }

  cursor[keys[keys.length - 1]] = value

  return root as HermesConfigRecord
}

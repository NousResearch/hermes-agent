import type { HermesConfigRecord, ToolsetInfo } from '@/types/hermes'

import { BUILTIN_PERSONALITIES, ENUM_OPTIONS, PROVIDER_GROUPS } from './constants'

export const asText = (v: unknown): string => (typeof v === 'string' ? v : v == null ? '' : String(v))

export const includesQuery = (v: unknown, q: string) => asText(v).toLowerCase().includes(q)

export const prettyName = (v: string) => v.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())

export const toolNames = (t: ToolsetInfo) => (Array.isArray(t.tools) ? t.tools.map(asText).filter(Boolean) : [])

export const withoutKey = <T>(record: Record<string, T>, key: string) => {
  const next = { ...record }
  delete next[key]

  return next
}

export const redactedValue = (v: string) => (v.length <= 8 ? '••••' : `${v.slice(0, 4)}...${v.slice(-4)}`)

export const providerGroup = (key: string) => PROVIDER_GROUPS.find(g => key.startsWith(g.prefix))?.name ?? 'Other'

export const providerPriority = (name: string) => PROVIDER_GROUPS.find(g => g.name === name)?.priority ?? 99

const POLLUTING_PATH_PARTS = new Set(['__proto__', 'constructor', 'prototype'])

function isSafePart(part: string): boolean {
  return part.length > 0 && !POLLUTING_PATH_PARTS.has(part)
}

function configPathParts(path: string): string[] {
  const parts = path.split('.')

  if (!parts.every(isSafePart)) {
    throw new Error(`Unsafe config path: ${path}`)
  }

  return parts
}

function safeSet(target: Record<string, unknown>, key: string, value: unknown): void {
  if (key === '__proto__' || key === 'constructor' || key === 'prototype' || !key) {
    throw new Error(`Unsafe config key: ${key}`)
  }

  Object.defineProperty(target, key, {
    value,
    writable: true,
    enumerable: true,
    configurable: true
  })
}

export function getNested(obj: HermesConfigRecord, path: string): unknown {
  let cur: unknown = obj

  for (const part of configPathParts(path)) {
    if (cur == null || typeof cur !== 'object') {
      return undefined
    }

    if (!Object.prototype.hasOwnProperty.call(cur, part)) {
      return undefined
    }

    cur = (cur as Record<string, unknown>)[part]
  }

  return cur
}

export function setNested(obj: HermesConfigRecord, path: string, value: unknown): HermesConfigRecord {
  const clone = structuredClone(obj)
  const parts = configPathParts(path)
  let cur: Record<string, unknown> = clone

  for (let i = 0; i < parts.length - 1; i += 1) {
    const part = parts[i]

    if (!isSafePart(part)) {
      throw new Error(`Unsafe config path part: ${part}`)
    }

    const existing = Object.prototype.hasOwnProperty.call(cur, part) ? cur[part] : undefined

    if (existing == null || typeof existing !== 'object') {
      safeSet(cur, part, {})
    }

    cur = cur[part] as Record<string, unknown>
  }

  safeSet(cur, parts[parts.length - 1], value)

  return clone
}

function personalityOptions(config: HermesConfigRecord): string[] {
  const custom = getNested(config, 'agent.personalities')

  const customNames =
    custom && typeof custom === 'object' && !Array.isArray(custom) ? Object.keys(custom as Record<string, unknown>) : []

  return [...new Set(['', ...BUILTIN_PERSONALITIES, ...customNames])]
}

export function enumOptionsFor(
  key: string,
  value: unknown,
  config: HermesConfigRecord,
  dynamicOptions?: string[]
): string[] | undefined {
  const opts = dynamicOptions ?? (key === 'display.personality' ? personalityOptions(config) : ENUM_OPTIONS[key])

  if (!opts) {
    return undefined
  }

  const current = asText(value)

  return current && !opts.includes(current) ? [...opts, current] : opts
}

import { asText } from '@/lib/text'
import type { ConfigFieldSchema, HermesConfigRecord, ToolsetInfo } from '@/types/hermes'

import { BUILTIN_PERSONALITIES, ENUM_OPTIONS, PROVIDER_GROUPS, SECTIONS } from './constants'

// Canonical implementations live in @/lib/text; re-exported here so the many
// settings/capabilities call sites keep their import path.
export { asText, includesQuery, prettyName } from '@/lib/text'

/** Strip leading emoji from toolset titles (CLI registry prefixes labels with icons). */
export const stripToolsetLabel = (label: string): string =>
  label.replace(/^[\p{Emoji}\p{Extended_Pictographic}\s]+/u, '').trim() || label

export const toolsetDisplayLabel = (toolset: Pick<ToolsetInfo, 'label' | 'name'>): string =>
  stripToolsetLabel(asText(toolset.label || toolset.name))

export const toolNames = (t: ToolsetInfo) => (Array.isArray(t.tools) ? t.tools.map(asText).filter(Boolean) : [])

export const withoutKey = <T>(record: Record<string, T>, key: string) => {
  const next = { ...record }
  delete next[key]

  return next
}

export const redactedValue = (v: string) => (v.length <= 8 ? '••••' : `${v.slice(0, 4)}...${v.slice(-4)}`)

// Longest-prefix match so a more specific group like ``MINIMAX_CN_`` is
// chosen over its shorter parent ``MINIMAX_``. Falls back to the bucket
// "Other" used by the Keys settings view for un-grouped env vars.
export const providerGroup = (key: string) => {
  let best: (typeof PROVIDER_GROUPS)[number] | undefined

  for (const candidate of PROVIDER_GROUPS) {
    if (!key.startsWith(candidate.prefix)) {
      continue
    }

    if (!best || candidate.prefix.length > best.prefix.length) {
      best = candidate
    }
  }

  return best?.name ?? 'Other'
}

export const providerMeta = (name: string) =>
  PROVIDER_GROUPS.find(g => g.name === name && (g.description || g.docsUrl)) ??
  PROVIDER_GROUPS.find(g => g.name === name)

export const providerPriority = (name: string) => providerMeta(name)?.priority ?? 99

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

export function inferFieldSchema(value: unknown): ConfigFieldSchema {
  if (typeof value === 'boolean') {
    return { type: 'boolean' }
  }

  if (typeof value === 'number') {
    return { type: 'number' }
  }

  if (Array.isArray(value)) {
    return { type: 'list' }
  }

  return { type: 'string' }
}

// Backend schema omits some declared keys (e.g. memory.provider); config presence is the availability signal.
export function sectionFieldEntries(
  schema: Record<string, ConfigFieldSchema>,
  config: HermesConfigRecord
): Map<string, [string, ConfigFieldSchema][]> {
  return new Map(
    SECTIONS.map(s => [
      s.id,
      s.keys.flatMap(k => {
        const value = getNested(config, k)
        const field = schema[k] ?? (value === undefined ? undefined : inferFieldSchema(value))

        return field ? [[k, field] as [string, ConfigFieldSchema]] : []
      })
    ])
  )
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

// Built-in memory (MEMORY.md/USER.md) is controlled by memory_enabled, not
// memory.provider — only a real external plugin name gets provider-shaped
// affordances (config panel, OAuth connect). See #49513.
export function isExternalMemoryProvider(value: unknown): value is string {
  if (typeof value !== 'string') {
    return false
  }

  const normalized = value.trim().toLowerCase()

  return normalized !== '' && normalized !== 'builtin' && normalized !== 'built-in' && normalized !== 'none'
}

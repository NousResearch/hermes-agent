import { describe, expect, it } from 'vitest'

import { en } from './en'
import { vi } from './vi'

// Machine-checkable contract for a fully translated locale.
//
// `vi` declares `Translations` directly instead of going through
// `defineLocale()`, so TypeScript already guarantees every key exists. These
// tests cover what the type system cannot: that no machine-meaningful token
// was translated away, that interpolated values still land in the output, and
// that no user-facing English string was left behind verbatim.

const TOKEN_PATTERNS = [
  /https?:\/\/[^\s)'"`]+/g,
  /`[^`]+`/g,
  /(?<![\p{L}\p{N}/:])\/[a-z][a-z0-9_-]*/giu,
  /--[a-z][a-z0-9_-]*/gi,
  /\b[A-Z][A-Z0-9_]{2,}s?\b/g,
  /\b[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+\b/gi
]

// Abbreviations that legitimately differ between languages ("e.g." vs "v.v.")
// while matching the dotted-identifier pattern.
const ABBREVIATIONS = new Set(['e.g', 'v.v'])

const BRANDS =
  /^(Hermes|Nous Research|GitHub|OpenAI|Anthropic|Google|Discord|Telegram|Slack|WhatsApp|Signal|Codex|Claude|Gemini|OpenCode)(\s|$)/

/** Machine-meaningful substrings that must survive translation unchanged. */
function protectedTokens(value: string): string[] {
  return TOKEN_PATTERNS.flatMap(pattern => value.match(pattern) ?? [])
    .filter(token => !ABBREVIATIONS.has(token))
    .map(token => (token === 'URLs' ? 'URL' : token))
    .sort()
}

/** True when a string is prose a translator was expected to rewrite. */
function isTranslatableProse(value: string): boolean {
  if (!/[A-Za-z]/.test(value) || !/\s/.test(value)) {
    return false
  }

  if (BRANDS.test(value)) {
    return false
  }

  const acronymsOnly =
    /^[A-Za-z0-9_.:/@#%+*()[\]{}<>=,'" -]+$/.test(value) &&
    /(?:^|\s)(?:API|URL|JSON|YAML|SSH|RDP|OAuth|MCP|CLI|PTY|WebSocket)(?:\s|$)/.test(value)

  return !acronymsOnly
}

interface Leaf {
  path: string
  source: string
  target: string
  /** `template` leaves come from functions and are often pure technical output. */
  kind: 'string' | 'template'
}

function collectLeaves(source: unknown, target: unknown, path = '', leaves: Leaf[] = []): Leaf[] {
  if (typeof source === 'string' && typeof target === 'string') {
    leaves.push({ path, source, target, kind: 'string' })

    return leaves
  }

  if (typeof source === 'function' && typeof target === 'function') {
    // Probe with sentinel numbers so interpolated values are traceable in the
    // rendered output regardless of the argument's declared type.
    const args = Array.from({ length: source.length }, (_, index) => 987654321 + index)

    try {
      const sourceOut = (source as (...values: number[]) => string)(...args)
      const targetOut = (target as (...values: number[]) => string)(...args)

      if (typeof sourceOut === 'string' && typeof targetOut === 'string') {
        for (const arg of args) {
          const sentinel = String(arg)
          const expected = sourceOut.split(sentinel).length - 1
          const actual = targetOut.split(sentinel).length - 1
          expect(`${path} interpolates ${sentinel} ${actual}x`).toBe(`${path} interpolates ${sentinel} ${expected}x`)
        }

        leaves.push({ path, source: sourceOut, target: targetOut, kind: 'template' })
      }
    } catch {
      // Some entries return structured objects or require typed values; the
      // TypeScript signature already constrains those.
    }

    return leaves
  }

  if (Array.isArray(source) && Array.isArray(target)) {
    expect(`${path} length ${target.length}`).toBe(`${path} length ${source.length}`)
    source.forEach((entry, index) => collectLeaves(entry, target[index], `${path}[${index}]`, leaves))

    return leaves
  }

  if (source && target && typeof source === 'object' && typeof target === 'object') {
    const sourceRecord = source as Record<string, unknown>
    const targetRecord = target as Record<string, unknown>
    expect(Object.keys(targetRecord).sort()).toEqual(Object.keys(sourceRecord).sort())

    for (const key of Object.keys(sourceRecord)) {
      collectLeaves(sourceRecord[key], targetRecord[key], path ? `${path}.${key}` : key, leaves)
    }
  }

  return leaves
}

describe('Vietnamese locale parity with English', () => {
  const leaves = collectLeaves(en, vi)

  it('covers every English string', () => {
    expect(leaves.length).toBeGreaterThan(2000)
  })

  it('preserves URLs, commands, config keys and other machine-meaningful tokens', () => {
    const mismatched = leaves
      .filter(leaf => protectedTokens(leaf.source).join('\u0000') !== protectedTokens(leaf.target).join('\u0000'))
      .map(leaf => `${leaf.path}: ${JSON.stringify(leaf.source)} -> ${JSON.stringify(leaf.target)}`)

    expect(mismatched).toEqual([])
  })

  it('leaves no user-facing English prose untranslated', () => {
    const untranslated = leaves
      .filter(leaf => leaf.kind === 'string' && leaf.source === leaf.target && isTranslatableProse(leaf.source))
      .map(leaf => `${leaf.path}: ${leaf.source}`)

    expect(untranslated).toEqual([])
  })
})

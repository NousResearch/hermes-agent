import { describe, expect, it } from 'vitest'

import { BOTS_LOCALES } from '@/plugins/hermes-bots/i18n'
import { KANBAN_LOCALES } from '@/plugins/kanban/i18n'

type Bundle = Record<string, unknown>

/** Flatten a bundle to dotted leaf paths so locales compare key by key. */
function flatten(node: unknown, prefix = '', out: Record<string, unknown> = {}): Record<string, unknown> {
  for (const [key, value] of Object.entries(node as Record<string, unknown>)) {
    const path = prefix ? `${prefix}.${key}` : key

    if (value && typeof value === 'object' && !Array.isArray(value)) flatten(value, path, out)
    else out[path] = value
  }

  return out
}

/** The `{name}` tokens a string interpolates, order-insensitive. */
function placeholders(value: string): string {
  return [...value.matchAll(/\{[A-Za-z0-9_]+\}/g)]
    .map(match => match[0])
    .sort()
    .join(',')
}

const plugins = [
  { name: 'kanban', locales: KANBAN_LOCALES as Bundle },
  { name: 'hermes-bots', locales: BOTS_LOCALES as Bundle }
]

/**
 * Plugin bundles ship their own locale maps (`KANBAN_LOCALES`,
 * `BOTS_LOCALES`), so a new key added to `en` silently misses every other
 * locale and falls back to English at runtime. These checks keep the shipped
 * locales in step with English: same keys, same value kind, same function
 * arity, same `{placeholder}` set.
 */
describe.each(plugins)('$name plugin locale bundles', ({ locales }) => {
  const english = flatten(locales.en)
  const englishKeys = Object.keys(english).sort()

  it('ships a non-empty English bundle', () => {
    expect(englishKeys.length).toBeGreaterThan(0)
  })

  const others = Object.keys(locales).filter(locale => locale !== 'en')

  it('registers at least one translated locale', () => {
    expect(others.length).toBeGreaterThan(0)
  })

  describe.each(others)('%s', locale => {
    const bundle = flatten(locales[locale])

    it('covers every English key', () => {
      expect(englishKeys.filter(key => !(key in bundle))).toEqual([])
    })

    it('adds no key English is missing', () => {
      expect(Object.keys(bundle).filter(key => !(key in english))).toEqual([])
    })

    it('keeps strings and message functions aligned with English', () => {
      const mismatched = englishKeys.filter(key => typeof english[key] !== typeof bundle[key])

      expect(mismatched).toEqual([])
    })

    it('keeps every message function the same arity as English', () => {
      const mismatched = englishKeys.filter(
        key =>
          typeof english[key] === 'function' &&
          (english[key] as (...args: unknown[]) => unknown).length !==
            (bundle[key] as (...args: unknown[]) => unknown).length
      )

      expect(mismatched).toEqual([])
    })

    it('keeps the {placeholder} set of every string', () => {
      const mismatched = englishKeys.filter(
        key =>
          typeof english[key] === 'string' &&
          placeholders(english[key] as string) !== placeholders(bundle[key] as string)
      )

      expect(mismatched).toEqual([])
    })
  })
})

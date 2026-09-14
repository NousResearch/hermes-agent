/**
 * The English bundle is the message shape. Every other shipped bundle must
 * cover the same leaves so a locale switch never falls through to a raw key —
 * and the interpolators must still splice their arguments, not drop them.
 *
 * The non-English bundles are derived from BOTS_LOCALES rather than listed by
 * name: a catalog that names its locales one by one silently stops covering
 * the next one added.
 */

import { describe, expect, it } from 'vitest'

import { BOTS_LOCALES } from './i18n'

type Leaf = string | ((...args: never[]) => string)

function leafEntries(node: unknown, prefix = ''): Array<[string, Leaf]> {
  if (typeof node === 'function' || typeof node === 'string') {
    return [[prefix, node as Leaf]]
  }

  return Object.entries(node as Record<string, unknown>).flatMap(([key, value]) =>
    leafEntries(value, prefix ? `${prefix}.${key}` : key)
  )
}

const en = BOTS_LOCALES.en
const translated = Object.entries(BOTS_LOCALES).filter(([id]) => id !== 'en')

describe('BOTS_LOCALES', () => {
  it('covers the English key tree in every shipped locale', () => {
    // ja / zh / zh-hant shipped before this assertion existed; the count
    // guards against the filter silently matching nothing.
    expect(translated.length).toBeGreaterThanOrEqual(3)

    const enPaths = leafEntries(en).map(([path]) => path)

    for (const [id, locale] of translated) {
      expect(locale, `${id} bundle is missing`).toBeDefined()
      expect(leafEntries(locale).map(([path]) => path), `${id} key tree`).toEqual(enPaths)
    }
  })

  it('translates user-visible chrome instead of echoing English', () => {
    // 'tools.skillsHub' was a sample until a Latin-script locale shipped:
    // 'Hermes Skills Hub' is a product name and stays identical in German, so
    // asserting it differs from English tests the brand, not the translation.
    const samples = ['roster.emptyTitle', 'bot.newTitle', 'group.manageTitle'] as const
    const enByPath = Object.fromEntries(leafEntries(en))

    for (const [id, locale] of translated) {
      const byPath = Object.fromEntries(leafEntries(locale))

      for (const path of samples) {
        expect(byPath[path], `${id}.${path}`).not.toBe(enByPath[path])
      }
    }
  })

  it('keeps interpolator arguments in the translated string', () => {
    const sentinel = 'QUERY_SENTINEL'
    const gateway = 'GATEWAY_SENTINEL'

    for (const [, locale] of Object.entries(BOTS_LOCALES)) {
      const byPath = Object.fromEntries(leafEntries(locale))
      const queryFn = byPath['roster.noMatchQuery'] as (query: string) => string
      const bothFn = byPath['roster.noMatchQueryOn'] as (query: string, gateway: string) => string
      const reasonFn = byPath['roster.rosterUnavailable'] as (reason: string) => string

      expect(queryFn(sentinel)).toContain(sentinel)
      expect(bothFn(sentinel, gateway)).toContain(sentinel)
      expect(bothFn(sentinel, gateway)).toContain(gateway)
      expect(reasonFn(sentinel)).toContain(sentinel)
    }
  })
})

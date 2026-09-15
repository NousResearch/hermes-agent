/**
 * The English bundle is the message shape. ko / ja / zh / zh-hant must cover the
 * same leaves so a locale switch never falls through to a raw key — and the
 * interpolators must still splice their arguments, not drop them.
 */

import type { PluginContext } from '@hermes/plugin-sdk'
import { describe, expect, it } from 'vitest'

// The harness supplies the host's registry and runtime locale.
// eslint-disable-next-line no-restricted-imports
import { createPluginI18n } from '@/i18n'
// eslint-disable-next-line no-restricted-imports
import { setRuntimeI18nLocale } from '@/i18n/runtime'

import { BOTS_LOCALES, botsText } from './i18n'
import { setPluginCtx } from './shared'

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
const ko = BOTS_LOCALES.ko
const ja = BOTS_LOCALES.ja
const zh = BOTS_LOCALES.zh
const zhHant = BOTS_LOCALES['zh-hant']

describe('BOTS_LOCALES', () => {
  it('refreshes both string and function messages when a stable context translator changes locale', () => {
    const i18n = createPluginI18n('hermes-bots', dispose => dispose)
    const dispose = i18n.register(BOTS_LOCALES)
    setPluginCtx({ i18n } as PluginContext)

    try {
      setRuntimeI18nLocale('en')
      expect(botsText().cron.unitMinutes).toBe('minute(s)')
      expect(botsText().cron.everyNMinutes(7)).toBe('Every 7m')

      setRuntimeI18nLocale('ko')
      expect(botsText().cron.unitMinutes).toBe('분')
      expect(botsText().cron.everyNMinutes(7)).toBe('7분마다')

      setRuntimeI18nLocale('en')
      expect(botsText().cron.unitMinutes).toBe('minute(s)')
    } finally {
      dispose()
      setPluginCtx(null)
      setRuntimeI18nLocale('en')
    }
  })

  it('covers the English key tree in every shipped locale', () => {
    expect(ko).toBeDefined()
    expect(ja).toBeDefined()
    expect(zh).toBeDefined()
    expect(zhHant).toBeDefined()

    const enPaths = leafEntries(en).map(([path]) => path)

    expect(leafEntries(ko).map(([path]) => path)).toEqual(enPaths)
    expect(leafEntries(ja).map(([path]) => path)).toEqual(enPaths)
    expect(leafEntries(zh).map(([path]) => path)).toEqual(enPaths)
    expect(leafEntries(zhHant).map(([path]) => path)).toEqual(enPaths)
  })

  it('translates user-visible chrome instead of echoing English', () => {
    const samples = ['roster.emptyTitle', 'bot.newTitle', 'group.manageTitle', 'tools.skillsHub'] as const
    const enByPath = Object.fromEntries(leafEntries(en))

    for (const locale of [ko, ja, zh, zhHant]) {
      const byPath = Object.fromEntries(leafEntries(locale))

      for (const path of samples) {
        expect(byPath[path]).not.toBe(enByPath[path])
      }
    }
  })

  it('keeps interpolator arguments in the translated string', () => {
    const sentinel = 'QUERY_SENTINEL'
    const gateway = 'GATEWAY_SENTINEL'

    for (const locale of [en, ko, ja, zh, zhHant]) {
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

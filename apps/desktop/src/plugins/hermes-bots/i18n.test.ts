/**
 * The English bundle is the message shape. ja / zh / zh-hant must cover the
 * same leaves so a locale switch never falls through to a raw key — and the
 * interpolators must still splice their arguments, not drop them.
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
const ja = BOTS_LOCALES.ja
const zh = BOTS_LOCALES.zh
const zhHant = BOTS_LOCALES['zh-hant']

describe('BOTS_LOCALES', () => {
  it('covers the English key tree in every shipped locale', () => {
    expect(ja).toBeDefined()
    expect(zh).toBeDefined()
    expect(zhHant).toBeDefined()

    const enPaths = leafEntries(en).map(([path]) => path)

    expect(leafEntries(ja).map(([path]) => path)).toEqual(enPaths)
    expect(leafEntries(zh).map(([path]) => path)).toEqual(enPaths)
    expect(leafEntries(zhHant).map(([path]) => path)).toEqual(enPaths)
  })

  it('translates user-visible chrome instead of echoing English', () => {
    const samples = ['roster.emptyTitle', 'bot.newTitle', 'group.manageTitle', 'tools.skillsHub'] as const
    const enByPath = Object.fromEntries(leafEntries(en))

    for (const locale of [ja, zh, zhHant]) {
      const byPath = Object.fromEntries(leafEntries(locale))

      for (const path of samples) {
        expect(byPath[path]).not.toBe(enByPath[path])
      }
    }
  })

  it('keeps interpolator arguments in the translated string', () => {
    const sentinel = 'QUERY_SENTINEL'
    const gateway = 'GATEWAY_SENTINEL'

    for (const locale of [en, ja, zh, zhHant]) {
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

  it('keeps automatic continuity copy concise and free of gateway jargon', () => {
    const byPath = Object.fromEntries(leafEntries(en!))
    const copy = (path: string) => byPath[`group.${path}`] as (value: string) => string
    const text = (path: string) => byPath[`group.${path}`] as string

    const samples = [
      copy('hostedFallbackToDesktop')('Studio'),
      copy('hostedQueued')('Studio'),
      copy('hostedQueuedHint')('Studio'),
      copy('hostedSendFailed')('Studio'),
      copy('hostedRenameQueued')('Studio'),
      copy('hostedRenameFailed')('Studio'),
      copy('hostUpdateNeeded')('Studio'),
      copy('hostReconnectToContinue')('Studio'),
      copy('hostedReconnectToStop')('Studio'),
      copy('hostedReconnectToDelete')('Studio'),
      text('hostedSending'),
      text('hostedWorking'),
      text('hostedNeedsAttention'),
      text('hostedStopping'),
      text('hostedStopped'),
      text('hostedDeleted'),
      text('hostedDeleteLocally'),
      text('hostedMembersFixed'),
      text('hostRouteMissing'),
      text('hostedSyncing'),
      text('botsNeedOneHost'),
      text('desktopStorageUnavailable'),
      text('hostRejectedCommand')
    ]

    expect(byPath).not.toHaveProperty('group.keepRunningTitle')

    for (const sample of samples) {
      expect(sample).not.toMatch(/gateway/i)
      expect(sample.length).toBeLessThanOrEqual(110)
    }
  })
})

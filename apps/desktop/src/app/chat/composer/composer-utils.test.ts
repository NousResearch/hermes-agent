import type { Unstable_TriggerItem } from '@assistant-ui/core'
import { describe, expect, it } from 'vitest'

import {
  acceptsTriggerCompletion,
  implicitSlashAcceptIndex,
  isPendingDraftPersistCurrent,
  type PendingDraftPersist,
  pickPlaceholder,
  slashArgStage,
  slashChipKindForItem,
  slashCommandToken,
  type TriggerAcceptInput
} from './composer-utils'

const item = (group: string): Unstable_TriggerItem =>
  ({ id: 'x', type: 'slash', label: 'x', metadata: { group } }) as unknown as Unstable_TriggerItem

describe('slashArgStage', () => {
  it('is true only once the query is past the command name', () => {
    expect(slashArgStage('personality')).toBe(false)
    expect(slashArgStage('personality alice')).toBe(true)
  })
})

describe('slashCommandToken', () => {
  it('extracts the lowercased /command token', () => {
    expect(slashCommandToken('Personality alice')).toBe('/personality')
    expect(slashCommandToken('model')).toBe('/model')
  })

  it('handles an empty query', () => {
    expect(slashCommandToken('')).toBe('/')
  })
})

describe('slashChipKindForItem', () => {
  it('maps completion groups to chip kinds', () => {
    expect(slashChipKindForItem(item('Skills'))).toBe('skill')
    expect(slashChipKindForItem(item('Themes'))).toBe('theme')
    expect(slashChipKindForItem(item('Commands'))).toBe('command')
  })
})

describe('acceptsTriggerCompletion', () => {
  const press = (key: string, overrides: Partial<TriggerAcceptInput> = {}) =>
    acceptsTriggerCompletion({
      activeExplicit: false,
      freeTextArgStage: false,
      key,
      kind: '/',
      query: 'personality alic',
      ...overrides
    })

  it('accepts on Enter / Tab / Space for a finite option list', () => {
    expect(press('Enter')).toBe(true)
    expect(press('Tab')).toBe(true)
    expect(press(' ')).toBe(true)
  })

  it('ignores keys that are neither navigation nor acceptance', () => {
    expect(press('a')).toBe(false)
    expect(press('Escape')).toBe(false)
  })

  it('lets an `@` mention take a literal space', () => {
    expect(press(' ', { kind: '@', query: 'src/comp' })).toBe(false)
    expect(press('Enter', { kind: '@', query: 'src/comp' })).toBe(true)
  })

  it('types a space on a bare `/ ` instead of accepting', () => {
    expect(press(' ', { query: '' })).toBe(false)
  })

  // The `/goal <prose>` class: the popover may be live over free-form text, so
  // the keys that mean something else in prose must keep meaning it.
  it('sends the prose rather than the unchosen first row', () => {
    expect(press('Enter', { freeTextArgStage: true, query: 'goal ship the redesign' })).toBe(false)
    expect(press(' ', { freeTextArgStage: true, query: 'goal ship the' })).toBe(false)
  })

  it('accepts on Enter once the user has arrowed to a row deliberately', () => {
    expect(press('Enter', { activeExplicit: true, freeTextArgStage: true, query: 'goal stat' })).toBe(true)
  })

  it('keeps Tab as the explicit accept even over free text', () => {
    expect(press('Tab', { freeTextArgStage: true, query: 'goal stat' })).toBe(true)
  })
})

describe('implicitSlashAcceptIndex', () => {
  const rows = ['/compress', '/review', '/resume']

  it('completes a prefix of the highlighted row', () => {
    expect(implicitSlashAcceptIndex('com', rows, 0, false)).toBe(0)
  })

  it('keeps a fully typed command even when another row is highlighted', () => {
    expect(implicitSlashAcceptIndex('review', rows, 0, false)).toBe(1)
  })

  it('does not steal when the typed token is not a prefix of any row', () => {
    expect(implicitSlashAcceptIndex('review', ['/compress', '/resume'], 0, false)).toBeNull()
  })

  it('takes the only prefix match when the highlight is a leftover', () => {
    expect(implicitSlashAcceptIndex('rev', ['/compress', '/review', '/resume'], 0, false)).toBe(1)
  })

  it('honours an arrowed pick even when it is not a prefix', () => {
    expect(implicitSlashAcceptIndex('review', rows, 0, true)).toBe(0)
  })

  it('matches an arg-stage prefix against the full completion text', () => {
    expect(implicitSlashAcceptIndex('personality alic', ['/personality alice', '/personality none'], 0, false)).toBe(0)
  })
})

describe('pickPlaceholder', () => {
  it('returns a member of the pool', () => {
    const pool = ['a', 'b', 'c'] as const
    expect(pool).toContain(pickPlaceholder(pool))
  })
})

describe('isPendingDraftPersistCurrent (#54527 integrity guard)', () => {
  it('accepts a write when the pending entry still matches what was captured', () => {
    const entry: PendingDraftPersist = { scope: 'session-a', text: 'hello' }

    expect(isPendingDraftPersistCurrent(entry, entry)).toBe(true)
    expect(isPendingDraftPersistCurrent({ scope: 'session-a', text: 'hello' }, entry)).toBe(true)
  })

  it('rejects when the pending slot was cleared (session swap / newer flush already committed)', () => {
    const entry: PendingDraftPersist = { scope: 'session-a', text: 'hello' }

    expect(isPendingDraftPersistCurrent(null, entry)).toBe(false)
  })

  it('rejects when the pending slot now belongs to a different session (the #54527 misroute shape)', () => {
    const captured: PendingDraftPersist = { scope: 'session-a', text: 'carefully composed prompt' }
    const supersededBy: PendingDraftPersist = { scope: 'session-b', text: 'different draft' }

    expect(isPendingDraftPersistCurrent(supersededBy, captured)).toBe(false)
  })

  it('rejects when the pending slot was replaced by a newer keystroke in the same session', () => {
    const captured: PendingDraftPersist = { scope: 'session-a', text: 'first draft' }
    const supersededBy: PendingDraftPersist = { scope: 'session-a', text: 'first draft continued' }

    expect(isPendingDraftPersistCurrent(supersededBy, captured)).toBe(false)
  })

  it('rejects when nothing was ever captured', () => {
    expect(isPendingDraftPersistCurrent(null, null)).toBe(false)
  })
})

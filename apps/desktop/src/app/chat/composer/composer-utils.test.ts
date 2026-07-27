import type { Unstable_TriggerItem } from '@assistant-ui/core'
import { describe, expect, it } from 'vitest'

import {
  isPendingDraftPersistCurrent,
  type PendingDraftPersist,
  pickPlaceholder,
  slashArgStage,
  slashChipKindForItem,
  slashCommandToken
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

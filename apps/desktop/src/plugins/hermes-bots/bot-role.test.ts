import { describe, expect, it } from 'vitest'

import { BOT_ROLE_MAX, botRole, normalizeBotRole } from './bot-role'
import { editedLook } from './edit-profile-look'
import type { RosterRow } from './types'

describe('a bot role is a short single-line label', () => {
  it.each([
    ['  Deep Research  ', 'Deep Research'],
    ['Deep\n\tResearch', 'Deep Research'],
    ['', ''],
    [undefined, ''],
    [['a', 'b'], ''],
    [42, '']
  ])('normalizes %j to %j', (raw, expected) => {
    expect(normalizeBotRole(raw)).toBe(expected)
  })

  it('caps by character, never splitting an emoji, and never ends on a space', () => {
    expect(normalizeBotRole('x'.repeat(BOT_ROLE_MAX + 25))).toHaveLength(BOT_ROLE_MAX)
    expect(Array.from(normalizeBotRole('🛰️'.repeat(BOT_ROLE_MAX))).length).toBeLessThanOrEqual(BOT_ROLE_MAX)
    expect(normalizeBotRole(`${'x'.repeat(BOT_ROLE_MAX - 1)} tail`)).toBe('x'.repeat(BOT_ROLE_MAX - 1))
  })
})

describe('the role a row shows', () => {
  const bot = { name: 'scout' } as RosterRow

  it('prefers this desktop’s meta, then what the row’s own backend reports', () => {
    const served = { ...bot, ui_meta: { 'hermes-bots': { role: 'Scouting' } } } as RosterRow

    expect(botRole(served, { role: 'Deep Research' })).toBe('Deep Research')
    expect(botRole(served, {})).toBe('Scouting')
    expect(botRole(bot, null)).toBe('')
    expect(botRole(null, undefined)).toBe('')
  })

  it('a role cleared on this desktop stays cleared, not revived from the backend’s older copy', () => {
    const served = { ...bot, ui_meta: { 'hermes-bots': { role: 'Old role' } } } as RosterRow

    expect(botRole(served, { role: '' })).toBe('')
    expect(botRole(served, { role: '   ' })).toBe('')
  })

  it('never falls back to the description or the title: those are not roles', () => {
    expect(botRole({ ...bot, description: 'Finds sources', title: 'Scout' }, { title: 'Scout' })).toBe('')
  })
})

describe('Edit Profile writes a role only when the user changed it', () => {
  const opened = { color: '#ff0000', image: null, role: 'Deep Research', shape: 'circle' as const, title: 'Scout' }

  it('leaves an untouched role out of the patch, so a role set elsewhere is not reverted', () => {
    expect(editedLook(opened, { ...opened, color: '#0000ff' })).toEqual({ color: '#0000ff', custom: true })
    expect(editedLook(opened, { ...opened, role: '  Deep   Research ' })).toBeNull()
  })

  it('sends a normalized role, and "" when it is cleared (an omitted key would keep the old one)', () => {
    expect(editedLook(opened, { ...opened, role: ' Scouting ' })).toEqual({ role: 'Scouting' })
    expect(editedLook(opened, { ...opened, role: '' })).toEqual({ role: '' })
  })

  it('does not mark the avatar as customised when only the role changed', () => {
    expect(editedLook(opened, { ...opened, role: 'Scouting' })).not.toHaveProperty('custom')
  })
})

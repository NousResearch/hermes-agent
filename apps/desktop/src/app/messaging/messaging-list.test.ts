import { describe, expect, it } from 'vitest'

import {
  filterAndSortMessagingPlatforms,
  messagingStatusFilter,
  parseMessagingQuery,
  setMessagingQueryToken
} from './messaging-list'

const rows = [
  { id: 'telegram', name: 'Telegram', description: 'Chat', enabled: true, state: 'connected' },
  { id: 'discord', name: 'Discord', description: 'Community', enabled: true, state: 'pending_restart' },
  { id: 'slack', name: 'Slack', description: 'Work', enabled: false, state: 'disabled' },
  { id: 'matrix', name: 'Matrix', description: 'Bridge', enabled: true, state: 'fatal' }
]

const label = (state: null | string | undefined) => state ?? ''

describe('messaging list query', () => {
  it('parses status and sort operators while preserving free text', () => {
    expect(parseMessagingQuery('disc status:warning sort:status')).toEqual({
      sort: 'status',
      status: 'attention',
      text: 'disc'
    })
  })

  it('keeps unknown operators as ordinary search text', () => {
    expect(parseMessagingQuery('kind:bot telegram')).toEqual({
      sort: 'default',
      status: 'all',
      text: 'kind:bot telegram'
    })
  })

  it.each(['constructor', '__proto__'])('keeps inherited alias names as literal text: %s', value => {
    expect(parseMessagingQuery(`status:${value}`)).toEqual({ sort: 'default', status: 'all', text: `status:${value}` })
    expect(parseMessagingQuery(`sort:${value}`)).toEqual({ sort: 'default', status: 'all', text: `sort:${value}` })
    expect(filterAndSortMessagingPlatforms(rows, `status:${value}`, label)).toEqual([])
    expect(filterAndSortMessagingPlatforms(rows, `sort:${value}`, label)).toEqual([])
  })

  it('updates one query operator without destroying free text', () => {
    expect(setMessagingQueryToken('discord status:error', 'status', 'connected')).toBe('discord status:connected')
    expect(setMessagingQueryToken('discord status:error sort:status', 'sort', 'default')).toBe('discord status:error')
  })

  it('preserves unknown sort and status tokens while changing or resetting a control', () => {
    expect(setMessagingQueryToken('telegram sort:recent', 'sort', 'name-asc')).toBe('telegram sort:recent sort:name-asc')
    expect(setMessagingQueryToken('telegram sort:recent sort:name-asc', 'sort', 'default')).toBe('telegram sort:recent')
    expect(setMessagingQueryToken('telegram status:offline', 'status', 'connected')).toBe('telegram status:offline status:connected')
    expect(setMessagingQueryToken('telegram status:offline status:connected', 'status', 'all')).toBe('telegram status:offline')

    expect(filterAndSortMessagingPlatforms(rows, 'telegram sort:recent', label)).toEqual([])
    expect(filterAndSortMessagingPlatforms(rows, 'telegram sort:recent sort:name-asc', label)).toEqual([])
  })

  it('classifies platform state into stable user-facing groups', () => {
    expect(messagingStatusFilter(rows[0])).toBe('connected')
    expect(messagingStatusFilter(rows[1])).toBe('attention')
    expect(messagingStatusFilter(rows[2])).toBe('inactive')
    expect(messagingStatusFilter(rows[3])).toBe('error')
  })

  it('filters by status and sorts status groups with names as a tiebreaker', () => {
    expect(filterAndSortMessagingPlatforms(rows, 'status:attention', label).map(row => row.id)).toEqual(['discord'])
    expect(filterAndSortMessagingPlatforms(rows, 'sort:status', label).map(row => row.id)).toEqual([
      'matrix',
      'discord',
      'telegram',
      'slack'
    ])
  })

  it('searches localized state labels as well as platform fields', () => {
    const localized = (state: null | string | undefined) => (state === 'connected' ? 'verbunden' : (state ?? ''))

    expect(filterAndSortMessagingPlatforms(rows, 'verbunden', localized).map(row => row.id)).toEqual(['telegram'])
  })

  it('sorts by name in both directions', () => {
    expect(filterAndSortMessagingPlatforms(rows, 'sort:name', label).map(row => row.name)).toEqual([
      'Discord',
      'Matrix',
      'Slack',
      'Telegram'
    ])
    expect(filterAndSortMessagingPlatforms(rows, 'sort:name-desc', label).map(row => row.name)).toEqual([
      'Telegram',
      'Slack',
      'Matrix',
      'Discord'
    ])
  })
})

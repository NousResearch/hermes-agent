/**
 * ⌘K "Open Screen" resolves its bot at CLICK time, so the row it opens with is the
 * one the Bots pane's snapshot published — the row that carries the connection
 * binding the Screen pane's ws URL is built from. A row captured at registration
 * (from `profiles.list`, on a launch that never opened the pane) has no connection
 * id, and the pane then connects to a fallback endpoint, receives nothing and paints
 * black.
 */
import { expect, it } from 'vitest'

import type { RosterRow } from './types'
import { pickScreenBot } from './screen-target'

const rows = [{ name: 'default' }, { name: 'mo' }, { name: 'jian' }] as unknown as RosterRow[]

it('finds the bot the command names', () => {
  expect(pickScreenBot(rows, 'mo')).toBe(rows[1])
})

it('trims the name it is given', () => {
  expect(pickScreenBot(rows, '  jian ')).toBe(rows[2])
})

it('returns nothing for a bot the current roster does not list', () => {
  expect(pickScreenBot(rows, 'ying')).toBeNull()
})

it('returns nothing when the roster has not landed yet', () => {
  expect(pickScreenBot(null, 'default')).toBeNull()
  expect(pickScreenBot(undefined, 'default')).toBeNull()
  expect(pickScreenBot([], 'default')).toBeNull()
})

it('returns nothing for an empty name', () => {
  expect(pickScreenBot(rows, '')).toBeNull()
  expect(pickScreenBot(rows, '   ')).toBeNull()
})

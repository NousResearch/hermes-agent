import { expect, it } from 'vitest'

import { matchOldName } from './old-name-aliases'

const destination = (query: string, local?: Parameters<typeof matchOldName>[1]) =>
  matchOldName(query, local)?.alias.destination ?? null

it('finds each renamed place by its classic word', () => {
  expect(destination('sessions')).toBe('chats')
  expect(destination('profiles')).toBe('agents')
  expect(destination('artifacts')).toBe('library')
  expect(destination('cron')).toBe('routines')
  expect(destination('cron jobs')).toBe('routines')
  expect(destination('gateways')).toBe('machines')
})

it('ignores case, outer space and doubled spaces, and says whether the word was typed whole', () => {
  expect(matchOldName('  Cron   Jobs ')).toMatchObject({ alias: { code: 'cron' }, exact: true })
  expect(matchOldName('sess')).toMatchObject({ alias: { code: 'sessions' }, exact: false })
})

it('stays quiet for fragments too short to mean one word, and for unrelated queries', () => {
  expect(destination('se')).toBeNull()
  expect(destination('chats')).toBeNull()
  expect(destination('sessions about pricing')).toBeNull()
})

it("accepts a locale's own classic word", () => {
  expect(destination('profils', { profiles: 'Profils' })).toBe('agents')
  expect(matchOldName('profils', { profiles: 'Profils' })?.exact).toBe(true)
})

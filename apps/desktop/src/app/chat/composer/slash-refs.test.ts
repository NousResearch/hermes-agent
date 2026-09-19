import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { rememberDesktopCommandsCatalog } from '@/lib/desktop-slash-commands'

import { slashCommandMatches } from './slash-refs'

const commands = (text: string, options?: Parameters<typeof slashCommandMatches>[1]) =>
  slashCommandMatches(text, options).map(match => `${match.kind}:${match.command}`)

describe('slashCommandMatches', () => {
  beforeEach(() => {
    rememberDesktopCommandsCatalog({
      commands: { '/goal': { argument_mode: 'mixed', desktop: null } }
    })
  })

  afterEach(() => {
    rememberDesktopCommandsCatalog(undefined)
  })

  it('recognizes a leading command and a skill named mid-prose', () => {
    expect(commands('/some-skill clean this with /other-skill please')).toEqual([
      'skill:/some-skill',
      'skill:/other-skill'
    ])
  })

  it('leaves a path alone — /usr/local/bin is not a command', () => {
    expect(commands('see /usr/local/bin ')).toEqual([])
  })

  it('holds a trailing token as still-typed unless the text is inert', () => {
    expect(commands('/some-skill')).toEqual([])
    expect(commands('/some-skill', { trailingCommitted: true })).toEqual(['skill:/some-skill'])
  })

  it('leaves an arg-taking command as text — its tail may be prose', () => {
    expect(commands('/goal ship the redesign')).toEqual([])
  })

  it('leaves a command with no desktop surface as text', () => {
    expect(commands('/exit now')).toEqual([])
  })

  it('offers a built-in only as an invocation, never mid-message', () => {
    // Mirrors what the popover offers: `/new` acts on the app, so it means
    // nothing dropped into a sentence, while a skill reads as "handle this
    // part with X".
    expect(commands('/new ')).toEqual(['command:/new'])
    expect(commands('start over with /new ')).toEqual([])
    expect(commands('start over with /some-skill ')).toEqual(['skill:/some-skill'])
  })

  it('disqualifies a leading token when the text lands mid-word', () => {
    expect(commands('/some-skill ', { boundaryBefore: false })).toEqual([])
  })
})

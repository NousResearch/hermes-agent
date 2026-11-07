import { describe, expect, it } from 'vitest'

import { commandLineIsHermesRuntime } from './hermes-process'

// HX003: process identity comes from a whole argv TOKEN, never an argv substring.
describe('commandLineIsHermesRuntime (HX003 token matching)', () => {
  it('matches a real Hermes gateway command line', () => {
    expect(commandLineIsHermesRuntime('hermes\0gateway\0run')).toBe(true)
    expect(commandLineIsHermesRuntime('hermes gateway run')).toBe(true)
    expect(commandLineIsHermesRuntime('hermes --profile work gateway run')).toBe(true)
  })

  it('matches a Hermes launched by an interpreter (basename of a path token)', () => {
    expect(commandLineIsHermesRuntime('/venv/bin/python\0/venv/bin/hermes\0gateway\0run')).toBe(true)
    expect(commandLineIsHermesRuntime('C:\\Hermes\\hermes.exe gateway run')).toBe(true)
  })

  // The bug this replaces: a bare `.includes('hermes')` matched any of these.
  it('does NOT match an unrelated process whose path merely contains "hermes"', () => {
    expect(commandLineIsHermesRuntime('vim /home/user/hermes-notes.txt')).toBe(false)
    expect(commandLineIsHermesRuntime('/opt/hermesbrew/bin/tool --run')).toBe(false)
    expect(commandLineIsHermesRuntime('node /srv/hermes-dashboard/server.js')).toBe(false)
    expect(commandLineIsHermesRuntime('cat ~/.hermes/config.yaml')).toBe(false)
  })
})

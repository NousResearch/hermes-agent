import { expect, test } from 'vitest'

import { bootstrapSnapshot } from './bootstrap-state'
import { type InstallStamp } from './install-stamp'

function stamp(payload: InstallStamp['payload']): InstallStamp {
  return {
    schemaVersion: 2, commit: null, commitDate: null, branch: null,
    builtAt: null, dirty: false, source: 'build', distribution: 'desktop-app',
    updateMechanism: 'self', baseVersion: null, displayVersion: null,
    distance: null, payload, tag: null
  }
}

test.each(['bundled', 'bootstrap', 'light'] as const)('bootstrap state retains the %s artifact fact across failure and reset', payload => {
  const artifact = stamp(payload)
  const state = { error: 'backend exited before setup', setupChoice: null }
  expect(bootstrapSnapshot(state, artifact)).toEqual({ ...state, bundled: payload === 'bundled' })
  expect(bootstrapSnapshot({ error: null, setupChoice: null }, artifact).bundled).toBe(payload === 'bundled')
  expect(state).not.toHaveProperty('bundled')
  expect(bootstrapSnapshot({ ...state, bundled: payload !== 'bundled' }, artifact).bundled).toBe(payload === 'bundled')
})

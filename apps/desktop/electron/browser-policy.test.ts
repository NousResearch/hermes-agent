import assert from 'node:assert/strict'
import test from 'node:test'
import { validateBrowserPrepare } from './browser-policy'

test('explicit driver-owned profile preparation is permitted', () => {
  validateBrowserPrepare({ allow_launch: true, profile: { mode: 'isolated_new' } })
  validateBrowserPrepare({ allow_launch: true, profile: { mode: 'isolated_named', name: 'athena-test' } })
})
test('private profile requests never accept existing-window parameters', () => {
  for (const args of [
    { pid: 10, strategy: { kind: 'existing_profile' } },
    { allow_launch: true, profile: { mode: 'isolated_new' }, pid: 10 },
    { allow_launch: true, profile: { mode: 'isolated_new' }, window_id: 20 },
    { allow_launch: true, profile: { mode: 'existing_profile' } },
    { profile: { mode: 'isolated_new' } }
  ]) assert.throws(() => validateBrowserPrepare(args))
})
test('account modes are explicit and existing windows require exact identity', () => {
  validateBrowserPrepare({ browser: 'edge', allow_launch: true, profile: { mode: 'athena_profile' } })
  validateBrowserPrepare({ browser: 'edge', allow_launch: false, profile: { mode: 'existing_profile' }, pid: 10, window_id: 20 })
  for (const args of [
    { browser: 'firefox', allow_launch: false, profile: { mode: 'existing_profile' }, pid: 10, window_id: 20 },
    { browser: 'edge', allow_launch: false, profile: { mode: 'existing_profile' }, pid: 10 },
    { browser: 'edge', allow_launch: true, profile: { mode: 'athena_profile' }, pid: 10 },
    { browser: 'edge', allow_launch: true, profile: { mode: 'existing_profile' }, pid: 10, window_id: 20 }
  ]) assert.throws(() => validateBrowserPrepare(args))
})

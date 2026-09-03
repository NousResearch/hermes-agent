import assert from 'node:assert/strict'
import path from 'node:path'
import test from 'node:test'
import { resolveConfig } from 'vite'

import { sessionOpenPerfFixtureEntry } from './vite-conditionals.mjs'

const root = path.resolve(import.meta.dirname, '../../..')

test('normal production aliases the session-open perf fixture to its typed no-op', () => {
  assert.match(sessionOpenPerfFixtureEntry(root, 'build', {}), /session-open-perf-fixture\.noop\.ts$/)
})

test('dev and opt-in perf production builds keep the real session-open perf fixture', () => {
  assert.match(sessionOpenPerfFixtureEntry(root, 'serve', {}), /session-open-perf-fixture\.ts$/)
  assert.doesNotMatch(sessionOpenPerfFixtureEntry(root, 'serve', {}), /\.noop\.ts$/)
  assert.match(
    sessionOpenPerfFixtureEntry(root, 'build', { VITE_PERF_PROBE: '1' }),
    /session-open-perf-fixture\.ts$/
  )
  assert.doesNotMatch(
    sessionOpenPerfFixtureEntry(root, 'build', { VITE_PERF_PROBE: '1' }),
    /\.noop\.ts$/
  )
})

test('Vite resolves the internal fixture alias to the production no-op', async () => {
  const config = await resolveConfig(
    { configFile: path.join(root, 'vite.config.ts'), mode: 'production' },
    'build'
  )
  const fixtureAlias = config.resolve.alias.find(
    ({ find }) => find === '@/app/session/hooks/use-session-actions/session-open-perf-fixture'
  )

  assert.ok(fixtureAlias)
  assert.equal(fixtureAlias.replacement, sessionOpenPerfFixtureEntry(root, 'build', {}))
})

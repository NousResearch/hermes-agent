import assert from 'node:assert/strict'

import { test } from 'vitest'

import { sessionProfileSpawnSpec } from './session-profile-spawn'

test('pins the default session explicitly at the root home', () => {
  const spec = sessionProfileSpawnSpec({ selectedProfile: 'default', hermesHome: '/tmp/hermes' })

  assert.deepEqual(spec.argvProfileFlag, ['--profile', 'default'])
  assert.equal(spec.envOverlay.HERMES_HOME, '/tmp/hermes')
})

test('pins an empty session selection to the default profile', () => {
  const spec = sessionProfileSpawnSpec({ selectedProfile: '  ', hermesHome: '/tmp/hermes' })

  assert.deepEqual(spec.argvProfileFlag, ['--profile', 'default'])
  assert.equal(spec.envOverlay.HERMES_HOME, '/tmp/hermes')
})


test('pins a named session without changing its resolved home', () => {
  const spec = sessionProfileSpawnSpec({
    selectedProfile: 'iridian-agent',
    hermesHome: '/tmp/hermes/profiles/iridian-agent'
  })

  assert.deepEqual(spec.argvProfileFlag, ['--profile', 'iridian-agent'])
  assert.equal(spec.envOverlay.HERMES_HOME, '/tmp/hermes/profiles/iridian-agent')
})

test('parent HERMES_HOME cannot override the resolved child overlay', () => {
  const foreignHome = '/foreign/parent-home'
  const spec = sessionProfileSpawnSpec({ selectedProfile: 'default', hermesHome: '/resolved/root' })

  assert.equal(spec.envOverlay.HERMES_HOME, '/resolved/root')
  assert.notEqual(spec.envOverlay.HERMES_HOME, foreignHome)
})

test('does not rewrite remote or URL backend control paths', () => {
  assert.equal(sessionProfileSpawnSpec({ selectedProfile: 'default', hermesHome: null, backendKind: 'remote' }), null)
  assert.equal(sessionProfileSpawnSpec({ selectedProfile: 'named', hermesHome: null, backendKind: 'url' }), null)
})

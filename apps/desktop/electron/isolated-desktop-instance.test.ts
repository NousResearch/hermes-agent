import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  assertIsolatedManifestMatches,
  DEFAULT_AUMID,
  isolatedDesktopLaunchArguments,
  isolatedDesktopLaunchEnv,
  isolatedInstanceSpecFromSsh,
  parseInstanceDeepLink,
  resolveAppUserModelId,
  shouldRegisterGlobalShortcuts,
  shouldRegisterProtocolClient,
  slugFromLabel
} from './isolated-desktop-instance'

test('slugFromLabel strips a Hermes prefix', () => {
  assert.equal(slugFromLabel('Hermes Athena'), 'athena')
  assert.equal(slugFromLabel('Work laptop'), 'work-laptop')
})

test('isolatedInstanceSpecFromSsh maps the full SSH dial contract', () => {
  const spec = isolatedInstanceSpecFromSsh({
    connectionId: 'c1',
    host: 'bear-agent',
    kind: 'ssh',
    keyPath: '/home/bear/.ssh/id_ed25519',
    label: 'Hermes Athena',
    port: 2222,
    remoteHermesPath: '/opt/hermes/bin/hermes',
    remoteProfile: 'default',
    user: 'bear'
  })

  assert.equal(spec.name, 'athena')
  assert.equal(spec.connectionId, 'c1')
  assert.equal(spec.sshHost, 'bear-agent')
  assert.equal(spec.sshUser, 'bear')
  assert.equal(spec.sshPort, 2222)
  assert.equal(spec.sshKeyPath, '/home/bear/.ssh/id_ed25519')
  assert.equal(spec.remoteHermesPath, '/opt/hermes/bin/hermes')
  assert.equal(spec.displayName, 'Hermes Athena')
  assert.equal(spec.aumid, 'com.nousresearch.hermes.instance.athena')
})

test('same-host SSH rows stay distinct when user/port/key/path/profile differ', () => {
  const alice = isolatedInstanceSpecFromSsh({
    connectionId: 'alice-box',
    host: 'lab.example',
    kind: 'ssh',
    keyPath: '/keys/alice',
    label: 'Lab',
    port: 22,
    remoteHermesPath: '/opt/hermes/bin/hermes',
    remoteProfile: 'default',
    user: 'alice'
  })

  const bob = isolatedInstanceSpecFromSsh({
    connectionId: 'bob-box',
    host: 'lab.example',
    kind: 'ssh',
    keyPath: '/keys/bob',
    label: 'Lab',
    port: 2200,
    remoteHermesPath: '/opt/hermes/bin/hermes',
    remoteProfile: 'research',
    user: 'bob'
  })

  assert.notEqual(alice.dialIdentity, bob.dialIdentity)
  assert.notEqual(alice.connectionId, bob.connectionId)
})

test('a stale isolated manifest must fail closed against a retargeted Connection', () => {
  const current = isolatedInstanceSpecFromSsh({
    connectionId: 'alice-box',
    host: 'lab.example',
    kind: 'ssh',
    keyPath: '/keys/alice',
    label: 'Lab',
    port: 2200,
    remoteHermesPath: '/opt/hermes/bin/hermes',
    remoteProfile: 'default',
    user: 'alice'
  })

  const stale = {
    connectionId: 'alice-box',
    dialIdentity: isolatedInstanceSpecFromSsh({
      connectionId: 'alice-box',
      host: 'lab.example',
      kind: 'ssh',
      keyPath: '/keys/alice',
      label: 'Lab',
      port: 22,
      remoteHermesPath: '/opt/hermes/bin/hermes',
      remoteProfile: 'default',
      user: 'alice'
    }).dialIdentity
  }

  assert.throws(() => assertIsolatedManifestMatches(stale, current), /no longer matches/)
})

test('isolatedInstanceSpecFromSsh rejects shared-shell kinds and relative paths', () => {
  assert.throws(() => isolatedInstanceSpecFromSsh({ kind: 'remote', label: 'box', host: 'x' }), /SSH/)
  assert.throws(
    () => isolatedInstanceSpecFromSsh({ host: 'lab', kind: 'ssh', label: 'box', remoteHermesPath: 'rel' }),
    /absolute/
  )
})

test('parseInstanceDeepLink extracts the slug and remainder', () => {
  const parsed = parseInstanceDeepLink('hermes://instance/grace/blueprint/morning')

  assert.deepEqual(parsed, { instanceName: 'grace', remainder: 'hermes://blueprint/morning' })
  assert.equal(parseInstanceDeepLink('hermes://blueprint/morning'), null)
  assert.equal(parseInstanceDeepLink('hermes://instance/desktop/blueprint/x'), null)
})

test('slugFromLabel rejects reserved names before the CLI round-trip', () => {
  assert.throws(() => slugFromLabel('Hermes Desktop'), /reserved/)
  assert.throws(() => slugFromLabel('local'), /reserved/)
})

test('isolated launch arguments put a deep link on argv for a warm second-instance', () => {
  const args = isolatedDesktopLaunchArguments('/u', 'hermes://blueprint/morning')

  assert.deepEqual(args, ['--user-data-dir=/u', 'hermes://blueprint/morning'])
  assert.deepEqual(isolatedDesktopLaunchArguments('/u'), ['--user-data-dir=/u'])
})

test('isolated launch env disables global hotkey and protocol capture', () => {
  const env = isolatedDesktopLaunchEnv(
    isolatedInstanceSpecFromSsh({
      connectionId: 'grace-id',
      host: 'grace',
      kind: 'ssh',
      label: 'Hermes Grace',
      remoteHermesPath: '/opt/hermes',
      remoteProfile: 'default'
    }),
    { cwd: '/tmp', hermesHome: '/h', runtimeRoot: '/r', userData: '/u' }
  )

  assert.equal(env.HERMES_DESKTOP_DISABLE_GLOBAL_SHORTCUTS, '1')
  assert.equal(env.HERMES_DESKTOP_SKIP_PROTOCOL_REGISTER, '1')
  assert.equal(shouldRegisterGlobalShortcuts(env), false)
  assert.equal(shouldRegisterProtocolClient(env), false)
  assert.equal(resolveAppUserModelId(env), 'com.nousresearch.hermes.instance.grace')
  assert.equal(shouldRegisterGlobalShortcuts({}), true)
  assert.equal(resolveAppUserModelId({}), DEFAULT_AUMID)
})

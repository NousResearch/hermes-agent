import assert from 'node:assert/strict'

import { test } from 'vitest'

import { normalizeSshConfig } from './connection-config'
import { createManagedSshLifecycleRuntime } from './managed-ssh-lifecycle-runtime'
import { ManagedConnectionUpdateGate } from './managed-ssh-update'

test('a restored managed SSH profile keeps the captured host and maps its remote profile', () => {
  const gate = new ManagedConnectionUpdateGate()

  const runtime = createManagedSshLifecycleRuntime({
    managedConnectionUpdateGate: gate,
    normalizeSshConfig
  })

  const source = {
    id: 'homelab',
    kind: 'ssh',
    host: 'mini.example',
    user: 'axl',
    port: 22,
    remoteProfile: '',
    keyPath: ''
  }

  assert.deepEqual(runtime.managedSshConfig(source, 'mara'), {
    mode: 'ssh',
    host: 'mini.example',
    user: 'axl',
    remoteProfile: 'mara'
  })
  assert.equal(runtime.managedConnectionUpdateGate, gate)
})

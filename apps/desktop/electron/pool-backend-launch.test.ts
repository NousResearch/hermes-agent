import assert from 'node:assert/strict'
import test from 'node:test'

import { preparePoolBackendLaunch } from './pool-backend-launch'

test('remote profile pool launch bypasses update wait and local runtime resolution', async () => {
  const events: string[] = []
  const remote = {
    authMode: 'token',
    baseUrl: 'https://remote.example',
    mode: 'remote',
    source: 'profile',
    token: 'tok',
    wsUrl: 'wss://remote.example/api/ws?token=tok'
  }

  const launch = await preparePoolBackendLaunch('remote-profile', {
    createToken: () => {
      events.push('token')
      return 'local-token'
    },
    ensureRuntime: async backend => {
      events.push('ensure')
      return backend
    },
    getBackendArgsForRuntime: backend => backend.args,
    resolveHermesBackend: args => {
      events.push('resolve')
      return { args }
    },
    resolveRemoteBackend: async () => {
      events.push('remote')
      return remote
    },
    waitForHermes: async () => {
      events.push('wait-hermes')
    },
    waitForUpdateToFinish: async () => {
      events.push('wait-update')
    }
  })

  assert.deepEqual(launch, { kind: 'remote', remote })
  assert.deepEqual(events, ['remote', 'wait-hermes'])
})

test('local profile pool launch waits for update before resolving runtime', async () => {
  const events: string[] = []

  const launch = await preparePoolBackendLaunch('local-profile', {
    createToken: () => {
      events.push('token')
      return 'local-token'
    },
    ensureRuntime: async backend => {
      events.push('ensure')
      return backend
    },
    getBackendArgsForRuntime: backend => {
      events.push('runtime-args')
      return ['served']
    },
    resolveHermesBackend: args => {
      events.push(`resolve:${args.join(' ')}`)
      return { args }
    },
    resolveRemoteBackend: async () => {
      events.push('remote')
      return null
    },
    waitForHermes: async () => {
      events.push('wait-hermes')
    },
    waitForUpdateToFinish: async () => {
      events.push('wait-update')
    }
  })

  assert.deepEqual(launch, {
    kind: 'local',
    backend: { args: ['served'] },
    token: 'local-token'
  })
  assert.equal(events[0], 'remote')
  assert.equal(events[1], 'wait-update')
  assert.match(events[3], /^resolve:--profile local-profile serve --host 127\.0\.0\.1 --port 0$/)
  assert.deepEqual(events.slice(4), ['ensure', 'runtime-args'])
  assert.ok(!events.includes('wait-hermes'))
})

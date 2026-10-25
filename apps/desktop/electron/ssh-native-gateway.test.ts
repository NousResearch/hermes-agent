import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'

import { afterEach, expect, test } from 'vitest'

import { localGatewayDial, nativeGatewayHttpHeaders, restartBoundGateway } from './local-gateway'
import { closeGatewayTicketBridges } from './local-gateway-python'
import { backendOwnedByApp } from './quit-guard'
import type { SshConnection } from './ssh-connection'
import { connectNativeSshGateway, remoteTicketCommand } from './ssh-native-gateway'

const endpoint = {
  profile_id: '/remote/hermes',
  instance_id: 'owner-1',
  authority_epoch: 3,
  runtime_protocol: 1,
  api_origin: 'http://127.0.0.1:12345',
  capabilities: ['session-authority-v1'],
  supervisor: 'systemd'
}

class Helper extends EventEmitter {
  stdin = new PassThrough()
  stdout = new PassThrough()
  stderr = new PassThrough()
  requests: any[] = []
  killed = false

  constructor() {
    super()
    this.stdin.on('data', data => {
      const request = JSON.parse(data.toString())
      this.requests.push(request)
      this.stdout.write(JSON.stringify({ id: request.id, ticket: `test-ticket-${request.id}` }) + '\n')
    })
  }

  kill() {
    this.killed = true
    this.emit('close')

    return true
  }
}

function fixture(override: Record<string, unknown> = {}) {
  const helper = new Helper()
  const commands: string[] = []
  const forwards: number[][] = []

  const ssh = {
    async exec(command: string) {
      commands.push(command)

      if (command.startsWith('[ -x')) {
        return 'OK'
      }

      if (command.includes('--print-runtime-command')) {
        return JSON.stringify(['/usr/bin/python3', '-I', '-c', 'pass'])
      }

      return JSON.stringify({ state: 'ready', endpoint, ...override })
    },
    async execResult(command: string) {
      return { code: override.state === 'draining' ? 6 : 0, stdout: await this.exec(command), stderr: '' }
    },
    openPipe(command: string) {
      commands.push(command)

      return helper
    },
    async forward(local: number, remote: number) {
      forwards.push([local, remote])
    },
    async cancelForward() {
      /* only the tunnel is owned */
    }
  }

  return {
    helper,
    commands,
    forwards,
    connect: () =>
      connectNativeSshGateway({
        ssh: ssh as unknown as SshConnection,
        profile: 'default',
        remoteHermesPath: '/bin/hermes',
        platform: { os: 'Linux' },
        pickLocalPort: async () => 23456
      })
  }
}

afterEach(() => closeGatewayTicketBridges())

test('native SSH reuses the owner, routes fresh tickets privately, and disposes only its helper', async () => {
  const f = fixture()
  const connection = await f.connect()
  expect(connection).not.toBeNull()

  if (!connection) {
    throw new Error('Expected native SSH descriptor')
  }

  expect(f.forwards).toEqual([[23456, 12345]])
  expect(f.commands.some(command => command.includes(' serve '))).toBe(false)
  expect(f.helper.requests).toHaveLength(0)

  const headers = await nativeGatewayHttpHeaders(connection, connection.baseUrl + '/api/config?profile=work')
  expect(headers['X-Hermes-Gateway-Ticket']).toBe('test-ticket-1')
  expect(f.helper.requests[0]).toMatchObject({
    endpoint: { profile_id: '/remote/hermes/profiles/work', control_home: '/remote/hermes', instance_id: 'owner-1' },
    purpose: 'native-http',
    resolve_profile: true
  })

  await localGatewayDial({
    ensure: async () => connection,
    forget: () => undefined,
    launchHome: '/not-the-remote-home',
    profile: 'default',
    prepare: (base, ticket) => `${base}/${ticket}`
  })
  expect(f.helper.requests[1].purpose).toBe('interactive')
  expect(f.helper.requests[1].endpoint.profile_id).toBe('/remote/hermes')

  await expect(nativeGatewayHttpHeaders(connection, 'https://elsewhere.test/api/config')).rejects.toThrow('origin')
  expect(f.helper.requests).toHaveLength(2)
  // Only data crosses Electron IPC; the private ticket client is keyed by endpoint object identity.
  expect(() =>
    structuredClone({ baseUrl: connection.baseUrl, gatewayEndpoint: connection.gatewayEndpoint })
  ).not.toThrow()
  connection.dispose()
  expect(f.helper.killed).toBe(true)
  await expect(nativeGatewayHttpHeaders(connection, connection.baseUrl + '/api/config')).rejects.toThrow('closed')
  expect(f.commands.some(command => /gateway (stop|restart)|kill /.test(command))).toBe(false)
})

test('non-ready and non-loopback owners cannot open a forward or mint', async () => {
  for (const override of [
    { state: 'inaccessible', reason_code: 'authorization' },
    { endpoint: { ...endpoint, api_origin: 'https://elsewhere.test:443' } },
    { endpoint: { ...endpoint, instance_id: '' } },
    { endpoint: { ...endpoint, capabilities: [] } }
  ]) {
    const f = fixture(override)
    await expect(f.connect()).rejects.toThrow()
    expect(f.forwards).toHaveLength(0)
    expect(f.helper.requests).toHaveLength(0)
  }
})

test('an after-turn restart wait remains attachable despite ensure exit 6', async () => {
  const f = fixture({ state: 'draining', reason_code: 'restart_wait' })
  const connection = await f.connect()
  expect(connection?.gatewayEndpoint.instance_id).toBe('owner-1')
  expect(f.forwards).toEqual([[23456, 12345]])
  connection?.dispose()
})

test('WS tickets keep the resolved remote profile while explicit restart remains available', async () => {
  const f = fixture({ endpoint: { ...endpoint, profile_id: '/remote/hermes/profiles/writer', control_home: '/remote/hermes' } })
  const connection = await f.connect()

  if (!connection) { throw new Error('Expected native connection') }

  try {
    await localGatewayDial({
      ensure: async () => connection, forget: () => undefined, launchHome: '/local', profile: 'default',
      prepare: (_base, ticket) => ticket
    })
    expect(f.helper.requests[0].endpoint.profile_id).toBe('/remote/hermes/profiles/writer')
    await restartBoundGateway(connection.gatewayEndpoint)
    expect(f.commands.some(command => command.includes("--profile 'default' gateway restart --detach"))).toBe(true)
  } finally { connection.dispose() }
})

test('runtime bootstrap is shell quoted and rejects an unsupported launcher shape', () => {
  const command = remoteTicketCommand(
    JSON.stringify(['/path with spaces/python', '-I', '-c', "import os; value='quoted'"])
  )

  expect(command).toContain("'/path with spaces/python'")
  expect(command).toContain('from hermes_cli.gateway_client import _session_ticket')

  for (const raw of ['null', '{}', '["relative-python","-I","-c","pass"]', '["/python","-m","other"]']) {
    expect(() => remoteTicketCommand(raw)).toThrow()
  }
})

test('an attached SSH primary survives quit while any legacy owned backend still warns', () => {
  expect(backendOwnedByApp({ primaryRouteKind: 'ssh', primaryAttached: true, ownedBackendCount: 0 })).toBe(false)
  expect(backendOwnedByApp({ primaryRouteKind: 'ssh', primaryAttached: true, ownedBackendCount: 1 })).toBe(true)
  expect(backendOwnedByApp({ primaryRouteKind: 'ssh', ownedBackendCount: 0 })).toBe(true)
})

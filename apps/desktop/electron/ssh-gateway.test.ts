import { expect, test } from 'vitest'

import { mintLocalGatewayTicket, nativeGatewayHttpHeaders, routedGatewayEndpoint } from './local-gateway'
import { attachSshGateway } from './ssh-gateway'

const endpoint = { profile_id: '/home/remote/.hermes', instance_id: 'daemon', authority_epoch: 1,
  runtime_protocol: 1, api_origin: 'http://127.0.0.1:4321', capabilities: ['session-authority-v1'], supervisor: 'external' }

test('SSH attachment pins native credentials to its tunnel and retires no gateway', async () => {
  const commands: string[] = [], requests: any[] = [], forwards: any[] = []
  const ssh = {
    async exec(command: string, options?: { stdinData?: string }) {
      commands.push(command)
      if (options?.stdinData) {
        const request = JSON.parse(options.stdinData)
        requests.push(request)
        return JSON.stringify({ ...request, ticket: 'a'.repeat(40), runtime_protocol: 1 })
      }
      if (command.includes('[ -x')) {return 'OK'}
      if (command.endsWith('gateway --help')) {return '  ensure  Attach\n  ticket  Mint'}
      if (command.endsWith('gateway ensure --json')) {return JSON.stringify({ state: 'ready', endpoint })}
      return 'Hermes test'
    },
    async forward(...args: any[]) {forwards.push(args)},
    async cancelForward() {throw new Error('not closed yet')}
  }
  const connection = await attachSshGateway({ ssh, profile: 'remote-name', profileAlias: 'default', remoteHermesPath: '/usr/bin/hermes', pickLocalPort: async () => 8765 })
  expect(connection).not.toBeNull()
  expect(forwards).toEqual([[8765, 4321, '127.0.0.1']])
  expect(routedGatewayEndpoint(connection!.gatewayEndpoint, 'default', '/local/wrong')).toBe(connection!.gatewayEndpoint)
  const secondary = routedGatewayEndpoint(connection!.gatewayEndpoint, 'work', '/local/wrong')
  expect(secondary.profile_id).toBe(endpoint.profile_id)
  expect(secondary.ssh_profile).toBe('work')
  await mintLocalGatewayTicket(secondary)
  await nativeGatewayHttpHeaders(connection!, connection!.baseUrl + '/api/config?profile=work', '/local/wrong')
  expect(requests.map(request => request.purpose)).toEqual(['interactive', 'native-http'])
  expect(requests.every(request => request.profile_id === secondary.profile_id && request.profile === 'work' && request.instance_id === 'daemon')).toBe(true)
  await expect(nativeGatewayHttpHeaders(connection!, 'http://127.0.0.1:1234/api/config')).rejects.toThrow('origin mismatch')
  connection!.release()
  await expect(mintLocalGatewayTicket(secondary)).rejects.toThrow('retired')
  expect(commands.some(command => / serve |gateway stop|gateway restart/.test(command))).toBe(false)
  expect(commands.join('\n')).not.toContain('a'.repeat(40))
})

test('only confirmed older command capabilities permit classic SSH fallback', async () => {
  const ssh = { async exec(command: string): Promise<string> {
    if (command.includes('[ -x')) {return 'OK'}
    return '  start Start\n  serve Legacy'
  }, async forward() {throw new Error('must not forward')}, async cancelForward() {} }
  const options = { ssh, profile: '', remoteHermesPath: '/usr/bin/hermes', pickLocalPort: async () => 8765 }
  expect(await attachSshGateway(options)).toBeNull()
  ssh.exec = async command => command.includes('[ -x') ? 'OK' : '  ensure Attach'
  await expect(attachSshGateway(options)).rejects.toThrow('Update Hermes')
  ssh.exec = async command => {
    if (command.includes('[ -x')) {return 'OK'}
    throw new Error('SSH unavailable')
  }
  await expect(attachSshGateway(options)).rejects.toThrow('SSH unavailable')
})

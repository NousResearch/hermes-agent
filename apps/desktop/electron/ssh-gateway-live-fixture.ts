/** Driven by the native SSH daemon test; uses the production exec/forward paths. */
import { spawn } from 'node:child_process'
import fs from 'node:fs'

import WebSocket from 'ws'

import { mintLocalGatewayTicket, nativeGatewayHttpHeaders } from './local-gateway'
import { pickLocalPort, SshConnection } from './ssh-connection'
import { attachSshGateway } from './ssh-gateway'

const config = JSON.parse(fs.readFileSync(0, 'utf8'))
const logs: string[] = []
const ssh = new SshConnection({ host: '127.0.0.1', user: config.user, port: config.port, keyPath: config.key }, {
  controlDir: config.controlDir, rememberLog: (line: string) => logs.push(line),
  spawnFn: (command: string, args: string[], options: object) => spawn(command,
    ['-F', '/dev/null', '-o', `UserKnownHostsFile=${config.knownHosts}`, '-o', 'StrictHostKeyChecking=yes', ...args], options)
})
let attached: Awaited<ReturnType<typeof attachSshGateway>>
try {
  await ssh.open()
  attached = await attachSshGateway({ ssh, profile: '', remoteHermesPath: config.hermes,
    pickLocalPort: async () => Number(await pickLocalPort()) })
  if (!attached) {throw new Error('Expected canonical SSH attachment')}
  const ticket = await mintLocalGatewayTicket(attached.gatewayEndpoint)
  const ws = new WebSocket(attached.baseUrl.replace('http:', 'ws:') + '/api/ws',
    ['hermes-gateway-v1', 'hermes-gateway-ticket.' + ticket])
  await new Promise<void>((resolve, reject) => { ws.once('open', resolve); ws.once('error', reject) })
  const result = await new Promise<any>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('RPC timed out')), 10000)
    ws.on('message', data => {
      const value = JSON.parse(data.toString())
      if (value.id === 1) {clearTimeout(timer); resolve(value)}
    })
    ws.send(JSON.stringify({ id: 1, method: 'groups.capabilities', params: {} }))
  })
  ws.close()
  if (!result.result?.driver || !result.result.methods.includes('groups.discard')) {throw new Error('Wrong room surface')}
  const headers = await nativeGatewayHttpHeaders(attached, attached.baseUrl + '/api/config')
  const response = await fetch(attached.baseUrl + '/api/config', { headers })
  if (!response.ok) {throw new Error(`HTTP ${response.status}`)}
  await response.text()
  if (JSON.stringify(logs).includes(ticket) || JSON.stringify(logs).includes(headers['X-Hermes-Gateway-Ticket'])) {
    throw new Error('Private credential leaked into SSH log')
  }
  process.stdout.write(JSON.stringify({ instance_id: attached.gatewayEndpoint.instance_id, canonical: true, http: response.status }))
} finally {
  attached?.release()
  await ssh.close()
}

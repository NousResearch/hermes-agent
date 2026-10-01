/** Driven by the native SSH daemon test; uses the production exec/forward paths. */
import { spawn } from 'node:child_process'
import fs from 'node:fs'

import WebSocket from 'ws'

import { mintLocalGatewayTicket, nativeGatewayHttpHeaders } from './local-gateway'
import { listRemoteHermesProfiles, readRemoteInstallId } from './remote-lifecycle'
import { pickLocalPort, SshConnection } from './ssh-connection'
import { attachSshGateway, inspectSshGatewayCommands } from './ssh-gateway'
import { readSshRosterInventory } from './ssh-roster-inventory'

const config = JSON.parse(fs.readFileSync(0, 'utf8'))
const logs: string[] = []
const ssh = new SshConnection({ host: '127.0.0.1', user: config.user, port: config.port, keyPath: config.key }, {
  controlDir: config.controlDir, rememberLog: (line: string) => logs.push(line),
  spawnFn: (command: string, args: string[], options: object) => spawn(command,
    ['-F', '/dev/null', '-o', `UserKnownHostsFile=${config.knownHosts}`, '-o', 'StrictHostKeyChecking=yes', ...args], options)
})
const commands: string[] = []
const execute = ssh.exec.bind(ssh)
ssh.exec = (command, options) => {commands.push(command); return execute(command, options)}
let attached: Awaited<ReturnType<typeof attachSshGateway>>
let secondary: Awaited<ReturnType<typeof attachSshGateway>>
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
  // The old raw-shell probe sees only the deliberately synthetic ambient
  // home. The production native inventory must use the attached owner instead.
  const classic = await inspectSshGatewayCommands(ssh, config.classicHermes)
  if (classic.canonical) {throw new Error('Older runtime was not recognized for cold named-profile discovery')}
  const ambient = await listRemoteHermesProfiles(ssh)
  if (!ambient.includes('ambient-only') || ambient.includes('selected-only')) {throw new Error('Synthetic homes were not distinct')}
  const ambientId = await readRemoteInstallId(ssh)
  const states = new Map([['peer', { ...attached, registryConnectionId: 'peer' }]])
  const beforeInventory = commands.length
  const inventory = await readSshRosterInventory({ connectionId: 'peer', states,
    request: async (descriptor, requestPath) => {
      const url = descriptor.baseUrl + requestPath
      const reply = await fetch(url, { headers: await nativeGatewayHttpHeaders(descriptor, url) })
      if (!reply.ok) {throw new Error('Native inventory refused')}
      return reply.json()
    }
  })
  if (inventory.kind !== 'canonical' || !inventory.profiles.includes('selected-only') ||
      inventory.profiles.includes('ambient-only') || !inventory.installId || inventory.installId === ambientId) {
    throw new Error('Canonical inventory escaped the configured owner')
  }
  if (commands.slice(beforeInventory).some(command => command.includes('${HERMES_HOME:-'))) {throw new Error('Canonical inventory read the SSH shell home')}
  secondary = await attachSshGateway({ ssh, profile: 'selected-only', profileAlias: 'display-alias', remoteHermesPath: config.hermes,
    pickLocalPort: async () => Number(await pickLocalPort()) })
  if (secondary?.gatewayEndpoint.profile_id !== config.selectedHome) {throw new Error('Secondary attachment did not bind its real home')}
  const secondaryInventory = await readSshRosterInventory({ connectionId: 'secondary',
    states: new Map([['secondary', { ...secondary, registryConnectionId: 'secondary' }]]),
    request: async (descriptor, requestPath) => {
      const configUrl = descriptor.baseUrl + '/api/config'
      const scoped = await fetch(configUrl, { headers: await nativeGatewayHttpHeaders(descriptor, configUrl) })
      if (!scoped.ok || !(await scoped.text()).includes('selected-model')) {throw new Error('Inventory ticket escaped its served secondary')}
      const url = descriptor.baseUrl + requestPath
      const response = await fetch(url, { headers: await nativeGatewayHttpHeaders(descriptor, url) })
      if (!response.ok) {throw new Error('Secondary inventory refused')}
      return response.json()
    }
  })
  if (secondaryInventory.kind !== 'canonical' || !secondaryInventory.profiles.includes('selected-only') ||
      secondaryInventory.profiles.includes('ambient-only')) {throw new Error('Secondary inventory used the ambient home')}

  process.stdout.write(JSON.stringify({ instance_id: attached.gatewayEndpoint.instance_id, canonical: true, http: response.status, selectedInventoryOnly: true }))
} finally {
  secondary?.release()
  attached?.release()
  await ssh.close()
}

/** Same-user SSH attachment to the existing canonical gateway. The tunnel is
 * Desktop-owned; the daemon is never a child or a teardown target. */
import crypto from 'node:crypto'

import { ensureLocalGateway, registerGatewayTicketTransport } from './local-gateway'
import type { GatewayEndpoint } from './local-gateway'
import { expandRemotePath, locateHermes, probeHermesVersion } from './remote-lifecycle'

// This launcher is POSIX-only; Windows SSH hosts are not supported here.
const quote = (value: string) => `'${value.replace(/'/g, `'\\''`)}'`
interface Ssh {
  exec(command: string, options?: { timeoutMs?: number; stdinData?: string }): Promise<string>
  forward(localPort: number, remotePort: number, remoteHost?: string): Promise<void>
  cancelForward(localPort: number, remotePort: number, remoteHost?: string): Promise<void>
}

/** Read-only capability inspection through the selected launcher, never a
 * login-home probe or a gateway start. Shared by attach and cold inventory. */
export async function inspectSshGatewayCommands(ssh: Pick<Ssh, 'exec'>, remoteHermesPath: string) {
  const hermesPath = await locateHermes(ssh, remoteHermesPath)
  const help = await ssh.exec(`${expandRemotePath(hermesPath)} gateway --help`, { timeoutMs: 15000 })
  const canonical = /^\s+ensure\s+/m.test(help)
  // A successful banner or wrapper message does not prove an older runtime.
  const classic = /^usage:.*\bgateway\b/im.test(help) && /^\s+(?:run|start)\s+/m.test(help) && /^\s+(?:stop|status)\s+/m.test(help)
  if (!canonical && !classic) {throw new Error('Could not determine SSH gateway capabilities. Check the configured Hermes launcher, then reconnect.')}
  return { hermesPath, canonical, tickets: /^\s+ticket\s+/m.test(help) }
}

export async function attachSshGateway(options: {
  ssh: Ssh; profile: string; remoteHermesPath: string; pickLocalPort: () => Promise<number>
  signal?: AbortSignal; profileAlias?: string
}) {
  const { ssh, profile } = options
  const commands = await inspectSshGatewayCommands(ssh, options.remoteHermesPath)
  const hermesPath = commands.hermesPath
  const command = expandRemotePath(hermesPath)
  // Only confirmed absence permits the existing classic connection path. A
  // transport, import, or authentication failure is never "old runtime".
  if (!commands.canonical) {return null}
  if (!commands.tickets) {throw new Error('Update Hermes on the SSH host to support private native tickets, then reconnect.')}

  const scoped = `${command}${profile ? ` --profile ${quote(profile)}` : ''}`
  const connection = await ensureLocalGateway(async () => ({
    code: 0, stdout: await ssh.exec(`${scoped} gateway ensure --json`, { timeoutMs: 70000 })
  }))
  options.signal?.throwIfAborted()
  const remote = new URL(connection.baseUrl)
  if (remote.protocol !== 'http:') {throw new Error('Unsupported canonical SSH listener protocol')}
  const remotePort = Number(remote.port)
  const remoteHost = remote.hostname.replace(/^\[|\]$/g, '')
  const localPort = await options.pickLocalPort()
  try {
    await ssh.forward(localPort, remotePort, remoteHost)
    options.signal?.throwIfAborted()
  } catch (error) {
    await ssh.cancelForward(localPort, remotePort, remoteHost).catch(() => undefined)
    throw error
  }
  const baseUrl = `http://127.0.0.1:${localPort}`
  const transportId = crypto.randomUUID()
  const gatewayEndpoint: GatewayEndpoint = { ...connection.gatewayEndpoint, ssh_transport_id: transportId, ssh_profile_alias: options.profileAlias || profile || 'default' }
  const release = registerGatewayTicketTransport(transportId, baseUrl, async (endpoint, purpose) => {
    const mintCommand = `${scoped} gateway ticket`
    let reply

    try {
      reply = JSON.parse(await ssh.exec(mintCommand, { timeoutMs: 15000, stdinData: JSON.stringify({
        profile_id: endpoint.profile_id, instance_id: endpoint.instance_id, purpose, profile: endpoint.ssh_profile || null
      }) }))
    } catch { throw new Error('Gateway ticket bootstrap failed') }
    if (reply.instance_id !== endpoint.instance_id || reply.profile !== (endpoint.ssh_profile || null) ||
        reply.runtime_protocol !== 1 || typeof reply.ticket !== 'string' || !/^[A-Za-z0-9_-]{32,256}$/.test(reply.ticket)) {
      throw new Error('Invalid gateway ticket response')
    }

    return reply.ticket
  })

  try {
    options.signal?.throwIfAborted()

    return { ...connection, gatewayEndpoint, baseUrl,
      wsUrl: `${baseUrl.replace(/^http/, 'ws')}/api/ws?native_dial=unminted`,
      canonical: true, localPort, remotePort, remoteHost, release,
      hermesPath, hermesHome: gatewayEndpoint.profile_id,
      hermesVersion: await probeHermesVersion(ssh, hermesPath), reused: true }
  } catch (error) {
    release()
    await ssh.cancelForward(localPort, remotePort, remoteHost)
    throw error
  }
}

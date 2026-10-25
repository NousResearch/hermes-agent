import crypto from 'node:crypto'

import {
  assertLocalGatewayEndpoint,
  bindGatewayTicketClient,
  type GatewayEndpoint,
  type GatewayTicketMinter,
  parseGatewayEnsureOutput,
  routedGatewayEndpoint
} from './local-gateway'
import { closeGatewayTicketBridge, mintGatewayTicketWithChild, TICKET_SCRIPT } from './local-gateway-python'
import { backendProfileArg } from './profile-id-guard'
import { expandRemotePath, locateHermes, openForward, shq } from './remote-lifecycle'
import type { SshConnection } from './ssh-connection'

/** Use the launcher's own bootstrap, without guessing its Python/store paths. */
export function remoteTicketCommand(raw: string): string {
  if (raw.length > 65536) {
    throw new Error('Remote runtime command exceeded limit')
  }

  const argv: unknown = JSON.parse(raw)

  if (
    !Array.isArray(argv) ||
    argv.length !== 4 ||
    !argv.every(arg => typeof arg === 'string') ||
    !argv[0].startsWith('/') ||
    /[\0\r\n]/.test(argv[0]) ||
    argv[1] !== '-I' ||
    argv[2] !== '-c'
  ) {
    throw new Error('Remote Hermes did not report a supported runtime command')
  }

  // hermes_constants is inert; the launcher's bootstrap still owns dependency/profile setup.
  return [argv[0], '-u', '-I', '-c', `${argv[3]}\n${TICKET_SCRIPT}`].map(shq).join(' ')
}

export interface NativeSshGateway {
  baseUrl: string
  wsUrl: string
  authMode: 'native'
  token: ''
  gatewayEndpoint: GatewayEndpoint
  dispose: () => void
  localPort: number
  remotePort: number
  hermesPath: string
  hermesHome: string
  reused: true
  platform: { os: string }
}

/** A failed reconnect can retire its own old tunnel, never the gateway behind it. */
export async function retireNativeSshConnection(previous, ssh: SshConnection | undefined): Promise<void> {
  if (previous?.gatewayEndpoint && previous.ssh === ssh) {
    previous.dispose?.()
    await ssh!.cancelForward(previous.localPort, previous.remotePort)
  }
}

export async function connectSshGateway<T>(
  opts: Parameters<typeof connectNativeSshGateway>[0],
  legacy: () => Promise<T>
): Promise<NativeSshGateway | T> {
  if (opts.platform.os === 'Windows') {
    return legacy()
  }

  return (await connectNativeSshGateway(opts)) ?? legacy()
}

/** An authenticated SSH channel may mint, but never owns/stops the shared gateway. */
export async function connectNativeSshGateway(opts: {
  ssh: SshConnection
  profile: string
  remoteHermesPath: string
  platform: { os: string }
  pickLocalPort: () => Promise<number>
  signal?: AbortSignal
}): Promise<NativeSshGateway | null> {
  const { ssh } = opts
  const profile = backendProfileArg(opts.profile || 'default')

  if (!profile) {
    throw new Error('Invalid native SSH profile')
  }

  const hermesPath = await locateHermes(ssh, opts.remoteHermesPath)
  const launcher = expandRemotePath(hermesPath)
  let output: string

  try {
    output = await ssh.exec(`${launcher} --profile ${shq(profile)} gateway ensure --json`, { timeoutMs: 75_000 })
  } catch (error) {
    // Only positively identified older runtimes keep their existing legacy path.
    if (/invalid choice.*ensure|unrecognized arguments:.*--json/i.test(String((error as Error).message))) {
      return null
    }

    throw error
  }

  const payload = parseGatewayEnsureOutput({ code: 0, stdout: output })

  if (payload.state !== 'ready' && !(payload.state === 'draining' && payload.reason_code === 'restart_wait')) {
    throw new Error(`Remote Hermes gateway ${payload.state}: ${payload.reason_code || 'not_ready'}`)
  }

  const endpoint = payload.endpoint as GatewayEndpoint
  assertLocalGatewayEndpoint(endpoint)
  const command = remoteTicketCommand(await ssh.exec(`${launcher} --print-runtime-command --module hermes_constants`))
  opts.signal?.throwIfAborted()
  const remotePort = Number(new URL(endpoint.api_origin).port)

  const localPort = await openForward(
    { pickLocalPort: opts.pickLocalPort, forward: (local, remote) => ssh.forward(local, remote) },
    remotePort
  )

  const baseUrl = `http://127.0.0.1:${localPort}`
  const key = `ssh-gateway:${crypto.randomUUID()}`
  const forwardedEndpoint = { ...endpoint, api_origin: baseUrl }
  let closed = false

  const dispose = () => {
    closed = true
    closeGatewayTicketBridge(key)
  }

  const mintGatewayTicket: GatewayTicketMinter = (route, purpose, method) => {
    if (closed) {
      return Promise.reject(new Error('Gateway ticket SSH connection closed'))
    }

    const routed = routedGatewayEndpoint(endpoint, route, endpoint.profile_id, method)

    return mintGatewayTicketWithChild(key, () => ssh.openPipe(command), routed, purpose, true)
  }

  bindGatewayTicketClient(forwardedEndpoint, mintGatewayTicket)

  try {
    opts.signal?.throwIfAborted()

    return {
      baseUrl,
      wsUrl: `${baseUrl.replace(/^http/, 'ws')}/api/ws?native_dial=unminted`,
      authMode: 'native',
      token: '',
      gatewayEndpoint: forwardedEndpoint,
      dispose,
      localPort,
      remotePort,
      hermesPath,
      hermesHome: endpoint.profile_id,
      reused: true,
      platform: opts.platform
    }
  } catch (error) {
    dispose()
    await ssh.cancelForward(localPort, remotePort)
    throw error
  }
}

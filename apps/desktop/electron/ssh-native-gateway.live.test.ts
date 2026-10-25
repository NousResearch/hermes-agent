import { expect, test } from 'vitest'

import { localGatewayDial, nativeGatewayHttpHeaders } from './local-gateway'
import { pickLocalPort, SshConnection } from './ssh-connection'
import { connectNativeSshGateway } from './ssh-native-gateway'

/** Opt-in read-only probe of an already configured SSH host; never submits agent work. */
test.runIf(Boolean(process.env.HERMES_SSH_LIVE_HOST))(
  'native SSH authenticates HTTP/WS and leaves the shared owner alive',
  async () => {
    const ssh = new SshConnection({ host: process.env.HERMES_SSH_LIVE_HOST }, { ownershipId: 'native-gateway-proof' })
    let connection: Awaited<ReturnType<typeof connectNativeSshGateway>>
    let socket: WebSocket | undefined

    try {
      await ssh.open()
      connection = await connectNativeSshGateway({
        ssh,
        profile: 'default',
        remoteHermesPath: '',
        platform: { os: 'Linux' },
        pickLocalPort
      })

      if (!connection) {
        throw new Error('Native gateway protocol is unavailable')
      }

      const before = connection.gatewayEndpoint.instance_id

      const response = await fetch(connection.baseUrl + '/api/status', {
        headers: await nativeGatewayHttpHeaders(connection, connection.baseUrl + '/api/status')
      })

      expect(response.status).toBe(200)
      await response.arrayBuffer()

      // Match the desktop socket factory: credentials stay in the private subprotocol, not the URL.
      await localGatewayDial({
        ensure: async () => connection!,
        forget: () => undefined,
        launchHome: '/not-a-remote-filesystem',
        profile: 'default',
        prepare: (baseUrl, ticket) => {
          socket = new WebSocket(baseUrl.replace(/^http/, 'ws') + '/api/ws', [
            'hermes-gateway-v1',
            `hermes-gateway-ticket.${ticket}`
          ])

          return 'private-native-dial'
        }
      })
      await new Promise<void>((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error('Native SSH ready event timed out')), 15_000)

        socket!.onerror = () => {
          clearTimeout(timer)
          reject(new Error('Native SSH socket failed'))
        }

        socket!.onmessage = event => {
          const frame = JSON.parse(String(event.data))

          if (frame?.params?.type === 'gateway.ready') {
            clearTimeout(timer)
            expect(frame.params.payload.session_authority).toBe(true)
            resolve()
          }
        }
      })
      socket!.close()
      connection.dispose()
      // No restart/start verb: independently discover the same live owner after the viewer left.
      const identify = await ssh.exec(`hermes --profile default gateway ensure --json`)
      expect(JSON.parse(identify).endpoint.instance_id).toBe(before)
    } finally {
      socket?.close()
      connection?.dispose()
      await ssh.close()
    }
  },
  90_000
)

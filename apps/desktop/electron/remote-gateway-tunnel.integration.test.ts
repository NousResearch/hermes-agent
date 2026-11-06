import { once } from 'node:events'
import http from 'node:http'
import type { AddressInfo, Socket } from 'node:net'

import { describe, expect, it } from 'vitest'
import { WebSocket, WebSocketServer } from 'ws'

import { RemoteGatewayTunnels } from './remote-gateway-tunnel'

// Opt-in: requires key/agent authentication to an SSH server on this machine.
// Only fixture HTTP/WS servers are forwarded; no Hermes process is touched.
const target = process.env.HERMES_TEST_TUNNEL_SSH_HOST

describe.skipIf(!target)('remote gateway real SSH transport', () => {
  it('carries HTTP and WebSocket concurrently and closing one tunnel leaves the other alive', async () => {
    const server = http.createServer((_req, res) => {
      res.setHeader('Content-Type', 'application/json')
      res.end(JSON.stringify({ fixture: true }))
    })

    const sockets = new Set<Socket>()
    server.on('connection', socket => {
      sockets.add(socket)
      socket.on('close', () => sockets.delete(socket))
    })
    const wss = new WebSocketServer({ server })
    wss.on('connection', ws => ws.on('message', data => ws.send(data)))
    server.listen(0, '127.0.0.1')
    await once(server, 'listening')
    const port = (server.address() as AddressInfo).port
    const logs: string[] = []
    const tunnels = new RemoteGatewayTunnels({ rememberLog: chunk => logs.push(chunk) })

    const source = (id: string) => ({
      id,
      url: `http://127.0.0.1:${port}`,
      sshTunnel: { host: target! }
    })

    const probe = async (url: string) => {
      expect(await (await fetch(url)).json()).toEqual({ fixture: true })
      const ws = new WebSocket(url.replace('http:', 'ws:'))

      try {
        await once(ws, 'open')
        ws.send('tunnel-fixture')
        const [data] = await once(ws, 'message')
        expect(data.toString()).toBe('tunnel-fixture')
      } finally {
        ws.terminate()
      }

      return url
    }

    try {
      const [a, b] = await Promise.all([
        tunnels.connect(source('fixture-a'), 'a/default', probe),
        tunnels.connect(source('fixture-b'), 'b/default', probe)
      ])

      expect(a).not.toBe(b)
      expect(logs.join('\n')).toContain(`-> 127.0.0.1:${port}`)
      await tunnels.closeOne('fixture-a')
      expect(await tunnels.connect(source('fixture-b'), 'b/bot', probe)).toBe(b)
    } finally {
      await tunnels.closeAll()

      for (const socket of sockets) {
        socket.destroy()
      }

      wss.close()
      await new Promise<void>(resolve => server.close(() => resolve()))
    }
  }, 60000)
})

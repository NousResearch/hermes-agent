/**
 * A local stand-in for the internet, for a sandbox that must stay offline.
 *
 * The sandbox's proxy variables point here. Every request a proxy-aware client sends (an HTTPS
 * tunnel or a plain HTTP request) is refused with 503 and only its `host:port` is kept: never a
 * path, header or body, since those can carry tokens. A spec asserts the list stays empty.
 */

import * as http from 'node:http'
import type { AddressInfo } from 'node:net'

export interface EgressRecorder {
  /** `http://127.0.0.1:<port>`: the value for HTTPS_PROXY / HTTP_PROXY / ALL_PROXY. */
  url: string
  /** Every `host:port` a client asked for, in order. */
  hosts: () => string[]
  close: () => Promise<void>
}

function plainRequestHost(request: http.IncomingMessage): string {
  try {
    const target = new URL(request.url ?? '', `http://${request.headers.host ?? 'unknown'}`)

    return `${target.hostname}:${target.port || (target.protocol === 'https:' ? '443' : '80')}`
  } catch {
    return 'unparseable'
  }
}

export async function startEgressRecorder(): Promise<EgressRecorder> {
  const seen: string[] = []

  const server = http.createServer((request, response) => {
    seen.push(plainRequestHost(request))
    response.writeHead(503, { connection: 'close' }).end()
  })

  server.on('connect', (request, socket) => {
    seen.push(request.url ?? 'unknown')
    socket.end('HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\n\r\n')
  })

  await new Promise<void>((resolve, reject) => {
    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => resolve())
  })

  const { port } = server.address() as AddressInfo

  return {
    url: `http://127.0.0.1:${port}`,
    hosts: () => [...seen],
    close: () =>
      new Promise<void>(resolve => {
        server.closeAllConnections()
        server.close(() => resolve())
      })
  }
}

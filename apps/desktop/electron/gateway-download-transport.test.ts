import fs from 'node:fs/promises'
import http from 'node:http'
import net from 'node:net'
import os from 'node:os'
import path from 'node:path'

import { expect, test } from 'vitest'

import { destroyKeepaliveAgents } from './api-transport'
import { downloadViaTokenToFile } from './gateway-download-transport'

test('download transport never follows redirects or retries after the save phase begins', async () => {
  let requests = 0
  let finalized = 0

  const server = http.createServer((req, res) => {
    requests++

    if (req.url === '/redirect') { res.writeHead(302, { Location: '/leak' }); res.end();

 return }

    res.end('bytes')
  })

  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const url = `http://127.0.0.1:${(server.address() as net.AddressInfo).port}`

  const finish = async (res: http.IncomingMessage) => {
    finalized++
    res.resume()
    throw Object.assign(new Error('write ECONNRESET'), { code: 'ECONNRESET' })
  }

  try {
    await expect(downloadViaTokenToFile(url + '/redirect', 'remote-static', {}, finish)).rejects.toThrow('redirect')
    expect(finalized).toBe(0)
    expect(requests).toBe(1)
    await expect(downloadViaTokenToFile(url + '/file', 'remote-static', {}, finish)).rejects.toThrow('write ECONNRESET')
    expect(finalized).toBe(1)
    expect(requests).toBe(2)
  } finally {
    destroyKeepaliveAgents()
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})

test('the connect timeout is dropped once headers arrive so a slow body streams to completion', async () => {
  // The transport hands the raw response stream to the finalizer instead of
  // buffering it, so the socket must stay open for as long as the save takes.
  // A body that trickles in slower than the connect timeout must still land
  // in full; only the wait for headers is bounded.
  const server = http.createServer((_req, res) => {
    res.writeHead(200, { 'Content-Type': 'application/octet-stream' })
    res.write('head-')
    setTimeout(() => res.end('tail'), 120)
  })

  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const url = `http://127.0.0.1:${(server.address() as net.AddressInfo).port}/file`

  const finish = async (res: http.IncomingMessage) => {
    const chunks: Buffer[] = []

    for await (const chunk of res) { chunks.push(Buffer.from(chunk)) }

    return Buffer.concat(chunks).toString()
  }

  try {
    await expect(downloadViaTokenToFile(url, 'remote-static', {}, finish, { timeoutMs: 30 })).resolves.toBe('head-tail')
  } finally {
    destroyKeepaliveAgents()
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})

test.skipIf(process.platform === 'win32')('download retries mint a new private grant for every wire attempt and preserve remote auth', async () => {
  const home = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'download-auth-')))
  const grants: string[] = []
  const wire: http.IncomingHttpHeaders[] = []
  const controls: any[] = []

  const control = net.createServer(socket => socket.once('data', chunk => {
    controls.push(JSON.parse(chunk.toString()).params)
    const ticket = `private-${controls.length}`
    grants.push(ticket)
    socket.end(JSON.stringify({ protocol: 1, id: 1, ok: true, result: { profile_id: home, instance_id: 'owner', ticket } }) + '\n')
  }))

  await new Promise<void>(resolve => control.listen(path.join(home, 'gateway.sock'), resolve))
  await fs.chmod(path.join(home, 'gateway.sock'), 0o600)

  const server = http.createServer((req, res) => {
    wire.push(req.headers)

    if (wire.length === 1) { req.socket.destroy();

 return }

    res.end(Buffer.from([0, 255, 1, 2, 128]))
  })

  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const baseUrl = `http://127.0.0.1:${(server.address() as net.AddressInfo).port}`
  const gatewayEndpoint = { profile_id: home, instance_id: 'owner', authority_epoch: 1, runtime_protocol: 1, api_origin: baseUrl, capabilities: ['session-authority-v1'], supervisor: 'none' }

  const finish = async (res: http.IncomingMessage) => {
    const chunks: Buffer[] = []

    for await (const chunk of res) { chunks.push(Buffer.from(chunk)) }

    return Buffer.concat(chunks)
  }

  try {
    const bytes = await downloadViaTokenToFile(baseUrl + '/api/fs/download', '', {}, finish, { gatewayDescriptor: { baseUrl, gatewayEndpoint } })
    expect(bytes).toEqual(Buffer.from([0, 255, 1, 2, 128]))
    expect(wire.map(h => h['x-hermes-gateway-ticket'])).toEqual(grants)
    expect(new Set(grants).size).toBe(2)
    expect(controls).toEqual(grants.map(() => ({ profile_id: home, instance_id: 'owner', purpose: 'native-http' })))
    expect(wire.every(h => !h.origin && !h['x-hermes-session-token'] && !h.authorization)).toBe(true)
    await expect(downloadViaTokenToFile(baseUrl + '/api/fs/download', '', {}, finish, { gatewayDescriptor: { baseUrl: baseUrl + '/wrong', gatewayEndpoint } })).rejects.toThrow('origin')
    expect(wire).toHaveLength(2)
    await downloadViaTokenToFile(baseUrl, 'remote-static', {}, finish)
    await downloadViaTokenToFile(baseUrl, null, {}, finish, { bearer: 'remote-bearer' })
    expect(wire[2]['x-hermes-session-token']).toBe('remote-static')
    expect(wire[3].authorization).toBe('Bearer remote-bearer')
    expect(wire.slice(2).every(h => !h['x-hermes-gateway-ticket'])).toBe(true)
    expect(grants).toHaveLength(2)
  } finally {
    destroyKeepaliveAgents()
    await Promise.all([new Promise<void>(resolve => server.close(() => resolve())), new Promise<void>(resolve => control.close(() => resolve()))])
    await fs.rm(home, { recursive: true, force: true })
  }
})

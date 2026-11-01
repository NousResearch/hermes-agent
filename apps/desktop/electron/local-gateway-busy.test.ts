import fs from 'node:fs/promises'
import net from 'node:net'
import path from 'node:path'

import { expect, test, vi } from 'vitest'

import { type GatewayEndpoint, isStaleLocalGatewayError, mintLocalGatewayTicket, redialLocalGateway } from './local-gateway'
import { shortSocketTmpDir } from './local-gateway.test-helpers'

async function ticketOwner(stallFirst = false) {
  const home = await shortSocketTmpDir('gw-busy-')
  let requests = 0
  let connections = 0
  let received!: () => void
  let disconnected!: () => void
  const firstRequest = new Promise<void>(resolve => { received = resolve })
  const firstDisconnect = new Promise<void>(resolve => { disconnected = resolve })
  const sockets = new Set<net.Socket>()

  const server = net.createServer(socket => {
    const first = connections++ === 0
    sockets.add(socket)
    socket.on('close', () => {
      sockets.delete(socket)

      if (first) { disconnected() }
    })
    socket.once('data', chunk => {
      const request = JSON.parse(chunk.toString())
      requests++
      received()

      if (stallFirst && requests === 1) { return }
      socket.end(JSON.stringify({ protocol: 1, id: request.id, ok: true,
        result: { profile_id: home, instance_id: 'alive', ticket: 'test-ticket' } }) + '\n')
    })
  })

  await new Promise<void>(resolve => server.listen(path.join(home, 'gateway.sock'), resolve))
  await fs.chmod(path.join(home, 'gateway.sock'), 0o600)

  const endpoint: GatewayEndpoint = { profile_id: home, instance_id: 'alive', authority_epoch: 1,
    runtime_protocol: 1, api_origin: 'http://127.0.0.1:1', capabilities: ['session-authority-v1'], supervisor: 'none' }

  return { endpoint, firstRequest, firstDisconnect, requests: () => requests,
    close: async () => {
      for (const socket of sockets) { socket.destroy() }
      await new Promise<void>(resolve => server.close(() => resolve()))
      await fs.rm(home, { recursive: true, force: true })
    } }
}

test.skipIf(process.platform === 'win32')('a connected but silent owner times out without forgetting or re-ensuring it', async () => {
  const owner = await ticketOwner(true)
  let ensures = 0
  let forgets = 0
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] })

  try {
    const result = redialLocalGateway({
      ensure: async () => { ensures++;

 return owner.endpoint },
      forget: () => { forgets++ },
      use: endpoint => mintLocalGatewayTicket(endpoint, 'native-http')
    }).then(ticket => ({ ticket, error: null }), error => ({ ticket: null, error }))

    await owner.firstRequest
    await vi.advanceTimersByTimeAsync(5001)
    const outcome = await result
    expect({ ensures, forgets, requests: owner.requests(), ticket: outcome.ticket }).toEqual({
      ensures: 1, forgets: 0, requests: 1, ticket: null
    })
    expect(outcome.error).toMatchObject({ name: 'GatewayTicketTimeoutError' })
    expect(isStaleLocalGatewayError(outcome.error)).toBe(false)
    await owner.firstDisconnect
    await expect(mintLocalGatewayTicket(owner.endpoint, 'native-http')).resolves.toBe('test-ticket')
    expect(owner.requests()).toBe(2)
  } finally {
    vi.useRealTimers()
    await owner.close()
  }
})

test.skipIf(process.platform === 'win32')('a responsive owner still returns a fresh ticket directly', async () => {
  const owner = await ticketOwner()

  try {
    await expect(mintLocalGatewayTicket(owner.endpoint, 'native-http')).resolves.toBe('test-ticket')
    expect(owner.requests()).toBe(1)
  } finally {
    await owner.close()
  }
})

test.skipIf(process.platform === 'win32')('a missing control socket still forgets once and attaches to its replacement', async () => {
  const owner = await ticketOwner()
  const deadHome = await shortSocketTmpDir('gw-gone-')
  let ensures = 0
  let forgets = 0

  try {
    const ticket = await redialLocalGateway({
      ensure: async () => ++ensures === 1 ? { ...owner.endpoint, profile_id: deadHome } : owner.endpoint,
      forget: () => { forgets++ },
      use: endpoint => mintLocalGatewayTicket(endpoint, 'native-http')
    })

    expect({ ticket, ensures, forgets, requests: owner.requests() }).toEqual({
      ticket: 'test-ticket', ensures: 2, forgets: 1, requests: 1
    })
  } finally {
    await fs.rm(deadHome, { recursive: true, force: true })
    await owner.close()
  }
})

import net, { type Socket } from 'node:net'

import type { LinkTitleAddress } from './link-title-dns'

export interface LinkTitleSocksGateway {
  proxyUrl: string
  close(): Promise<void>
}

export interface LinkTitleSocksGatewayController {
  close(): Promise<void>
  get(): Promise<LinkTitleSocksGateway>
}

export function createLinkTitleSocksGatewayController(options: {
  clearPins(): void
  start(): Promise<LinkTitleSocksGateway>
}): LinkTitleSocksGatewayController {
  let closed = false
  let closePromise: Promise<void> | null = null
  let pending: Promise<LinkTitleSocksGateway> | null = null

  return {
    close() {
      if (!closePromise) {
        closed = true
        const current = pending
        pending = null

        closePromise = (async () => {
          try {
            await (await current?.catch(() => null))?.close()
          } finally {
            options.clearPins()
          }
        })()
      }

      return closePromise
    },
    get() {
      if (closed) {
        return Promise.reject(new Error('Link title SOCKS gateway controller is closed'))
      }

      if (!pending) {
        const attempt = options.start()

        const guarded = attempt.catch(error => {
          if (pending === guarded) {
            pending = null
          }

          throw error
        })

        pending = guarded
      }

      return pending
    }
  }
}

const MAX_HANDSHAKE_BYTES = 1_024

function reply(code: number): Buffer {
  return Buffer.from([0x05, code, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
}

function ipv6FromBytes(value: Buffer): string {
  const words: string[] = []

  for (let offset = 0; offset < value.length; offset += 2) {
    words.push(value.readUInt16BE(offset).toString(16))
  }

  return words.join(':')
}

function abortError(): Error {
  return new Error('Link title SOCKS operation was aborted')
}

function timeoutError(): Error {
  return new Error('Link title SOCKS operation timed out')
}

function awaitWithin<T>(operation: PromiseLike<T>, signal: AbortSignal, deadline: number): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    if (signal.aborted) {
      reject(abortError())

      return
    }

    const remainingMs = deadline - Date.now()

    if (remainingMs <= 0) {
      reject(timeoutError())

      return
    }

    let settled = false

    const cleanup = () => {
      clearTimeout(timer)
      signal.removeEventListener('abort', onAbort)
    }

    const finish = (callback: () => void) => {
      if (settled) {
        return
      }

      settled = true
      cleanup()
      callback()
    }

    const onAbort = () => finish(() => reject(abortError()))
    const timer = setTimeout(() => finish(() => reject(timeoutError())), remainingMs)

    signal.addEventListener('abort', onAbort, { once: true })
    Promise.resolve(operation).then(
      value => finish(() => resolve(value)),
      error => finish(() => reject(error))
    )
  })
}

async function connectToApprovedAddress(
  addresses: readonly LinkTitleAddress[],
  port: number,
  deadline: number,
  signal: AbortSignal,
  createConnection: typeof net.createConnection,
  peers: Set<Socket>
): Promise<Socket> {
  let lastError: unknown = new Error('Link title SOCKS resolver returned no approved addresses')

  for (const target of addresses) {
    if (signal.aborted || Date.now() >= deadline) {
      throw signal.aborted ? abortError() : timeoutError()
    }

    let upstream: Socket

    try {
      upstream = createConnection({ family: target.family, host: target.address, port })
    } catch (error) {
      lastError = error

      continue
    }

    peers.add(upstream)

    try {
      await new Promise<void>((resolve, reject) => {
        const remainingMs = deadline - Date.now()

        if (remainingMs <= 0) {
          peers.delete(upstream)
          upstream.destroy()
          reject(timeoutError())

          return
        }

        let settled = false

        const cleanup = () => {
          clearTimeout(timer)
          signal.removeEventListener('abort', onAbort)
          upstream.off('close', onClose)
          upstream.off('connect', onConnect)
          upstream.off('error', onError)
        }

        const fail = (error: Error) => {
          if (settled) {
            return
          }

          settled = true
          cleanup()
          peers.delete(upstream)
          upstream.destroy()
          reject(error)
        }

        const onAbort = () => fail(abortError())
        const onClose = () => fail(new Error('Link title SOCKS upstream closed before connecting'))

        const onConnect = () => {
          if (settled) {
            return
          }

          settled = true
          cleanup()
          resolve()
        }

        const onError = (error: Error) => fail(error)
        const timer = setTimeout(() => fail(timeoutError()), remainingMs)

        signal.addEventListener('abort', onAbort, { once: true })
        upstream.once('close', onClose)
        upstream.once('connect', onConnect)
        upstream.once('error', onError)
      })

      return upstream
    } catch (error) {
      lastError = error
    }
  }

  throw lastError
}

function handleClient(
  client: Socket,
  options: {
    resolve(hostname: string): Promise<readonly LinkTitleAddress[]>
    connectTimeoutMs: number
    createConnection: typeof net.createConnection
  },
  peers: Set<Socket>,
  controllers: Set<AbortController>,
  tasks: Set<Promise<void>>
): void {
  const controller = new AbortController()
  const deadline = Date.now() + options.connectTimeoutMs
  let buffer = Buffer.alloc(0)
  let phase: 'greeting' | 'request' | 'connecting' | 'relay' | 'closed' = 'greeting'
  let processing = false
  let upstream: Socket | null = null

  peers.add(client)
  controllers.add(controller)

  const handshakeTimer = setTimeout(() => {
    if (phase !== 'relay' && phase !== 'closed') {
      controller.abort()
      phase = 'closed'
      client.destroy()
    }
  }, options.connectTimeoutMs)

  const closeWithReply = (code: number) => {
    if (phase === 'closed') {
      return
    }

    controller.abort()
    phase = 'closed'
    clearTimeout(handshakeTimer)
    client.end(reply(code))
  }

  const beginRelay = async (hostname: string, port: number, pending: Buffer) => {
    try {
      const addresses = await awaitWithin(options.resolve(hostname), controller.signal, deadline)
      upstream = await connectToApprovedAddress(
        addresses,
        port,
        deadline,
        controller.signal,
        options.createConnection,
        peers
      )
    } catch {
      if (!controller.signal.aborted && phase !== 'closed' && !client.destroyed) {
        closeWithReply(0x04)
      } else {
        client.destroy()
      }

      return
    }

    if (controller.signal.aborted || phase === 'closed' || client.destroyed) {
      upstream.destroy()

      return
    }

    phase = 'relay'
    clearTimeout(handshakeTimer)

    upstream.once('close', () => {
      peers.delete(upstream as Socket)
      client.destroy()
    })
    upstream.once('error', () => client.destroy())
    client.once('error', () => upstream?.destroy())

    client.write(reply(0x00))

    if (pending.length) {
      upstream.write(pending)
    }

    client.pipe(upstream)
    upstream.pipe(client)
    client.resume()
  }

  const startRelay = (hostname: string, port: number, pending: Buffer) => {
    const task = beginRelay(hostname, port, pending).finally(() => tasks.delete(task))
    tasks.add(task)
  }

  const processBuffer = () => {
    if (processing || phase === 'connecting' || phase === 'relay' || phase === 'closed') {
      return
    }

    processing = true

    try {
      while (phase === 'greeting' || phase === 'request') {
        if (phase === 'greeting') {
          if (buffer.length < 2) {
            return
          }

          const methodCount = buffer[1]
          const greetingLength = 2 + methodCount

          if (buffer[0] !== 0x05 || methodCount === 0 || greetingLength > MAX_HANDSHAKE_BYTES) {
            controller.abort()
            phase = 'closed'
            clearTimeout(handshakeTimer)
            client.destroy()

            return
          }

          if (buffer.length < greetingLength) {
            return
          }

          const methods = buffer.subarray(2, greetingLength)
          buffer = buffer.subarray(greetingLength)

          if (!methods.includes(0x00)) {
            controller.abort()
            phase = 'closed'
            clearTimeout(handshakeTimer)
            client.end(Buffer.from([0x05, 0xff]))

            return
          }

          client.write(Buffer.from([0x05, 0x00]))
          phase = 'request'
        }

        if (phase !== 'request' || buffer.length < 4) {
          return
        }

        if (buffer[0] !== 0x05 || buffer[2] !== 0x00) {
          closeWithReply(0x01)

          return
        }

        if (buffer[1] !== 0x01) {
          closeWithReply(0x07)

          return
        }

        const addressType = buffer[3]
        let addressLength: number
        let addressOffset: number

        if (addressType === 0x01) {
          addressOffset = 4
          addressLength = 4
        } else if (addressType === 0x04) {
          addressOffset = 4
          addressLength = 16
        } else if (addressType === 0x03) {
          if (buffer.length < 5) {
            return
          }

          addressOffset = 5
          addressLength = buffer[4]

          if (addressLength === 0) {
            closeWithReply(0x08)

            return
          }
        } else {
          closeWithReply(0x08)

          return
        }

        const requestLength = addressOffset + addressLength + 2

        if (requestLength > MAX_HANDSHAKE_BYTES) {
          controller.abort()
          phase = 'closed'
          clearTimeout(handshakeTimer)
          client.destroy()

          return
        }

        if (buffer.length < requestLength) {
          return
        }

        const rawAddress = buffer.subarray(addressOffset, addressOffset + addressLength)

        const hostname =
          addressType === 0x01
            ? [...rawAddress].join('.')
            : addressType === 0x04
              ? ipv6FromBytes(rawAddress)
              : rawAddress.toString('ascii')

        const port = buffer.readUInt16BE(addressOffset + addressLength)
        const pending = buffer.subarray(requestLength)
        buffer = Buffer.alloc(0)
        phase = 'connecting'
        client.pause()
        startRelay(hostname, port, pending)

        return
      }
    } finally {
      processing = false
    }
  }

  client.on('data', chunk => {
    if (phase !== 'greeting' && phase !== 'request') {
      return
    }

    const next = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)

    if (buffer.length + next.length > MAX_HANDSHAKE_BYTES) {
      controller.abort()
      phase = 'closed'
      clearTimeout(handshakeTimer)
      client.destroy()

      return
    }

    buffer = Buffer.concat([buffer, next])
    processBuffer()
  })
  client.once('close', () => {
    phase = 'closed'
    controller.abort()
    controllers.delete(controller)
    clearTimeout(handshakeTimer)
    peers.delete(client)
    upstream?.destroy()
  })
  client.once('error', () => undefined)
}

export async function startLinkTitleSocksGateway(options: {
  resolve(hostname: string): Promise<readonly LinkTitleAddress[]>
  connectTimeoutMs: number
  createConnection?: typeof net.createConnection
}): Promise<LinkTitleSocksGateway> {
  const controllers = new Set<AbortController>()
  const peers = new Set<Socket>()
  const tasks = new Set<Promise<void>>()
  const createConnection = options.createConnection ?? net.createConnection

  const server = net.createServer(client =>
    handleClient(client, { ...options, createConnection }, peers, controllers, tasks)
  )

  await new Promise<void>((resolve, reject) => {
    let settled = false

    const cleanup = () => {
      clearTimeout(timer)
      server.off('error', onError)
      server.off('listening', onListening)
    }

    const finish = (callback: () => void) => {
      if (settled) {
        return
      }

      settled = true
      cleanup()
      callback()
    }

    const onError = (error: Error) => finish(() => reject(error))
    const onListening = () => finish(resolve)

    const timer = setTimeout(() => {
      server.close()
      finish(() => reject(new Error('Link title SOCKS gateway startup timed out')))
    }, options.connectTimeoutMs)

    server.once('error', onError)
    server.once('listening', onListening)
    server.listen(0, '127.0.0.1')
  })

  const address = server.address()

  if (!address || typeof address === 'string') {
    server.close()
    throw new Error('Link title SOCKS gateway did not bind a TCP port')
  }

  server.on('error', () => undefined)
  let closePromise: Promise<void> | null = null

  return {
    async close() {
      if (!closePromise) {
        closePromise = (async () => {
          for (const controller of controllers) {
            controller.abort()
          }

          controllers.clear()

          for (const peer of peers) {
            peer.destroy()
          }

          peers.clear()

          await new Promise<void>((resolve, reject) => {
            server.close(error => {
              if (error) {
                reject(error)
              } else {
                resolve()
              }
            })
          })

          await Promise.allSettled([...tasks])
        })()
      }

      return closePromise
    },
    proxyUrl: `socks5://127.0.0.1:${address.port}`
  }
}

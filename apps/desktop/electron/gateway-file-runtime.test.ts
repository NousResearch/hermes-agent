import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'
import fs from 'node:fs'
import { createServer, type RequestListener, type Server } from 'node:http'
import os from 'node:os'
import path from 'node:path'
import { Readable } from 'node:stream'

import { afterEach, test } from 'vitest'

import { createGatewayFileRuntime } from './gateway-file-runtime'

const servers: Server[] = []
const directories: string[] = []

async function loopback(onRequest: RequestListener) {
  const server = createServer(onRequest)
  servers.push(server)
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const address = server.address()
  assert.ok(address && typeof address !== 'string')

  return `http://127.0.0.1:${address.port}`
}

function destination() {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-gateway-file-'))
  directories.push(directory)

  return path.join(directory, 'saved.txt')
}

afterEach(async () => {
  await Promise.all(servers.splice(0).map(server => new Promise<void>(resolve => server.close(() => resolve()))))

  for (const directory of directories.splice(0)) {
    fs.rmSync(directory, { recursive: true, force: true })
  }
})

test('token download saves via the selected live window and only a 404 uses the data-URL fallback', async () => {
  let failureStatus = 0
  const observed: Array<{ path: string; token: string | string[] | undefined }> = []

  const baseUrl = await loopback((request, response) => {
    observed.push({ path: request.url || '', token: request.headers['x-hermes-session-token'] })

    if (failureStatus) {
      response.writeHead(failureStatus, { connection: 'close' })
      response.end('missing')

      return
    }

    response.writeHead(200, { 'content-disposition': 'attachment; filename="remote.txt"', connection: 'close' })
    response.end('streamed bytes')
  })

  const target = destination()
  const windows: unknown[] = []
  const fallbackPaths: string[] = []
  const currentWindow = { id: 7 }

  const runtime = createGatewayFileRuntime({
    dialog: {
      showSaveDialog: async (window: unknown) => {
        windows.push(window)

        return { canceled: false, filePath: target }
      }
    },
    electronNet: {
      request: () => {
        throw new Error('cookie transport should not run')
      }
    },
    ensureBackend: async () => ({ baseUrl, token: 'session-secret', authMode: 'token' }),
    ensureRegistryBackend: async () => {
      throw new Error('wrong backend')
    },
    ensureNativeAccessToken: async () => null,
    fetchJsonForBackend: async (_connection: unknown, requestPath: string) => {
      fallbackPaths.push(requestPath)

      return { dataUrl: 'data:text/plain;base64,ZmFsbGJhY2sgYnl0ZXM=' }
    },
    getMainWindow: () => currentWindow,
    getOauthSessionForUrl: () => null,
    profileRouteOptions: () => ({})
  })

  assert.deepEqual(await runtime.saveGatewayFile({ path: '/tmp/report.txt' }), { path: target, saved: true })
  assert.equal(fs.readFileSync(target, 'utf8'), 'streamed bytes')
  assert.deepEqual(windows, [currentWindow])
  assert.match(observed[0].path, /^\/api\/fs\/download\?path=/)
  assert.equal(observed[0].token, 'session-secret')

  failureStatus = 404
  assert.deepEqual(await runtime.saveGatewayFile({ path: '/tmp/report.txt' }), { path: target, saved: true })
  assert.equal(fs.readFileSync(target, 'utf8'), 'fallback bytes')
  assert.equal(fallbackPaths.length, 1)
  assert.match(fallbackPaths[0], /^\/api\/fs\/read-data-url\?path=/)

  failureStatus = 500
  await assert.rejects(runtime.saveGatewayFile({ path: '/tmp/report.txt' }), /500:/)
  assert.equal(fallbackPaths.length, 1)
  assert.equal(fs.readFileSync(target, 'utf8'), 'fallback bytes')
})

test('OAuth file download selects the current cookie partition and preserves cancellation', async () => {
  const selectedSession = { partition: 'target-specific' }
  const requests: Array<Record<string, unknown>> = []
  let aborted = false

  const runtime = createGatewayFileRuntime({
    dialog: { showSaveDialog: async () => ({ canceled: true }) },
    electronNet: {
      request: (options: Record<string, unknown>) => {
        requests.push(options)
        const request = new EventEmitter() as EventEmitter & { abort: () => void; end: () => void }

        request.abort = () => {
          aborted = true
        }

        request.end = () => {
          queueMicrotask(() => {
            const response = Readable.from(['response bytes']) as Readable & {
              statusCode: number
              headers: Record<string, string>
            }

            response.statusCode = 200
            response.headers = {}
            request.emit('response', response)
          })
        }

        return request
      }
    },
    ensureBackend: async () => ({ baseUrl: 'https://gateway.test', authMode: 'oauth' }),
    ensureRegistryBackend: async () => {
      throw new Error('wrong backend')
    },
    ensureNativeAccessToken: async () => null,
    fetchJsonForBackend: async () => {
      throw new Error('fallback should not run')
    },
    getMainWindow: () => null,
    getOauthSessionForUrl: () => selectedSession,
    profileRouteOptions: () => ({})
  })

  assert.deepEqual(await runtime.saveGatewayFile({ path: '/tmp/report.txt' }), { canceled: true, saved: false })
  assert.equal(requests.length, 1)
  assert.equal(requests[0].session, selectedSession)
  assert.equal(requests[0].useSessionCookies, true)
  assert.equal(requests[0].redirect, 'follow')
  assert.equal(aborted, true)
})

test('registered connection file request stays on its selected backend with profile and session scope', async () => {
  const observed: string[] = []

  const baseUrl = await loopback((request, response) => {
    observed.push(request.url || '')

    response.writeHead(200, { connection: 'close' })
    response.end('registered bytes')
  })

  const target = destination()
  const registrations: Array<[string, null | string]> = []

  const runtime = createGatewayFileRuntime({
    dialog: { showSaveDialog: async () => ({ canceled: false, filePath: target }) },
    electronNet: {
      request: () => {
        throw new Error('cookie transport should not run')
      }
    },
    ensureBackend: async () => {
      throw new Error('legacy backend must not be selected')
    },
    ensureRegistryBackend: async (id: string, profile: null | string) => {
      registrations.push([id, profile])

      return { baseUrl, token: 'registry-token', mode: 'local' }
    },
    ensureNativeAccessToken: async () => null,
    fetchJsonForBackend: async () => {
      throw new Error('fallback should not run')
    },
    getMainWindow: () => null,
    getOauthSessionForUrl: () => null,
    profileRouteOptions: () => ({})
  })

  assert.deepEqual(
    await runtime.saveGatewayFile({
      connectionId: 'machine-2',
      profile: 'alpha',
      path: '/tmp/a.txt',
      sessionId: 's-1'
    }),
    { path: target, saved: true }
  )
  assert.deepEqual(registrations, [['machine-2', 'alpha']])
  assert.equal(fs.readFileSync(target, 'utf8'), 'registered bytes')
  assert.equal(observed.length, 1)
  assert.match(observed[0], /profile=alpha/)
  assert.match(observed[0], /session_id=s-1/)
})

import { EventEmitter } from 'node:events'
import fs from 'node:fs'
import http from 'node:http'
import type { AddressInfo } from 'node:net'
import os from 'node:os'
import path from 'node:path'
import { PassThrough } from 'node:stream'

import { afterEach, beforeEach, expect, test, vi } from 'vitest'

import { downloadAgentFor } from './api-transport'
import { pathForRegistryBackendRequest } from './connection-config'
import type {
  GatewayFileSaveContext,
  GatewayFileSaveDeps,
  GatewayFileSaveResult,
  GatewaySaveDialogResult
} from './gateway-file-download'
import {
  finalizeGatewayDownload,
  fsPumpDeps,
  gatewayFileRequestPaths,
  saveGatewayDownload
} from './gateway-file-download'
import type {
  GatewayDownloadOptions,
  GatewayOauthDownloadDeps,
  GatewayOauthDownloadRequest,
  GatewayOauthRequestOptions
} from './gateway-file-download-transport'
import { downloadViaOauthSessionToFile, downloadViaTokenToFile } from './gateway-file-download-transport'

interface Deferred<T> {
  promise: Promise<T>
  resolve: (value: T) => void
}

function deferred<T>(): Deferred<T> {
  let resolve: (value: T) => void = (): void => {
    throw new Error('Promise not initialized')
  }

  const promise: Promise<T> = new Promise((done): void => {
    resolve = done
  })

  return { promise, resolve }
}

const context: GatewayFileSaveContext = { suggested: 'suggested.bin', fallbackName: 'fallback.bin' }
let directory: string
const servers: http.Server[] = []

beforeEach(async (): Promise<void> => {
  directory = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'gateway-transport-'))
})
afterEach(async (): Promise<void> => {
  vi.useRealTimers()

  for (const server of servers.splice(0)) {
    server.closeAllConnections()
    await new Promise<void>((resolve, reject): void => {
      server.close((error?: Error): void => {
        if (error) {
          reject(error)
        } else {
          resolve()
        }
      })
    })
  }

  await fs.promises.rm(directory, { recursive: true, force: true })
})

async function serve(handler: (request: http.IncomingMessage, response: http.ServerResponse) => void): Promise<string> {
  const server: http.Server = http.createServer(handler)
  servers.push(server)
  await new Promise<void>((resolve): void => {
    server.listen(0, '127.0.0.1', resolve)
  })
  // SAFETY: listen(0, '127.0.0.1') completed above, so this is a bound TCP address, not a pipe or closed server.
  const address: AddressInfo = server.address() as AddressInfo

  return `http://127.0.0.1:${address.port}`
}

function saveDialog(filePath: string): GatewayFileSaveDeps {
  return { showSaveDialog: async (): Promise<GatewaySaveDialogResult> => ({ canceled: false, filePath }) }
}

interface AuthCase {
  name: string
  options: GatewayDownloadOptions
  header: string
  value: string
}

const authCases: AuthCase[] = [
  { name: 'token', options: {}, header: 'x-hermes-session-token', value: 'session-token' },
  { name: 'bearer', options: { bearer: 'native-token' }, header: 'authorization', value: 'Bearer native-token' }
]

test.each(authCases)(
  '$name transport streams before EOF, only after the dialog, and drops the connection timeout',
  async ({ options, header, value }: AuthCase): Promise<void> => {
    const responseReady: Deferred<http.ServerResponse> = deferred<http.ServerResponse>()

    const baseUrl: string = await serve((request: http.IncomingMessage, response: http.ServerResponse): void => {
      expect(request.headers[header]).toBe(value)
      expect(request.headers[header === 'authorization' ? 'x-hermes-session-token' : 'authorization']).toBeUndefined()
      response.writeHead(200, { 'Content-Disposition': 'attachment; filename="server.bin"' })
      response.flushHeaders()
      responseReady.resolve(response)
    })

    const dialogEntered: Deferred<void> = deferred<void>()
    const decision: Deferred<GatewaySaveDialogResult> = deferred<GatewaySaveDialogResult>()
    const destination: string = path.join(directory, 'existing.bin')
    await fs.promises.writeFile(destination, 'original')

    const pending: Promise<GatewayFileSaveResult> = downloadViaTokenToFile(
      `${baseUrl}/download`,
      'session-token',
      context,
      {
        showSaveDialog: async (settings: { defaultPath: string; title: string }): Promise<GatewaySaveDialogResult> => {
          expect(settings).toEqual({ defaultPath: 'server.bin', title: 'Save File' })
          dialogEntered.resolve()

          return decision.promise
        }
      },
      { ...options, timeoutMs: 2000 }
    )

    const response: http.ServerResponse = await responseReady.promise
    await dialogEntered.promise
    expect(
      Object.values(downloadAgentFor('http:').sockets)
        .flat()
        .some((socket): boolean => socket?.timeout === 0)
    ).toBe(true)
    response.write('first chunk')
    expect(await fs.promises.readdir(directory)).toEqual(['existing.bin'])
    decision.resolve({ canceled: false, filePath: destination })
    await vi.waitFor(async (): Promise<void> => {
      const part: string | undefined = (await fs.promises.readdir(directory)).find((name: string): boolean =>
        name.endsWith('.part')
      )

      expect(part).toBeDefined()
      expect(await fs.promises.readFile(path.join(directory, part!), 'utf8')).toBe('first chunk')
    })
    expect(await fs.promises.readFile(destination, 'utf8')).toBe('original')
    response.end(' last chunk')
    expect(await pending).toEqual({ saved: true, path: destination })
    expect(await fs.promises.readFile(destination, 'utf8')).toBe('first chunk last chunk')
    expect(await fs.promises.readdir(directory)).toEqual(['existing.bin'])
  }
)

interface FixtureSession {
  partition: string
}

class CookieRequest extends EventEmitter implements GatewayOauthDownloadRequest {
  aborted: boolean = false
  ended: boolean = false
  abort(): void {
    this.aborted = true
  }
  end(): void {
    this.ended = true
  }
}

test('cookie transport preserves the session, waits for the dialog without a deadline, and cancels without writing', async (): Promise<void> => {
  vi.useFakeTimers()
  const session: FixtureSession = { partition: 'persist:gateway-test' }
  const request: CookieRequest = new CookieRequest()
  const decision: Deferred<GatewaySaveDialogResult> = deferred<GatewaySaveDialogResult>()
  const response = Object.assign(new PassThrough(), { statusCode: 200, headers: {} })

  const deps: GatewayOauthDownloadDeps<FixtureSession> = {
    getSession: (url: string): FixtureSession => {
      expect(url).toBe('https://gateway.example/file')

      return session
    },
    request: (options: GatewayOauthRequestOptions<FixtureSession>): GatewayOauthDownloadRequest => {
      expect(options).toEqual({
        method: 'GET',
        url: 'https://gateway.example/file',
        session,
        useSessionCookies: true,
        redirect: 'follow'
      })
      expect(options.session).toBe(session)

      return request
    },
    showSaveDialog: (): Promise<GatewaySaveDialogResult> => decision.promise
  }

  const pending: Promise<GatewayFileSaveResult> = downloadViaOauthSessionToFile(
    'https://gateway.example/file',
    context,
    deps,
    { timeoutMs: 2000 }
  )

  expect(request.ended).toBe(true)
  request.emit('response', response)
  await vi.advanceTimersByTimeAsync(10000)
  expect(request.aborted).toBe(false)
  expect(response.listenerCount('data')).toBe(0)
  decision.resolve({ canceled: true })
  expect(await pending).toEqual({ canceled: true, saved: false })
  expect(request.aborted).toBe(true)
  expect(await fs.promises.readdir(directory)).toEqual([])
  response.destroy()
})

test('cookie connection timeout aborts before headers and never opens a dialog', async (): Promise<void> => {
  vi.useFakeTimers()
  const request: CookieRequest = new CookieRequest()

  const pending: Promise<GatewayFileSaveResult> = downloadViaOauthSessionToFile(
    'https://gateway.example/file',
    context,
    {
      getSession: (): FixtureSession => ({ partition: 'persist:gateway-test' }),
      request: (): GatewayOauthDownloadRequest => request,
      showSaveDialog: async (): Promise<GatewaySaveDialogResult> => {
        throw new Error('unexpected dialog')
      }
    },
    { timeoutMs: 2000 }
  )

  const rejected: Promise<void> = expect(pending).rejects.toThrow('Timed out connecting to Hermes backend after 2000ms')
  await vi.advanceTimersByTimeAsync(2000)
  await rejected
  expect(request.aborted).toBe(true)
  expect(await fs.promises.readdir(directory)).toEqual([])
})

test('cookie transport streams bytes after approval and preserves HTTP status on errors', async (): Promise<void> => {
  for (const statusCode of [200, 404, 503]) {
    const request: CookieRequest = new CookieRequest()
    const destination: string = path.join(directory, 'cookie.bin')
    const response = Object.assign(new PassThrough(), { statusCode, headers: {} })

    const pending: Promise<GatewayFileSaveResult> = downloadViaOauthSessionToFile(
      'https://gateway.example/file',
      context,
      {
        ...saveDialog(destination),
        getSession: (): FixtureSession => ({ partition: 'persist:gateway-test' }),
        request: (): GatewayOauthDownloadRequest => request
      }
    )

    const rejection: Promise<void> | null =
      statusCode >= 400
        ? expect(pending).rejects.toMatchObject({ statusCode, message: `${statusCode}: cookie error` })
        : null

    request.emit('response', response)
    response.end(statusCode === 200 ? 'cookie payload' : 'cookie error')

    if (rejection) {
      await rejection
    } else {
      expect(await pending).toEqual({ saved: true, path: destination })
    }

    expect(await fs.promises.readFile(destination, 'utf8')).toBe('cookie payload')
    expect(await fs.promises.readdir(directory)).toEqual(['cookie.bin'])
  }
})

test.each([404, 401, 403, 500])(
  'HTTP %i preserves status and permits only the scoped 404 fallback',
  async (statusCode: number): Promise<void> => {
    const baseUrl: string = await serve((_request: http.IncomingMessage, response: http.ServerResponse): void => {
      response.writeHead(statusCode)
      response.end('backend error')
    })

    const destination: string = path.join(directory, 'saved.bin')

    const paths = gatewayFileRequestPaths(
      '/remote/report.bin',
      (requestPath: string): string => pathForRegistryBackendRequest(requestPath, 'acme', { sharedRemote: true }),
      'session-42'
    )

    const reads: string[] = []

    const pending: Promise<GatewayFileSaveResult> = saveGatewayDownload(paths, context, {
      ...saveDialog(destination),
      download: (requestPath: string, ctx: GatewayFileSaveContext): Promise<GatewayFileSaveResult> =>
        downloadViaTokenToFile(`${baseUrl}${requestPath}`, 'session-token', ctx, {
          showSaveDialog: async (): Promise<GatewaySaveDialogResult> => {
            throw new Error('HTTP errors must not open a dialog')
          }
        }),
      readDataUrl: async (requestPath: string): Promise<string> => {
        reads.push(requestPath)

        return 'data:application/octet-stream;base64,aGVsbG8='
      }
    })

    if (statusCode === 404) {
      expect(await pending).toEqual({ saved: true, path: destination })
      expect(reads).toEqual(['/api/fs/read-data-url?path=%2Fremote%2Freport.bin&session_id=session-42&profile=acme'])
      expect(await fs.promises.readFile(destination, 'utf8')).toBe('hello')
    } else {
      await expect(pending).rejects.toMatchObject({ statusCode, message: `${statusCode}: backend error` })
      expect(reads).toEqual([])
      expect(await fs.promises.readdir(directory)).toEqual([])
    }
  }
)

test('dialog-time and mid-stream failures abort without clobbering an existing destination', async (): Promise<void> => {
  const destination: string = path.join(directory, 'existing.bin')
  await fs.promises.writeFile(destination, 'original')

  for (const duringDialog of [true, false]) {
    const response = Object.assign(new PassThrough(), { statusCode: 200, headers: {} })
    const decision: Deferred<GatewaySaveDialogResult> = deferred<GatewaySaveDialogResult>()
    let aborted: boolean = false

    const pending: Promise<GatewayFileSaveResult> = finalizeGatewayDownload(
      response,
      context,
      (): void => {
        aborted = true
      },
      { showSaveDialog: (): Promise<GatewaySaveDialogResult> => decision.promise }
    )

    const rejected: Promise<void> = expect(pending).rejects.toThrow('socket failed')

    if (!duringDialog) {
      decision.resolve({ canceled: false, filePath: destination })
      await vi.waitFor((): void => {
        expect(response.listenerCount('data')).toBe(1)
      })
      response.write('partial')
    }

    response.destroy(new Error('socket failed'))
    await new Promise<void>((resolve): void => {
      response.once('close', resolve)
    })
    decision.resolve({ canceled: false, filePath: destination })
    await rejected
    expect(aborted).toBe(true)
    expect(await fs.promises.readFile(destination, 'utf8')).toBe('original')
    expect(await fs.promises.readdir(directory)).toEqual(['existing.bin'])
  }
})

test('data-URL fallback keeps a pre-existing temp collision and destination intact', async (): Promise<void> => {
  const destination: string = path.join(directory, 'existing.bin')
  const temp: string = path.join(directory, 'collision.part')
  await fs.promises.writeFile(destination, 'original')
  await fs.promises.writeFile(temp, 'other download')
  await expect(
    saveGatewayDownload({ download: '/download', dataUrl: '/data-url' }, context, {
      ...saveDialog(destination),
      pump: { ...fsPumpDeps(), tempPathFor: (): string => temp },
      download: async (): Promise<GatewayFileSaveResult> => {
        throw Object.assign(new Error('not found'), { statusCode: 404 })
      },
      readDataUrl: async (): Promise<string> => 'data:,replacement'
    })
  ).rejects.toMatchObject({ code: 'EEXIST' })
  expect(await fs.promises.readFile(destination, 'utf8')).toBe('original')
  expect(await fs.promises.readFile(temp, 'utf8')).toBe('other download')
})

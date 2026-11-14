// The Windows channel update downloads its pinned package through the real
// strategy → verifier → pinned-download chain and shows the bytes as `fetch`
// progress (a frozen `restart` overlay read as a hang), resuming a dropped
// transfer without the bar going backwards.

import { createHash } from 'node:crypto'
import { once } from 'node:events'
import { mkdtemp, readdir, readFile, rm, writeFile } from 'node:fs/promises'
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http'
import type { AddressInfo } from 'node:net'
import os from 'node:os'
import path from 'node:path'

import { afterEach, expect, test } from 'vitest'

import type { AppInstallerStrategy} from './app-installer';
import { createChannelAppInstallerStrategy } from './app-installer'
import type { ChannelTarget } from './channel'
import type { NativeCommandResult } from './channel-native'
import { verifyPreparedChannelInstaller } from './channel-windows-host'

const BYTES: Buffer = Buffer.alloc(3 * 1024 * 1024).map((_value: number, index: number): number => index % 251) as Buffer
const ETAG = '"v1"'
const cleanups: (() => Promise<void>)[] = []

afterEach(async (): Promise<void> => {
  for (const cleanup of cleanups.splice(0)) {
    await cleanup()
  }
})

const sleep = (ms: number): Promise<void> => new Promise(resolve => setTimeout(resolve, ms))

function channelTarget(url: string): ChannelTarget {
  const identity = {
    token: 'a'.repeat(16),
    displayName: 'Preview',
    appId: 'chat.nous.preview',
    appNamePascal: 'Preview',
    artifactNamePascal: 'Preview',
    cliName: 'preview',
    windowsExecutableName: 'preview',
    msixAppIdWithOrg: 'NousResearch.Preview'
  }

  return {
    channel: {
      schema: 1,
      state: 'active',
      name: 'preview',
      repository: 'NousResearch/hermes-agent',
      policy: 'preview',
      revision: 1,
      nextSequence: 2,
      head: null,
      identity
    },
    manifest: {
      schema: 1,
      packages: [],
      request: {
        schema: 1,
        buildId: 'b'.repeat(32),
        channel: 'preview',
        sequence: 1,
        repository: 'NousResearch/hermes-agent',
        commit: 'c'.repeat(40),
        sourceVersion: '1.0.0',
        version: '0.0.2',
        windowsVersion: '0.0.2.0',
        identity,
        bundleEnv: {},
        publicBase: 'https://example.com'
      }
    },
    package: {
      platform: 'win32',
      arch: 'x64',
      variant: 'bundled',
      version: '0.0.2.0',
      identity: identity.msixAppIdWithOrg,
      publisher: 'CN=Test',
      artifact: {
        key: 'stable.msixbundle',
        sha256: createHash('sha256').update(BYTES).digest('hex'),
        size: BYTES.length
      },
      feed: { key: 'stable.appinstaller', channel: 'stable' }
    },
    artifactUrl: url,
    feedUrl: 'https://example.com/stable.appinstaller',
    manifestSha256: 'd'.repeat(64)
  }
}

/** First request: ~2.5 s of steady bytes up to half, then the connection dies. Then a Range resume. */
async function serveDroppingOnce(): Promise<{ url: string; ranges: (string | null)[] }> {
  const ranges: (string | null)[] = []
  const half = BYTES.length / 2

  const server = createServer((request: IncomingMessage, response: ServerResponse): void => {
    ranges.push(request.headers.range ? String(request.headers.range) : null)
    const match = /^bytes=(\d+)-$/.exec(String(request.headers.range ?? ''))
    const start = match && request.headers['if-range'] === ETAG ? Number(match[1]) : 0
    response.writeHead(start ? 206 : 200, {
      'content-length': BYTES.length - start,
      etag: ETAG,
      ...(start ? { 'content-range': `bytes ${start}-${BYTES.length - 1}/${BYTES.length}` } : {})
    })

    if (start) {
      response.end(BYTES.subarray(start))

      return
    }
    void (async (): Promise<void> => {
      const step = half / 8

      for (let at = 0; at < half && !response.destroyed; at += step) {
        response.write(BYTES.subarray(at, at + step))
        await sleep(300)
      }

      response.socket?.destroy()
    })()
  })

  server.listen(0, '127.0.0.1')
  await once(server, 'listening')
  cleanups.push(async (): Promise<void> => {
    server.closeAllConnections()
    server.close()
  })

  return { url: `http://127.0.0.1:${(server.address() as AddressInfo).port}/stable.msixbundle`, ranges }
}

test('the pinned Windows package download shows byte progress, resumes a drop, then hands off', async (): Promise<void> => {
  const { url, ranges } = await serveDroppingOnce()
  const directory: string = await mkdtemp(path.join(os.tmpdir(), 'hermes-msix-progress-'))
  cleanups.push(() => rm(directory, { recursive: true, force: true }))
  const descriptor: string = path.join(directory, 'stable.appinstaller')
  const verified: Buffer[] = []
  const stages: { stage: string; percent: number | null }[] = []
  const calls: string[] = []

  const nativeVerify = async (_script: string, input: string): Promise<NativeCommandResult> => {
    verified.push(await readFile(JSON.parse(input).artifact))

    return { stdout: '', stderr: '' }
  }

  const strategy: AppInstallerStrategy = createChannelAppInstallerStrategy(
    {
      python: 'python.exe',
      module: 'check.module',
      run: async () => ({ code: 0, stdout: '{"available": true}' }),
      channel: 'preview',
      light: false,
      feedBaseUrl: '',
      installer: {
        prepare: async (): Promise<string> => {
          calls.push('prepare')
          await writeFile(descriptor, 'native descriptor checked by the verifier')

          return descriptor
        },
        open: async (): Promise<string> => {
          calls.push('open')

          return ''
        }
      },
      teardownBundledBackend: async (): Promise<void> => {
        calls.push('teardown')
      },
      restoreBundledBackend: async (): Promise<void> => {
        calls.push('restore')
      },
      emitUpdateProgress: ({ stage, percent }): void => {
        stages.push({ stage, percent })
      },
      appVersion: '0.0.1.0',
      quit: (): void => {
        calls.push('quit')
      },
      registerPendingRelaunch: async () => ({ automatic: true, cancel: async (): Promise<void> => {} })
    },
    channelTarget(url),
    (file, target, onBytes) => verifyPreparedChannelInstaller(file, target, nativeVerify, onBytes)
  )

  expect((await strategy.apply()).ok).toBe(true)

  expect(ranges).toEqual([null, `bytes=${BYTES.length / 2}-`])
  expect(verified).toEqual([BYTES])
  const fetches: number[] = stages.filter(s => s.stage === 'fetch').map(s => s.percent ?? -1)
  // Progress while the bytes arrive, never backwards across the resume, complete at the end.
  expect(fetches.length).toBeGreaterThanOrEqual(2)
  expect(fetches[0]).toBeLessThan(50)
  expect(fetches).toEqual([...fetches].sort((a, b) => a - b))
  expect(fetches.at(-1)).toBe(100)
  // The hand-off owns the overlay again after the download.
  expect(stages.at(-1)).toEqual({ stage: 'restart', percent: 100 })
  expect(calls).toEqual(['prepare', 'teardown', 'open', 'quit'])
  // The verified package's digest-keyed download directory is gone; only the descriptor stays.
  expect(await readdir(directory)).toEqual(['stable.appinstaller'])
}, 30_000)

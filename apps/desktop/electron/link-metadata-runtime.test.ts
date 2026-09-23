import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, test, vi } from 'vitest'

const native = vi.hoisted(() => ({
  browserWindows: vi.fn(),
  getPath: vi.fn(),
  netFetch: vi.fn(),
  spawn: vi.fn()
}))

vi.mock('electron', () => ({
  app: { getPath: native.getPath, isReady: () => true },
  BrowserWindow: native.browserWindows,
  net: { fetch: native.netFetch },
  session: { fromPartition: vi.fn() }
}))
vi.mock('node:child_process', () => ({ spawn: native.spawn }))

import { createLinkMetadataRuntime } from './link-metadata-runtime'

const temporaryDirectories: string[] = []

function freshRuntime() {
  const userData = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-link-metadata-'))
  temporaryDirectories.push(userData)
  native.getPath.mockReturnValue(userData)

  return createLinkMetadataRuntime()
}

afterEach(() => {
  native.browserWindows.mockReset()
  native.getPath.mockReset()
  native.netFetch.mockReset()
  native.spawn.mockReset()

  for (const directory of temporaryDirectories.splice(0)) {
    fs.rmSync(directory, { recursive: true, force: true })
  }
})

test('title resolution is initialized after userData selection and coalesces same-page requests', async () => {
  assert.equal(native.getPath.mock.calls.length, 0)
  const runtime = freshRuntime()
  assert.equal(native.getPath.mock.calls.length, 1)

  native.spawn.mockImplementation(() => {
    const child = new EventEmitter() as EventEmitter & { stdout: EventEmitter }
    child.stdout = new EventEmitter()
    queueMicrotask(() => {
      child.stdout.emit(
        'data',
        Buffer.from('<title>Fleet &amp; updates</title>\nhermes-url-effective:https://example.org/guide')
      )
      child.emit('close', 0)
    })

    return child
  })

  const first = runtime.fetchLinkTitle('https://www.example.org/guide/')
  const second = runtime.fetchLinkTitle('https://example.org/guide')
  assert.strictEqual(first, second)
  assert.deepEqual(await Promise.all([first, second]), ['Fleet & updates', 'Fleet & updates'])
  assert.equal(await runtime.fetchLinkTitle('https://example.org/guide'), 'Fleet & updates')
  assert.equal(native.spawn.mock.calls.length, 1)
  assert.equal(native.browserWindows.mock.calls.length, 0)
})

test('a favicon is fetched once per host and retained for another page on that host', async () => {
  const runtime = freshRuntime()

  const png = Buffer.from(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9WlFdX8AAAAASUVORK5CYII=',
    'base64'
  )

  native.netFetch.mockImplementation(async (url: string) =>
    url.endsWith('/favicon.ico')
      ? new Response(png, { status: 200, headers: { 'content-type': 'image/png' } })
      : new Response('', { status: 200, headers: { 'content-type': 'text/html' } })
  )

  const first = await runtime.resolveFaviconCached('https://www.example.org/one')
  const requests = native.netFetch.mock.calls.length
  const second = await runtime.resolveFaviconCached('https://example.org/two')
  assert.match(first, /^data:image\/png;base64,/)
  assert.equal(second, first)
  assert.equal(native.netFetch.mock.calls.length, requests)
})

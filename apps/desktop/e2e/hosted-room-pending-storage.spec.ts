import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'

import { chromium } from '@playwright/test'
import { createServer } from 'vite'

import { expect, test } from './test'

const modulePath = '/src/plugins/hermes-bots/hosted-room-pending-storage.ts'
const desktopRoot = resolve(import.meta.dirname, '..')

test('commits 25MB random attachment bytes, survives browser restart, and never rewrites bulk bytes with the manifest', async () => {
  const server = await createServer({ configFile: false, root: desktopRoot, appType: 'custom', optimizeDeps: { noDiscovery: true, include: [] }, server: { host: '127.0.0.1', port: 0 } })
  server.middlewares.use('/pending-test', (_req, res) => { res.setHeader('Content-Type', 'text/html'); res.end('<html></html>') })
  const profile = await mkdtemp(join(tmpdir(), 'hosted-pending-'))
  let browser: Awaited<ReturnType<typeof chromium.launchPersistentContext>> | undefined

  try {
    await server.listen()
    const url = `${server.resolvedUrls!.local[0]}pending-test`
    browser = await chromium.launchPersistentContext(profile)
    let page = await browser.newPage()
    await page.goto(url)

    const saved = await page.evaluate(async path => {
      const { writePendingInput } = await import(path)

      const attachments = [15_000_000, 10_000_000].map(size => {
        const bytes = new Uint8Array(size)

        for (let offset = 0; offset < size; offset += 65_536) {crypto.getRandomValues(bytes.subarray(offset, offset + 65_536))}
        let binary = ''

        for (let offset = 0; offset < size; offset += 8192) {binary += String.fromCharCode(...bytes.subarray(offset, offset + 8192))}

        return { kind: 'file', name: `${size}.bin`, uploadId: `${size}-upload`, data: `data:application/octet-stream;base64,${btoa(binary)}` }
      })

      const digests = await Promise.all(attachments.map(async attachment => Array.from(new Uint8Array(
        await crypto.subtle.digest('SHA-256', new TextEncoder().encode(attachment.data))))))

      const pending = { eventId: 'same-event', threadId: 'same-thread', text: 'saved', attachments }
      // Persistence must not serialize payloads or use localStorage on fresh writes.
      const stringify = JSON.stringify
      const setItem = Storage.prototype.setItem

      JSON.stringify = () => {throw new Error('renderer JSON serialization')}

      Storage.prototype.setItem = () => {throw new Error('renderer localStorage write')}
      let frames = 0
      let running = true

      const frame = () => {frames++;

 if (running) {requestAnimationFrame(frame)}}

      requestAnimationFrame(frame)
      const started = performance.now()

      try {
        await writePendingInput('large', pending)
        const WorkerClass = Worker
        // Re-saving a manifest must reuse the byte Blobs, without another encoding job.
        window.Worker = class { constructor() {throw new Error('re-encoded attachment bytes')} } as unknown as typeof Worker

        try { await writePendingInput('large', { ...pending, manifest: [{ attachment_id: 'receipt' }] }) }
        finally { window.Worker = WorkerClass }
      } finally { running = false; JSON.stringify = stringify; Storage.prototype.setItem = setItem }

      return { digests, frames, elapsed: performance.now() - started }
    }, modulePath)

    expect(saved.frames).toBeGreaterThan(0)
    await browser.close()
    browser = await chromium.launchPersistentContext(profile)
    page = await browser.newPage()
    await page.goto(url)

    const restored = await page.evaluate(async path => {
      const { readPendingInput, writePendingInput } = await import(path)
      const legacy = { getRaw: () => {throw new Error('reread legacy storage')}, remove: () => {} }
      const input = await readPendingInput('large', legacy)

      const digests = await Promise.all(input.attachments.map(async (attachment: { data: string }) => Array.from(new Uint8Array(
        await crypto.subtle.digest('SHA-256', new TextEncoder().encode(attachment.data))))))

      await writePendingInput('large', null)

      return { eventId: input.eventId, threadId: input.threadId, manifest: input.manifest, digests, removed: await readPendingInput('large', legacy) }
    }, modulePath)

    expect(restored).toEqual({ eventId: 'same-event', threadId: 'same-thread', manifest: [{ attachment_id: 'receipt' }], digests: saved.digests, removed: null })
    process.stdout.write(`25MB hosted pending browser persistence: ${saved.elapsed.toFixed(1)}ms, ${saved.frames} animation frames; SHA-256 matched after process restart\n`)
  } finally { await browser?.close(); await server.close(); await rm(profile, { recursive: true, force: true }) }
})

test('migrates shipped inputs only after commit; aborted writes preserve recovery and tombstones defeat stale legacy copies', async () => {
  const server = await createServer({ configFile: false, root: desktopRoot, appType: 'custom', optimizeDeps: { noDiscovery: true, include: [] }, server: { host: '127.0.0.1', port: 0 } })
  server.middlewares.use('/pending-test', (_req, res) => { res.setHeader('Content-Type', 'text/html'); res.end('<html></html>') })
  let browser: Awaited<ReturnType<typeof chromium.launch>> | undefined

  try {
    browser = await chromium.launch()
    await server.listen()
    const page = await browser.newPage()
    await page.goto(`${server.resolvedUrls!.local[0]}pending-test`)

    const result = await page.evaluate(async path => {
      const { readPendingInput, writePendingInput } = await import(path)

      const old = { eventId: 'shipped', threadId: 'reply', text: 'legacy', attachments: [
        { kind: 'file', name: 'old.txt', uploadId: 'old-upload', data: 'data:text/plain;base64,dGVzdA==' }
      ], manifest: [{ attachment_id: 'old-receipt' }] }

      localStorage.setItem('hosted-input:old', JSON.stringify(old))
      let removals = 0
      const legacy = { getRaw: (key: string) => localStorage.getItem(key), remove: () => {removals++} }
      let readFailed = false

      try {await readPendingInput('old', { ...legacy, getRaw: () => {throw new Error('Read denied')} })}
      catch {readFailed = true}

      localStorage.setItem('hosted-input:old', '{broken json')
      let corruptFailed = false

      try {await readPendingInput('old', legacy)} catch {corruptFailed = true}
      localStorage.setItem('hosted-input:old', JSON.stringify(old))
      const put = IDBObjectStore.prototype.put

      IDBObjectStore.prototype.put = function (...args) { const request = put.apply(this, args); this.transaction.abort();

 return request }

      let failed = false

      try {await readPendingInput('old', legacy)} catch {failed = true}
      const before = { failed, removals, legacy: JSON.parse(legacy.getRaw('hosted-input:old')!) }
      IDBObjectStore.prototype.put = put
      const parse = JSON.parse

      JSON.parse = () => {throw new Error('Legacy JSON parsed on renderer')}
      let migrated

      try {migrated = await readPendingInput('old', legacy)} finally {JSON.parse = parse}

      IDBObjectStore.prototype.put = function (...args) { const request = put.apply(this, args); this.transaction.abort();

 return request }

      try {await writePendingInput('old', null)} catch { /* Aborted discard must preserve input. */ }
      IDBObjectStore.prototype.put = put
      const afterAbort = await readPendingInput('old', legacy)
      await writePendingInput('old', null)
      const discarded = await readPendingInput('old', legacy)

      return { before, old, migrated, afterAbort, discarded, removals, readFailed, corruptFailed }
    }, modulePath)

    expect(result.before).toEqual({ failed: true, removals: 0, legacy: result.old })
    expect(result.readFailed).toBe(true)
    expect(result.corruptFailed).toBe(true)
    expect(result.migrated).toEqual(result.old)
    expect(result.afterAbort).toEqual(result.old)
    expect(result.discarded).toBeNull()
    expect(result.removals).toBe(1)
  } finally { await browser?.close(); await server.close() }
})

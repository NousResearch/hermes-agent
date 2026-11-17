import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import fs from 'node:fs/promises'
import http from 'node:http'
import { createRequire } from 'node:module'
import os from 'node:os'
import path from 'node:path'
import { setTimeout as delay } from 'node:timers/promises'
import { pathToFileURL } from 'node:url'
import { afterAll, beforeAll, test, vi } from 'vitest'
import { downloadPreparedElectron } from './prepare-packaging-tools.mjs'

// The builder's own download path (what acquirePackagingTools loads), against a local mirror that
// trickles the archive out over ~400 ms: long enough to outlast a short deadline, short for a test.
const require = createRequire(import.meta.url)
const builderDist = path.dirname(require.resolve('app-builder-lib'))
const content = Buffer.from('abcd'.repeat(2048))
const filename = 'electron-v40.0.0-linux-x64.zip'
/** @type {http.Server} */
let server
let origin = ''
let directory = ''
let corrupt = false
/** @type {any} */
let electronGet

beforeAll(async () => {
  electronGet = await import(pathToFileURL(path.join(builderDist, 'util/electronGet.js')).href)
  directory = await fs.mkdtemp(path.join(os.tmpdir(), 'electron-download-'))
  server = http.createServer(async (req, res) => {
    if (req.url === '/SHASUMS256.txt') {
      res.end(`${createHash('sha256').update(content).digest('hex')} *${filename}\n`)
      return
    }
    const bytes = corrupt ? Buffer.alloc(content.length) : content
    res.writeHead(200, { 'content-length': bytes.length })
    for (let i = 0; i < 8 && !res.destroyed; i++) {
      res.write(bytes.subarray(i * 1024, (i + 1) * 1024))
      await delay(50)
    }
    res.end()
  })
  await new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve(undefined)))
  const address = /** @type {import('node:net').AddressInfo} */ (server.address())
  origin = `http://127.0.0.1:${address.port}`
})

afterAll(async () => {
  server.closeAllConnections()
  await new Promise(resolve => server.close(() => resolve(undefined)))
  await fs.rm(directory, { recursive: true, force: true })
})

/** @param {string} name */
const details = name => ({
  version: '40.0.0', platformName: 'linux', arch: 'x64', artifactName: 'electron', cacheDir: path.join(directory, name),
  options: {
    mirrorOptions: { resolveAssetURL: async (/** @type {any} */ d) => `${origin}/${d.artifactName === 'SHASUMS256.txt' ? d.artifactName : filename}` },
    downloadOptions: { quiet: true },
  },
})

test('the prepared Electron download gets a one-hour deadline and keeps the caller options', async () => {
  /** @type {any} */
  let supplied
  const builder = { downloadElectronArtifactZip: async (/** @type {any} */ value) => { supplied = value; return 'archive' } }
  const timeout = vi.spyOn(AbortSignal, 'timeout')
  try {
    assert.equal(await downloadPreparedElectron(builder, details('options')), 'archive')
    assert.deepEqual(timeout.mock.calls, [[60 * 60 * 1000]])
  } finally {
    timeout.mockRestore()
  }
  assert.equal(supplied.options.downloadOptions.signal.aborted, false)
  assert.equal(supplied.options.downloadOptions.quiet, true)
  assert.equal(typeof supplied.options.mirrorOptions.resolveAssetURL, 'function')
  assert.equal(supplied.cacheDir, path.join(directory, 'options'))
})

test('the builder runs the transfer on the deadline we pass, not its own', async () => {
  // A 120 ms deadline ends a transfer that is still moving; a 3 s one lets the same transfer finish.
  // Under the builder's own ten-minute deadline neither would differ.
  await assert.rejects(downloadPreparedElectron(electronGet, details('short'), 120))
  const archive = await downloadPreparedElectron(electronGet, details('long'), 3000)
  assert.deepEqual(await fs.readFile(archive), content)
})

test('the builder still validates the checksum', async () => {
  corrupt = true
  try {
    await assert.rejects(downloadPreparedElectron(electronGet, details('bad'), 3000), /checksum/i)
  } finally {
    corrupt = false
  }
})

import assert from 'node:assert/strict'
import fs from 'node:fs'
import http from 'node:http'
import os from 'node:os'
import path from 'node:path'
import { afterEach, test } from 'vitest'
import { downloadFileWithResume } from './prepare-packaging-tools.mjs'

/** @type {string[]} */
const roots = []
afterEach(() => roots.splice(0).forEach(root => fs.rmSync(root, { recursive: true, force: true })))

test('downloadFileWithResume creates a missing cold-cache slot before opening its partial file', async () => {
  const payload = Buffer.from('cold cache fixture payload')
  const server = http.createServer((req, res) => {
    res.writeHead(200, { 'content-length': String(payload.length) })
    res.end(payload)
  })
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
  try {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'cold-cache-slot-'))
    roots.push(root)
    const destPath = path.join(root, 'electron', 'nested', 'electron.zip')
    assert.equal(fs.existsSync(path.dirname(destPath)), false)
    const result = await downloadFileWithResume(`http://127.0.0.1:${server.address().port}/electron.zip`, destPath, { label: 'cold-cache regression', attempts: 1 })
    assert.equal(result, destPath)
    assert.equal(fs.readFileSync(destPath, 'utf8'), payload.toString())
  } finally {
    server.close()
  }
})

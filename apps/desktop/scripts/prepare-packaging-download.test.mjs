import assert from 'node:assert/strict'
import fs from 'node:fs'
import http from 'node:http'
import os from 'node:os'
import path from 'node:path'
import { test } from 'vitest'
import { downloadFileWithResume } from './prepare-packaging-tools.mjs'

test('a download to a cold cache creates its destination directory', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'packaging-download-'))
  const destination = path.join(root, 'missing', 'cache-slot', 'artifact.zip')
  const content = Buffer.from('complete artifact fixture')
  const server = http.createServer((_request, response) => response.end(content))
  try {
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
    assert.equal(fs.existsSync(path.dirname(destination)), false)
    const result = await downloadFileWithResume(`http://127.0.0.1:${server.address().port}/artifact.zip`, destination,
      { attempts: 1, baseDelayMs: 0 })
    assert.equal(result, destination)
    assert.deepEqual(fs.readFileSync(destination), content)
    assert.equal(fs.existsSync(`${destination}.part`), false)
  } finally {
    server.closeAllConnections()
    await new Promise(resolve => server.close(resolve))
    fs.rmSync(root, { recursive: true, force: true })
  }
})

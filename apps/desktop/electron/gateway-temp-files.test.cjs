'use strict'

const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const test = require('node:test')

const {
  GATEWAY_TEMP_TTL_MS,
  ensureGatewayTempDir,
  stageGatewayDataUrlToTemp,
  sweepGatewayTempFiles
} = require('./gateway-temp-files.cjs')

function mkTmpDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-gateway-temp-test-'))
}

test('ensureGatewayTempDir creates a private per-user temp subdir', async () => {
  const root = mkTmpDir()

  try {
    const dir = path.join(root, 'gateway-files')
    await ensureGatewayTempDir(dir)

    const mode = fs.statSync(dir).mode & 0o777
    assert.equal(mode, 0o700)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('stageGatewayDataUrlToTemp writes decoded bytes with a safe extension', async () => {
  const root = mkTmpDir()

  try {
    const dir = path.join(root, 'gateway-files')
    const staged = await stageGatewayDataUrlToTemp({
      dataUrl: 'data:text/html;base64,PGh0bWw+PC9odG1sPg==',
      sourcePath: '/gateway/report.html',
      tempDir: dir,
      nowMs: 1_700_000_000_000,
      randomHex: 'abc123'
    })

    assert.equal(path.dirname(staged), dir)
    assert.match(path.basename(staged), /^gateway-1700000000000-abc123\.html$/)
    assert.equal(fs.readFileSync(staged, 'utf8'), '<html></html>')
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('sweepGatewayTempFiles removes only expired staged files', async () => {
  const root = mkTmpDir()

  try {
    const dir = path.join(root, 'gateway-files')
    fs.mkdirSync(dir, { mode: 0o700 })
    const fresh = path.join(dir, 'gateway-fresh.html')
    const expired = path.join(dir, 'gateway-expired.html')
    const unrelated = path.join(dir, 'manual-note.txt')
    fs.writeFileSync(fresh, 'fresh')
    fs.writeFileSync(expired, 'expired')
    fs.writeFileSync(unrelated, 'keep')

    const now = 1_700_000_000
    fs.utimesSync(fresh, now, now)
    fs.utimesSync(expired, now - GATEWAY_TEMP_TTL_MS / 1000 - 60, now - GATEWAY_TEMP_TTL_MS / 1000 - 60)
    fs.utimesSync(unrelated, now - GATEWAY_TEMP_TTL_MS / 1000 - 60, now - GATEWAY_TEMP_TTL_MS / 1000 - 60)

    const result = await sweepGatewayTempFiles({ tempDir: dir, nowMs: now * 1000 })

    assert.equal(result.removed, 1)
    assert.equal(fs.existsSync(fresh), true)
    assert.equal(fs.existsSync(expired), false)
    assert.equal(fs.existsSync(unrelated), true)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

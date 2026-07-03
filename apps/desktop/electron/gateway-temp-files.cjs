'use strict'

const crypto = require('node:crypto')
const fs = require('node:fs')
const path = require('node:path')

const GATEWAY_TEMP_TTL_MS = 24 * 60 * 60 * 1000
const GATEWAY_TEMP_SWEEP_INTERVAL_MS = 30 * 60 * 1000
const STAGED_FILE_PREFIX = 'gateway-'
const SAFE_EXT_RE = /^\.[a-z0-9]{1,16}$/i

function extensionFromSourcePath(sourcePath) {
  const ext = path.extname(String(sourcePath || '').split(/[?#]/, 1)[0]).toLowerCase()

  return SAFE_EXT_RE.test(ext) ? ext : '.bin'
}

function decodeDataUrl(dataUrl) {
  const text = String(dataUrl || '').trim()
  const comma = text.indexOf(',')

  if (!text.startsWith('data:') || comma < 0) {
    throw new Error('Invalid data URL')
  }

  const metadata = text.slice(5, comma).toLowerCase()
  const data = text.slice(comma + 1)

  if (metadata.includes(';base64')) {
    return Buffer.from(data, 'base64')
  }

  return Buffer.from(decodeURIComponent(data), 'utf8')
}

async function ensureGatewayTempDir(tempDir) {
  await fs.promises.mkdir(tempDir, { mode: 0o700, recursive: true })

  try {
    await fs.promises.chmod(tempDir, 0o700)
  } catch {
    // Best effort: Windows ignores POSIX mode bits; Unix mkdir mode is still the
    // security boundary we need.
  }

  return tempDir
}

async function stageGatewayDataUrlToTemp({ dataUrl, sourcePath, tempDir, nowMs = Date.now(), randomHex }) {
  await ensureGatewayTempDir(tempDir)

  const ext = extensionFromSourcePath(sourcePath)
  const nonce = randomHex || crypto.randomBytes(8).toString('hex')
  const filePath = path.join(tempDir, `${STAGED_FILE_PREFIX}${nowMs}-${nonce}${ext}`)

  await fs.promises.writeFile(filePath, decodeDataUrl(dataUrl), { mode: 0o600 })

  return filePath
}

async function sweepGatewayTempFiles({ tempDir, ttlMs = GATEWAY_TEMP_TTL_MS, nowMs = Date.now(), logger } = {}) {
  if (!tempDir) {
    return { removed: 0 }
  }

  let entries
  try {
    entries = await fs.promises.readdir(tempDir, { withFileTypes: true })
  } catch (error) {
    if (error?.code !== 'ENOENT') {
      logger?.(`[file] gateway temp sweep skipped: ${error.message}`)
    }

    return { removed: 0 }
  }

  let removed = 0
  for (const entry of entries) {
    if (!entry.isFile() || !entry.name.startsWith(STAGED_FILE_PREFIX)) {
      continue
    }

    const filePath = path.join(tempDir, entry.name)
    try {
      const stat = await fs.promises.stat(filePath)

      if (nowMs - stat.mtimeMs <= ttlMs) {
        continue
      }

      await fs.promises.unlink(filePath)
      removed += 1
      logger?.(`[file] gateway temp sweep removed expired staged file: ${filePath}`)
    } catch (error) {
      if (error?.code !== 'ENOENT') {
        logger?.(`[file] gateway temp sweep could not remove ${filePath}: ${error.message}`)
      }
    }
  }

  logger?.(`[file] gateway temp sweep ran: removed=${removed}`)

  return { removed }
}

module.exports = {
  GATEWAY_TEMP_SWEEP_INTERVAL_MS,
  GATEWAY_TEMP_TTL_MS,
  ensureGatewayTempDir,
  stageGatewayDataUrlToTemp,
  sweepGatewayTempFiles
}

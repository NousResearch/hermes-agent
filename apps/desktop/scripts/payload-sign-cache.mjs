import { createHash } from 'node:crypto'
import { execFile } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'
import { readSecurityDirectory } from './sanitize-pe-signatures.mjs'

const exec = promisify(execFile)
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex')

// Only checksum, security-directory entry and a trailing certificate can change.
// Keep every other byte, including overlays. Unknown layouts are not cacheable.
export function peContentHash(file) {
  const entry = readSecurityDirectory(file)
  if (!entry) return null
  const bytes = fs.readFileSync(file)
  const pe = bytes.readUInt32LE(0x3c)
  const checksum = pe + 24 + 64
  if (checksum + 4 > entry.offsetInFile) return null
  let end = bytes.length
  if (entry.certSize || entry.certOffset) {
    if (!entry.certSize || entry.certOffset < entry.offsetInFile + 8 ||
        entry.certOffset % 8 || entry.certOffset + entry.certSize !== bytes.length) return null
    end = entry.certOffset
  }
  const content = Buffer.from(bytes.subarray(0, end))
  content.fill(0, checksum, checksum + 4)
  content.fill(0, entry.offsetInFile, entry.offsetInFile + 8)
  const hash = createHash('sha256').update(content)
  // Signing aligns the certificate table to eight bytes.
  hash.update(Buffer.alloc((8 - content.length % 8) % 8))
  return hash.digest('hex')
}

export async function verifySignedPayloads(files, publisher, signtool = '') {
  if (!files.length) return new Set()
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'verify-signatures-'))
  try {
    const manifest = path.join(temp, 'files.json')
    fs.writeFileSync(manifest, JSON.stringify(files.map(file => ({ path: file, publisher }))))
    const { stdout } = await exec('powershell.exe', [
      '-NoProfile', '-NonInteractive', '-File',
      path.join(import.meta.dirname, 'verify-signed-payloads.ps1'), manifest,
      ...(signtool ? ['-SignTool', signtool] : [])
    ], { windowsHide: true, timeout: 600000, maxBuffer: 4 * 1024 * 1024 })
    const results = JSON.parse(stdout.replace(/^\uFEFF/, ''))
    if (!Array.isArray(results) || results.length !== files.length ||
        results.some((r, i) => r.path !== files[i] || typeof r.valid !== 'boolean')) {
      throw new Error('Invalid Authenticode verification response')
    }
    return new Set(results.filter(r => r.valid).map(r => r.path))
  } finally {
    fs.rmSync(temp, { recursive: true, force: true })
  }
}

export function createPayloadSignCache({ root, env, signtool, dlib, timestampUrl, verify = verifySignedPayloads }) {
  const publisher = env.AZURE_SIGN_PUBLISHER
  if (!root || !publisher) return null
  const policy = sha256(JSON.stringify({
    schema: 1, endpoint: env.AZURE_SIGN_ENDPOINT, account: env.AZURE_SIGN_ACCOUNT,
    profile: env.AZURE_SIGN_PROFILE, publisher, timestampUrl,
    digest: 'SHA256', timestampDigest: 'SHA256',
    signtool: sha256(fs.readFileSync(signtool)), dlib: sha256(fs.readFileSync(dlib))
  }))
  const directory = path.join(root, policy)
  const entryPath = key => path.join(directory, key)
  return {
    async prepare(files) {
      const groups = new Map()
      const uncached = []
      for (const file of files) {
        const content = peContentHash(file)
        if (!content) { uncached.push(file); continue }
        const key = sha256(fs.readFileSync(file))
        if (groups.has(key)) groups.get(key).files.push(file)
        else groups.set(key, { key, content, files: [file] })
      }
      const candidates = []
      for (const group of groups.values()) {
        const entry = entryPath(group.key)
        if (!fs.existsSync(entry)) continue
        try {
          const receipt = JSON.parse(fs.readFileSync(path.join(entry, 'receipt.json'), 'utf8'))
          const binary = path.join(entry, 'signed.exe')
          if (receipt.signedHash !== sha256(fs.readFileSync(binary)) || peContentHash(binary) !== group.content) {
            throw new Error('Cache content mismatch')
          }
          candidates.push({ group, binary })
        } catch {
          fs.rmSync(entry, { recursive: true, force: true })
        }
      }
      const valid = await verify(candidates.map(c => c.binary), publisher, signtool)
      let restored = 0
      for (const { group, binary } of candidates) {
        if (!valid.has(binary)) {
          fs.rmSync(entryPath(group.key), { recursive: true, force: true })
          continue
        }
        for (const file of group.files) fs.copyFileSync(binary, file)
        restored += group.files.length
        groups.delete(group.key)
      }
      const pending = [...groups.values()]
      return {
        files: [...uncached, ...pending.map(g => g.files[0])],
        pending, restored,
        duplicates: pending.reduce((n, g) => n + g.files.length - 1, 0)
      }
    },
    async publish(plan) {
      const files = plan.pending.map(g => g.files[0])
      const valid = await verify(files, publisher, signtool)
      for (const group of plan.pending) {
        const file = group.files[0]
        if (!valid.has(file) || peContentHash(file) !== group.content) {
          throw new Error(`Signed payload failed verification: ${file}`)
        }
        const signed = fs.readFileSync(file)
        fs.mkdirSync(directory, { recursive: true })
        const temp = fs.mkdtempSync(path.join(directory, '.tmp-'))
        try {
          fs.writeFileSync(path.join(temp, 'signed.exe'), signed)
          fs.writeFileSync(path.join(temp, 'receipt.json'), JSON.stringify({ signedHash: sha256(signed) }))
          const dest = entryPath(group.key)
          // A complete entry appears in one rename, never a half-written pair.
          if (!fs.existsSync(dest)) fs.renameSync(temp, dest)
        } finally {
          fs.rmSync(temp, { recursive: true, force: true })
        }
        for (const duplicate of group.files.slice(1)) fs.copyFileSync(file, duplicate)
      }
    }
  }
}

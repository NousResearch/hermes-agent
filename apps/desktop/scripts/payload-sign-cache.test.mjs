import { createHash } from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { execFileSync } from 'node:child_process'
import { expect, test, vi } from 'vitest'
import { batchSignBinaries } from './batch-sign-binaries.mjs'
import { ensureWindowsBundleTools } from './windows-bundle-tools.mjs'
import { createPayloadSignCache, peContentHash, verifySignedPayloads } from './payload-sign-cache.mjs'
import { readSecurityDirectory } from './sanitize-pe-signatures.mjs'

const hash = bytes => createHash('sha256').update(bytes).digest('hex')

// Minimal PE input for byte-binding tests, not a signature-validity fixture.
function pe(marker = 1) {
  const bytes = Buffer.alloc(1024)
  bytes.write('MZ')
  bytes.writeUInt32LE(128, 0x3c)
  bytes.writeUInt32LE(0x4550, 128)
  bytes.writeUInt16LE(240, 148)
  bytes.writeUInt16LE(0x20b, 152)
  bytes.writeUInt32LE(16, 260)
  bytes[600] = marker
  return bytes
}

function signatureBytes(input) {
  const result = Buffer.concat([input, Buffer.alloc(32, 42)])
  result.writeUInt32LE(12345, 216)
  result.writeUInt32LE(input.length, 296)
  result.writeUInt32LE(32, 300)
  return result
}

function fixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'payload-sign-cache-test-'))
  const tool = path.join(root, 'tool')
  fs.writeFileSync(tool, 'test tool bytes')
  const env = { AZURE_SIGN_ENDPOINT: 'https://test.invalid', AZURE_SIGN_ACCOUNT: 'account',
    AZURE_SIGN_PROFILE: 'profile', AZURE_SIGN_PUBLISHER: 'CN=Test', TEMP: root }
  const verify = vi.fn(async files => new Set(files))
  const opts = { root: path.join(root, 'cache'), env, signtool: tool, dlib: tool,
    timestampUrl: 'http://timestamp.test.invalid', verify }
  return { root, tool, env, verify, opts, cache: createPayloadSignCache(opts),
    cleanup: () => fs.rmSync(root, { recursive: true, force: true }) }
}

test.runIf(process.platform === 'win32')('native verification binds cached bytes to the publisher and timestamp', async () => {
  const f = fixture()
  try {
    const publisher = execFileSync('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command',
      '(Get-AuthenticodeSignature -LiteralPath $env:NATIVE_SIGN_TEST_INPUT).SignerCertificate.Subject'
    ], { encoding: 'utf8', windowsHide: true, env: { ...process.env, NATIVE_SIGN_TEST_INPUT: process.execPath } }).trim()
    expect(publisher).not.toBe('')
    const files = ['original.exe', 'renamed.dll'].map(name => path.join(f.root, name))
    for (const file of files) fs.copyFileSync(process.execPath, file)
    expect((await verifySignedPayloads(files, publisher)).size).toBe(2)
    expect((await verifySignedPayloads(files, 'CN=Wrong publisher')).size).toBe(0)
    const cache = createPayloadSignCache({ ...f.opts, verify: verifySignedPayloads,
      env: { ...f.env, AZURE_SIGN_PUBLISHER: publisher } })
    const cold = await cache.prepare(files)
    expect(cold.files).toHaveLength(1)
    await cache.publish(cold)
    const warm = await cache.prepare(files)
    expect(warm.restored).toBe(2)
    expect(warm.files).toEqual([])
    const corrupted = fs.readFileSync(files[1])
    corrupted[0x200] ^= 1
    fs.writeFileSync(files[1], corrupted)
    expect(await verifySignedPayloads(files, publisher)).toEqual(new Set([files[0]]))
  } finally { f.cleanup() }
}, 30000)

test.runIf(process.platform === 'win32')('catalog trust cannot mask the embedded signature', async () => {
  const f = fixture()
  vi.stubEnv('ELECTRON_BUILDER_CACHE', path.join(f.root, 'sdk-cache'))
  try {
    const { signtool: nativeSigntool } = await ensureWindowsBundleTools()
    const source = path.join(process.env.SystemRoot, 'System32', 'kernel32.dll')
    const publisher = execFileSync('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command',
      '[Security.Cryptography.X509Certificates.X509Certificate]::CreateFromSignedFile($env:NATIVE_SIGN_TEST_INPUT).Subject'
    ], { encoding: 'utf8', windowsHide: true, env: { ...process.env, NATIVE_SIGN_TEST_INPUT: source } }).trim()
    expect(publisher).not.toBe('')
    const files = ['original.dll', 'renamed.exe'].map(name => path.join(f.root, name))
    for (const file of files) fs.copyFileSync(source, file)
    const catalog = () => JSON.parse(execFileSync('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command',
      '$s = Get-AuthenticodeSignature -LiteralPath $env:NATIVE_SIGN_TEST_INPUT; [pscustomobject]@{ Status = [string]$s.Status; Type = [string]$s.SignatureType } | ConvertTo-Json -Compress'
    ], { encoding: 'utf8', windowsHide: true, env: { ...process.env, NATIVE_SIGN_TEST_INPUT: files[1] } }))
    expect(catalog()).toEqual({ Status: 'Valid', Type: 'Catalog' })
    expect(await verifySignedPayloads(files, publisher, nativeSigntool)).toEqual(new Set(files))
    expect(await verifySignedPayloads(files, 'CN=Wrong publisher', nativeSigntool)).toEqual(new Set())

    // Catalog hashes omit the certificate table. Destroy it without changing code.
    const bytes = fs.readFileSync(files[1])
    const { certOffset, certSize } = readSecurityDirectory(files[1])
    expect(certSize).toBeGreaterThan(8)
    bytes.fill(0, certOffset + 8, certOffset + certSize)
    fs.writeFileSync(files[1], bytes)
    expect(catalog()).toEqual({ Status: 'Valid', Type: 'Catalog' })
    expect(await verifySignedPayloads(files, publisher, nativeSigntool)).toEqual(new Set([files[0]]))

    fs.copyFileSync(source, files[1])
    execFileSync(nativeSigntool, ['remove', '/u', files[1]], { windowsHide: true })
    expect(catalog()).toEqual({ Status: 'Valid', Type: 'Catalog' })
    expect(await verifySignedPayloads(files, publisher, nativeSigntool)).toEqual(new Set([files[0]]))
  } finally {
    vi.unstubAllEnvs()
    f.cleanup()
  }
}, 240000)

test('input bytes select entries across paths, while policy and executable content stay binding', async () => {
  const f = fixture()
  try {
    const first = path.join(f.root, 'one.exe')
    const duplicate = path.join(f.root, 'different.dll')
    const input = pe()
    fs.writeFileSync(first, input)
    fs.writeFileSync(duplicate, input)
    const plan = await f.cache.prepare([first, duplicate])
    expect(plan.files).toEqual([first])
    expect(plan.duplicates).toBe(1)
    fs.writeFileSync(first, signatureBytes(input))
    await f.cache.publish(plan)
    expect(fs.readFileSync(duplicate)).toEqual(fs.readFileSync(first))
    fs.writeFileSync(duplicate, input)
    const warm = await createPayloadSignCache(f.opts).prepare([duplicate])
    expect(warm.files).toEqual([])
    expect(warm.restored).toBe(1)
    expect(fs.readFileSync(duplicate)).toEqual(signatureBytes(input))

    fs.writeFileSync(duplicate, input)
    const changedPolicy = createPayloadSignCache({ ...f.opts, env: { ...f.env, AZURE_SIGN_PROFILE: 'other' } })
    expect((await changedPolicy.prepare([duplicate])).restored).toBe(0)
    fs.writeFileSync(duplicate, pe(2))
    expect((await f.cache.prepare([duplicate])).restored).toBe(0)

    const policyDir = path.join(f.opts.root, fs.readdirSync(f.opts.root)[0])
    const entry = path.join(policyDir, hash(input))
    const candidate = path.join(entry, 'signed.exe')
    // A different signed program with a matching receipt must not replace input.
    const substituted = signatureBytes(pe(2))
    fs.writeFileSync(candidate, substituted)
    fs.writeFileSync(path.join(entry, 'receipt.json'), JSON.stringify({ signedHash: hash(substituted) }))
    fs.writeFileSync(duplicate, input)
    expect((await f.cache.prepare([duplicate])).restored).toBe(0)
    expect(fs.readFileSync(duplicate)).toEqual(input)
    expect(fs.existsSync(entry)).toBe(false)

    for (const mode of ['corrupt', 'untrusted']) {
      const pending = await f.cache.prepare([duplicate])
      fs.writeFileSync(duplicate, signatureBytes(input))
      await f.cache.publish(pending)
      fs.writeFileSync(duplicate, input)
      if (mode === 'corrupt') fs.appendFileSync(candidate, 'corruption')
      else f.verify.mockResolvedValueOnce(new Set())
      expect((await f.cache.prepare([duplicate])).restored).toBe(0)
      expect(fs.readFileSync(duplicate)).toEqual(input)
    }
  } finally { f.cleanup() }
})

test('batch cache publishes only after successful signing and timestamping, and warm hits skip both', async () => {
  const f = fixture()
  try {
    const file = path.join(f.root, 'input.exe')
    const input = pe()
    const options = { env: f.env, signtool: f.tool, dlib: f.tool, cache: f.cache,
      timestampAttempts: 1, timestampRetryDelayMs: 0 }
    for (const failAt of ['sign', 'timestamp']) {
      fs.writeFileSync(file, input)
      await expect(batchSignBinaries([file], { ...options, exec: async (_, args) => {
        if (args[0] === failAt) throw new Error(failAt)
        fs.writeFileSync(file, signatureBytes(input))
      } })).rejects.toThrow(failAt)
      expect(fs.existsSync(f.opts.root)).toBe(false)
    }
    fs.writeFileSync(file, input)
    f.verify.mockImplementation(async () => new Set())
    await expect(batchSignBinaries([file], { ...options, exec: async (_, args) => {
      if (args[0] === 'sign') fs.writeFileSync(file, signatureBytes(input))
    } })).rejects.toThrow('failed verification')
    expect(fs.existsSync(f.opts.root)).toBe(false)
    f.verify.mockImplementation(async files => new Set(files))
    fs.writeFileSync(file, input)
    const calls = []
    const cold = await batchSignBinaries([file], { ...options, exec: async (_, args) => {
      calls.push(args[0])
      if (args[0] === 'sign') fs.writeFileSync(file, signatureBytes(input))
    } })
    expect(cold.signed).toBe(1)
    expect(calls).toEqual(['sign', 'timestamp'])
    fs.writeFileSync(file, input)
    const warm = await batchSignBinaries([file], { ...options, exec: () => { throw new Error('not a hit') } })
    expect(warm.signed).toBe(0)
    expect(fs.readFileSync(file)).toEqual(signatureBytes(input))

    const malformed = path.join(f.root, 'not-pe.exe')
    fs.writeFileSync(malformed, 'not a PE')
    expect(peContentHash(malformed)).toBe(null)
    expect((await f.cache.prepare([malformed])).files).toEqual([malformed])
  } finally { f.cleanup() }
})

import { createHash } from 'node:crypto'
import { execFileSync } from 'node:child_process'
import { createServer } from 'node:http'
import { mkdtemp, readFile, readdir, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { afterEach, expect, test } from 'vitest'
import { stageBundleInputs } from '../tests/install/e2e-assets/bundle-inputs.mjs'
import { validateBundleInputs, validateDownloadedBundle } from '../tests/install/e2e-assets/bundle-manifest.cjs'
import { validateBundledManifest } from '../tests/install/e2e-assets/windows-bundled-helpers.mjs'
import { bundledMatrix, renderMarkdownResults } from '../scripts/sandbox/generate-e2e-matrix.mjs'

const cleanup = []
afterEach(async () => { while (cleanup.length) await cleanup.pop()() })
function fixture(platform = 'windows') {
  const extension = platform === 'windows' ? 'msixbundle' : 'zip'
  return { schema: 1, platform, arch: 'x64', old: {
    tag: 'v1.2.0', version: platform === 'windows' ? '1.2.0.0' : '1.2.0',
    commit: 'a'.repeat(40), identity: 'test.bundle', teamId: 'ABCDEFGHIJ', publisher: 'CN=Test', applicationId: 'Test',
    artifact: { url: `https://example.com/old.${extension}`, sha256: 'a'.repeat(64) }
  }, new: {
    tag: 'v1.3.0', version: platform === 'windows' ? '1.3.0.0' : '1.3.0',
    commit: 'b'.repeat(40), identity: 'test.bundle', teamId: 'ABCDEFGHIJ', publisher: 'CN=Test', applicationId: 'Test',
    artifact: { url: `https://example.com/new.${extension}`, sha256: 'b'.repeat(64) }
  } }
}

test('package transitions are ordered, identity-preserving, and report through the existing family', () => {
  for (const platform of ['windows', 'macos']) {
    const manifest = fixture(platform)
    expect(validateBundleInputs(manifest, platform, 'x64')).toBe(manifest)
    const arm = { ...manifest, arch: 'arm64' }
    expect(validateBundleInputs(arm, platform, 'arm64')).toBe(arm)
    const { include } = bundledMatrix(platform, manifest.old.tag)
    expect(include).toHaveLength(1)
    expect(include[0].update_method).toBe('open-app-update')
    expect(renderMarkdownResults([{ name: `${include[0].name} / e2e`, conclusion: 'failure' }])).toContain('1 failed')
  }
})

// Independent fault inputs apply to remote admission and resumed phases alike.
test.each(['windows', 'macos'])('%s rejects malformed and incompatible pairs', platform => {
  const cases = [
    ['schema', 2, /schema/], ['platform', 'linux', /platform/], ['arch', 'riscv', /architecture/],
    ['arch', 'arm64', /architecture/],
    ['new', null, /tag/], ['new.tag', 'main', /tag/], ['new.commit', 'not-a-commit', /commit/],
    ['new.identity', 4, /identity/], ['new.identity', 'other.bundle', /identity/],
    ['new.commit', 'a'.repeat(40), /commit/], ['new.artifact', null, /SHA-256/],
    ['new.artifact.sha256', ['b'.repeat(64)], /SHA-256/],
    ['new.artifact.sha256', 'A'.repeat(64), /SHA-256/], ['new.artifact.sha256', 'a'.repeat(64), /differ/],
    ['new.artifact.url', '', /url/], ['new.artifact.url', 'http://example.com/new.zip', /HTTPS/],
    ['new.artifact.url', 'https://user:pass@example.com/new.zip', /credentials/],
    ['new.artifact.url', 'https://example.com/new.exe', /artifact/],
    ...(platform === 'windows' ? [
      ['new.version', '1.2.0.0', /increase/], ['new.version', '1.1.9.9', /increase/],
      ['new.version', '1.3.0', /four numeric/], ['new.version', '1.3.65536.0', /16 bits/],
      ['new.publisher', '', /publisher/], ['new.publisher', 'CN=Other', /publisher/],
      ['new.applicationId', 7, /applicationId/], ['new.applicationId', 'Other', /applicationId/],
    ] : [
      ['new.identity', 'TEAM123456', /CFBundleIdentifier/], ['new.version', '1.9.0', /match/],
      ['new.teamId', ['ABCDEFGHIJ'], /teamId/], ['new.teamId', '', /teamId/], ['new.teamId', 'OTHER12345', /signing team/],
    ]),
  ]
  const oldCases = cases.filter(([field]) => field.startsWith('new') && field !== 'new.version')
    .map(([field, value, diagnostic]) => [field.replace(/^new/, 'old'),
      field === 'new.commit' && value === 'a'.repeat(40) ? 'b'.repeat(40) :
        field === 'new.artifact.sha256' && value === 'a'.repeat(64) ? 'b'.repeat(64) : value, diagnostic])
  oldCases.push(...(platform === 'windows' ? [
    ['old.version', '1.3.0.0', /increase/], ['old.version', '1.4.0.0', /increase/],
    ['old.version', '1.2.0', /four numeric/], ['old.version', '1.2.65536.0', /16 bits/],
  ] : [['old.version', '1.9.0', /match/]]))
  for (const [field, value, diagnostic] of [...cases, ...oldCases]) {
    const bad = fixture(platform), keys = field.split('.')
    const last = keys.pop()
    keys.reduce((object, key) => object[key], bad)[last] = value
    expect(() => validateBundleInputs(bad, platform, 'x64'), field).toThrow(diagnostic)
    expect(() => validateDownloadedBundle(bad, platform, 'x64'), field).toThrow(diagnostic)
  }
  expect(() => validateBundleInputs(fixture(platform), platform, 'riscv')).toThrow('Unsupported')
  expect(() => validateBundleInputs(fixture(platform), platform, 'arm64')).toThrow('architecture')
  const pair = fixture(platform)
  if (platform === 'macos') {
    pair.new.tag = 'v1.3.0-canary.20260907000000'; pair.new.version = pair.new.tag.slice(1)
    expect(() => validateBundleInputs(pair, platform, 'x64')).toThrow('channel')
    pair.new.tag = pair.old.tag; pair.new.version = pair.old.version
    expect(() => validateBundleInputs(pair, platform, 'x64')).toThrow('increase')
  } else {
    for (const [old, newer] of [['1.2.3.9', '1.2.3.10'], ['1.2.65535.0', '1.3.0.0']]) {
      pair.old.version = old; pair.new.version = newer
      expect(validateBundleInputs(pair, platform, 'x64')).toBe(pair)
    }
  }
})

test.each(['windows', 'macos'])('%s staging and resumed validation use actual files and reject tampering', async platform => {
  // These bytes test download integrity only, not native package acceptance.
  const bytes = { old: Buffer.from('old transport payload'), new: Buffer.from('new transport payload') }
  const manifest = fixture(platform)
  const server = createServer((req, res) => {
    res.end(req.url === '/manifest.json' ? JSON.stringify(manifest) : bytes[req.url.slice(1).split('.')[0]])
  })
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
  cleanup.push(() => new Promise(resolve => { server.closeAllConnections(); server.close(resolve) }))
  const directory = await mkdtemp(path.join(os.tmpdir(), 'bundle-input-test-'))
  cleanup.push(() => rm(directory, { recursive: true, force: true }))
  const base = `http://127.0.0.1:${server.address().port}`
  for (const slot of ['old', 'new']) {
    manifest[slot].artifact.url = `${base}/${slot}.${platform === 'windows' ? 'msixbundle' : 'zip'}`
    manifest[slot].artifact.sha256 = createHash('sha256').update(bytes[slot]).digest('hex')
  }
  const out = path.join(directory, 'good')
  const manifestSha256 = createHash('sha256').update(JSON.stringify(manifest)).digest('hex')
  await expect(stageBundleInputs({ manifestUrl: `${base}/manifest.json`, platform, arch: 'x64', out, manifestSha256: '0'.repeat(64) })).rejects.toThrow('manifest SHA-256')
  await expect(stageBundleInputs({ manifestUrl: `${base}/manifest.json`, platform, arch: 'x64', out, expectedCommit: 'c'.repeat(40) })).rejects.toThrow('workflow SHA')
  const filename = await stageBundleInputs({ manifestUrl: `${base}/manifest.json`, platform, arch: 'x64', out, manifestSha256 })
  const result = JSON.parse(await readFile(filename, 'utf8'))
  expect(validateDownloadedBundle(result, platform, 'x64')).toBe(result)
  if (platform === 'windows') {
    expect(validateBundledManifest(result, { expectedPublisher: 'CN=Test' })).toEqual({ ok: true, errors: [] })
    expect(validateBundledManifest(result, { expectedPublisher: 'CN=Other' }).errors.join()).toContain('OUT_OF_STORE_PUBLISHER')
  }
  // Native Mac resume uses require(), not an ESM-only import.
  const validator = fileURLToPath(new URL('../tests/install/e2e-assets/bundle-manifest.cjs', import.meta.url))
  execFileSync(process.execPath, ['-e', 'require(process.argv[1]).validateDownloadedBundle(require(process.argv[2]), process.argv[3], "x64")', validator, filename, platform])
  for (const slot of ['old', 'new']) {
    for (const missing of [undefined, path.join(directory, 'missing'), directory]) {
      const bad = structuredClone(result); bad[slot].artifact.path = missing
      expect(() => validateDownloadedBundle(bad, platform, 'x64')).toThrow('artifact.path')
      if (platform === 'windows') expect(validateBundledManifest(bad, { expectedPublisher: 'CN=Test' }).ok).toBe(false)
    }
  }
  const altered = structuredClone(result); altered.new.commit = altered.old.commit
  expect(() => validateDownloadedBundle(altered, platform, 'x64')).toThrow('commit')
  if (platform === 'windows') expect(validateBundledManifest(altered, { expectedPublisher: 'CN=Test' }).ok).toBe(false)
  for (const slot of ['old', 'new']) expect(await readFile(result[slot].artifact.path)).toEqual(bytes[slot])
  manifest.old.artifact.sha256 = 'c'.repeat(64)
  const bad = path.join(directory, 'bad')
  await expect(stageBundleInputs({ manifestUrl: `${base}/manifest.json`, platform, arch: 'x64', out: bad })).rejects.toThrow('SHA-256')
  expect(await readdir(bad)).toEqual([])
})

test('explicit bundled routes refuse absent package inputs instead of reporting a skipped pass', async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'bundle-plan-test-'))
  cleanup.push(() => rm(directory, { recursive: true, force: true }))
  const script = fileURLToPath(new URL('../tests/install/e2e-assets/bundle-plan.mjs', import.meta.url))
  const env = { ...process.env, BUNDLE_WINDOWS_MANIFEST: '', BUNDLE_MACOS_MANIFEST: '',
    GITHUB_OUTPUT: path.join(directory, 'output'), GITHUB_STEP_SUMMARY: path.join(directory, 'summary') }
  expect(() => execFileSync(process.execPath, [script], { env: { ...env, BUNDLE_ROUTE: 'windows-bundled' }, stdio: 'pipe' })).toThrow()
  execFileSync(process.execPath, [script], { env: { ...env, BUNDLE_ROUTE: 'all' }, stdio: 'pipe' })
  expect(await readFile(env.GITHUB_STEP_SUMMARY, 'utf8')).toContain('no signed package pair supplied')
  expect(await readFile(env.GITHUB_OUTPUT, 'utf8')).toContain('windows={"include":[]}')
})

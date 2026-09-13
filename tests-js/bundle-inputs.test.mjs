import { createHash } from 'node:crypto'
import { execFileSync } from 'node:child_process'
import { createServer } from 'node:http'
import { mkdtemp, readFile, readdir, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { afterEach, expect, test } from 'vitest'
import { stageBundleInputs, validateBundleInputs } from '../tests/install/e2e-assets/bundle-inputs.mjs'
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
    for (const change of [
      value => { value.new.version = value.old.version },
      value => { value.new.identity = 'other' },
      value => { value.new.commit = 'not-a-commit' },
      value => { value.new.artifact.sha256 = 'not-a-digest' },
      value => { value.new.artifact.url = 'http://example.com/package.zip' },
      value => { value.arch = 'arm64' }
    ]) {
      const bad = structuredClone(manifest)
      change(bad)
      expect(() => validateBundleInputs(bad, platform, 'x64')).toThrow()
    }
    const { include } = bundledMatrix(platform, manifest.old.tag)
    expect(include).toHaveLength(1)
    expect(include[0].update_method).toBe('open-app-update')
    expect(renderMarkdownResults([{ name: `${include[0].name} / e2e`, conclusion: 'failure' }])).toContain('1 failed')
  }
})

test('staging streams actual bytes, verifies hashes and removes rejected partial downloads', async () => {
  // These bytes test download integrity only, not native package acceptance.
  const bytes = { old: Buffer.from('old transport payload'), new: Buffer.from('new transport payload') }
  const manifest = fixture()
  const server = createServer((req, res) => {
    res.end(req.url === '/manifest.json' ? JSON.stringify(manifest) : bytes[req.url.slice(1).split('.')[0]])
  })
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
  cleanup.push(() => new Promise(resolve => { server.closeAllConnections(); server.close(resolve) }))
  const directory = await mkdtemp(path.join(os.tmpdir(), 'bundle-input-test-'))
  cleanup.push(() => rm(directory, { recursive: true, force: true }))
  const base = `http://127.0.0.1:${server.address().port}`
  for (const slot of ['old', 'new']) {
    manifest[slot].artifact.url = `${base}/${slot}.msixbundle`
    manifest[slot].artifact.sha256 = createHash('sha256').update(bytes[slot]).digest('hex')
  }
  const out = path.join(directory, 'good')
  const manifestSha256 = createHash('sha256').update(JSON.stringify(manifest)).digest('hex')
  await expect(stageBundleInputs({ manifestUrl: `${base}/manifest.json`, platform: 'windows', arch: 'x64', out, manifestSha256: '0'.repeat(64) })).rejects.toThrow('manifest SHA-256')
  const filename = await stageBundleInputs({ manifestUrl: `${base}/manifest.json`, platform: 'windows', arch: 'x64', out, manifestSha256 })
  const result = JSON.parse(await readFile(filename, 'utf8'))
  for (const slot of ['old', 'new']) expect(await readFile(result[slot].artifact.path)).toEqual(bytes[slot])
  manifest.old.artifact.sha256 = 'c'.repeat(64)
  const bad = path.join(directory, 'bad')
  await expect(stageBundleInputs({ manifestUrl: `${base}/manifest.json`, platform: 'windows', arch: 'x64', out: bad })).rejects.toThrow('SHA-256')
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

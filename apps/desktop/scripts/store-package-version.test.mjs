import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { afterEach, expect, test, vi } from 'vitest'
import { appIdentity, storeManifestTemplate, storePackageVersion, storePackageVersionAt } from '../../../scripts/msix-shared.mjs'
import { stageStoreManifest } from './before-build.mjs'
import { AppInfo } from '../../../node_modules/app-builder-lib/dist/appInfo.js'
import { substituteManifestMacros } from '../../../node_modules/app-builder-lib/dist/targets/win/winAppUtil.js'

const roots = []
afterEach(() => { for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true }) })
const desktop = fileURLToPath(new URL('../', import.meta.url))

function fixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'store-manifest-'))
  roots.push(root)
  const app = path.join(root, 'apps', 'desktop')
  fs.mkdirSync(path.join(app, 'assets'), { recursive: true })
  fs.copyFileSync(path.join(desktop, 'assets/msix-manifest.xml'), path.join(app, 'assets/msix-manifest.xml'))
  fs.writeFileSync(path.join(app, 'product-identity.cjs'), "module.exports={store:true,artifactNamePascal:'HermesBundled'}\n")
  fs.writeFileSync(path.join(app, 'package.json'), JSON.stringify({ name: 'hermes', version: '0.27.1' }))
  const env = { ...process.env, GIT_AUTHOR_NAME: 'Test', GIT_AUTHOR_EMAIL: 'test@example.invalid',
    GIT_COMMITTER_NAME: 'Test', GIT_COMMITTER_EMAIL: 'test@example.invalid',
    GIT_AUTHOR_DATE: '2026-09-07T00:18:00Z', GIT_COMMITTER_DATE: '2026-09-07T00:18:00Z' }
  for (const args of [['init', '-q'], ['add', '.'], ['-c', 'commit.gpgsign=false', 'commit', '-qm', 'fixture'], ['tag', 'v0.27.1']]) {
    execFileSync('git', args, { cwd: root, env, stdio: 'pipe' })
  }
  return { root, app }
}

test('Store manifest and envelope agree while executable app semver and sideload sequence remain unchanged', () => {
  const { app } = fixture()
  const tag = 'v0.27.1-canary.20260907001718'
  const identity = appIdentity(app, tag)
  const staged = fs.readFileSync(stageStoreManifest(app, tag), 'utf8')
  const appInfo = new AppInfo({ metadata: { name: 'hermes', version: tag.slice(1) }, config: { buildNumber: '32863' } }, undefined, {})
  const xml = substituteManifestMacros(staged, key => key === 'version' ? appInfo.getVersionInWeirdWindowsForm(false) : `test-${key}`)
  const version = /<Identity\b[^>]*Version="([^"]+)"/.exec(xml)[1]
  expect(version).toBe(identity.version)
  expect(version.split('.')[3]).toBe('0')
  expect(Number(version.split('.')[0])).toBeGreaterThan(0)
  expect(identity.fileVersion).toBe(tag.slice(1))
  expect(appInfo.version).toBe(tag.slice(1))
  expect(appInfo.getVersionInWeirdWindowsForm(true)).toBe('0.27.1.32863')
  const stable = appIdentity(app, 'v0.27.1').version.split('.').map(Number)
  expect(stable[2]).toBeGreaterThan(Number(version.split('.')[2]))
})

test('Store calendar ordering survives minute, hour, day and year boundaries and rejects reserved revision', () => {
  const seconds = [
    '2026-09-07T00:17:18Z', '2026-09-07T00:17:19Z', '2026-09-07T00:18:00Z',
    '2026-09-07T01:00:00Z', '2026-09-08T00:00:00Z', '2027-01-01T00:00:00Z'
  ].map(value => Date.parse(value) / 1000)
  const versions = seconds.map(storePackageVersionAt).map(value => value.split('.').map(Number))
  for (const parts of versions) {
    expect(parts[3]).toBe(0)
    expect(parts.every(value => value >= 0 && value <= 65535)).toBe(true)
  }
  for (let i = 1; i < versions.length; i++) {
    const first = versions[i].findIndex((value, index) => value !== versions[i - 1][index])
    expect(versions[i][first]).toBeGreaterThan(versions[i - 1][first])
  }
  expect(() => storeManifestTemplate('${version}', '0.27.1.32863')).toThrow()
  expect(() => storeManifestTemplate('${version}', '0.27.1.0')).toThrow()
  expect(() => storePackageVersionAt(NaN)).toThrow()
  expect(() => storePackageVersion('v0.27.1-canary.20260231000000', '.')).toThrow('Invalid canary')
})

test('commit builds cannot stage a Store manifest', () => {
  const { root, app } = fixture()
  const commit = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' }).trim()
  vi.stubEnv('HERMES_PAYLOAD_TAG', '')
  vi.stubEnv('HERMES_BUILD_COMMIT', commit)
  vi.stubEnv('HERMES_PAYLOAD_VERSION', '0.21.1')
  try {
    expect(() => appIdentity(app)).toThrow('Store packaging requires a stable release tag')
    expect(() => stageStoreManifest(app, '')).toThrow('Store packaging requires a stable release tag')
    expect(fs.existsSync(path.join(app, 'build/store-msix-manifest.xml'))).toBe(false)
  } finally {
    vi.unstubAllEnvs()
  }
})

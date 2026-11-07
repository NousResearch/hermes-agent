// Hermes shares %LOCALAPPDATA% and HKCU state with the unpackaged CLI/gateway,
// so every MSIX the repo produces must opt out of write virtualisation and the
// builder must request the restricted capability that permits it.
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import { createRequire } from 'node:module'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { afterEach, expect, test } from 'vitest'
import { stageChannelManifest } from '../../../scripts/msix-shared.mjs'
import { stageReleaseManifest, stageStoreManifest } from './before-build.mjs'

const require = createRequire(import.meta.url)
const desktop = fileURLToPath(new URL('../', import.meta.url))
const roots = []
afterEach(() => { for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true }) })

function virtualization(xml) {
  const read = name => new RegExp(`<desktop6:${name}>([^<]*)</desktop6:${name}>`).exec(xml)?.[1]
  return { fileSystem: read('FileSystemWriteVirtualization'), registry: read('RegistryWriteVirtualization') }
}

// A tagged app dir: the real template plus just enough for appIdentity.
function taggedApp(tag) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'msix-virt-'))
  roots.push(root)
  const app = path.join(root, 'apps', 'desktop')
  fs.mkdirSync(path.join(app, 'assets'), { recursive: true })
  fs.copyFileSync(path.join(desktop, 'assets/msix-manifest.xml'), path.join(app, 'assets/msix-manifest.xml'))
  fs.writeFileSync(path.join(app, 'product-identity.cjs'), "module.exports={store:true,artifactNamePascal:'HermesBundled'}\n")
  fs.writeFileSync(path.join(app, 'package.json'), JSON.stringify({ name: 'hermes', version: '0.27.1' }))
  const env = { ...process.env, GIT_AUTHOR_NAME: 'Test', GIT_AUTHOR_EMAIL: 'test@example.invalid',
    GIT_COMMITTER_NAME: 'Test', GIT_COMMITTER_EMAIL: 'test@example.invalid',
    GIT_AUTHOR_DATE: '2026-09-07T00:18:00Z', GIT_COMMITTER_DATE: '2026-09-07T00:18:00Z' }
  for (const args of [['init', '-q'], ['add', '.'], ['-c', 'commit.gpgsign=false', 'commit', '-qm', 'fixture'], ['tag', tag]]) {
    execFileSync('git', args, { cwd: root, env, stdio: 'pipe' })
  }
  return app
}

// A channel app dir: stageChannelManifest loads scripts/before-build.mjs from it,
// so link the real scripts directory (a junction needs no privilege on Windows).
function channelApp() {
  const app = fs.mkdtempSync(path.join(os.tmpdir(), 'msix-virt-channel-'))
  roots.push(app)
  fs.mkdirSync(path.join(app, 'assets'))
  fs.mkdirSync(path.join(app, 'build/agent-payload'), { recursive: true })
  fs.copyFileSync(path.join(desktop, 'assets/msix-manifest.xml'), path.join(app, 'assets/msix-manifest.xml'))
  fs.writeFileSync(path.join(app, 'build/agent-payload/manifest.json'), JSON.stringify({ launchers: ['hermes'] }))
  fs.writeFileSync(path.join(app, 'build/msix-extensions.xml'), '<Extensions />')
  fs.symlinkSync(path.join(desktop, 'scripts'), path.join(app, 'scripts'), 'junction')
  return app
}

test('store, release, canary and channel MSIX manifests keep write virtualisation disabled', () => {
  const store = taggedApp('v0.27.1')
  const canary = taggedApp('v0.27.1+canary.20260907T001800Z')
  fs.mkdirSync(path.join(canary, 'build'), { recursive: true })
  fs.copyFileSync(path.join(canary, 'assets/msix-manifest.xml'), path.join(canary, 'build/msix-manifest.xml'))
  const channel = channelApp()
  stageChannelManifest(channel, {
    windowsVersion: '0.0.7.0', identity: { appNamePascal: 'HermesChannelab12', displayName: 'Hermes preview' }
  })
  const manifests = {
    store: stageStoreManifest(store, 'v0.27.1'),
    release: stageReleaseManifest(taggedApp('v0.27.1'), 'v0.27.1'),
    canary: stageReleaseManifest(canary, 'v0.27.1+canary.20260907T001800Z'),
    channel: path.join(channel, 'build/msix-manifest.xml')
  }
  for (const [kind, file] of Object.entries(manifests)) {
    expect([kind, virtualization(fs.readFileSync(file, 'utf8'))]).toEqual([kind, { fileSystem: 'disabled', registry: 'disabled' }])
  }
})

test('the builder config requests unvirtualizedResources for MSIX packages', () => {
  const config = require('../electron-builder.config.cjs')
  expect(config.msix.capabilities).toContain('unvirtualizedResources')
})

import assert from 'node:assert/strict'
import { execFileSync, spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { ensureWindowsBundleTools } from './windows-bundle-tools.mjs'

const repo = path.resolve(import.meta.dirname, '../../..')
const sha256 = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex')

function run(command, args, cwd, env) {
  return execFileSync(command, args, { cwd, env, encoding: 'utf8', timeout: 60_000, stdio: ['ignore', 'pipe', 'pipe'] })
}

function fixture(kit) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'commit-msix-'))
  const desktop = path.join(root, 'apps/desktop')
  fs.mkdirSync(path.join(desktop, 'scripts'), { recursive: true })
  fs.mkdirSync(path.join(root, 'scripts'))
  for (const file of ['stage-msixbundle.mjs', 'bundle-store-msixbundle.mjs', 'msix-shared.mjs', 'release-content-types.json']) {
    fs.copyFileSync(path.join(repo, 'scripts', file), path.join(root, 'scripts', file))
  }
  fs.copyFileSync(path.join(repo, 'apps/desktop/scripts/windows-bundle-tools.mjs'), path.join(desktop, 'scripts/windows-bundle-tools.mjs'))
  fs.copyFileSync(path.join(repo, 'apps/desktop/product-identity.cjs'), path.join(desktop, 'product-identity.cjs'))
  fs.writeFileSync(path.join(desktop, 'package.json'), JSON.stringify({ name: 'fixture', version: '0.21.1' }))
  fs.writeFileSync(path.join(desktop, 'electron-builder.config.cjs'), `module.exports=${JSON.stringify({ directories: { buildResources: kit }, toolsets: { winCodeSign: { url: 'file://' + kit } } })}\n`)
  fs.symlinkSync(path.join(repo, 'node_modules'), path.join(root, 'node_modules'), 'junction')
  const env = Object.fromEntries(Object.entries(process.env).filter(([key]) =>
    !/^(AZURE_|CLOUDFLARE_|HERMES_|GITHUB_|GH_|NODE_OPTIONS$)/i.test(key)))
  Object.assign(env, { HERMES_PAYLOAD_TAG: '', CI: '', GIT_CONFIG_GLOBAL: path.join(root, 'git-config'),
    GIT_CONFIG_NOSYSTEM: '1', GIT_AUTHOR_NAME: 'Fixture', GIT_AUTHOR_EMAIL: 'fixture@example.invalid',
    GIT_COMMITTER_NAME: 'Fixture', GIT_COMMITTER_EMAIL: 'fixture@example.invalid',
    GIT_AUTHOR_DATE: '2026-09-10T10:20:30Z', GIT_COMMITTER_DATE: '2026-09-10T10:20:30Z' })
  for (const args of [['init', '-q'], ['add', 'scripts', 'apps'], ['-c', 'commit.gpgsign=false', 'commit', '-qm', 'fixture']]) run('git', args, root, env)
  const commit = run('git', ['rev-parse', 'HEAD'], root, env).trim()
  env.HERMES_BUILD_COMMIT = commit
  env.HERMES_PAYLOAD_VERSION = '0.21.1'
  const release = path.join(desktop, 'release')
  fs.mkdirSync(release)
  return { root, desktop, env, commit, release }
}

function identity(root, env, variant) {
  return JSON.parse(run(process.execPath, ['--input-type=module', '-e',
    "import{appIdentity}from'./scripts/msix-shared.mjs';console.log(JSON.stringify(appIdentity(process.cwd()+'/apps/desktop','')))"],
  root, { ...env, HERMES_DESKTOP_VARIANT: variant }))
}

function makePackage(makeappx, root, release, metadata, variant, arch, env) {
  const content = path.join(root, `${variant}-${arch}`)
  fs.mkdirSync(content)
  fs.mkdirSync(path.join(content, 'assets'))
  for (const name of ['StoreLogo.png', 'Square150x150Logo.png', 'Square44x44Logo.png']) {
    fs.copyFileSync(path.join(repo, 'apps/desktop/assets/appx', name), path.join(content, 'assets', name))
  }
  const name = metadata.identity.store ? metadata.identity.storeMsix.identityName : metadata.identity.msixAppIdWithOrg
  const publisher = metadata.identity.store ? metadata.identity.storeMsix.publisher : 'CN=Fixture'
  fs.writeFileSync(path.join(content, 'index.html'), '<!doctype html><title>Assembly fixture</title>Not an installed Hermes application.')
  fs.writeFileSync(path.join(content, 'AppxManifest.xml'), `<?xml version="1.0" encoding="utf-8"?>
<Package xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10" xmlns:uap="http://schemas.microsoft.com/appx/manifest/uap/windows10" IgnorableNamespaces="uap">
  <Identity Name="${name}" Publisher="${publisher}" Version="${metadata.version}" ProcessorArchitecture="${arch}" />
  <Properties><DisplayName>Assembly fixture</DisplayName><PublisherDisplayName>Fixture</PublisherDisplayName><Logo>assets\\StoreLogo.png</Logo></Properties>
  <Resources><Resource Language="en-us" /></Resources>
  <Dependencies><TargetDeviceFamily Name="Windows.Desktop" MinVersion="10.0.22621.0" MaxVersionTested="10.0.26100.0" /></Dependencies>
  <Applications><Application Id="Fixture" StartPage="index.html"><uap:VisualElements DisplayName="Assembly fixture" Description="No installed application" BackgroundColor="transparent" Square150x150Logo="assets\\Square150x150Logo.png" Square44x44Logo="assets\\Square44x44Logo.png" /></Application></Applications>
</Package>\n`)
  const file = path.join(release, `${variant === 'store' ? 'Store-' : ''}${metadata.name}-${metadata.fileVersion}-win-${arch}.msix`)
  run(makeappx, ['pack', '/o', '/d', content, '/p', file], root, env)
  assert.ok(fs.statSync(file).size > 0)
  return file
}

test.runIf(process.platform === 'win32')('commit assembly preserves per-arch packages and rejects Store bundles', { timeout: 180_000 }, async () => {
  const tools = await ensureWindowsBundleTools()
  const { root, env, commit, release } = fixture(path.dirname(path.dirname(tools.makeappx)))
  try {
    const info = identity(root, env, 'bundled')
    assert.throws(() => identity(root, env, 'store'), /Store.*stable/)
    const inputs = {}
    for (const arch of ['x64', 'arm64']) {
      const file = makePackage(tools.makeappx, root, release, info, 'bundled', arch, env)
      inputs[file] = sha256(file)
    }
    const script = path.join(root, 'scripts/stage-msixbundle.mjs')
    const args = ['--commit', commit, '--version', '0.21.1', '--variant', 'bundled', '--no-upload']
    const result = spawnSync(process.execPath, [script, ...args], { cwd: root, env, encoding: 'utf8', timeout: 60_000 })
    assert.equal(result.status, 0, result.stdout + result.stderr)
    const bundle = path.join(release, `${info.name}-${info.version}-win.msixbundle`)
    assert.ok(fs.statSync(bundle).size > 0)
    const expanded = path.join(root, 'expanded')
    run(tools.makeappx, ['unbundle', '/o', '/p', bundle, '/d', expanded], root, env)
    const xml = fs.readFileSync(path.join(expanded, 'AppxMetadata/AppxBundleManifest.xml'), 'utf8')
    const identityTag = /<Identity\b[^>]*>/.exec(xml)[0]
    assert.ok(identityTag.includes(`Version="${info.version}"`), xml)
    const bundledNames = [...xml.matchAll(/FileName="([^"]+\.msix)"/g)].map(match => match[1]).sort()
    const expected = Object.keys(inputs).map(file => path.basename(file)).sort()
    assert.deepEqual(bundledNames, expected)
    for (const name of expected) assert.equal(sha256(path.join(expanded, name)), inputs[path.join(release, name)])
    const digest = sha256(bundle)
    const modified = fs.statSync(bundle).mtimeMs
    const invalid = [
      [...args, '--candidate'], [...args, '--tag', 'v0.21.1'], [...args, '--unknown'],
      ['--commit', commit.slice(0, 8), '--version', '0.21.1', '--no-upload'],
      ['--commit', commit, '--version', '1.65536.0', '--no-upload'],
      args.filter(value => value !== '--no-upload'),
    ]
    for (const badArgs of invalid) {
      const rejected = spawnSync(process.execPath, [script, ...badArgs], { cwd: root, env, encoding: 'utf8', timeout: 30_000 })
      assert.notEqual(rejected.status, 0, `accepted ${badArgs.join(' ')}\n${rejected.stdout}${rejected.stderr}`)
      assert.equal(sha256(bundle), digest)
      assert.equal(fs.statSync(bundle).mtimeMs, modified)
    }
    const before = fs.readdirSync(release).sort()
    const store = spawnSync(process.execPath, [path.join(root, 'scripts/bundle-store-msixbundle.mjs'), '--commit', commit], { cwd: root, env, encoding: 'utf8' })
    assert.notEqual(store.status, 0)
    assert.match(store.stderr, /Unknown option.*commit/)
    assert.deepEqual(fs.readdirSync(release).sort(), before)
    const missing = path.join(release, expected[0])
    fs.renameSync(missing, `${missing}.held`)
    try {
      const refused = spawnSync(process.execPath, [script, ...args], { cwd: root, env, encoding: 'utf8', timeout: 30_000 })
      assert.notEqual(refused.status, 0)
      assert.match(refused.stdout + refused.stderr, /need both per-arch/)
      assert.equal(sha256(bundle), digest)
      assert.equal(fs.statSync(bundle).mtimeMs, modified)
    } finally {
      fs.renameSync(`${missing}.held`, missing)
    }
    assert.equal(fs.readdirSync(release).some(name => name.endsWith('.appinstaller')), false)
    for (const [file, inputDigest] of Object.entries(inputs)) assert.equal(sha256(file), inputDigest)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

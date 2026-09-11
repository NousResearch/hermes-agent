import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test } from 'vitest'
import { spawnSync } from 'node:child_process'
import * as stamps from './write-build-stamp.mjs'
import * as msix from './before-build.mjs'
const { appExecutionAliasExtensions } = msix

test('the Store bundler rejects nonstable selectors before platform tools or staging', () => {
  for (const args of [['--commit', 'a'.repeat(40)], ['--tag', 'v0.28.0-canary.20260818']]) {
    const result = spawnSync(process.execPath, [path.resolve(import.meta.dirname, '../../../scripts/bundle-store-msixbundle.mjs'), ...args], {
      encoding: 'utf8', env: { ...process.env, HERMES_BUILD_COMMIT: '', HERMES_PAYLOAD_TAG: '' }
    })
    assert.notEqual(result.status, 0)
    assert.match(result.stderr, /Unknown option.*commit|requires a stable release tag/)
  }
})

test('the embedded Python checkout receives the same commit provenance as Electron', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'desktop-stamp-'))
  try {
    const commit = 'a'.repeat(40)
    const repo = path.join(root, 'agent-payload/hermes-agent')
    fs.mkdirSync(repo, { recursive: true })
    const built = stamps.buildStampPayload({ commit, source: 'local', branch: 'main' }, {
      HERMES_DESKTOP_VARIANT: 'bundled', HERMES_BUILD_COMMIT: commit, HERMES_PAYLOAD_VERSION: '0.28.0'
    }, 'win32', { runtime: { repoDir: 'hermes-agent', commands: { hermes: 'bin/hermes-aaaaaaa.exe' } } })
    stamps.writeDesktopStamp(root, built)
    const pythonStamp = JSON.parse(fs.readFileSync(path.join(repo, 'install-stamp.json'), 'utf8'))
    assert.deepEqual(pythonStamp, JSON.parse(fs.readFileSync(path.join(root, 'install-stamp.json'), 'utf8')))
    assert.equal(pythonStamp.source, 'commit-build')
    assert.equal(pythonStamp.commit, commit)
    assert.equal(pythonStamp.tag, null)
    assert.equal(pythonStamp.baseVersion, '0.28.0')
    assert.equal(pythonStamp.displayVersion, '0.28.0')
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('nonstable MSIX CLI aliases each activate their own entrypoint, never the GUI', () => {
  const launchers = ['hermes-canary', 'hermes-canary-acp']
  const applications = msix.appExecutionAliasApplications(launchers, {
    appNamePascal: 'HermesBundledCanary', displayName: 'Hermes Agent Canary'
  })
  const apps = applications.match(/<Application[\s\S]*?<\/Application>/g)
  assert.equal(apps.length, launchers.length)
  for (const [index, name] of launchers.entries()) {
    assert.ok(apps[index].includes(`Executable="app\\resources\\agent-payload\\bin\\${name}.exe"`))
    assert.ok(apps[index].includes(`Alias="${name}.exe"`))
    assert.equal((apps[index].match(/<uap5:ExecutionAlias /g) || []).length, 1)
    assert.ok(apps[index].includes('AppListEntry="none"'))
  }
})

test('desktop payload exposes only identity-qualified launchers while preserving canonical command keys', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'desktop-launchers-'))
  try {
    for (const cliName of ['hermes-canary', 'hermes-abcdef1', 'hermes-1234567']) {
      const payload = path.join(root, cliName)
      fs.mkdirSync(path.join(payload, 'bin'), { recursive: true })
      const commands = { hermes: 'bin/hermes.exe', 'hermes-acp': 'bin/hermes-acp.exe' }
      for (const [name, file] of Object.entries(commands)) fs.writeFileSync(path.join(payload, file), `PE bytes for ${name}`)
      fs.writeFileSync(path.join(payload, 'manifest.json'), JSON.stringify({
        target: 'win32-x64', launchers: Object.keys(commands), runtime: { commands }
      }))
      const manifest = stamps.stageDesktopLaunchers(payload, { cliName })
      assert.deepEqual(manifest.launchers, [cliName, `${cliName}-acp`])
      assert.deepEqual(Object.keys(manifest.runtime.commands), Object.keys(commands))
      for (const [name, file] of Object.entries(manifest.runtime.commands)) {
        assert.equal(fs.readFileSync(path.join(payload, file), 'utf8'), `PE bytes for ${name}`)
        assert.ok(file.includes(cliName))
      }
      assert.equal(fs.existsSync(path.join(payload, 'bin/hermes.exe')), false)
      assert.equal(fs.existsSync(path.join(payload, 'bin/hermes-acp.exe')), false)
      assert.deepEqual(stamps.stageDesktopLaunchers(payload, { cliName }), manifest)
      const xml = appExecutionAliasExtensions(manifest.launchers)
      assert.ok(xml.includes(`Alias="${cliName}.exe"`))
      assert.ok(!xml.includes('Alias="hermes.exe"'))
    }
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

import { expect, test } from 'vitest'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { execFileSync, spawnSync } from 'node:child_process'
import { bundleIdentity, verifyBundleStamp, verifyMacMetadata } from '../tests/install/e2e-assets/bundle-smoke-metadata.mjs'
import { stampAssertions } from '../tests/install/e2e-assets/mac-bundled-manifest.cjs'
import { randomBytes } from 'node:crypto'
import { buildStampPayload } from '../apps/desktop/scripts/write-build-stamp.mjs'

const commit = 'a'.repeat(40)

test('fresh-install provenance admits tagless commit stamps without relaxing update acceptance', () => {
  const stamp = { commit, tag: null, source: 'commit-build', branch: null, dirty: false,
    distribution: 'desktop-app', payload: 'bundled', updateMechanism: 'external', displayVersion: '1.2.3' }
  for (const platform of ['darwin', 'win32']) {
    expect(verifyBundleStamp(stamp, { commit, platform })).toBe('1.2.3')
    for (const [key, value] of [['commit', 'b'.repeat(40)], ['tag', 'v1.2.3'], ['payload', 'light'],
      ['source', 'git'], ['branch', 'main'], ['dirty', true], ['updateMechanism', 'electron-updater']]) {
      expect(() => verifyBundleStamp({ ...stamp, [key]: value }, { commit, platform })).toThrow(key)
    }
    const tag = 'v1.2.3-canary.20260913000000'
    const tagged = { ...stamp, tag, source: 'git', branch: 'main', displayVersion: tag.slice(1),
      updateMechanism: platform === 'darwin' ? 'electron-updater' : 'app-installer' }
    expect(verifyBundleStamp(tagged, { commit, tag, platform })).toBe(tag.slice(1))
    expect(() => verifyBundleStamp({ ...tagged, updateMechanism: 'external' }, { commit, tag, platform })).toThrow('updateMechanism')
  }
  expect(stampAssertions(stamp, { commit, tag: null }).join(';')).toContain('electron-updater')
})

test('identity uses the production bundled variant and exact input tokens, not inherited build flags', () => {
  const stable = bundleIdentity(commit, 'v1.2.3')
  const canary = bundleIdentity(commit, 'v1.2.3-canary.20260913000000')
  const tagless = bundleIdentity(commit)
  expect(stable.appId).not.toBe(canary.appId)
  expect(stable.msixIdentity).not.toBe(tagless.msixIdentity)
  expect(tagless.appId).toContain(commit.slice(0, 7))
  expect(tagless.publisher).toBe(stable.publisher)
  for (const bad of [commit.slice(0, 7), ` ${commit}`, commit.toUpperCase()]) {
    expect(() => bundleIdentity(bad)).toThrow('commit')
  }
  expect(() => bundleIdentity(commit, 'v1.2.3 ')).toThrow('tag')
  const hostileEnvironment = { ...process.env, HERMES_DESKTOP_VARIANT: 'store', HERMES_BUILD_COMMIT: 'b'.repeat(40), HERMES_PAYLOAD_TAG: 'v9.9.9', _HERMES_CHANNEL_REQUEST_JSON: '{"invalid":"inherited"}' }
  const cli = path.resolve(import.meta.dirname, '../tests/install/e2e-assets/bundle-smoke-metadata.mjs')
  expect(JSON.parse(execFileSync(process.execPath, [cli, 'identity', '--commit', commit], { env: hostileEnvironment, encoding: 'utf8' }))).toEqual(tagless)
})

test('channel smoke binds the complete admitted request, not a commit-build identity or display version', () => {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'channel-smoke-'))
  const token = randomBytes(8).toString('hex'), sequence = 65537
  const request = { schema: 1, buildId: randomBytes(16).toString('hex'), channel: `smoke-${token}`,
    repository: 'NousResearch/hermes-agent', publicBase: 'https://releases.example.test', commit,
    sourceVersion: '1.2.3', sequence, version: `0.0.${sequence}`,
    windowsVersion: `0.${Math.floor(sequence / 65536)}.${sequence % 65536}.0`, bundleEnv: {},
    identity: { token, displayName: 'Smoke Channel', appId: `com.example.preview-${token}`,
      appNamePascal: `Smoke${token}`, artifactNamePascal: `Artifact${token}`,
      msixAppIdWithOrg: `Example.Smoke${token}`, cliName: `smoke-${token}`, windowsExecutableName: `smoke-${token}` } }
  const cli = path.resolve(import.meta.dirname, '../tests/install/e2e-assets/bundle-smoke-metadata.mjs')
  const requestPath = path.join(temp, 'request.json'), stampPath = path.join(temp, 'stamp.json')
  const run = (command, args = []) => spawnSync(process.execPath,
    [cli, command, '--commit', commit, '--channel-request', requestPath, ...args], { encoding: 'utf8' })
  try {
    fs.writeFileSync(requestPath, JSON.stringify(request))
    const identity = run('identity')
    expect(identity.status, identity.stderr).toBe(0)
    expect(JSON.parse(identity.stdout)).toMatchObject({ appId: request.identity.appId,
      msixIdentity: request.identity.msixAppIdWithOrg, applicationId: request.identity.appNamePascal,
      windowsVersion: request.windowsVersion })
    const env = { ...process.env, HERMES_DESKTOP_VARIANT: 'bundled', HERMES_BUILD_COMMIT: '',
      HERMES_PAYLOAD_TAG: '', HERMES_PAYLOAD_VERSION: '', _HERMES_CHANNEL_REQUEST_JSON: JSON.stringify(request) }
    for (const platform of ['darwin', 'win32']) {
      const stamp = buildStampPayload({ commit, dirty: false }, env, platform,
        { runtime: { commands: { hermes: 'bin/hermes' } } })
      fs.writeFileSync(stampPath, JSON.stringify(stamp))
      const result = run('stamp', ['--platform', platform, '--stamp', stampPath])
      expect(result.status, result.stderr).toBe(0)
      expect(JSON.parse(result.stdout)).toBe(platform === 'darwin' ? request.version : request.windowsVersion)
      const options = { commit, platform, channelRequest: request }
      for (const key of Object.keys(request)) {
        const changed = { ...stamp, channelBuild: { ...request, [key]: null } }
        expect(() => verifyBundleStamp(changed, options), key).toThrow('channelBuild')
      }
      for (const [key, value] of [['source', 'commit-build'], ['tag', 'v1.2.3'], ['branch', 'main'],
        ['dirty', true], ['updateMechanism', 'external'], ['displayVersion', request.version], ['baseVersion', request.version]]) {
        expect(() => verifyBundleStamp({ ...stamp, [key]: value }, options), key).toThrow(key)
      }
      expect(() => verifyBundleStamp(stamp, { commit, platform })).toThrow()
      if (platform === 'darwin') {
        const plist = { CFBundleIdentifier: request.identity.appId, CFBundleShortVersionString: request.version,
          CFBundleVersion: request.version, CFBundleExecutable: request.identity.displayName }
        expect(verifyMacMetadata(plist, stamp, options)).toBe(request.version)
        for (const key of ['CFBundleIdentifier', 'CFBundleShortVersionString', 'CFBundleVersion']) {
          expect(() => verifyMacMetadata({ ...plist, [key]: 'wrong' }, stamp, options)).toThrow(key)
        }
        expect(stampAssertions(stamp, { commit, tag: null, channelRequest: request })).toEqual([])
        expect(stampAssertions(stamp, { commit, tag: null }).join(';')).toContain('channelBuild')
        expect(stampAssertions({ ...stamp, channelBuild: { ...request, bundleEnv: { HERMES_MODEL: 'other' } } },
          { commit, tag: null, channelRequest: request }).join(';')).toContain('channelBuild')
      }
    }
    expect(run('identity', ['--tag', 'v1.2.3']).status).not.toBe(0)
    for (const raw of ['', JSON.stringify(request).replace('{', '{"channel":"duplicate",')]) {
      fs.writeFileSync(requestPath, raw)
      expect(run('identity').status).not.toBe(0)
    }
    for (const invalid of [{ ...request, commit: 'b'.repeat(40) }, { ...request, channel: 'smoke/invalid' },
      { ...request, windowsVersion: '1.2.3.0' }, { ...request, identity: { token } }]) {
      fs.writeFileSync(requestPath, JSON.stringify(invalid))
      expect(run('identity').status).not.toBe(0)
    }
  } finally { fs.rmSync(temp, { recursive: true, force: true }) }
}, 15_000)

test('workspace admission rejects reuse and symlink escapes before creating anything outside runner temp', () => {
  const temp = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'smoke-paths-')))
  const cli = path.resolve(import.meta.dirname, '../tests/install/e2e-assets/bundle-smoke-metadata.mjs')
  const run = (work, out) => spawnSync(process.execPath, [cli, 'prepare', '--work', work, '--out', out],
    { env: { ...process.env, RUNNER_TEMP: temp }, encoding: 'utf8' })
  try {
    const work = path.join(temp, 'work'), out = path.join(temp, 'out')
    expect(run(work, out).status).toBe(0)
    expect(run(work, out).status).toBe(1)
    expect(run(path.join(temp, '..', 'escape'), out).status).toBe(1)
    expect(run(path.join(temp, 'another-work'), path.join(temp, 'another-work')).status).toBe(1)
    fs.symlinkSync(os.tmpdir(), path.join(temp, 'link'), process.platform === 'win32' ? 'junction' : 'dir')
    expect(run(path.join(temp, 'link', 'escape'), out).status).toBe(1)
    const marker = path.join(work, 'keep')
    fs.writeFileSync(marker, 'untouched')
    expect(run(work, out).status).toBe(1)
    expect(fs.readFileSync(marker, 'utf8')).toBe('untouched')
    const alias = path.join(temp, 'alias')
    fs.symlinkSync(temp, alias, process.platform === 'win32' ? 'junction' : 'dir')
    expect(run(path.join(alias, 'fresh'), out).status).toBe(0)
  } finally { fs.rmSync(temp, { recursive: true, force: true }) }
})

test.runIf(process.platform === 'win32')('PowerShell admits only the pinned package and native slice', () => {
  const script = path.resolve(import.meta.dirname, '../tests/install/e2e-assets/windows-bundle-metadata.test.ps1')
  expect(execFileSync('powershell.exe', ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', script],
    { encoding: 'utf8' })).toContain('PASS: MSIX identity/executable and MSIXBUNDLE native-slice admission')
})
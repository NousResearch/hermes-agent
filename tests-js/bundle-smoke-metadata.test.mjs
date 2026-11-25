import { expect, test } from 'vitest'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { execFileSync, spawnSync } from 'node:child_process'
import { bundleIdentity, shouldVerifyMacDmg, verifyBundleStamp, verifyMacDmg, verifyMacMetadata } from '../tests/install/e2e-assets/bundle-smoke-metadata.mjs'
import { stampAssertions } from '../tests/install/e2e-assets/mac-bundled-manifest.cjs'
import { randomBytes } from 'node:crypto'
import { buildStampPayload } from '../apps/desktop/scripts/write-build-stamp.mjs'
import { channelBuildRequest, releaseVersion } from '../scripts/msix-shared.mjs'

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
    const tag = 'canary.20260913T000000Z-v1.2.3'
    const tagged = { ...stamp, tag, source: 'git', branch: 'main', displayVersion: releaseVersion(tag),
      updateMechanism: platform === 'darwin' ? 'electron-updater' : 'app-installer' }
    expect(verifyBundleStamp(tagged, { commit, tag, platform })).toBe(releaseVersion(tag))
    expect(() => verifyBundleStamp({ ...tagged, updateMechanism: 'external' }, { commit, tag, platform })).toThrow('updateMechanism')
  }
  expect(stampAssertions(stamp, { commit, tag: null }).join(';')).toContain('electron-updater')
})

test('identity uses the production bundled variant and exact input tokens, not inherited build flags', () => {
  const stable = bundleIdentity(commit, 'v1.2.3')
  const canary = bundleIdentity(commit, 'canary.20260913T000000Z-v1.2.3')
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

test.skipIf(process.platform === 'win32')('channel admission validates each distinct request once and re-checks every rejection', () => {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'channel-admit-'))
  try {
    const log = path.join(temp, 'calls'), python = path.join(temp, 'python')
    const quote = value => `'${value.replaceAll("'", "'\\''")}'`
    fs.writeFileSync(python, `#!/bin/sh\necho x >> ${quote(log)}\nexec ${quote(process.env.HERMES_PYTHON || 'python')} "$@"\n`,
      { mode: 0o755 })
    const calls = () => fs.existsSync(log) ? fs.readFileSync(log, 'utf8').split('\n').filter(Boolean).length : 0
    const token = randomBytes(8).toString('hex')
    const request = { schema: 1, buildId: randomBytes(16).toString('hex'), channel: `smoke-${token}`,
      repository: 'NousResearch/hermes-agent', publicBase: 'https://releases.example.test', commit,
      sourceVersion: '1.2.3', sequence: 1, version: '0.0.1', windowsVersion: '0.0.1.0', bundleEnv: {},
      identity: { token, displayName: 'Smoke Channel', appId: `com.example.preview-${token}`,
        appNamePascal: `Smoke${token}`, artifactNamePascal: `Artifact${token}`,
        msixAppIdWithOrg: `Example.Smoke${token}`, cliName: `smoke-${token}`, windowsExecutableName: `smoke-${token}` } }
    const admit = value => channelBuildRequest({ HERMES_PYTHON: python, HERMES_DESKTOP_VARIANT: 'bundled',
      PATH: process.env.PATH, _HERMES_CHANNEL_REQUEST_JSON: JSON.stringify(value) })
    expect(admit(request)).toEqual(request)
    expect(admit(request)).toEqual(request)
    expect(calls()).toBe(1)
    const insecure = { ...request, publicBase: 'http://releases.example.test' }
    expect(() => admit(insecure)).toThrow()
    expect(() => admit(insecure)).toThrow()
    expect(calls()).toBe(3)
    expect(admit({ ...request, buildId: randomBytes(16).toString('hex') }).channel).toBe(request.channel)
    expect(calls()).toBe(4)
  } finally { fs.rmSync(temp, { recursive: true, force: true }) }
})

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

test('shouldVerifyMacDmg enforces outer DMG checks on all releases/channels and signedDmg commit claims while allowing older commit claims', () => {
  expect(shouldVerifyMacDmg({ tag: 'v1.2.3', stamp: {} })).toBe(true)
  expect(shouldVerifyMacDmg({ tag: 'canary.20261125T000000Z-v1.2.3', stamp: {} })).toBe(true)
  expect(shouldVerifyMacDmg({ channelRequest: { channel: 'preview' }, stamp: {} })).toBe(true)
  expect(shouldVerifyMacDmg({ tag: 'commit.20261125T000000Z-v1.2.3', stamp: { signedDmg: true } })).toBe(true)
  expect(shouldVerifyMacDmg({ tag: '', stamp: { signedDmg: true } })).toBe(true)
  expect(shouldVerifyMacDmg({ tag: 'commit.20261125T000000Z-v1.2.3', stamp: {} })).toBe(false)
  expect(shouldVerifyMacDmg({ tag: '', stamp: {} })).toBe(false)

  const commit = 'b'.repeat(40)
  const runtime = { commands: { hermes: 'bin/hermes' } }
  expect(buildStampPayload({ commit, dirty: false }, { HERMES_DESKTOP_VARIANT: 'bundled' }, 'darwin', { runtime }).signedDmg).toBe(true)
  expect(buildStampPayload({ commit, dirty: false }, { HERMES_DESKTOP_VARIANT: 'bundled' }, 'win32', { runtime }).signedDmg).toBeUndefined()
  expect(buildStampPayload({ commit, dirty: false }, { HERMES_DESKTOP_VARIANT: 'light' }, 'darwin', { runtime }).signedDmg).toBeUndefined()
})

test('verifyMacDmg enforces codesign, team ID, stapler ticket, and spctl primary-signature open policy', () => {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'dmg-smoke-'))
  try {
    const dmg = path.join(temp, 'HermesBundled-1.2.3-mac-arm64.dmg')
    fs.writeFileSync(dmg, 'dmg-fixture')
    const calls = []
    const exec = (cmd, args) => { calls.push([cmd, ...args]) }
    const spawn = (cmd, args) => {
      calls.push([cmd, ...args])
      return { status: 0, stderr: 'Authority=Developer ID Application: Nous Research (TEAM123456)\nTeamIdentifier=TEAM123456\n' }
    }
    expect(verifyMacDmg(dmg, { exec, spawn })).toEqual({ dmg, teamIdentifier: 'TEAM123456' })
    expect(calls).toEqual([
      ['/usr/bin/codesign', '--verify', '--verbose=2', dmg],
      ['/usr/bin/codesign', '-dv', '--verbose=4', dmg],
      ['/usr/bin/xcrun', 'stapler', 'validate', dmg],
      ['/usr/sbin/spctl', '-a', '-vv', '-t', 'open', '--context', 'context:primary-signature', dmg],
    ])

    expect(() => verifyMacDmg(path.join(temp, 'missing.dmg'), { exec, spawn })).toThrow(/existing \.dmg/)
    expect(() => verifyMacDmg(path.join(temp, 'app.zip'), { exec, spawn })).toThrow(/existing \.dmg/)
    expect(() => verifyMacDmg(dmg, {
      exec,
      spawn: () => ({ status: 0, stderr: 'TeamIdentifier=not set\n' }),
    })).toThrow(/signing team/)
    expect(() => verifyMacDmg(dmg, {
      exec,
      spawn: () => ({ status: 1, stderr: 'code object is not signed at all' }),
    })).toThrow(/codesign display failed/)
    expect(() => verifyMacDmg(dmg, {
      exec: (cmd) => { if (cmd === '/usr/sbin/spctl') throw new Error('rejected') },
      spawn,
    })).toThrow(/rejected/)
  } finally {
    fs.rmSync(temp, { recursive: true, force: true })
  }
})

test.runIf(process.platform === 'win32')('PowerShell admits only the pinned package and native slice', () => {
  const script = path.resolve(import.meta.dirname, '../tests/install/e2e-assets/windows-bundle-metadata.test.ps1')
  expect(execFileSync('powershell.exe', ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', script],
    { encoding: 'utf8' })).toContain('PASS: MSIX identity/executable and MSIXBUNDLE native-slice admission')
})
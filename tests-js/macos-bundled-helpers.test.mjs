import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createRequire } from 'node:module'

import { expect, it, describe } from 'vitest'

const require = createRequire(import.meta.url)
const {
  validateBundleManifest,
  codesignTeam,
  channelFromTag,
  stampAssertions,
} = require('../tests/install/e2e-assets/mac-bundled-manifest.cjs')

const side = (over = {}) => ({
  tag: 'v0.28.0',
  version: '0.28.0',
  commit: 'a'.repeat(40),
  identity: 'com.nousresearch.hermes-bundled',
  teamId: 'TEAM123456',
  artifact: { url: 'https://example.com/HermesBundled-0.28.0-mac-arm64.zip', sha256: 'b'.repeat(64), path: '/tmp/old.zip' },
  ...over,
})
const manifest = (over = {}) => ({
  schema: 1,
  platform: 'macos',
  arch: 'arm64',
  old: side(),
  new: side({
    tag: 'v0.29.0',
    version: '0.29.0',
    commit: 'c'.repeat(40),
    artifact: { url: 'https://example.com/HermesBundled-0.29.0-mac-arm64.zip', sha256: 'd'.repeat(64), path: '/tmp/new.zip' },
  }),
  ...over,
})
const want = { platform: 'macos', arch: 'arm64' }

describe('validateBundleManifest', () => {
  it('accepts a well-formed schema-1 pair', () => {
    expect(() => validateBundleManifest(manifest(), want)).not.toThrow()
  })

  it('rejects wrong schema, platform and arch', () => {
    expect(() => validateBundleManifest(manifest({ schema: 2 }), want)).toThrow(/schema/)
    expect(() => validateBundleManifest(manifest({ platform: 'windows' }), want)).toThrow(/platform/)
    expect(() => validateBundleManifest(manifest({ arch: 'x64' }), want)).toThrow(/arch/)
    expect(() => validateBundleManifest(manifest(), { platform: 'macos', arch: 'riscv' })).toThrow(/arch/)
  })

  it('rejects version/tag and commit-shaped lies', () => {
    expect(() => validateBundleManifest(manifest({ old: side({ version: '0.99.0' }) }), want)).toThrow(/version/)
    expect(() => validateBundleManifest(manifest({ old: side({ tag: 'notatag' }) }), want)).toThrow(/tag/)
    expect(() => validateBundleManifest(manifest({ old: side({ commit: 'zz' }) }), want)).toThrow(/commit/)
  })

  it('requires an explicit teamId on BOTH sides (10-char signing team)', () => {
    expect(() => validateBundleManifest(manifest({ old: side({ teamId: undefined }) }), want)).toThrow(/teamId/)
    expect(() => validateBundleManifest(manifest({ old: side({ teamId: 'tooshort' }) }), want)).toThrow(/teamId/)
    expect(() => validateBundleManifest(manifest({ new: side({ teamId: 'OTHER99999', commit: 'c'.repeat(40) }) }), want)).toThrow(/Squirrel/)
  })

  it('treats identity as the exact CFBundleIdentifier, not a namespace', () => {
    expect(() => validateBundleManifest(manifest({ old: side({ identity: 'TEAM123456' }) }), want)).toThrow(/CFBundleIdentifier/)
    expect(() => validateBundleManifest(manifest({ old: side({ identity: 'com.nousresearch.hermes-bundled' }) }), want)).not.toThrow()
    expect(() => validateBundleManifest(
      manifest({ new: side({ identity: 'com.nousresearch.hermes-other', tag: 'v0.29.0', version: '0.29.0', commit: 'c'.repeat(40), artifact: { url: 'u', sha256: 'd'.repeat(64), path: '/tmp/n' } }) }),
      want,
    )).toThrow(/same application bundle/)
  })

  it('requires the resolver to have downloaded the artifact', () => {
    expect(() => validateBundleManifest(
      manifest({ old: side({ artifact: { url: 'x', sha256: 'b'.repeat(64) } }) }), want,
    )).toThrow(/artifact.path/)
    expect(() => validateBundleManifest(
      manifest({ old: side({ artifact: { url: 'x', sha256: 'nothex', path: '/tmp/x' } }) }), want,
    )).toThrow(/sha256/)
  })

  it('rejects a no-op pair and a channel-crossing pair', () => {
    expect(() => validateBundleManifest(manifest({ new: side() }), want)).toThrow(/no update/)
    expect(() => validateBundleManifest(
      manifest({ new: side({ tag: 'v0.29.0-canary.20260907000000', version: '0.29.0-canary.20260907000000', commit: 'c'.repeat(40), artifact: { url: 'u', sha256: 'd'.repeat(64), path: '/tmp/n' } }) }),
      want,
    )).toThrow(/channel/)
  })
})

describe('channelFromTag', () => {
  it('maps canary tags to the canary channel, everything else to stable', () => {
    expect(channelFromTag('v0.29.0-canary.20260907000000')).toBe('canary')
    expect(channelFromTag('v0.29.0')).toBe('stable')
  })
})

describe('codesignTeam', () => {
  it('parses the team from codesign -dv STDERR output', () => {
    const out = [
      'Executable=/tmp/Hermes Bundled.app/Contents/MacOS/Hermes Bundled',
      'Identifier=com.nousresearch.hermes-bundled',
      'TeamIdentifier=TEAM123456',
    ].join('\n')
    expect(codesignTeam(out)).toBe('TEAM123456')
  })
  it('returns null for unsigned / ad-hoc signatures', () => {
    expect(codesignTeam('Identifier=com.x\nTeamIdentifier=not set')).toBeNull()
    expect(codesignTeam('Identifier=com.x\nTeamIdentifier=')).toBeNull()
    expect(codesignTeam('')).toBeNull()
  })
  it('never matches the word "not" as a team id', () => {
    expect(codesignTeam('TeamIdentifier=not set\nFoo=not')).toBeNull()
  })
})

describe('stampAssertions', () => {
  const sideArg = { commit: 'a'.repeat(40), tag: 'v0.28.0' }
  const good = {
    schemaVersion: 2,
    commit: 'a'.repeat(40),
    branch: 'main',
    payload: 'bundled',
    distribution: 'desktop-app',
    updateMechanism: 'electron-updater',
    tag: 'v0.28.0',
  }
  it('accepts the bundled electron-updater stamp', () => {
    expect(stampAssertions(good, sideArg)).toEqual([])
  })
  it('flags wrong mechanism, payload, commit and tag', () => {
    const problems = stampAssertions(
      { ...good, updateMechanism: 'external', payload: 'light', commit: 'b'.repeat(40), tag: null },
      sideArg,
    )
    expect(problems).toHaveLength(4)
  })
  it('rejects non-objects and store submissions', () => {
    expect(stampAssertions(null, sideArg)).toHaveLength(1)
    expect(stampAssertions({ ...good, updateMechanism: 'external' }, sideArg).join(' ')).toMatch(/updateMechanism/)
  })
})

describe('mac-bundled-feed materializer', () => {
  it('emits the production update-feed contract for the real NEW zip', async () => {
    const { materializeFeed, feedFileNames, sha512Base64 } = await import('../tests/install/e2e-assets/mac-bundled-feed.mjs')
    const { createHash } = await import('node:crypto')

    // The feed layout comes from the ONE production contract, not a copy.
    const { darwinFeed } = require('../apps/desktop/update-feed.cjs')
    expect(darwinFeed('stable').directory).toBe('releases/darwin/stable')
    expect(feedFileNames('arm64', 'stable').prefixed).toBe('arm64-stable-mac.yml')
    expect(feedFileNames('x64', 'stable').prefixed).toBe('stable-mac.yml')
    expect(feedFileNames('arm64', 'canary').directory).toBe('releases/darwin/canary')
    expect(feedFileNames('arm64', 'canary').prefixed).toBe('arm64-canary-mac.yml')

    const outDir = fs.mkdtempSync(path.join(os.tmpdir(), 'mac-feed-'))
    const zip = path.join(outDir, 'HermesBundled-0.29.0-mac-arm64.zip')
    const zipBytes = Buffer.from('fake signed zip for feed shape tests')
    fs.writeFileSync(zip, zipBytes)
    const expectedSha512 = createHash('sha512').update(zipBytes).digest('base64')
    // The streamed hash matches a whole-buffer hash (streams, no full read).
    expect(await sha512Base64(zip)).toBe(expectedSha512)

    const receipt = await materializeFeed({
      outDir,
      zipPath: zip,
      version: '0.29.0',
      tag: 'v0.29.0',
      arch: 'arm64',
      releaseDate: '2026-09-07T00:00:00.000Z',
    })

    // The channel comes from the tag, never a hard-coded 'stable'.
    expect(receipt.channel).toBe('stable')
    expect(receipt.artifactUrlPath).toBe('/releases/tag/v0.29.0/HermesBundled-0.29.0-mac-arm64.zip')
    expect(receipt.written).toEqual([
      'releases/darwin/stable/arm64-stable-mac.yml',
      'releases/darwin/stable/stable-mac.yml',
    ])
    const yml = fs.readFileSync(path.join(outDir, 'releases/darwin/stable/arm64-stable-mac.yml'), 'utf8')
    expect(yml).toContain('version: 0.29.0')
    expect(yml).toContain('  - url: /releases/tag/v0.29.0/HermesBundled-0.29.0-mac-arm64.zip')
    expect(yml).toContain(`    sha512: ${expectedSha512}`)
    expect(yml).toContain(`    size: ${zipBytes.length}`)
    expect(yml).toContain(`path: /releases/tag/v0.29.0/HermesBundled-0.29.0-mac-arm64.zip`)
    expect(yml).toContain(`sha512: ${expectedSha512}`)
    // The served artifact bytes are the real NEW zip, copied verbatim.
    expect(fs.readFileSync(path.join(outDir, 'releases/tag/v0.29.0/HermesBundled-0.29.0-mac-arm64.zip')))
      .toEqual(zipBytes)
  })

  it('serves canary tags from the canary channel directory', async () => {
    const { materializeFeed } = await import('../tests/install/e2e-assets/mac-bundled-feed.mjs')
    const outDir = fs.mkdtempSync(path.join(os.tmpdir(), 'mac-feed-canary-'))
    const zip = path.join(outDir, 'HermesBundled-0.30.0-canary-mac-arm64.zip')
    fs.writeFileSync(zip, 'canary zip')
    const receipt = await materializeFeed({
      outDir, zipPath: zip, version: '0.30.0',
      tag: 'v0.30.0-canary.20260907000000', arch: 'x64',
      releaseDate: '2026-09-07T00:00:00.000Z',
    })
    expect(receipt.channel).toBe('canary')
    expect(receipt.feedKey).toBe('releases/darwin/canary/canary-mac.yml')
    expect(fs.existsSync(path.join(outDir, 'releases/darwin/canary/canary-mac.yml'))).toBe(true)
    expect(fs.existsSync(path.join(outDir, 'releases/darwin/canary/arm64-canary-mac.yml'))).toBe(false)
  })
})

describe('mac-bundled-serve path safety', () => {
  it('resolves in-root paths and rejects traversal outside the feed root', async () => {
    const { safeJoin } = await import('../tests/install/e2e-assets/mac-bundled-serve.mjs')
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'mac-serve-'))
    expect(safeJoin(root, '/releases/darwin/stable/stable-mac.yml'))
      .toBe(path.join(root, 'releases/darwin/stable/stable-mac.yml'))
    // Classic traversal: the old startsWith(root) check passed these.
    expect(safeJoin(root, '/../escape.yml')).toBeNull()
    expect(safeJoin(root, '/%2e%2e/escape.yml')).toBeNull()
    expect(safeJoin(root, '/..%2fescape.yml')).toBeNull()
  })
})

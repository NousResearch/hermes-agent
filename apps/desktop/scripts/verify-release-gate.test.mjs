import assert from 'node:assert/strict'
import { describe, test } from 'vitest'

import { classifyArtifact } from '../scripts/release-manifest.mjs'
import {
  evaluateGate,
  MIN_ARTIFACT_BYTES,
  normalizeVersion,
  renderReport,
  REQUIRED_ARTIFACTS
} from '../scripts/verify-release-gate.mjs'

const BIG = MIN_ARTIFACT_BYTES * 40

function artifact(overrides) {
  return {
    file: 'x',
    platform: 'win',
    arch: 'x64',
    kind: 'nsis',
    signable: true,
    signed: true,
    size: BIG,
    sha256: 'deadbeef',
    ...overrides
  }
}

/** A release that should pass cleanly, so each test can break exactly one thing. */
function goodManifest() {
  return {
    product: 'AI Evolution Jarvis',
    version: '0.17.2',
    signing: { windows: 'azure', macos: 'api-key' },
    artifacts: [
      artifact({ file: 'jarvis-win-x64.exe', kind: 'nsis' }),
      artifact({ file: 'jarvis-win-x64.msi', kind: 'msi' }),
      artifact({ file: 'jarvis-mac-arm64.dmg', platform: 'mac', arch: 'arm64', kind: 'dmg' }),
      artifact({ file: 'jarvis-mac-x64.dmg', platform: 'mac', arch: 'x64', kind: 'dmg' }),
      artifact({ file: 'jarvis-mac-arm64.zip', platform: 'mac', arch: 'arm64', kind: 'zip', signable: false, signed: false }),
      artifact({ file: 'jarvis-mac-x64.zip', platform: 'mac', arch: 'x64', kind: 'zip', signable: false, signed: false }),
      artifact({ file: 'jarvis-linux-x64.AppImage', platform: 'linux', kind: 'appimage', signable: false, signed: false }),
      artifact({ file: 'jarvis-linux-x64.deb', platform: 'linux', kind: 'deb', signable: false, signed: false }),
      artifact({ file: 'jarvis-linux-x64.rpm', platform: 'linux', kind: 'rpm', signable: false, signed: false })
    ]
  }
}

describe('normalizeVersion', () => {
  test('a v-prefixed tag and a bare version are the same release', () => {
    assert.equal(normalizeVersion('v0.17.2'), '0.17.2')
    assert.equal(normalizeVersion(' 0.17.2 '), '0.17.2')
    assert.equal(normalizeVersion('v2026.3.15.1'), '2026.3.15.1')
    assert.equal(normalizeVersion(undefined), '')
  })
})

describe('evaluateGate', () => {
  test('a complete signed release passes', () => {
    const result = evaluateGate({ manifest: goodManifest(), version: 'v0.17.2' })
    assert.deepEqual(result.problems, [])
    assert.equal(result.ok, true)
  })

  // Building 0.17.2 and tagging 0.18.0 is silent today; it must not be.
  test('a tag that disagrees with the build fails', () => {
    const result = evaluateGate({ manifest: goodManifest(), version: 'v0.18.0' })
    assert.equal(result.ok, false)
    assert.ok(result.problems.some(p => /version mismatch/.test(p)))
  })

  test('an unsigned Windows installer fails', () => {
    const manifest = goodManifest()
    manifest.artifacts[0].signed = false

    const result = evaluateGate({ manifest, version: '0.17.2' })
    assert.ok(result.problems.some(p => /jarvis-win-x64\.exe is unsigned/.test(p)))
  })

  test('--allow-unsigned is honoured for dry runs', () => {
    const manifest = goodManifest()
    manifest.artifacts[0].signed = false

    assert.equal(evaluateGate({ manifest, version: '0.17.2', allowUnsigned: true }).ok, true)
  })

  // An Intel-only dmg excludes every Mac sold since 2020.
  test('a missing arm64 dmg fails even though a dmg exists', () => {
    const manifest = goodManifest()
    manifest.artifacts = manifest.artifacts.filter(a => a.file !== 'jarvis-mac-arm64.dmg')

    const result = evaluateGate({ manifest, version: '0.17.2' })
    assert.ok(result.problems.some(p => /missing mac dmg for arm64/.test(p)))
  })

  test('a universal build satisfies both mac slices', () => {
    const manifest = goodManifest()
    manifest.artifacts = manifest.artifacts.filter(a => !(a.platform === 'mac' && a.kind === 'dmg'))
    manifest.artifacts.push(
      artifact({ file: 'jarvis-mac-universal.dmg', platform: 'mac', arch: 'universal', kind: 'dmg' })
    )

    const result = evaluateGate({ manifest, version: '0.17.2' })
    assert.deepEqual(result.problems, [])
  })

  test('a missing msi fails', () => {
    const manifest = goodManifest()
    manifest.artifacts = manifest.artifacts.filter(a => a.kind !== 'msi')

    assert.ok(evaluateGate({ manifest, version: '0.17.2' }).problems.some(p => /missing win msi/.test(p)))
  })

  test('a platform with no artifacts at all is named once, clearly', () => {
    const manifest = goodManifest()
    manifest.artifacts = manifest.artifacts.filter(a => a.platform !== 'linux')

    const result = evaluateGate({ manifest, version: '0.17.2' })
    assert.ok(result.problems.some(p => /no linux artifacts/.test(p)))
  })

  test('a per-architecture leg only owes its own slice', () => {
    const manifest = goodManifest()
    manifest.artifacts = manifest.artifacts.filter(a => a.platform !== 'mac' || a.arch === 'arm64')

    assert.ok(evaluateGate({ manifest, version: '0.17.2', platforms: ['mac'] }).problems.some(p => /for x64/.test(p)))
    assert.deepEqual(evaluateGate({ manifest, version: '0.17.2', platforms: ['mac'], archs: ['arm64'] }).problems, [])
  })

  test('narrowing --platforms skips the legs that were not built', () => {
    const manifest = goodManifest()
    manifest.artifacts = manifest.artifacts.filter(a => a.platform === 'linux')

    const result = evaluateGate({ manifest, version: '0.17.2', platforms: ['linux'] })
    assert.deepEqual(result.problems, [])
  })

  // Checksums cannot catch truncation: the digest of half a file is valid.
  test('a truncated artifact fails on size', () => {
    const manifest = goodManifest()
    manifest.artifacts[0].size = 4096

    const result = evaluateGate({ manifest, version: '0.17.2' })
    assert.ok(result.problems.some(p => /truncated/.test(p)))
  })

  // The first real release attempt died here: every installer was intact, but
  // the updater's own sidecars (a 765-byte `latest-linux.yml`, 124 KB
  // `.blockmap` files) were judged against the 1 MiB installer floor.
  test('small updater sidecars do not trip the truncation floor', () => {
    const manifest = goodManifest()
    manifest.artifacts.push(
      artifact({ file: 'latest-linux.yml', platform: 'linux', kind: 'update-metadata', size: 765, signable: false, signed: false }),
      artifact({ file: 'latest-mac.yml', platform: 'mac', kind: 'update-metadata', size: 885, signable: false, signed: false }),
      artifact({ file: 'AI-Evolution-Jarvis-0.17.2-win-x64.exe.blockmap', platform: 'win', kind: 'blockmap', size: 123810, signable: false, signed: false }),
      artifact({ file: 'AI-Evolution-Jarvis-0.17.2-mac-arm64.dmg.blockmap', platform: 'mac', kind: 'blockmap', size: 139624, signable: false, signed: false })
    )

    assert.deepEqual(evaluateGate({ manifest, version: '0.17.2' }).problems, [])
  })

  // The exemption must not become a hole: a genuinely truncated installer of an
  // exempt-kind's size still fails.
  test('a truncated installer of any size still fails', () => {
    const manifest = goodManifest()
    manifest.artifacts.find(a => a.kind === 'dmg').size = 139624

    assert.ok(evaluateGate({ manifest, version: '0.17.2' }).problems.some(p => /\.dmg is only 139624/.test(p)))
  })

  test('digests are re-verified against disk when asked', () => {
    const result = evaluateGate({
      manifest: goodManifest(),
      version: '0.17.2',
      verifyDigests: true,
      digest: file => (file === 'jarvis-win-x64.msi' ? 'tampered' : 'deadbeef')
    })

    assert.ok(result.problems.some(p => /jarvis-win-x64\.msi checksum mismatch/.test(p)))
  })

  test('every problem is reported, not just the first', () => {
    const manifest = goodManifest()
    manifest.version = '0.1.0'
    manifest.artifacts[0].signed = false
    manifest.artifacts[1].size = 10

    const result = evaluateGate({ manifest, version: '0.17.2' })
    assert.equal(result.problems.length, 3)
  })

  test('a malformed manifest fails instead of throwing', () => {
    assert.equal(evaluateGate({ manifest: null }).ok, false)
    assert.equal(evaluateGate({ manifest: { version: '1' } }).ok, false)
  })

  test('an unknown platform name is surfaced', () => {
    const result = evaluateGate({ manifest: goodManifest(), platforms: ['solaris'] })
    assert.ok(result.problems.some(p => /unknown platform "solaris"/.test(p)))
  })
})

describe('REQUIRED_ARTIFACTS', () => {
  // A platform the gate requires but the manifest builder can never label is a
  // gate nothing can satisfy: every release would fail with "no X artifacts".
  test('every required platform is one the manifest can produce', () => {
    const producible = new Set(
      ['a-win-x64.exe', 'a-mac-arm64.dmg', 'a-linux-x64.deb']
        .map(file => classifyArtifact(file).platform)
    )

    for (const platform of Object.keys(REQUIRED_ARTIFACTS)) {
      assert.ok(producible.has(platform), `nothing can be classified as ${platform}`)
    }
  })

  // Same contract in the other direction: a kind the gate demands must be one
  // classifyArtifact emits, or a present artifact reads as missing.
  test('every required kind is one the manifest can produce', () => {
    const kinds = new Set(
      [
        'a-win-x64.exe',
        'a-win-x64.msi',
        'a-mac-arm64.dmg',
        'a-mac-arm64.zip',
        'a-linux-x64.AppImage',
        'a-linux-x64.deb',
        'a-linux-x64.rpm'
      ].map(file => classifyArtifact(file).kind)
    )

    for (const [platform, required] of Object.entries(REQUIRED_ARTIFACTS)) {
      for (const { kind } of required) {
        assert.ok(kinds.has(kind), `${platform} requires "${kind}", which nothing produces`)
      }
    }
  })
})

describe('renderReport', () => {
  test('a passing report says so', () => {
    const manifest = goodManifest()
    const text = renderReport(evaluateGate({ manifest, version: '0.17.2' }), manifest)

    assert.match(text, /PASS/)
    assert.match(text, /windows signing: azure/)
  })

  test('a failing report lists each problem', () => {
    const manifest = goodManifest()
    manifest.artifacts[0].signed = false
    const text = renderReport(evaluateGate({ manifest, version: '0.17.2' }), manifest)

    assert.match(text, /FAIL — 1 problem/)
    assert.match(text, /✗ jarvis-win-x64\.exe is unsigned/)
  })
})

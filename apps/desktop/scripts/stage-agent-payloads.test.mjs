import assert from 'node:assert/strict'
import path from 'node:path'
import { test } from 'vitest'

import {
  assertBanner,
  bannerExpectations,
  buildManifest,
  bundlePthLines,
  PAYLOAD_SCHEMA_VERSION,
  pipTargetArgs,
  pythonDirPattern,
  pythonRequest,
  resolveTag,
  resolveTargets
} from '../scripts/stage-agent-payloads.mjs'

// ─── resolveTargets ────────────────────────────────────────────────

test('resolveTargets covers every shipping (platform, arch) pair', () => {
  for (const [platform, arch] of [
    ['linux', 'x64'],
    ['linux', 'arm64'],
    ['darwin', 'x64'],
    ['darwin', 'arm64'],
    ['win32', 'x64'],
    ['win32', 'arm64']
  ]) {
    const t = resolveTargets(platform, arch)
    // Invariant: every target specifies all three toolchain descriptors.
    assert.ok(t.uvTarget && t.pythonPlatform && t.nodeDist, `${platform}-${arch}`)
    assert.equal(t.platform, platform)
    assert.equal(t.arch, arch)
  }
})

test('resolveTargets rejects unknown pairs (no universal2, no ia32)', () => {
  assert.throws(() => resolveTargets('darwin', 'universal'), /unsupported/)
  assert.throws(() => resolveTargets('win32', 'ia32'), /unsupported/)
})

test('windows targets map to msvc toolchains, darwin to apple, linux to gnu', () => {
  assert.match(resolveTargets('win32', 'x64').pythonPlatform, /windows-msvc$/)
  assert.match(resolveTargets('darwin', 'arm64').pythonPlatform, /apple-darwin$/)
  assert.match(resolveTargets('linux', 'x64').pythonPlatform, /linux-gnu$/)
})

// ─── pipTargetArgs ─────────────────────────────────────────────────

test('site-packages install refuses sdists and targets the payload dir', () => {
  const args = pipTargetArgs({ sitePackagesDir: '/out/site-packages' })
  // Invariants: the requirements come from the frozen lockfile, and the
  // install is binary-only. An sdist would try to compile on the build
  // runner for packages we did not explicitly allow-list. The install is
  // native, so no --platform cross-tags belong here.
  assert.equal(args[0], 'install')
  assert.ok(args.includes('--only-binary'))
  assert.equal(args[args.indexOf('-r') + 1], 'requirements-payload.txt')
  assert.equal(args[args.indexOf('--target') + 1], '/out/site-packages')
  assert.ok(!args.includes('--platform'))
})

// ─── bundlePthLines ────────────────────────────────────────────────

test('bundle .pth entries are relative and name repo before site-packages', () => {
  // POSIX layout: purelib nests three levels under the payload root, so
  // the entries climb out with ../ segments — never absolute paths.
  const payload = '/build/agent-payload'
  const purelib = '/build/agent-payload/python/cpython-3.11.15-macos-aarch64-none/lib/python3.11/site-packages'
  const lines = bundlePthLines(purelib, payload, path.posix)
  assert.equal(lines.length, 2)
  assert.ok(lines.every((line) => !path.posix.isAbsolute(line)), lines.join(','))
  assert.match(lines[0], /repo$/)
  assert.match(lines[1], /site-packages$/)

  // Windows layout (Lib/site-packages) stays relative too.
  const winLines = bundlePthLines(
    'C:\\b\\agent-payload\\python\\cpython-3.11.15-windows-x86_64-none\\Lib\\site-packages',
    'C:\\b\\agent-payload',
    path.win32
  )
  assert.ok(winLines.every((line) => !path.win32.isAbsolute(line)), winLines.join(','))
  assert.match(winLines[0], /repo$/)
})

// ─── resolveTag ────────────────────────────────────────────────────

test('explicit --tag wins and must be a final release', () => {
  assert.equal(resolveTag(['--tag=v1.2.3'], () => null), 'v1.2.3')
  assert.throws(() => resolveTag(['--tag=v1.2.3-rc1'], () => null), /final release/)
  assert.throws(() => resolveTag(['--tag=main'], () => null), /final release/)
})

test('falls back to git describe only for exact release tags', () => {
  assert.equal(resolveTag([], () => 'v0.17.0'), 'v0.17.0')
  assert.throws(() => resolveTag([], () => 'v0.17.0-14-gdeadbeef'), /no release tag/)
  assert.throws(() => resolveTag([], () => null), /no release tag/)
})

// ─── buildManifest ─────────────────────────────────────────────────

test('the manifest is a complete-payload sentinel: schema, tag, commit', () => {
  const target = resolveTargets('linux', 'x64')
  const manifest = buildManifest({
    tag: 'v1.0.0',
    commit: 'a'.repeat(40),
    target
  })

  assert.equal(manifest.schemaVersion, PAYLOAD_SCHEMA_VERSION)
  assert.equal(manifest.tag, 'v1.0.0')
  assert.equal(manifest.commit, 'a'.repeat(40))
  assert.equal(manifest.platform, 'linux')
  // No per-item status exists: completeness is a build invariant, not a
  // runtime question. The external stub is the only other manifest shape.
  assert.equal('items' in manifest, false)
})

// ─── arch guards ────────────────────────────────────────────────────

test('assertBanner passes on a matching triple and throws on a foreign one', () => {
  const target = resolveTargets('win32', 'arm64')
  const expect = bannerExpectations(target)

  assert.doesNotThrow(() =>
    assertBanner('uv', 'uv 0.12.1 (329541a50 aarch64-pc-windows-msvc)', expect.uv)
  )
  // The exact failure from the first Windows test build: an x64 uv from
  // PATH staged into an arm64 artifact (it ran via emulation).
  assert.throws(
    () => assertBanner('uv', 'uv 0.12.1 (329541a50 x86_64-pc-windows-msvc)', expect.uv),
    /wrong-architecture/
  )
})

test('banner expectations name the target, not the build host', () => {
  const linuxArm = resolveTargets('linux', 'arm64')
  assert.equal(bannerExpectations(linuxArm).uv, 'aarch64-unknown-linux-gnu')
  assert.equal(bannerExpectations(linuxArm).node, 'arm64')
  assert.ok(bannerExpectations(linuxArm).pythonAny.includes('aarch64'))
})

test('python install requests name the full build, not just the version', () => {
  // A bare "3.11" lets uv substitute another architecture when the native
  // build is missing — the silent x86_64-on-arm64 failure. The request
  // must pin cpython-<ver>-<os>-<arch>-<libc>.
  assert.equal(pythonRequest(resolveTargets('win32', 'arm64'), '3.11'), 'cpython-3.11-windows-aarch64-none')
  assert.equal(pythonRequest(resolveTargets('linux', 'x64'), '3.11'), 'cpython-3.11-linux-x86_64-gnu')
  assert.equal(pythonRequest(resolveTargets('darwin', 'arm64'), '3.12'), 'cpython-3.12-macos-aarch64-none')
})

test('python dir matcher accepts patch-versioned installs and rejects foreign builds', () => {
  const winArm = resolveTargets('win32', 'arm64')
  const pattern = pythonDirPattern(winArm, '3.11')

  // uv creates the patch-versioned directory plus a minor-version alias.
  assert.ok(pattern.test('cpython-3.11.15-windows-aarch64-none'))
  assert.ok(pattern.test('cpython-3.11-windows-aarch64-none'))
  // Another arch, another version, or a partial name must not match.
  assert.ok(!pattern.test('cpython-3.11.15-windows-x86_64-none'))
  assert.ok(!pattern.test('cpython-3.12.1-windows-aarch64-none'))
  assert.ok(!pattern.test('cpython-3.115-windows-aarch64-none'))
})

test('source-build exceptions override only-binary for the named packages only', () => {
  // Fully wheel-covered targets keep the pure only-binary shape.
  const linux = resolveTargets('linux', 'x64')
  assert.deepEqual(pipTargetArgs({ sitePackagesDir: '/sp', sourceBuild: linux.sourceBuild ?? [] }), [
    'install', '--only-binary', ':all:', '-r', 'requirements-payload.txt',
    '--target', '/sp', '--upgrade', '--no-compile'
  ])

  // win32-arm64 names the packages with no published win_arm64 wheel;
  // pip's later --no-binary overrides --only-binary per package, so
  // exactly these build from sdist and everything else stays wheels-only.
  const winArm = resolveTargets('win32', 'arm64')
  const args = pipTargetArgs({ sitePackagesDir: '/sp', sourceBuild: winArm.sourceBuild })
  const noBinary = args[args.indexOf('--no-binary') + 1]
  assert.ok(args.indexOf('--no-binary') > args.indexOf('--only-binary'))
  assert.equal(noBinary, 'cryptography,httptools,ruamel-yaml-clib,pywinpty,pyyaml')
})

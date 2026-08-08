import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  EMBEDDED_RUNTIME_ITEMS,
  findEmbeddedPython,
  latestReleaseFromLsRemote,
  PAYLOAD_SCHEMA_VERSION,
  resolvePayload,
  updateChannelFromConfig
} from '../electron/bundled-runtime'

// ─── resolvePayload ────────────────────────────────────────────────

const readerFor = (manifest: unknown) => (p: string) => {
  if (!p.endsWith('manifest.json')) {
    throw new Error('ENOENT')
  }

  return JSON.stringify(manifest)
}

const allDirsExist = () => true
const noDirsExist = () => false

const completeManifest = { schemaVersion: PAYLOAD_SCHEMA_VERSION, tag: 'v1.2.3', commit: 'a'.repeat(40) }

test('resolvePayload returns null for dev runs, external stubs, and garbage', () => {
  assert.equal(resolvePayload(null), null)
  assert.equal(resolvePayload(undefined), null)
  assert.equal(
    resolvePayload('/res', readerFor({ schemaVersion: PAYLOAD_SCHEMA_VERSION, external: true }), allDirsExist),
    null
  )
  assert.equal(
    resolvePayload(
      '/res',
      () => {
        throw new Error('ENOENT')
      },
      allDirsExist
    ),
    null
  )
  assert.equal(resolvePayload('/res', readerFor('not-an-object'), allDirsExist), null)
})

test('resolvePayload rejects old-schema manifests', () => {
  // A schema-2 manifest comes from a pre-embedded artifact. The app and
  // its payload travel together, so a mismatch means a foreign artifact.
  assert.equal(
    resolvePayload('/res', readerFor({ schemaVersion: 2, tag: 'v1.0.0', items: { repo: { status: 'staged' } } }), allDirsExist),
    null
  )
})

test('resolvePayload rejects a payload with a missing item directory', () => {
  // Completeness is a build invariant; a missing directory here means a
  // damaged or truncated artifact.
  assert.equal(resolvePayload('/res', readerFor(completeManifest), noDirsExist), null)

  // One missing item out of five is still a rejection.
  const allButUv = (p: string) => !p.endsWith('/uv')

  assert.equal(resolvePayload('/res', readerFor(completeManifest), allButUv), null)
})

test('resolvePayload returns dir + tag for a complete payload', () => {
  const p = resolvePayload('/res', readerFor(completeManifest), allDirsExist)

  assert.ok(p)
  assert.match(p.dir, /agent-payload$/)
  assert.equal(p.tag, 'v1.2.3')
})

test('the required items include uv — plugin lazy installs are mandatory', () => {
  assert.deepEqual([...EMBEDDED_RUNTIME_ITEMS].sort(), ['node', 'python', 'repo', 'site-packages', 'uv'])
})

// ─── findEmbeddedPython ────────────────────────────────────────────

test('findEmbeddedPython picks the patch-versioned dir and needs a real binary', () => {
  const fsStub = (dirs: string[], files: string[]) => ({
    readdirSync: (p: string) => {
      if (!p.endsWith('python')) {
        throw new Error('ENOENT')
      }

      return dirs
    },
    existsSync: (p: string) => files.some(f => p === f)
  })

  // Patch-versioned real dir wins over the minor alias (reverse sort).
  const python = findEmbeddedPython(
    '/res/agent-payload',
    'darwin',
    fsStub(
      ['cpython-3.11-macos-aarch64-none', 'cpython-3.11.15-macos-aarch64-none'],
      ['/res/agent-payload/python/cpython-3.11.15-macos-aarch64-none/bin/python3']
    ) as never
  )

  assert.match(String(python), /3\.11\.15.*bin\/python3$/)

  // No python dir at all → null, not a throw.
  assert.equal(
    findEmbeddedPython('/res/agent-payload', 'darwin', {
      readdirSync: () => {
        throw new Error('ENOENT')
      },
      existsSync: () => false
    } as never),
    null
  )

  // Windows binary lives at the install root, not bin/. The
  // implementation joins with the HOST path module, so the test builds
  // its expected path the same way to stay host-agnostic.
  const winRoot = 'win-res/agent-payload'
  const winExpected = ['win-res/agent-payload', 'python', 'cpython-3.11.15-windows-x86_64-none', 'python.exe'].join('/')

  const winPython = findEmbeddedPython(
    winRoot,
    'win32',
    fsStub(['cpython-3.11.15-windows-x86_64-none'], [winExpected]) as never
  )

  assert.match(String(winPython), /python\.exe$/)
})

// ─── updateChannelFromConfig ───────────────────────────────────────

test('channel comes from update.channel in config.yaml; absent means main', () => {
  assert.equal(updateChannelFromConfig('update:\n  channel: stable\n'), 'stable')
  assert.equal(updateChannelFromConfig('update:\n  channel: "stable"\n'), 'stable')
  assert.equal(updateChannelFromConfig('update:\n  channel: main\n'), 'main')
  assert.equal(updateChannelFromConfig('model:\n  provider: nous\n'), 'main')
  assert.equal(updateChannelFromConfig(null), 'main')
  assert.equal(updateChannelFromConfig(''), 'main')
})

test('channel parsing stays inside the update block', () => {
  // A channel key in ANOTHER block must not leak into the answer.
  const text = 'gateway:\n  channel: stable\nupdate:\n  interval: 1\nmodel:\n  channel: stable\n'

  assert.equal(updateChannelFromConfig(text), 'main')

  // The update block ends at the next top-level key.
  const ended = 'update:\n  interval: 1\nother:\n  channel: stable\n'

  assert.equal(updateChannelFromConfig(ended), 'main')
})

// ── latestReleaseFromLsRemote ───────────────────────────────────────

test('release picking is numeric, skips prereleases, prefers peeled shas', () => {
  const output = [
    `${'a'.repeat(40)}\trefs/tags/v0.9.0`,
    `${'b'.repeat(40)}\trefs/tags/v0.10.0`,
    `${'c'.repeat(40)}\trefs/tags/v0.10.0^{}`,
    `${'d'.repeat(40)}\trefs/tags/v0.11.0-rc1`,
    `${'e'.repeat(40)}\trefs/tags/v2026.7.20`
  ].join('\n')

  const latest = latestReleaseFromLsRemote(output)

  // v0.10.0 beats v0.9.0 numerically (a lexicographic sort would invert
  // it), the rc prerelease is skipped, and the CalVer tag is excluded by
  // the three-digit major cap — otherwise 2026 would beat every SemVer
  // release forever.
  assert.equal(latest?.tag, 'v0.10.0')
  assert.equal(latest?.sha, 'c'.repeat(40))

  const semverOnly = latestReleaseFromLsRemote(
    [`${'a'.repeat(40)}\trefs/tags/v0.9.0`, `${'b'.repeat(40)}\trefs/tags/v0.10.0`, `${'c'.repeat(40)}\trefs/tags/v0.10.0^{}`].join('\n')
  )

  assert.equal(semverOnly?.tag, 'v0.10.0')
  assert.equal(semverOnly?.sha, 'c'.repeat(40))
})

test('release picking returns null when no final release tag exists', () => {
  assert.equal(latestReleaseFromLsRemote(''), null)
  assert.equal(latestReleaseFromLsRemote(`${'d'.repeat(40)}\trefs/tags/v1.0.0-beta.2`), null)
})

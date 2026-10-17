import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createHash } from 'node:crypto'
import yaml from 'js-yaml'
import { afterEach, expect, test } from 'vitest'
import { codesignTeam, stampAssertions } from '../tests/install/e2e-assets/mac-bundled-manifest.cjs'
import { materializeFeed } from '../tests/install/e2e-assets/mac-bundled-feed.mjs'
import { safeJoin } from '../tests/install/e2e-assets/mac-bundled-serve.mjs'

const directories = []
afterEach(() => { for (const dir of directories.splice(0)) fs.rmSync(dir, { recursive: true, force: true }) })
function scratch() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'mac-feed-'))
  directories.push(dir)
  return dir
}

test('codesign stderr identifies the team but never blesses an unsigned/ad-hoc signature', () => {
  expect(codesignTeam('Identifier=com.x\nTeamIdentifier=TEAM123456')).toBe('TEAM123456')
  for (const output of ['', 'Identifier=com.x', 'TeamIdentifier=', 'TeamIdentifier=not set\nFoo=not', 'TeamIdentifier=-']) {
    expect(codesignTeam(output)).toBeNull()
  }
})

test('installed stamp independently binds payload, mechanism, commit and tag', () => {
  const side = { commit: 'a'.repeat(40), tag: 'v0.28.0' }
  const good = { ...side, payload: 'bundled', updateMechanism: 'electron-updater' }
  expect(stampAssertions(good, side)).toEqual([])
  expect(stampAssertions(null, side)).toEqual(['install-stamp.json: not an object'])
  for (const [key, value] of [['payload', 'light'], ['updateMechanism', 'external'], ['commit', 'b'.repeat(40)], ['tag', null]]) {
    const problems = stampAssertions({ ...good, [key]: value }, side)
    expect(problems).toHaveLength(1)
    expect(problems[0]).toContain(`stamp.${key}`)
  }
})

test.each([
  ['arm64', 'stable', 'v0.29.0', 'arm64-stable-mac.yml'],
  ['x64', 'canary', 'v0.30.0+canary.20260907T000000Z', 'canary-mac.yml'],
])('%s %s feed binds parsed YAML to copied artifact bytes', async (arch, channel, tag, name) => {
  const outDir = scratch()
  const zip = path.join(outDir, 'payload.zip')
  const bytes = Buffer.from('transport fixture, not a signed native package')
  fs.writeFileSync(zip, bytes)
  const sha512 = createHash('sha512').update(bytes).digest('base64')
  const version = tag.slice(1), releaseDate = '2026-09-07T00:00:00.000Z'
  const receipt = await materializeFeed({ outDir, zipPath: zip, version, tag, arch, releaseDate })
  const directory = `releases/darwin/${channel}`, url = `/releases/tag/${tag}/payload.zip`
  expect(receipt).toMatchObject({ channel, feedKey: `${directory}/${name}`, artifactUrlPath: url, sha512 })
  expect(fs.readdirSync(path.join(outDir, directory)).sort()).toEqual([...new Set([name, `${channel}-mac.yml`])].sort())
  for (const filename of [name, `${channel}-mac.yml`]) {
    expect(yaml.load(fs.readFileSync(path.join(outDir, directory, filename), 'utf8'))).toEqual({
      version, files: [{ url, sha512, size: bytes.length }], path: url, sha512, releaseDate,
    })
  }
  expect(fs.readFileSync(path.join(outDir, url))).toEqual(bytes)
})

test('feed server rejects literal and encoded traversal', () => {
  const root = scratch()
  expect(safeJoin(root, '/releases/darwin/stable/stable-mac.yml')).toBe(path.join(root, 'releases/darwin/stable/stable-mac.yml'))
  for (const attack of ['/../escape.yml', '/%2e%2e/escape.yml', '/..%2fescape.yml']) expect(safeJoin(root, attack)).toBeNull()
})

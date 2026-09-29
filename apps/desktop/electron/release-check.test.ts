import assert from 'node:assert/strict'

import { test } from 'vitest'

import { compareVersions, evaluateRelease, pickInstallerAsset } from './release-check'

const asset = (name: string) => ({ browser_download_url: `https://github.com/x/y/releases/download/v1/${name}`, name, size: 1 })

const release = (over: Record<string, unknown> = {}) => ({
  assets: [
    asset('Agent-Czesiek-Setup-0.22.0.exe'),
    asset('Agent-Czesiek-Setup-0.22.0.exe.blockmap'),
    asset('Agent-Czesiek-0.22.0-arm64.dmg'),
    asset('Agent-Czesiek-0.22.0-x64.dmg'),
    asset('Agent-Czesiek-0.22.0.AppImage')
  ],
  body: '\n## Co nowego\n\n- Vault\n',
  html_url: 'https://github.com/aievolutionpl/AGENT_CZESIEK/releases/tag/v0.22.0',
  published_at: '2026-10-01T10:00:00Z',
  tag_name: 'v0.22.0',
  ...over
})

const win = { arch: 'x64', currentVersion: '0.21.1', platform: 'win32' }

test('a newer tag is an update, an equal or older one is not, and versions compare numerically', () => {
  assert.equal(evaluateRelease(release(), win).updateAvailable, true)
  assert.equal(evaluateRelease(release(), { ...win, currentVersion: '0.22.0' }).updateAvailable, false)
  assert.equal(evaluateRelease(release(), { ...win, currentVersion: '0.30.1' }).updateAvailable, false)
  // 0.9 → 0.10 is newer even though "0.10" sorts before "0.9" as text.
  assert.equal(compareVersions('v0.10.0', '0.9.5'), 1)
  assert.equal(compareVersions('nonsense', '0.9.5'), 0)
})

test('picks the installer that fits this machine and never the blockmap', () => {
  const assets = evaluateRelease(release(), win).asset
  assert.equal(assets?.name, 'Agent-Czesiek-Setup-0.22.0.exe')

  const mac = (arch: string) => evaluateRelease(release(), { ...win, arch, platform: 'darwin' }).asset?.name
  assert.equal(mac('arm64'), 'Agent-Czesiek-0.22.0-arm64.dmg')
  assert.equal(mac('x64'), 'Agent-Czesiek-0.22.0-x64.dmg')

  assert.equal(evaluateRelease(release(), { ...win, platform: 'linux' }).asset?.name, 'Agent-Czesiek-0.22.0.AppImage')
  assert.equal(pickInstallerAsset([], 'win32', 'x64'), null)
})

test('refuses a payload that is not a usable published release', () => {
  assert.throws(() => evaluateRelease({ message: 'Not Found' }, win))
  assert.throws(() => evaluateRelease(release({ draft: true }), win))
  assert.throws(() => evaluateRelease(release({ prerelease: true }), win))
  assert.throws(() => evaluateRelease(release({ html_url: 'http://insecure' }), win))
})

test('keeps the release page as the fallback and a short excerpt of the notes', () => {
  const result = evaluateRelease(release({ assets: [] }), win)
  assert.equal(result.asset, null)
  assert.equal(result.pageUrl, 'https://github.com/aievolutionpl/AGENT_CZESIEK/releases/tag/v0.22.0')
  assert.equal(result.notes, '## Co nowego\n- Vault')
})

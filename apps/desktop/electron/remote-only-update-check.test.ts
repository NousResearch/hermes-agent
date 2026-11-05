import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'

import { afterEach, describe, expect, test, vi } from 'vitest'

import {
  compareVersions,
  createRemoteOnlyUpdateChecker,
  isNewerVersion,
  parseReleaseTag,
  REMOTE_ONLY_RELEASES_URL
} from './remote-only-update-check'

const dirs: string[] = []

async function cachePath(): Promise<string> {
  const dir = await mkdtemp(path.join(tmpdir(), 'hermes-remote-only-'))
  dirs.push(dir)

  return path.join(dir, 'update-check.json')
}

afterEach(async () => {
  await Promise.all(dirs.splice(0).map(dir => rm(dir, { recursive: true, force: true })))
})

/** A fetch stand-in that answers like GitHub's /releases/latest redirect. */
function redirectTo(location: string, status = 302) {
  return vi.fn(async () => new Response(null, { status, headers: { location } }))
}

describe('parseReleaseTag', () => {
  test('reads the tag from absolute and relative redirect locations', () => {
    expect(parseReleaseTag('https://github.com/NousResearch/hermes-agent/releases/tag/v0.22.9')).toBe('v0.22.9')
    expect(parseReleaseTag('/NousResearch/hermes-agent/releases/tag/v0.22.9')).toBe('v0.22.9')
    expect(parseReleaseTag('https://github.com/NousResearch/hermes-agent/releases/tag/v0.22.9?foo=1')).toBe('v0.22.9')
  })

  test('ignores anything that is not a release tag', () => {
    expect(parseReleaseTag(null)).toBeNull()
    expect(parseReleaseTag(undefined)).toBeNull()
    expect(parseReleaseTag('https://github.com/NousResearch/hermes-agent/releases')).toBeNull()
    expect(parseReleaseTag('https://github.com/NousResearch/hermes-agent/releases/tag/')).toBeNull()
  })
})

describe('compareVersions', () => {
  test('orders releases and prereleases', () => {
    expect(compareVersions('v0.22.9', 'v0.22.7')).toBe(1)
    expect(compareVersions('0.22.7', 'v0.22.9')).toBe(-1)
    expect(compareVersions('v0.22.7', 'v0.22.7')).toBe(0)
    expect(compareVersions('v1.0.0', 'v0.99.99')).toBe(1)
    expect(compareVersions('v0.22.9-rc.1', 'v0.22.9')).toBe(-1)
    expect(compareVersions('v0.22.9-rc.2', 'v0.22.9-rc.1')).toBe(1)
  })

  test('never claims an update when a side is not semver', () => {
    expect(compareVersions('nightly', 'v0.22.7')).toBe(0)
    expect(isNewerVersion('nightly', '0.22.7')).toBe(false)
  })

  test('isNewerVersion only fires for a strictly newer release', () => {
    expect(isNewerVersion('v0.22.9', '0.22.7')).toBe(true)
    expect(isNewerVersion('v0.22.7', '0.22.7')).toBe(false)
    expect(isNewerVersion(null, '0.22.7')).toBe(false)
  })
})

describe('createRemoteOnlyUpdateChecker', () => {
  test('follows the redirect itself and caches the tag', async () => {
    const file = await cachePath()
    const fetchImpl = redirectTo('https://github.com/NousResearch/hermes-agent/releases/tag/v0.22.9')
    const checker = createRemoteOnlyUpdateChecker({
      currentVersion: '0.22.7',
      cachePath: file,
      fetch: fetchImpl as unknown as typeof fetch,
      now: () => 1_000
    })

    const status = await checker.check()

    expect(status).toMatchObject({ supported: true, current: '0.22.7', latest: 'v0.22.9', newer: true, checkedAt: 1_000 })
    expect(status.releaseUrl).toBe(`${REMOTE_ONLY_RELEASES_URL}/tag/v0.22.9`)
    expect(fetchImpl).toHaveBeenCalledWith(
      'https://github.com/NousResearch/hermes-agent/releases/latest',
      expect.objectContaining({ redirect: 'manual' })
    )
    expect(JSON.parse(await readFile(file, 'utf8'))).toEqual({ checkedAt: 1_000, latest: 'v0.22.9' })
  })

  test('serves a fresh cache without touching the network', async () => {
    const file = await cachePath()
    await writeFile(file, JSON.stringify({ checkedAt: 5_000, latest: 'v0.22.9' }), 'utf8')
    const fetchImpl = redirectTo('https://github.com/NousResearch/hermes-agent/releases/tag/v0.23.0')
    const checker = createRemoteOnlyUpdateChecker({
      currentVersion: '0.22.7',
      cachePath: file,
      fetch: fetchImpl as unknown as typeof fetch,
      now: () => 5_000 + 60_000
    })

    const status = await checker.check()

    expect(status.latest).toBe('v0.22.9')
    expect(fetchImpl).not.toHaveBeenCalled()
  })

  test('force bypasses the cache', async () => {
    const file = await cachePath()
    await writeFile(file, JSON.stringify({ checkedAt: 5_000, latest: 'v0.22.7' }), 'utf8')
    const fetchImpl = redirectTo('https://github.com/NousResearch/hermes-agent/releases/tag/v0.23.0')
    const checker = createRemoteOnlyUpdateChecker({
      currentVersion: '0.22.7',
      cachePath: file,
      fetch: fetchImpl as unknown as typeof fetch,
      now: () => 5_000 + 60_000
    })

    await expect(checker.check({ force: true })).resolves.toMatchObject({ latest: 'v0.23.0', newer: true })
  })

  test('an expired cache is refreshed', async () => {
    const file = await cachePath()
    await writeFile(file, JSON.stringify({ checkedAt: 0, latest: 'v0.22.7' }), 'utf8')
    const checker = createRemoteOnlyUpdateChecker({
      currentVersion: '0.22.7',
      cachePath: file,
      fetch: redirectTo('https://github.com/NousResearch/hermes-agent/releases/tag/v0.23.0') as unknown as typeof fetch,
      now: () => 24 * 60 * 60 * 1000
    })

    await expect(checker.check()).resolves.toMatchObject({ latest: 'v0.23.0' })
  })

  test('a failed check reports the error and falls back to the stale cache', async () => {
    const file = await cachePath()
    await writeFile(file, JSON.stringify({ checkedAt: 0, latest: 'v0.22.8' }), 'utf8')
    const checker = createRemoteOnlyUpdateChecker({
      currentVersion: '0.22.7',
      cachePath: file,
      fetch: vi.fn(async () => new Response(null, { status: 200 })) as unknown as typeof fetch,
      now: () => 24 * 60 * 60 * 1000
    })

    const status = await checker.check()

    expect(status).toMatchObject({ latest: 'v0.22.8', newer: true })
    expect(status.error).toMatch(/did not redirect/)
  })

  test('a failed check with no cache reports "unknown" instead of throwing', async () => {
    const file = await cachePath()
    const checker = createRemoteOnlyUpdateChecker({
      currentVersion: '0.22.7',
      cachePath: file,
      fetch: vi.fn(async () => {
        throw new Error('offline')
      }) as unknown as typeof fetch
    })

    const status = await checker.check()

    expect(status).toMatchObject({ latest: null, newer: false, supported: true })
    expect(status.error).toBe('offline')
    expect(status.releaseUrl).toBe(REMOTE_ONLY_RELEASES_URL)
  })
})

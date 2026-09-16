import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  compareReleaseTagsNewestFirst,
  fetchOfficialGithubReleases,
  normaliseGithubRelease,
  parseRemoteReleaseTags,
  resolveDesktopReleaseStatus
} from './update-releases'

const SHA_OLD = '1'.repeat(40)
const SHA_MIDDLE = '2'.repeat(40)
const SHA_LATEST_TAG_OBJECT = '3'.repeat(40)
const SHA_LATEST = '4'.repeat(40)

function release(tag: string) {
  return {
    body: `Notes for ${tag}`,
    draft: false,
    html_url: `https://github.com/NousResearch/hermes-agent/releases/tag/${tag}`,
    name: tag,
    prerelease: false,
    published_at: '2026-07-20T00:00:00Z',
    tag_name: tag
  }
}

test('release ordering handles three- and four-component date tags numerically', () => {
  const tags = ['v2026.7.9', 'v2026.7.20', 'v2026.7.7.2', 'v2026.7.7']

  assert.deepEqual(tags.sort(compareReleaseTagsNewestFirst), [
    'v2026.7.20',
    'v2026.7.9',
    'v2026.7.7.2',
    'v2026.7.7'
  ])
})

test('normaliseGithubRelease rejects drafts, prereleases, and unsupported tags', () => {
  assert.equal(normaliseGithubRelease({ ...release('v2026.7.20'), draft: true }), null)
  assert.equal(normaliseGithubRelease({ ...release('v2026.7.20'), prerelease: true }), null)
  assert.equal(normaliseGithubRelease(release('v1.0.0')), null)
  assert.equal(normaliseGithubRelease(release('v2026.7.20'))?.tag, 'v2026.7.20')
})

test('parseRemoteReleaseTags prefers peeled commits for annotated tags', () => {
  const tags = parseRemoteReleaseTags(
    [
      `${SHA_OLD}\trefs/tags/v2026.7.7`,
      `${SHA_LATEST_TAG_OBJECT}\trefs/tags/v2026.7.20`,
      `${SHA_LATEST}\trefs/tags/v2026.7.20^{}`,
      `not-a-sha\trefs/tags/v2026.7.21`,
      `${SHA_MIDDLE}\trefs/heads/main`
    ].join('\n')
  )

  assert.equal(tags.get('v2026.7.7'), SHA_OLD)
  assert.equal(tags.get('v2026.7.20'), SHA_LATEST)
  assert.equal(tags.has('v2026.7.21'), false)
})

test('resolveDesktopReleaseStatus counts zero, one, and multiple releases behind', () => {
  const records = [release('v2026.7.20'), release('v2026.7.14'), release('v2026.7.7')]
    .map(record => normaliseGithubRelease(record))
    .filter(record => record !== null)

  const remoteTags = new Map([
    ['v2026.7.20', SHA_LATEST],
    ['v2026.7.14', SHA_MIDDLE],
    ['v2026.7.7', SHA_OLD]
  ])

  assert.equal(resolveDesktopReleaseStatus({ currentSha: SHA_LATEST, releaseRecords: records, remoteTags }).behind, 0)
  assert.equal(resolveDesktopReleaseStatus({ currentSha: SHA_MIDDLE, releaseRecords: records, remoteTags }).behind, 1)

  const old = resolveDesktopReleaseStatus({ currentSha: SHA_OLD, releaseRecords: records, remoteTags })
  assert.equal(old.behind, 2)
  assert.equal(old.currentRelease, 'v2026.7.7')
  assert.equal(old.targetRelease, 'v2026.7.20')
  assert.equal(old.targetSha, SHA_LATEST)
  assert.equal(old.countKnown, true)
})

test('unknown checkouts degrade to a generic single available release', () => {
  const latest = normaliseGithubRelease(release('v2026.7.20'))
  assert.ok(latest)

  const status = resolveDesktopReleaseStatus({
    currentSha: 'f'.repeat(40),
    releaseRecords: [latest],
    remoteTags: new Map([['v2026.7.20', SHA_LATEST]])
  })

  assert.equal(status.behind, 1)
  assert.equal(status.countKnown, false)
  assert.equal(status.currentRelease, null)
})

test('fetchOfficialGithubReleases follows pagination and de-duplicates tags', async () => {
  const calls: string[] = []
  const firstPage = [release('v2026.7.14'), ...Array.from({ length: 99 }, (_, index) => release(`invalid-${index}`))]
  const secondPage = [release('v2026.7.20'), release('v2026.7.14')]

  const fetchImpl = async input => {
    calls.push(String(input))
    const payload = calls.length === 1 ? firstPage : secondPage

    return new Response(JSON.stringify(payload), {
      status: 200,
      headers: { 'Content-Type': 'application/json' }
    })
  }

  const releases = await fetchOfficialGithubReleases(fetchImpl)

  assert.equal(calls.length, 2)
  assert.deepEqual(
    releases.map(item => item.tag),
    ['v2026.7.20', 'v2026.7.14']
  )
})

test('fetchOfficialGithubReleases fails closed on HTTP errors', async () => {
  await assert.rejects(
    () => fetchOfficialGithubReleases(async () => new Response('{}', { status: 503 })),
    /503/
  )
})

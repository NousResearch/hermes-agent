import { afterEach, describe, expect, it } from 'vitest'

import type { ArtifactDetection } from '@/lib/artifact-detect'

import {
  $artifactRegistry,
  $hostedArtifacts,
  artifactsForSession,
  clearArtifactRegistry,
  getHostedArtifact,
  recordHostedArtifactVersion,
  upsertArtifact
} from './artifacts'

const ID = '01j9z8y7x6w5v4t3s2r1q0p9n8'

const sighting = (version: number, extra: { contentUrl?: string; title?: string } = {}) => ({
  contentUrl: extra.contentUrl ?? `https://${ID}.frame.example/__av?t=v${version}`,
  id: ID,
  title: extra.title,
  version,
  viewerUrl: `https://portal.example/a/${ID}`
})

afterEach(() => clearArtifactRegistry())

describe('hosted artifact cache', () => {
  it('keys by server artifact id and lets the latest version win', () => {
    recordHostedArtifactVersion(sighting(1, { title: 'Draft' }), 1)
    recordHostedArtifactVersion(sighting(3, { title: 'Final' }), 2)

    const record = getHostedArtifact(ID)!

    expect(record.headVersion).toBe(3)
    expect(record.title).toBe('Final')
    expect(record.versions.map(v => v.version)).toEqual([1, 3])
  })

  it('retains an older version seen later without moving the head back', () => {
    recordHostedArtifactVersion(sighting(3, { title: 'Final' }), 1)
    recordHostedArtifactVersion(sighting(2, { title: 'Middle' }), 2)

    const record = getHostedArtifact(ID)!

    expect(record.headVersion).toBe(3)
    expect(record.title).toBe('Final')
    expect(record.versions.map(v => v.version)).toEqual([2, 3])
  })

  it('refreshes the ticket when the same version is shown again', () => {
    recordHostedArtifactVersion(sighting(2), 1)
    recordHostedArtifactVersion(sighting(2, { contentUrl: `https://${ID}.frame.example/__av?t=fresh` }), 2)

    const record = getHostedArtifact(ID)!

    expect(record.versions).toHaveLength(1)
    expect(record.versions[0]!.contentUrl).toBe(`https://${ID}.frame.example/__av?t=fresh`)
  })

  it('writes nothing for a re-render of the same result', () => {
    recordHostedArtifactVersion(sighting(2), 1)
    const before = $hostedArtifacts.get()

    recordHostedArtifactVersion(sighting(2), 2)

    expect($hostedArtifacts.get()).toBe(before)
  })

  it('keeps at most the twenty newest versions per artifact', () => {
    for (let version = 1; version <= 25; version += 1) {
      recordHostedArtifactVersion(sighting(version), version)
    }

    const versions = getHostedArtifact(ID)!.versions.map(v => v.version)

    expect(versions).toHaveLength(20)
    expect(versions[0]).toBe(6)
    expect(versions.at(-1)).toBe(25)
  })

  it('refuses a version that is not a positive integer', () => {
    expect(recordHostedArtifactVersion(sighting(0))).toBeNull()
    expect(recordHostedArtifactVersion(sighting(1.5))).toBeNull()
    expect(getHostedArtifact(ID)).toBeNull()
  })

  it('is cleared with the rest of the artifact state', () => {
    recordHostedArtifactVersion(sighting(1))
    clearArtifactRegistry()

    expect($hostedArtifacts.get()).toEqual({})
  })
})

describe('local artifacts next to hosted ones', () => {
  const detection: ArtifactDetection = { kind: 'html', language: 'html', title: 'Dashboard' }

  it('keeps registering and versioning transcript artifacts exactly as before', () => {
    recordHostedArtifactVersion(sighting(1, { title: 'Dashboard' }))

    const first = upsertArtifact('session-1', detection, '<p>one</p>')
    const second = upsertArtifact('session-1', detection, '<p>two</p>')

    expect(first?.artifactId).toBe('session-1:' + first?.record.slug)
    expect(second?.artifactId).toBe(first?.artifactId)
    expect(artifactsForSession('session-1')[0]!.versions.map(v => v.content)).toEqual(['<p>one</p>', '<p>two</p>'])
    // Hosted ids never leak into the per-session registry, and vice versa.
    expect(Object.keys($artifactRegistry.get())).toEqual(['session-1'])
    expect(Object.keys($hostedArtifacts.get())).toEqual([ID])
  })
})

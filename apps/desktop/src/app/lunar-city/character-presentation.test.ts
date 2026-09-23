import { describe, expect, it } from 'vitest'

import actualManifest from '../../../public/lunar-city/v2/world-manifest.v2.json'

import { characterPresentationForEntity } from './character-presentation'
import { parseWorldManifest } from './manifest'
import type { EntityKey, LunarEntity } from './model'

function profile(index: number, groupName: string, connectionId = 'local'): LunarEntity {
  const profileName = `worker-${index}`
  const key = `profile:connection=${connectionId}:profile=${profileName}` as EntityKey

  return {
    animation: 'idle',
    authority: 'authoritative',
    destination: 'project',
    identity: { kind: 'profile', connectionId, profile: profileName },
    key,
    observedAt: 1,
    presentation: {
      groups: [{ id: `opaque-${index}`, name: groupName }],
      metadata: { source: `profiles:${connectionId}`, state: 'fresh' },
      placement: { lodHint: 0, overflow: false, primaryGroupId: `opaque-${index}`, slot: index }
    }
  }
}

describe('characterPresentationForEntity', () => {
  const manifest = parseWorldManifest(structuredClone(actualManifest))

  it('binds all 19 exact configured group names to manifest-declared kits without using profile display names', () => {
    const presentations = manifest.characterAssets.groupKits.map((kit, index) =>
      characterPresentationForEntity(profile(index, kit.group), manifest.characterAssets)
    )

    expect(presentations.map(value => value?.kitId)).toEqual(manifest.characterAssets.groupKits.map(kit => kit.kitId))
    expect(presentations.every(value => value?.signature?.silhouetteAccessory)).toBe(true)
  })

  it('gives 320 exact profiles collision-free signatures that survive reorder and incremental additions', () => {
    const workers = Array.from({ length: 320 }, (_, index) => profile(index, 'Engineering Guild'))

    const first = new Map(
      workers.map(worker => [
        worker.key,
        characterPresentationForEntity(worker, manifest.characterAssets)?.visibleSignature
      ])
    )

    const reordered = new Map(
      [...workers]
        .reverse()
        .map(worker => [worker.key, characterPresentationForEntity(worker, manifest.characterAssets)?.visibleSignature])
    )

    const added = characterPresentationForEntity(profile(999, 'Engineering Guild'), manifest.characterAssets)

    expect(new Set(first.values())).toHaveLength(320)
    expect(reordered).toEqual(first)
    expect(added?.visibleSignature).not.toBe(undefined)
    expect([...first.entries()].every(([key, value]) => reordered.get(key) === value)).toBe(true)
  })

  it('keeps duplicate profile names on different exact connections visibly distinct', () => {
    const local = profile(4, 'Research Lab', 'local')
    const remote = profile(4, 'Research Lab', 'remote')

    expect(characterPresentationForEntity(local, manifest.characterAssets)?.visibleSignature).not.toBe(
      characterPresentationForEntity(remote, manifest.characterAssets)?.visibleSignature
    )
  })

  it('keeps a no-group exact profile neutral while retaining a physical base accent', () => {
    const entity = profile(4, 'Research Lab')
    entity.presentation = {
      groups: [{ id: 'still-present-but-unavailable', name: 'Research Lab' }],
      metadata: { source: 'profiles:local', state: 'unavailable' },
      placement: { lodHint: 0, overflow: false, primaryGroupId: 'still-present-but-unavailable', slot: 7 }
    }

    const result = characterPresentationForEntity(entity, manifest.characterAssets)

    expect(result?.kitId).toBeUndefined()
    expect(result?.signature).toMatchObject({
      body: expect.any(String),
      head: expect.any(String),
      palette: expect.any(String)
    })
    expect(result?.signature?.emblem).toBeUndefined()
    expect(result?.signature?.silhouetteAccessory).toBeUndefined()
    expect(result?.accentCode).toBe(7)
  })

  it('keeps aggregate-only unavailable rows identifiable without promoting their LOD', () => {
    const entity = profile(4, 'Research Review Board')
    entity.authority = 'stale'
    entity.position = undefined
    entity.presentation = {
      ...entity.presentation!,
      placement: { lodHint: 2, overflow: true, primaryGroupId: 'opaque-4' }
    }

    const result = characterPresentationForEntity(entity, manifest.characterAssets)

    expect(result).toEqual({ lod: 'far', renderMode: 'aggregate' })
  })
})

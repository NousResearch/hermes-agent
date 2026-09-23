import { afterEach, describe, expect, it, vi } from 'vitest'

import actualManifest from '../../../public/lunar-city/v2/world-manifest.v2.json'

import { APPROVED_SOURCE_SHA256, loadWorldManifest, parseWorldManifest } from './manifest'

const cloneActualManifest = (): unknown => structuredClone(actualManifest)

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('parseWorldManifest', () => {
  it('normalizes the actual v2 asset contract without discarding placement, LOD, animation, or budget metadata', () => {
    const manifest = parseWorldManifest(cloneActualManifest())
    const library = manifest.models.find(model => model.id === 'library')
    const leaders = manifest.models.find(model => model.id === 'leaders')

    expect(manifest.version).toBe(2)
    expect(manifest.assetVersion).toBe('2.0.0')
    expect(manifest.source.sha256).toBe(APPROVED_SOURCE_SHA256)
    expect(library).toMatchObject({
      uri: 'models/library.glb',
      transform: {
        position: { x: -28, y: 4, z: -18 },
        rotation: { x: 0, y: 0.785, z: 0 },
        scale: { x: 1, y: 1, z: 1 }
      },
      lods: [
        { distance: 0, node: 'library:lod:near' },
        { distance: 112, node: 'library:lod:far' }
      ],
      requiredClips: ['lights-idle'],
      maxTriangles: 28_000,
      maxDrawCalls: 8
    })
    expect(leaders?.statistics.animationClips).toContain('leader:fox:talking')
    expect(manifest.navigation.meshUri).toBe('models/navigation.glb')
    expect(manifest.destinations.review).toEqual({ x: 33, y: 0, z: 10 })
    expect(manifest.projectSlots[0]).toMatchObject({
      id: 'compound-inner-1',
      position: { x: 16, y: 0, z: 38 }
    })
    expect(manifest.qualityBudgets.balancedOverview).toEqual({
      drawCalls: 180,
      visibleTriangles: 1_500_000,
      gpuMiB: 256
    })
    expect(manifest.textures[0]).toMatchObject({
      uri: 'textures/approved-palette.png',
      source: 'generated-approved-palette'
    })
    expect(manifest.characterAssets).toMatchObject({
      fleetIdentityFloor: 128,
      sharedResourceStrategy: {
        animationClips: 'shared',
        gpuBuffers: 'shared',
        materials: 'shared',
        rig: 'worker:shared-rig'
      }
    })
    expect(manifest.characterAssets.leaders).toHaveLength(6)
    expect(manifest.characterAssets.groupKits).toHaveLength(19)
  })

  it.each([
    [
      'duplicate leader silhouette',
      (fixture: any) =>
        (fixture.characterAssets.leaders[1].silhouetteId = fixture.characterAssets.leaders[0].silhouetteId)
    ],
    ['unknown group body', (fixture: any) => (fixture.characterAssets.groupKits[0].signature.body = 'not-declared')],
    [
      'duplicate group kit',
      (fixture: any) => (fixture.characterAssets.groupKits[1].kitId = fixture.characterAssets.groupKits[0].kitId)
    ],
    [
      'duplicate complete kit signature',
      (fixture: any) =>
        (fixture.characterAssets.groupKits[1].signature = structuredClone(
          fixture.characterAssets.groupKits[0].signature
        ))
    ],
    ['unknown group name', (fixture: any) => (fixture.characterAssets.groupKits[0].group = 'Display Name Guess')],
    ['insufficient fleet floor', (fixture: any) => (fixture.characterAssets.fleetIdentityFloor = 127)],
    [
      'per-profile material allocation',
      (fixture: any) => (fixture.characterAssets.sharedResourceStrategy.perProfile.materials = 1)
    ],
    [
      'missing physical activation scale',
      (fixture: any) => delete fixture.characterAssets.physicalVariantRoots.activationScale['worker:head-variant:visor']
    ],
    [
      'collapsed physical activation scale',
      (fixture: any) =>
        (fixture.characterAssets.physicalVariantRoots.activationScale['worker:head-variant:visor'] = [0, 1, 1])
    ]
  ])('rejects an unsafe character asset contract: %s', (_label, mutate) => {
    const fixture = structuredClone(actualManifest)
    mutate(fixture)

    expect(() => parseWorldManifest(fixture)).toThrow(/characterAssets/)
  })

  it('sorts and freezes LOD thresholds once while rejecting duplicate distances', () => {
    const fixture = structuredClone(actualManifest) as any
    fixture.models[0].lods.reverse()
    const manifest = parseWorldManifest(fixture)

    expect(manifest.models[0]?.lods.map(lod => lod.distance)).toEqual([0, 96])
    expect(Object.isFrozen(manifest.models[0]?.lods)).toBe(true)

    fixture.models[0].lods[1].distance = fixture.models[0].lods[0].distance
    expect(() => parseWorldManifest(fixture)).toThrow(/LOD distances must be distinct/)
  })

  it('accepts optional host decorations and rejects duplicate decoration identities', () => {
    const fixture = structuredClone(actualManifest) as any
    fixture.externalDecorations = [
      {
        id: 'kenney-crate',
        uri: 'models/kenney-crate.glb',
        position: [1, 2, 3],
        rotation: [0, 0.5, 0],
        scale: [1, 1, 1]
      }
    ]

    const manifest = parseWorldManifest(fixture)
    expect(manifest.externalDecorations).toEqual(
      fixture.externalDecorations.map((entry: any) => ({
        id: entry.id,
        uri: entry.uri,
        position: { x: 1, y: 2, z: 3 },
        rotation: { x: 0, y: 0.5, z: 0 },
        scale: { x: 1, y: 1, z: 1 }
      }))
    )

    fixture.externalDecorations.push(structuredClone(fixture.externalDecorations[0]))
    expect(() => parseWorldManifest(fixture)).toThrow(/external decoration id kenney-crate is duplicated/)
  })

  it.each([
    ['model', (fixture: any) => (fixture.models[0].uri = '../moon-settlement-approved.jpg?cache=1')],
    ['navigation', (fixture: any) => (fixture.navigation.meshUri = './MOON-SETTLEMENT-APPROVED.JPG#nav')],
    ['texture', (fixture: any) => (fixture.textures[0].uri = 'moon-settlement-approved.jpg?cache=1')],
    [
      'future runtime URI',
      (fixture: any) => (fixture.runtimeExtension = { billboardUri: 'moon-settlement-approved.jpg' })
    ]
  ])('rejects the approved JPG in the %s runtime URI field', (_label, mutate) => {
    const fixture = structuredClone(actualManifest)
    mutate(fixture)

    expect(() => parseWorldManifest(fixture)).toThrow(/approved source cannot be a runtime asset/)
  })

  it.each([
    ['remote model URL', (fixture: any) => (fixture.models[0].uri = 'https://example.test/terrain.glb')],
    ['model traversal', (fixture: any) => (fixture.models[0].uri = '../models/terrain.glb')],
    ['wrong model family', (fixture: any) => (fixture.models[0].uri = 'textures/terrain.glb')],
    ['navigation traversal', (fixture: any) => (fixture.navigation.meshUri = 'models/../navigation.glb')],
    ['wrong texture family', (fixture: any) => (fixture.textures[0].uri = 'models/approved-palette.png')]
  ])('rejects a runtime asset outside the declared v2 pack: %s', (_label, mutate) => {
    const fixture = structuredClone(actualManifest)
    mutate(fixture)

    expect(() => parseWorldManifest(fixture)).toThrow(/must be a relative v2 (model|texture) asset/)
  })

  it('rejects malformed transforms instead of silently replacing them', () => {
    const fixture = structuredClone(actualManifest) as any
    fixture.models[0].transform.position = [0, Number.NaN, 0]

    expect(() => parseWorldManifest(fixture)).toThrow(/models\[0\]\.transform\.position\[1\] must be a finite number/)
  })

  it('rejects duplicate model identities', () => {
    const fixture = structuredClone(actualManifest) as any
    fixture.models[1].id = fixture.models[0].id

    expect(() => parseWorldManifest(fixture)).toThrow(/model id terrain is duplicated/)
  })
})

describe('loadWorldManifest', () => {
  it('fetches and parses the manifest through the provided URL', async () => {
    const fetcher = vi.fn(async () => new Response(JSON.stringify(actualManifest), { status: 200 }))
    vi.stubGlobal('fetch', fetcher)

    const manifest = await loadWorldManifest('./lunar-city/v2/world-manifest.v2.json')

    expect(fetcher).toHaveBeenCalledWith('./lunar-city/v2/world-manifest.v2.json', {
      signal: undefined
    })
    expect(manifest.models).toHaveLength(15)
  })

  it('reports an unsuccessful manifest response without parsing it', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response('missing', { status: 404, statusText: 'Not Found' }))
    )

    await expect(loadWorldManifest('./lunar-city/v2/world-manifest.v2.json')).rejects.toThrow(
      /Lunar City manifest request failed: 404 Not Found/
    )
  })
})

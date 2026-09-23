import { describe, expect, it, vi } from 'vitest'

import actualManifest from '../../../../public/lunar-city/v2/world-manifest.v2.json'
import { parseWorldManifest } from '../manifest'
import type {
  BabylonImportResultLike,
  BabylonNodeLike,
  LunarCitySnapshot,
  LunarCityWorldModules,
  LunarEntity,
  ModelManifestEntry,
  NavigationManifest,
  RecastRuntimeLike,
  Vec3,
  WorldBounds
} from '../model'

import { createProjectCompoundController } from './project-compounds'
import { createManifestNavigationQuery, createRouteNavigationQuery } from './world-navigation'
import {
  createBabylonEntityFactory,
  leaderCameraAnchor,
  transformManifestPoint,
  worldBoundsFromModel
} from './world-scene'

describe('physical leader camera anchors', () => {
  it('derives a distinct world anchor from each structured leader node instead of sharing the pack anchor', () => {
    const owl = { name: 'owl', getWorldMatrix: () => ({ m: [0, 2, 0, 0, -3, 0, 0, 0, 0, 0, 4, 0, 7, 3, 24, 1] }) }
    const fox = { name: 'fox', getWorldMatrix: () => ({ m: [0, 2, 0, 0, -3, 0, 0, 0, 0, 0, 4, 0, 15, 3, 18, 1] }) }

    expect(leaderCameraAnchor(owl, { x: 0, y: 0, z: 0 })).toEqual({ x: 7, y: 3, z: 24 })
    expect(leaderCameraAnchor(fox, { x: 0, y: 0, z: 0 })).toEqual({ x: 15, y: 3, z: 18 })
  })
})

const model = {
  transform: {
    position: { x: 11, y: -3, z: 5 },
    rotation: { x: 0.37, y: -0.61, z: 0.82 },
    scale: { x: 2, y: 3, z: 4 }
  },
  bounds: {
    min: { x: -1, y: -2, z: -0.5 },
    max: { x: 4, y: 1, z: 3 }
  }
} as Pick<ModelManifestEntry, 'bounds' | 'transform'>

function expectVectorClose(actual: Vec3, expected: Vec3): void {
  expect(actual.x).toBeCloseTo(expected.x, 5)
  expect(actual.y).toBeCloseTo(expected.y, 5)
  expect(actual.z).toBeCloseTo(expected.z, 5)
}

function expectBoundsClose(actual: WorldBounds, expected: WorldBounds): void {
  expectVectorClose(actual.min, expected.min)
  expectVectorClose(actual.max, expected.max)
}

describe('manifest world transforms', () => {
  it('applies Babylon y-x-z rotation and non-uniform scale to a local camera anchor', () => {
    expectVectorClose(transformManifestPoint(model, { x: 3, y: -2, z: 1 }), {
      x: 15.75357037782669,
      y: -4.172778964042664,
      z: 13.001759648323059
    })
  })

  it('builds an occlusion world AABB from all eight rotated asymmetric local corners', () => {
    expectBoundsClose(worldBoundsFromModel(model), {
      min: { x: 1.5535336136817932, y: -12.51904046535492, z: 1.6066522002220154 },
      max: { x: 19.773607850074768, y: 5.084729433059692, z: 20.330265641212463 }
    })
  })
})

describe('manifest navigation query', () => {
  it('routes the library and laboratory destinations over the authored links', () => {
    const manifest = parseWorldManifest(actualManifest)
    const query = createManifestNavigationQuery(manifest.navigation)

    for (const destination of ['library', 'lab']) {
      const target = manifest.destinations[destination]!
      expect(target).toBeDefined()
      const route = query.computePath(manifest.destinations.bus!, target)
      expect(route?.at(-1)).toEqual(target)
    }
  })

  const navigation: NavigationManifest = {
    areas: ['walkable'],
    links: [
      { bidirectional: true, from: { x: 0, y: 0, z: 0 }, to: { x: 2, y: 0, z: 0 } },
      { bidirectional: false, from: { x: 2, y: 0, z: 0 }, to: { x: 4, y: 0, z: 0 } }
    ],
    meshUri: 'models/navigation.glb'
  }

  it('uses only declared navigation links and never invents a direct route through city geometry', () => {
    const query = createManifestNavigationQuery(navigation)

    expect(query.computePath({ x: 0, y: 0, z: 0 }, { x: 4, y: 0, z: 0 })).toEqual([
      { x: 0, y: 0, z: 0 },
      { x: 2, y: 0, z: 0 },
      { x: 4, y: 0, z: 0 }
    ])
    expect(query.computePath({ x: 4, y: 0, z: 0 }, { x: 0, y: 0, z: 0 })).toBeUndefined()
    expect(query.computePath({ x: 1, y: 0, z: 0 }, { x: 4, y: 0, z: 0 })).toBeUndefined()
  })

  it('builds the route-local Recast query from the declared GLB and releases the temporary mesh', async () => {
    const build = vi.fn()
    const destroy = vi.fn()
    const dispose = vi.fn()
    const vectors: FakeVector3[] = []
    const configs: FakeConfig[] = []

    class FakeVector3 {
      constructor(
        public readonly x: number,
        public readonly y: number,
        public readonly z: number
      ) {
        vectors.push(this)
      }

      destroy = vi.fn()
    }

    class FakeNavMesh {
      build = build
      computePath = vi.fn(() => ({
        getPoint: (index: number) => {
          if (index === 0) {
            return new FakeVector3(0, 0, 0)
          }

          if (index === 1) {
            return new FakeVector3(4, 0, 0)
          }

          return undefined
        },
        getPointCount: () => 2,
        destroy: vi.fn()
      }))
      destroy = destroy
    }

    class FakeConfig {
      constructor() {
        configs.push(this)
      }

      destroy = vi.fn()
      set_bmax = vi.fn()
      set_bmin = vi.fn()
    }

    const result: BabylonImportResultLike = {
      animationGroups: [],
      meshes: [
        {
          dispose,
          getIndices: () => [0, 1, 2],
          getVerticesData: () => [0, 0, 0, 4, 0, 0, 0, 0, 4],
          getWorldMatrix: () => ({
            m: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 0, 0, 1]
          }),
          name: 'navigation:surface',
          setEnabled: vi.fn()
        }
      ],
      transformNodes: []
    }

    const modules = {
      ImportMeshAsync: vi.fn(async () => result),
      createRecastNavigation: vi.fn(async () => ({ NavMesh: FakeNavMesh, Vec3: FakeVector3, rcConfig: FakeConfig }))
    } as unknown as Pick<LunarCityWorldModules, 'ImportMeshAsync' | 'createRecastNavigation'>

    const query = await createRouteNavigationQuery(navigation, modules, {} as never, uri => `route/${uri}`)

    expect(modules.ImportMeshAsync).toHaveBeenCalledWith('route/models/navigation.glb', expect.anything())
    expect(build).toHaveBeenCalledWith(expect.any(Float32Array), 3, expect.any(Uint32Array), 3, expect.any(FakeConfig))
    expect(Array.from(build.mock.calls[0]?.[0] ?? [])).toEqual([10, 0, 0, 14, 0, 0, 10, 0, 4])
    expect(configs[0]?.set_bmin.mock.calls).toEqual([
      [0, 10],
      [1, 0],
      [2, 0]
    ])
    expect(configs[0]?.set_bmax.mock.calls).toEqual([
      [0, 14],
      [1, 0],
      [2, 4]
    ])
    expect(configs[0]?.destroy).toHaveBeenCalledOnce()
    expect(query.computePath({ x: 0, y: 0, z: 0 }, { x: 4, y: 0, z: 0 })).toEqual([
      { x: 0, y: 0, z: 0 },
      { x: 4, y: 0, z: 0 }
    ])
    expect(dispose).toHaveBeenCalledOnce()
    expect(vectors.map(vector => vector.destroy.mock.calls.length)).toEqual([1, 1, 0, 0])

    query.dispose?.()
    expect(destroy).toHaveBeenCalledOnce()
  })

  it('builds and disposes a nonzero route through the actual Recast WASM runtime', async () => {
    const imported = await import('recast-detour')
    const createRecast = imported.default as unknown as () => Promise<RecastRuntimeLike>
    const runtime = await createRecast()
    const OriginalNavMesh = runtime.NavMesh
    const destroy = vi.fn()

    class TrackingNavMesh {
      private readonly delegate = new OriginalNavMesh()

      build(
        positions: Float32Array,
        positionCount: number,
        indices: Uint32Array,
        indexCount: number,
        configuration: unknown
      ): void {
        this.delegate.build(positions, positionCount, indices, indexCount, configuration)
      }

      computePath(from: unknown, to: unknown) {
        return this.delegate.computePath(from, to)
      }

      destroy(): void {
        destroy()
        this.delegate.destroy?.()
      }
    }

    const result: BabylonImportResultLike = {
      animationGroups: [],
      meshes: [
        {
          dispose: vi.fn(),
          getIndices: () => [0, 1, 2, 1, 3, 2],
          getVerticesData: () => [0, 0, 0, 4, 0, 0, 0, 0, 4, 4, 0, 4],
          name: 'navigation:surface',
          setEnabled: vi.fn()
        }
      ],
      transformNodes: []
    }

    const modules = {
      ImportMeshAsync: vi.fn(async () => result),
      createRecastNavigation: vi.fn(async () => ({ ...runtime, NavMesh: TrackingNavMesh }))
    } as unknown as Pick<LunarCityWorldModules, 'ImportMeshAsync' | 'createRecastNavigation'>

    const query = await createRouteNavigationQuery(navigation, modules, {} as never, uri => uri)
    const path = query.computePath({ x: 0.1, y: 0, z: 0.1 }, { x: 3.9, y: 0, z: 3.9 })

    expect(path).toHaveLength(2)
    expect(path?.[0]?.x).toBeGreaterThanOrEqual(0)
    expect(path?.at(-1)?.z).toBeGreaterThan(3)

    query.dispose?.()
    expect(destroy).toHaveBeenCalledOnce()
  })
})

describe('retained project compounds', () => {
  it('retains one declared-position compound per canonical connection/project and excludes unplaced work', () => {
    const snapshot: LunarCitySnapshot = {
      entities: new Map([
        [
          'kanban:one' as never,
          {
            animation: 'work',
            authority: 'authoritative',
            destination: 'project',
            identity: {
              board: 'main',
              connectionId: 'source-a',
              kind: 'kanban',
              profile: 'default',
              taskId: 'task-one'
            },
            key: 'kanban:one' as never,
            observedAt: 1,
            position: { x: 16, y: 0, z: 38 },
            projectId: 'project-alpha'
          }
        ],
        [
          'kanban:two' as never,
          {
            animation: 'work',
            authority: 'authoritative',
            destination: 'project',
            identity: {
              board: 'main',
              connectionId: 'source-a',
              kind: 'kanban',
              profile: 'research',
              runId: 'run-two',
              taskId: 'task-two'
            },
            key: 'kanban:two' as never,
            observedAt: 1,
            position: { x: 16, y: 0, z: 38 },
            projectId: 'project-alpha'
          }
        ],
        [
          'kanban:overflow' as never,
          {
            animation: 'unavailable',
            authority: 'partial',
            destination: 'unknown',
            identity: {
              board: 'main',
              connectionId: 'source-a',
              kind: 'kanban',
              profile: 'default',
              taskId: 'task-overflow'
            },
            key: 'kanban:overflow' as never,
            observedAt: 1,
            projectId: 'project-overflow'
          }
        ]
      ]),
      observedAt: 1,
      revision: 1,
      sources: []
    }

    expect(createProjectCompoundController(parseWorldManifest(structuredClone(actualManifest)).projectSlots.slice(0, 1)).update(snapshot).anchors).toEqual([
      {
        connectionId: 'source-a',
        key: 'compound:connection:string:8:source-a:project:string:13:project-alpha',
        position: { x: 16, y: 0, z: 38 },
        projectId: 'project-alpha'
      }
    ])
  })
})

describe('worker GLB presentation', () => {
  it('fails closed when a manifest-declared physical signature root is absent from the worker GLB', () => {
    const assets = parseWorldManifest(structuredClone(actualManifest)).characterAssets

    const result: BabylonImportResultLike = {
      animationGroups: [],
      meshes: [],
      transformNodes: [{ name: 'workers:root', setEnabled: vi.fn() }]
    }

    expect(() =>
      createBabylonEntityFactory(
        {
          id: 'workers',
          instancing: { eligible: true, variants: [] },
          lods: [],
          occlusionGroup: 'workers'
        } as unknown as ModelManifestEntry,
        result,
        { TransformNode: class {} } as unknown as LunarCityWorldModules,
        {} as never,
        assets
      )
    ).toThrow(/missing manifest-declared physical signature root/)
  })

  it.each(['missing', 'duplicate'] as const)('fails closed when a manifest-declared kit root is %s', fault => {
    const assets = parseWorldManifest(structuredClone(actualManifest)).characterAssets

    const engineeringAssets = {
      ...assets,
      groupKits: assets.groupKits.filter(kit => kit.kitId === 'engineering-guild')
    }

    const signatureRoots = [
      ...Object.values(assets.physicalVariantRoots.body),
      ...Object.values(assets.physicalVariantRoots.head),
      ...Object.values(assets.physicalVariantRoots.palette)
    ].map(name => ({ name, setEnabled: vi.fn() }))

    const kitNames = [
      'worker:group-kit:engineering-guild',
      'worker:group-kit:engineering-guild:silhouette',
      'worker:group-kit:engineering-guild:emblem',
      'worker:group-kit:engineering-guild:identity-accent'
    ]

    const kitRoots = kitNames.map(name => ({ name, setEnabled: vi.fn() }))

    if (fault === 'missing') {
      kitRoots.pop()
    } else {
      kitRoots.push({ name: kitNames[1]!, setEnabled: vi.fn() })
    }

    const result: BabylonImportResultLike = {
      animationGroups: [],
      meshes: [],
      transformNodes: [{ name: 'workers:root', setEnabled: vi.fn() }, ...signatureRoots, ...kitRoots]
    }

    expect(() =>
      createBabylonEntityFactory(
        {
          id: 'workers',
          instancing: { eligible: true, variants: [] },
          lods: [],
          occlusionGroup: 'workers'
        } as unknown as ModelManifestEntry,
        result,
        { TransformNode: class {} } as unknown as LunarCityWorldModules,
        {} as never,
        engineeringAssets
      )
    ).toThrow(/missing or duplicate manifest-declared kit roots/)
  })

  it('instances a shared mid-LOD base and exact identity accent without allocating profile resources', () => {
    const assets = parseWorldManifest(structuredClone(actualManifest)).characterAssets

    const engineeringAssets = {
      ...assets,
      groupKits: assets.groupKits.filter(kit => kit.kitId === 'engineering-guild')
    }

    const template = { name: 'workers:root', setEnabled: vi.fn() }
    const near = { name: 'workers:lod:near', setEnabled: vi.fn() }
    const mid = { name: 'workers:lod:mid', setEnabled: vi.fn() }
    const far = { name: 'workers:lod:far', setEnabled: vi.fn() }
    const bodyCompact = { name: 'worker:body-variant:compact', setEnabled: vi.fn() }
    const bodyStandard = { name: 'worker:body-variant:standard', setEnabled: vi.fn() }
    const headOrb = { name: 'worker:head-variant:orb', setEnabled: vi.fn() }
    const headVisor = { name: 'worker:head-variant:visor', setEnabled: vi.fn() }

    const paletteRust = {
      name: 'worker:palette:rust-bone',
      rotation: { set: vi.fn() },
      scaling: { set: vi.fn() },
      setEnabled: vi.fn()
    }

    const paletteViolet = { name: 'worker:palette:violet-cyan', setEnabled: vi.fn() }
    const kit = { name: 'worker:group-kit:engineering-guild', setEnabled: vi.fn() }

    const silhouette = {
      name: 'worker:group-kit:engineering-guild:silhouette',
      parent: kit,
      setEnabled: vi.fn()
    }

    const emblem = { name: 'worker:group-kit:engineering-guild:emblem', parent: kit, setEnabled: vi.fn() }

    const accent = {
      name: 'worker:group-kit:engineering-guild:identity-accent',
      parent: emblem,
      setEnabled: vi.fn()
    }

    const baseInstance = { dispose: vi.fn(), name: 'mid-base:instance', parent: null, setEnabled: vi.fn() }

    const accentInstance = {
      dispose: vi.fn(),
      name: 'mid-accent:instance',
      parent: null,
      rotation: { set: vi.fn() },
      scaling: { set: vi.fn() },
      setEnabled: vi.fn()
    }

    const midBase = {
      createInstance: vi.fn(() => baseInstance),
      name: 'workers:lod:mid:body',
      parent: mid,
      setEnabled: vi.fn()
    }

    const accentMesh = {
      createInstance: vi.fn(() => accentInstance),
      name: 'worker:group-kit:engineering-guild:identity-accent:mesh',
      parent: accent,
      setEnabled: vi.fn()
    }

    const result: BabylonImportResultLike = {
      animationGroups: [],
      meshes: [midBase, accentMesh],
      transformNodes: [
        template,
        near,
        mid,
        far,
        bodyCompact,
        bodyStandard,
        headOrb,
        headVisor,
        paletteRust,
        paletteViolet,
        kit,
        silhouette,
        emblem,
        accent
      ]
    }

    class TransformNode {
      dispose = vi.fn()
      name: string
      position = { set: vi.fn() }
      constructor(name: string) {
        this.name = name
      }
    }

    const factory = createBabylonEntityFactory(
      {
        id: 'workers',
        instancing: { eligible: true, variants: [] },
        lods: [
          { distance: 0, node: near.name },
          { distance: 28, node: mid.name },
          { distance: 80, node: far.name }
        ],
        occlusionGroup: 'workers'
      } as unknown as ModelManifestEntry,
      result,
      { TransformNode } as unknown as LunarCityWorldModules,
      {} as never,
      engineeringAssets
    )

    const group = factory.createInstancedGroup('worker:idle:kit:engineering-guild:lod:1')

    group.sync([
      {
        animation: 'idle',
        character: {
          accentCode: 42,
          identityAccent: 'deadbeef',
          kitId: 'engineering-guild',
          lod: 'mid',
          renderMode: 'instanced',
          signature: {
            body: 'standard',
            emblem: 'engineering-bridge',
            head: 'visor',
            palette: 'violet-cyan',
            silhouetteAccessory: 'engineering-hammer'
          },
          visibleSignature: 'standard:visor:engineering-hammer:violet-cyan:engineering-bridge:deadbeef'
        },
        identity: { connectionId: 'local', kind: 'profile', profile: 'worker' },
        key: 'profile:worker' as never,
        position: { x: 1, y: 0, z: 2 },
        variant: undefined
      }
    ])

    expect(midBase.createInstance).toHaveBeenCalledOnce()
    expect(accentMesh.createInstance).toHaveBeenCalledOnce()
    expect(accentInstance.rotation.set).toHaveBeenCalledOnce()
    expect(accentInstance.scaling.set).toHaveBeenCalledOnce()
  })

  it('activates exactly the selected physical variant subtree and releases cloned animation groups', () => {
    const orbital = {
      metadata: { gltf: { extras: { activationScale: [1, 1, 1] } } },
      name: 'worker:variant:orbital',
      setEnabled: vi.fn()
    }

    const builder = {
      metadata: { gltf: { extras: { activationScale: [1, 1, 1] } } },
      name: 'worker:variant:builder',
      setEnabled: vi.fn()
    }

    const near = { name: 'workers:lod:near', setEnabled: vi.fn() }
    const far = { name: 'workers:lod:far', setEnabled: vi.fn() }
    const engineering = { name: 'worker:group-kit:engineering-guild', setEnabled: vi.fn() }
    const research = { name: 'worker:group-kit:research-lab', setEnabled: vi.fn() }
    const engineeringSilhouette = { name: 'worker:group-kit:engineering-guild:silhouette', setEnabled: vi.fn() }
    const engineeringEmblem = { name: 'worker:group-kit:engineering-guild:emblem', setEnabled: vi.fn() }
    const researchSilhouette = { name: 'worker:group-kit:research-lab:silhouette', setEnabled: vi.fn() }
    const researchEmblem = { name: 'worker:group-kit:research-lab:emblem', setEnabled: vi.fn() }
    const bodyCompact = { name: 'worker:body-variant:compact', setEnabled: vi.fn() }
    const bodyStandard = { name: 'worker:body-variant:standard', setEnabled: vi.fn() }
    const headOrb = { name: 'worker:head-variant:orb', setEnabled: vi.fn() }
    const headVisor = { name: 'worker:head-variant:visor', setEnabled: vi.fn() }

    const paletteRust = {
      name: 'worker:palette:rust-bone',
      rotation: { set: vi.fn() },
      scaling: { set: vi.fn() },
      setEnabled: vi.fn()
    }

    const paletteViolet = { name: 'worker:palette:violet-cyan', setEnabled: vi.fn() }

    const accent = {
      name: 'worker:group-kit:engineering-guild:identity-accent',
      rotation: { set: vi.fn() },
      scaling: { set: vi.fn() },
      setEnabled: vi.fn()
    }

    const researchAccent = {
      name: 'worker:group-kit:research-lab:identity-accent',
      rotation: { set: vi.fn() },
      scaling: { set: vi.fn() },
      setEnabled: vi.fn()
    }

    const variantClones = new Map<object, { name: string; setEnabled: ReturnType<typeof vi.fn> }>()

    const template = {
      instantiateHierarchy: (
        parent: BabylonNodeLike,
        _options: unknown,
        onNewNodeCreated: (source: unknown, clone: BabylonNodeLike) => void
      ) => {
        const root = { name: 'worker-clone', dispose: vi.fn(), parent, setEnabled: vi.fn() }
        onNewNodeCreated(template, root)

        for (const source of [
          orbital,
          builder,
          near,
          far,
          engineering,
          research,
          engineeringSilhouette,
          engineeringEmblem,
          researchSilhouette,
          researchEmblem,
          bodyCompact,
          bodyStandard,
          headOrb,
          headVisor,
          paletteRust,
          paletteViolet,
          accent,
          researchAccent
        ]) {
          const clone = {
            name: `${source.name}:clone`,
            rotation: 'rotation' in source ? { set: vi.fn() } : undefined,
            scaling: 'scaling' in source ? { set: vi.fn() } : undefined,
            setEnabled: vi.fn()
          }

          variantClones.set(source, clone)
          onNewNodeCreated(source, clone)
        }

        return root
      },
      name: 'workers:root',
      setEnabled: vi.fn()
    }

    const sceneAnimationGroups: unknown[] = []

    const clonedAnimation = {
      dispose: vi.fn(() => {
        sceneAnimationGroups.splice(sceneAnimationGroups.indexOf(clonedAnimation), 1)
      }),
      start: vi.fn(),
      stop: vi.fn()
    }

    const animation = {
      clone: vi.fn(() => {
        sceneAnimationGroups.push(clonedAnimation)

        return clonedAnimation
      }),
      name: 'work'
    }

    const result: BabylonImportResultLike = {
      animationGroups: [animation],
      meshes: [],
      transformNodes: [
        template,
        orbital,
        builder,
        near,
        far,
        engineering,
        research,
        engineeringSilhouette,
        engineeringEmblem,
        researchSilhouette,
        researchEmblem,
        bodyCompact,
        bodyStandard,
        headOrb,
        headVisor,
        paletteRust,
        paletteViolet,
        accent,
        researchAccent
      ]
    }

    const model = {
      id: 'workers',
      instancing: { eligible: true, variants: ['orbital', 'builder'] },
      lods: [
        { distance: 0, node: 'workers:lod:near' },
        { distance: 28, node: 'workers:lod:far' }
      ],
      occlusionGroup: 'workers'
    } as Pick<ModelManifestEntry, 'id' | 'instancing' | 'lods'>

    class TransformNode {
      dispose = vi.fn()
      name: string
      position = { set: vi.fn() }
      constructor(name: string) {
        this.name = name
      }
    }

    const factory = createBabylonEntityFactory(
      model as ModelManifestEntry,
      result,
      { TransformNode } as unknown as LunarCityWorldModules,
      { animationGroups: sceneAnimationGroups } as never,
      {
        ...parseWorldManifest(structuredClone(actualManifest)).characterAssets,
        groupKits: parseWorldManifest(structuredClone(actualManifest)).characterAssets.groupKits.filter(kit =>
          ['engineering-guild', 'research-lab'].includes(kit.kitId)
        )
      }
    )

    const entity = {
      animation: 'work',
      authority: 'authoritative',
      destination: 'project',
      identity: { connectionId: 'local', kind: 'session', profile: 'worker', sessionId: 'one' },
      key: 'session:one' as never,
      observedAt: 1
    } as LunarEntity

    const character = {
      accentCode: 42,
      identityAccent: 'deadbeef',
      kitId: 'engineering-guild',
      lod: 'near' as const,
      renderMode: 'animated' as const,
      signature: {
        body: 'standard',
        emblem: 'engineering-bridge',
        head: 'orb',
        palette: 'rust-bone',
        silhouetteAccessory: 'engineering-hammer'
      },
      visibleSignature: 'standard:orb:engineering-hammer:rust-bone:engineering-bridge:deadbeef'
    }

    const visual = factory.createAnimated(entity, 'builder', character)
    visual.setAnimation?.('work')
    expect(sceneAnimationGroups).toHaveLength(1)
    visual.dispose?.()

    expect(variantClones.get(builder)?.setEnabled).toHaveBeenCalledWith(true)
    expect(variantClones.get(orbital)?.setEnabled).toHaveBeenCalledWith(false)
    expect(variantClones.get(engineering)?.setEnabled).toHaveBeenCalledWith(true)
    expect(variantClones.get(research)?.setEnabled).toHaveBeenCalledWith(false)
    expect(variantClones.get(bodyStandard)?.setEnabled).toHaveBeenCalledWith(true)
    expect(variantClones.get(bodyCompact)?.setEnabled).toHaveBeenCalledWith(false)
    expect(variantClones.get(headOrb)?.setEnabled).toHaveBeenCalledWith(true)
    expect(variantClones.get(headVisor)?.setEnabled).toHaveBeenCalledWith(false)
    expect(variantClones.get(paletteRust)?.setEnabled).toHaveBeenCalledWith(true)
    expect(variantClones.get(paletteViolet)?.setEnabled).toHaveBeenCalledWith(false)

    const accentClone = variantClones.get(accent) as
      { rotation?: { set: ReturnType<typeof vi.fn> }; scaling?: { set: ReturnType<typeof vi.fn> } } | undefined

    expect(accentClone?.rotation?.set).toHaveBeenCalledTimes(1)
    expect(accentClone?.scaling?.set).toHaveBeenCalledTimes(1)
    expect((variantClones.get(builder) as { metadata?: Record<string, unknown> } | undefined)?.metadata).toMatchObject({
      lunarCity: {
        entityKey: entity.key,
        focusEntityKey: entity.key,
        identity: entity.identity,
        kind: 'worker',
        character,
        occlusionGroup: 'workers',
        selectable: true
      }
    })
    expect(clonedAnimation.start).toHaveBeenCalledOnce()
    expect(clonedAnimation.dispose).toHaveBeenCalledOnce()
    expect(sceneAnimationGroups).toHaveLength(0)

    const neutral = factory.createAnimated(entity, 'builder', {
      accentCode: 7,
      identityAccent: 'neutral-key',
      lod: 'near',
      renderMode: 'animated',
      signature: { body: 'standard', head: 'orb', palette: 'rust-bone' },
      visibleSignature: 'standard:orb:neutral:rust-bone:neutral:neutral-key'
    })

    const neutralPalette = variantClones.get(paletteRust) as
      { rotation?: { set: ReturnType<typeof vi.fn> }; scaling?: { set: ReturnType<typeof vi.fn> } } | undefined

    expect(variantClones.get(engineering)?.setEnabled).toHaveBeenCalledWith(false)
    expect(neutralPalette?.rotation?.set).toHaveBeenCalledOnce()
    expect(neutralPalette?.scaling?.set).toHaveBeenCalledTimes(2)
    neutral.dispose?.()
  })
})

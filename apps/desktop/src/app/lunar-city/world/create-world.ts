import type { Scene } from '@babylonjs/core/scene'

import { assertWorldManifestRuntimeAssets } from '../manifest'
import type { LunarCityIntent, LunarCityWorldHandle, LunarCityWorldModules, WorldManifestV2 } from '../model'
import { buildRuntimeAssetDigests, importRuntimeAsset } from '../runtime-asset-integrity'

import { bindCameraInput, type CameraInputRelease } from './camera-controller'
import { createWorldScene } from './world-scene'

const DEFAULT_MANIFEST_URL = './lunar-city/v2/world-manifest.v2.json'

export function isLocalInteriorReview(manifestUrl: URL, documentUrl: URL): boolean {
  return manifestUrl.origin === documentUrl.origin &&
    ['http:', 'https:', 'file:'].includes(manifestUrl.protocol) &&
    manifestUrl.pathname.endsWith('/lunar-city/v2-review/world-manifest.v2.json')
}

export async function loadBabylonModules(): Promise<LunarCityWorldModules> {
  // Modular Babylon leaves Scene.pick as a no-op until ray side effects register.
  await import('@babylonjs/core/Culling/ray')

  const [
    engine,
    scene,
    sceneLoader,
    camera,
    vector,
    color,
    directionalLight,
    hemisphericLight,
    glowLayer,
    shadowGenerator,
    transformNode,
    instrumentation,
    ,
    recast
  ] = await Promise.all([
    import('@babylonjs/core/Engines/engine'),
    import('@babylonjs/core/scene'),
    import('@babylonjs/core/Loading/sceneLoader'),
    import('@babylonjs/core/Cameras/arcRotateCamera'),
    import('@babylonjs/core/Maths/math.vector'),
    import('@babylonjs/core/Maths/math.color'),
    import('@babylonjs/core/Lights/directionalLight'),
    import('@babylonjs/core/Lights/hemisphericLight'),
    import('@babylonjs/core/Layers/glowLayer'),
    import('@babylonjs/core/Lights/Shadows/shadowGenerator'),
    import('@babylonjs/core/Meshes/transformNode'),
    import('@babylonjs/core/Instrumentation/sceneInstrumentation'),
    import('@babylonjs/loaders/glTF'),
    import('recast-detour')
  ])

  const { createProjectMarker } = await import('./project-marker')
  const { createAtmosphere } = await import('./atmosphere')

  return {
    createProjectMarker,
    createAtmosphere,
    ArcRotateCamera: camera.ArcRotateCamera,
    Color3: color.Color3,
    Color4: color.Color4,
    DirectionalLight: directionalLight.DirectionalLight,
    HemisphericLight: hemisphericLight.HemisphericLight,
    GlowLayer: glowLayer.GlowLayer,
    ShadowGenerator: shadowGenerator.ShadowGenerator,
    Engine: engine.Engine,
    ImportMeshAsync: sceneLoader.ImportMeshAsync,
    Scene: scene.Scene,
    SceneInstrumentation: instrumentation.SceneInstrumentation,
    TransformNode: transformNode.TransformNode,
    Vector3: vector.Vector3,
    createRecastNavigation: recast.default
  } as unknown as LunarCityWorldModules
}

function absoluteManifestUrl(manifestUrl: string): URL {
  const documentBase = typeof document === 'undefined' ? 'http://localhost/' : document.baseURI

  try {
    return new URL(manifestUrl, documentBase)
  } catch {
    return new URL(manifestUrl, 'http://localhost/')
  }
}

export async function createLunarCityWorld(
  canvas: HTMLCanvasElement,
  manifest: WorldManifestV2,
  emit: (intent: LunarCityIntent) => void,
  modules?: LunarCityWorldModules,
  manifestUrl = DEFAULT_MANIFEST_URL
): Promise<LunarCityWorldHandle> {
  assertWorldManifestRuntimeAssets(manifest)
  const loaded = modules ?? (await loadBabylonModules())

  const engine = new loaded.Engine(canvas, true, {
    alpha: true,
    powerPreference: 'low-power',
    preserveDrawingBuffer: false,
    stencil: false
  })

  const resolvedManifestUrl = absoluteManifestUrl(manifestUrl)
  let world: Awaited<ReturnType<typeof createWorldScene>> | undefined
  let observer: ResizeObserver | undefined
  let resizeListener: (() => void) | undefined
  let releaseCameraInput: CameraInputRelease | undefined
  let contextLostListener: ((event: Event) => void) | undefined
  let contextRestoredListener: (() => void) | undefined
  let ownedListenerCount = 0
  let destroyed = false

  try {
    let runtimeModules = loaded

    if (!modules) {
      const hasReviews = !!(manifest.reviewLeaderAssets?.length || manifest.reviewWorkerAssets?.length)
      const receiptsResponse = hasReviews ? await fetch(new URL('review-sources.json', resolvedManifestUrl)) : undefined

      if (receiptsResponse && !receiptsResponse.ok) {throw new Error('Lunar City review receipts unavailable')}
      const receipts = receiptsResponse ? await receiptsResponse.json() : undefined
      const digests = buildRuntimeAssetDigests(manifest, resolvedManifestUrl.href, receipts)
      runtimeModules = { ...loaded, ImportMeshAsync: (url, scene) => importRuntimeAsset(url, digests,
        (verifiedUrl, pluginExtension) => loaded.ImportMeshAsync(verifiedUrl, scene, { pluginExtension })) }

      if (isLocalInteriorReview(resolvedManifestUrl, new URL(typeof document === 'undefined' ? 'http://localhost/' : document.baseURI))) {
        const { eligibleInteriorPlans, interiorPlanProvenance } = await import('./interior-plan-data')
        const interiorPlans = eligibleInteriorPlans(manifest)
        const { createInteriorCutaway } = await import('./interior-cutaway')
        runtimeModules = { ...runtimeModules, interiorPlans,
          createInteriorCutaway: (scene, meshes) => createInteriorCutaway(
            scene as unknown as Scene, interiorPlans, meshes, interiorPlanProvenance) }
      }
    }

    world = await createWorldScene(engine, manifest, emit, runtimeModules, uri => new URL(uri, resolvedManifestUrl).toString())
    releaseCameraInput = bindCameraInput(canvas, {
      dispatch(intent) {
        world?.dispatchCamera(intent)
      },
      pick(clientX, clientY) {
        return world?.pick(clientX, clientY)
      }
    })

    contextLostListener = event => {
      event.preventDefault()
      world?.setVisible(false)
    }

    contextRestoredListener = () => world?.setVisible(true)
    canvas.addEventListener('webglcontextlost', contextLostListener)
    canvas.addEventListener('webglcontextrestored', contextRestoredListener)
    ownedListenerCount += 2

    const resize = () => {
      if (destroyed) {
        return
      }

      engine.resize()
      world?.render()
    }

    resizeListener = resize
    window.addEventListener('resize', resize)
    ownedListenerCount += 1

    if (typeof ResizeObserver !== 'undefined') {
      observer = new ResizeObserver(resize)
      observer.observe(canvas)
    }

    world.render()

    const handle: LunarCityWorldHandle = {
      setLeaderLifeMode: (id, mode) => world?.setLeaderLifeMode(id, mode),
      getLeaderLife: id => world?.getLeaderLife(id),
      setInteriorBuilding: id => world?.setInteriorBuilding(id) ?? false,
      getInteriorBuildings: () => world?.getInteriorBuildings() ?? [],
      leaderStateClips: world.leaderStateClips,
      requestWorkerInteraction: (key, gesture) => world?.requestWorkerInteraction(key, gesture) ?? false,
      getWorkerEncounters: () => world?.getWorkerEncounters() ?? [],
      getWorkerPresentation: key => world?.getWorkerPresentation(key),
      reviewWorkerClips: world.reviewWorkerClips,
      setReviewWorkerAnimation(id, state) {
        world?.setReviewWorkerAnimation(id, state)
      },
      applySnapshot(snapshot) {
        world?.applySnapshot(snapshot)
      },
      dispatchCamera(intent) {
        world?.dispatchCamera(intent)
      },
      getEntityCameraOrder() {
        return world?.getEntityCameraOrder() ?? []
      },
      getCameraState() {
        return world?.getCameraState() ?? { focusedEntityKey: undefined, following: false }
      },
      projectWorldPoint(point) {
        return world?.projectWorldPoint?.(point)
      },
      getPerfSnapshot() {
        const metrics = world?.getPerfSnapshot()

        return metrics
          ? {
              ...metrics,
              listeners: metrics.listeners + ownedListenerCount + (releaseCameraInput?.activeListenerCount() ?? 0)
            }
          : {
              activeAnimations: 0,
              drawCalls: 0,
              entities: 0,
              frameMs: 0,
              frameTimestampsMs: [],
              listeners: 0,
              rafs: 0,
              renderFrames: 0,
              targetFps: 0,
              textures: 0,
              timers: 0,
              visibleTriangles: 0,
              worldUpdateMs: 0,
              worldUpdateTimestampsMs: []
            }
      },
      setLeaderAnimation(leaderId, state) {
        world?.setLeaderAnimation(leaderId, state)
      },
      setQuality(tier) {
        world?.setQuality(tier)
      },
      setReducedMotion(reduced) {
        world?.setReducedMotion(reduced)
      },
      destroy() {
        if (destroyed) {
          return
        }

        destroyed = true

        if (resizeListener) {
          window.removeEventListener('resize', resizeListener)
          ownedListenerCount -= 1
        }

        releaseCameraInput?.()

        if (contextLostListener) {
          canvas.removeEventListener('webglcontextlost', contextLostListener)
          ownedListenerCount -= 1
        }

        if (contextRestoredListener) {
          canvas.removeEventListener('webglcontextrestored', contextRestoredListener)
          ownedListenerCount -= 1
        }

        observer?.disconnect()
        world?.dispose()
        engine.dispose()
        world = undefined
        observer = undefined
        resizeListener = undefined
        releaseCameraInput = undefined
        contextLostListener = undefined
        contextRestoredListener = undefined
      }
    }

    return handle
  } catch (error) {
    destroyed = true

    if (resizeListener) {
      window.removeEventListener('resize', resizeListener)
      ownedListenerCount -= 1
    }

    releaseCameraInput?.()

    if (contextLostListener) {
      canvas.removeEventListener('webglcontextlost', contextLostListener)
      ownedListenerCount -= 1
    }

    if (contextRestoredListener) {
      canvas.removeEventListener('webglcontextrestored', contextRestoredListener)
      ownedListenerCount -= 1
    }

    observer?.disconnect()
    world?.dispose()
    engine.dispose()
    throw error
  }
}

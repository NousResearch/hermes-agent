interface BabylonPerfCounterLike {
  current?: unknown
}

interface BabylonScenePerfLike {
  _activeIndices?: BabylonPerfCounterLike
  meshes?: readonly unknown[]
  textures?: readonly unknown[]
}

interface BabylonInstrumentationLike {
  dispose(): void
  drawCallsCounter?: BabylonPerfCounterLike
}

function natural(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : 0
}

/** The only adapter allowed to translate Babylon counters into receipt facts. */
export function createBabylonPerfAdapter(
  scene: BabylonScenePerfLike,
  instrumentation: BabylonInstrumentationLike | undefined
) {
  return {
    dispose(): void {
      instrumentation?.dispose()
    },
    snapshot() {
      return {
        drawCalls: natural(instrumentation?.drawCallsCounter?.current),
        entities: scene.meshes?.length ?? 0,
        textures: scene.textures?.length ?? 0,
        visibleTriangles: natural(scene._activeIndices?.current) / 3
      }
    }
  }
}

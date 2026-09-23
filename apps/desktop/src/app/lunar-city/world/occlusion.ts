import type { Vec3 } from '../model'

export interface OcclusionMaterial {
  alpha?: number
  clone?(): OcclusionMaterial | undefined
}

export interface OcclusionCamera {
  position: Vec3
}

export interface OcclusionSelection {
  cameraAnchor: Vec3
  occlusionGroup: string
}

export interface OcclusionCandidate {
  group: string
  material: OcclusionMaterial
  isolateMaterial?: boolean
  assignMaterial?(material: OcclusionMaterial): void
  intersectsFocusRay(camera: OcclusionCamera, selection: OcclusionSelection): boolean
}

export interface OcclusionController {
  clear(): void
  update(camera: OcclusionCamera, selection: OcclusionSelection | undefined): void
}

interface ManagedCandidate {
  candidate: OcclusionCandidate
  fadedMaterial: OcclusionMaterial | undefined
  isolatedMaterial: OcclusionMaterial | undefined
  originalAlpha: number
  originalMaterial: OcclusionMaterial
}

const FADED_ALPHA = 0.26

function occludes(group: string): boolean {
  return /(?:roof|wall)/iu.test(group)
}

export function createOcclusionController(candidates: readonly OcclusionCandidate[]): OcclusionController {
  const managed = candidates.map<ManagedCandidate>(candidate => ({
    candidate,
    fadedMaterial: undefined,
    isolatedMaterial: undefined,
    originalAlpha: candidate.material.alpha ?? 1,
    originalMaterial: candidate.material
  }))

  const faded = new Set<ManagedCandidate>()

  const restore = (): void => {
    for (const candidate of faded) {
      const material = candidate.fadedMaterial ?? candidate.originalMaterial

      material.alpha = candidate.originalAlpha

      if (material !== candidate.originalMaterial) {
        candidate.candidate.assignMaterial?.(candidate.originalMaterial)
      }

      candidate.fadedMaterial = undefined
    }

    faded.clear()
  }

  return {
    clear: restore,
    update(camera, selection) {
      restore()

      if (!selection) {
        return
      }

      for (const candidate of managed) {
        if (
          candidate.candidate.group !== selection.occlusionGroup &&
          occludes(candidate.candidate.group) &&
          candidate.candidate.intersectsFocusRay(camera, selection)
        ) {
          const material = candidate.candidate.isolateMaterial
            ? (candidate.isolatedMaterial ??= candidate.originalMaterial.clone?.() ?? candidate.originalMaterial)
            : candidate.originalMaterial

          if (material !== candidate.originalMaterial) {
            candidate.candidate.assignMaterial?.(material)
          }

          material.alpha = Math.min(candidate.originalAlpha, FADED_ALPHA)
          candidate.fadedMaterial = material
          faded.add(candidate)
        }
      }
    }
  }
}

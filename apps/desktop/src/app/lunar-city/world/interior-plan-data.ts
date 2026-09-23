import authoredPlans from '../../../../public/lunar-city/interior-plans-v1/plans.json'
import type { WorldManifestV2 } from '../model'

export const interiorPlanProvenance = authoredPlans.worldManifestSha256
export type AuthoredInteriorPlan = (typeof authoredPlans.plans)[number]

/** Bundled plans are review-only and remain bound to their recorded source placement. */
export function eligibleInteriorPlans(manifest: WorldManifestV2): readonly AuthoredInteriorPlan[] {
  return authoredPlans.plans.filter(plan => {
    const model = manifest.models.find(candidate => candidate.id === plan.id)

    if (!model || model.uri !== plan.sourceModelUri || model.statistics.sha256 !== plan.sourceModelSha256) {return false}

    return (['position', 'rotation', 'scale'] as const).every(field =>
      (['x', 'y', 'z'] as const).every((axis, index) => model.transform[field][axis] === plan.exteriorTransform[field][index]))
  })
}

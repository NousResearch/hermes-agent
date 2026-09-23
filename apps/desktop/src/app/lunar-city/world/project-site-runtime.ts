import type { LunarCitySnapshot, ProjectSlotManifestEntry } from '../model'

import { createProjectCompoundController, type ProjectCompoundAnchor } from './project-compounds'

/** Retains one physical site sign per exact project identity; scene owns final disposal. */
export function createProjectSiteRuntime<Node extends { dispose?(): void }>(
  slots: readonly ProjectSlotManifestEntry[],
  nodes: Map<string, Node>,
  create: (anchor: ProjectCompoundAnchor) => Node,
  onLayoutChange: (revision: number) => void
) {
  const controller = createProjectCompoundController(slots)
  let revision = 0

  return {
    update(snapshot: LunarCitySnapshot) {
      const report = controller.update(snapshot)
      const desired = new Set(report.anchors.map(anchor => anchor.key))
      let changed = false

      for (const [key, node] of nodes) {
        if (!desired.has(key)) { node.dispose?.(); nodes.delete(key); changed = true }
      }

      for (const anchor of report.anchors) {
        if (!nodes.has(anchor.key)) { nodes.set(anchor.key, create(anchor)); changed = true }
      }

      if (changed) {onLayoutChange(++revision)}

      return report
    }
  }
}

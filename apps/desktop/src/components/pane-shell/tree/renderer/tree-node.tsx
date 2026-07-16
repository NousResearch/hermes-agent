import type { LayoutNode } from '../model'

import { TreeGroup } from './tree-group'
import { TreeSplit } from './tree-split'

/** Dispatch a layout node to its renderer — the split/group recursion point.
 *  `root` marks the tree's top split (side collapse applies only there).
 *  `rootRow` marks the row split that owns the side columns — usually the root
 *  itself, but in a column-root layout (Terminal deck, Quad) it's the row
 *  child holding sessions/workspace/files. Side collapse (⌘B/⌘J) applies here.
 *  `parentAxis` is the containing split's orientation — a group collapses
 *  ALONG that axis, so it picks the minimized form (row → vertical rail,
 *  column → horizontal header). `railSide` is which half of that row the
 *  child sits in — the rail's divider stroke faces the content side. */
export function TreeNode({
  node,
  parentAxis,
  railSide,
  root,
  rootRow
}: {
  node: LayoutNode
  parentAxis?: 'column' | 'row'
  railSide?: 'left' | 'right'
  root?: boolean
  rootRow?: boolean
}) {
  return node.type === 'split' ? (
    <TreeSplit node={node} root={root} rootRow={rootRow} />
  ) : (
    <TreeGroup node={node} parentAxis={parentAxis} railSide={railSide} />
  )
}

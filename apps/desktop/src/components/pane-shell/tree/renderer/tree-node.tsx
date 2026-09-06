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
 *  column → horizontal header). `lockAxisRow`/`lockAxisColumn` indicate
 *  whether a row/column split exists ABOVE this node in the tree. The
 *  ZoneMenu uses them to decide which lock actions to offer: a zone inside
 *  a column-within-a-row gets BOTH Lock width (row ancestor) and Lock height
 *  (column parent). `railSide` is which half of that row the child sits in —
 *  the rail's divider stroke faces the content side. */
export function TreeNode({
  node,
  lockAxisColumn,
  lockAxisRow,
  parentAxis,
  railSide,
  root,
  rootRow
}: {
  node: LayoutNode
  lockAxisColumn?: boolean
  lockAxisRow?: boolean
  parentAxis?: 'column' | 'row'
  railSide?: 'left' | 'right'
  root?: boolean
  rootRow?: boolean
}) {
  if (node.type === 'split') {
    // A row split means every descendant has a row ancestor (width lockable);
    // a column split means every descendant has a column ancestor (height lockable).
    return (
      <TreeSplit
        lockAxisColumn={lockAxisColumn || node.orientation === 'column'}
        lockAxisRow={lockAxisRow || node.orientation === 'row'}
        node={node}
        root={root}
        rootRow={rootRow}
      />
    )
  }

  return (
    <TreeGroup
      lockAxisColumn={lockAxisColumn}
      lockAxisRow={lockAxisRow}
      node={node}
      parentAxis={parentAxis}
      railSide={railSide}
    />
  )
}

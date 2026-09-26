// The import's pure decisions, kept apart from the Electron/SDK wiring in
// web-import.ts so they can be exercised without a live guest.

export type PenImportMode = 'page' | 'selection'

export interface PenImportOptions {
  /** CSS selector of the one element to import; a live pick or the whole page when absent. */
  selector?: string
  mode?: PenImportMode
  /** The canvas was opened for this import: the editor's empty starter frame makes way. */
  fresh?: boolean
}

/** A selector wins; otherwise `mode`, defaulting to the whole page. */
export function resolveImportMode(options: PenImportOptions): PenImportMode {
  return options.selector || options.mode === 'selection' ? 'selection' : 'page'
}

/**
 * Selector for the ancestor `steps` levels above `selector`, via `:has(> …)`
 * — the pick's path only carries display labels, never selectors, so the
 * crumbs hover by climbing from the picked element. 0 steps is the element
 * itself; a negative step (a crumb below the pick) has no target.
 */
export function ancestorSelector(selector: string, steps: number): string | undefined {
  if (steps < 0) {
    return undefined
  }

  if (steps === 0) {
    return selector
  }

  return `*:has(> ${'* > '.repeat(steps - 1)}${selector})`
}

export interface PenCanvasNode {
  id: string
  name: string
  /** No children — the editor's starter frame, or a frame the user has not filled yet. */
  empty: boolean
}

/** Marker the top-level probe prints one node per line under; see `topLevelNodesProbe`. */
export const NODE_LINE = 'hermes-node'

/** `execute` input listing every top-level node as `hermes-node ["id","name",childCount]`. */
export const topLevelNodesProbe = `Get((n, c) => { if (c.depth === 0) Print(${JSON.stringify(NODE_LINE)}, JSON.stringify([n.id, n.name || '', n.children ? n.children.length : 0])); c.skipChildren() })`

/** The probe's lines out of an `execute` response; anything else in the text is ignored. */
export function parseTopLevelNodes(text: string): PenCanvasNode[] {
  return text
    .split('\n')
    .filter(line => line.startsWith(`${NODE_LINE} `))
    .map(line => {
      const [id, name, children] = JSON.parse(line.slice(NODE_LINE.length + 1)) as [string, string, number]

      return { empty: children === 0, id, name }
    })
}

/** What an import added: the top-level nodes that were not there before it. */
export function importedNodes(before: PenCanvasNode[], after: PenCanvasNode[]): PenCanvasNode[] {
  const known = new Set(before.map(node => node.id))

  return after.filter(node => !known.has(node.id))
}

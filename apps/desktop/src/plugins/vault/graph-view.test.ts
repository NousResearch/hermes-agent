import { describe, expect, it } from 'vitest'

import { connectedVaultNodeIds, filterVaultGraph } from './graph-view'
import type { VaultGraph } from './types'

const graph: VaultGraph = {
  nodes: [
    { id: 'A', label: 'A', path: 'A.md', tags: ['project'], weight: 1, group: 'project' },
    { id: 'B', label: 'B', path: 'B.md', tags: ['project'], weight: 1, group: 'project' },
    { id: 'C', label: 'C', path: 'C.md', tags: ['other'], weight: 1, group: 'other' }
  ],
  edges: [
    { source: 'A', target: 'B' },
    { source: 'B', target: 'C' }
  ]
}

describe('Vault graph projections', () => {
  it('filters nodes and edges consistently by tag', () => {
    expect(filterVaultGraph(graph, 'project')).toEqual({
      nodes: graph.nodes.slice(0, 2),
      edges: [graph.edges[0]]
    })
  })

  it('returns both incoming and outgoing neighbours for highlighting', () => {
    expect([...connectedVaultNodeIds(graph, 'B')].sort()).toEqual(['A', 'B', 'C'])
  })
})

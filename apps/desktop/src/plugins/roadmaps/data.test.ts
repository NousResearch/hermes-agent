/**
 * Roadmaps plugin — data layer (data.ts) tests: pure derivation over a
 * snapshot, error-code extraction, and validators. No SDK import.
 */

import { describe, expect, it } from 'vitest'

import {
  copilotSections,
  criticalChain,
  depsSatisfied,
  extractPlanJsonBlock,
  groupMilestones,
  isValidIdentifier,
  mapRelations,
  milestoneNodes,
  nextAction,
  nodeDepsInfo,
  nodeLabel,
  planPreviewFromJson,
  plural,
  rpcError,
  threadNodes,
  validatePlanPayload,
  validateProgress,
  validateRoadmapTitle
} from './data'
import type { RoadmapNode, RoadmapRelation, RoadmapVersion } from './types'

let seq = 0

function node(partial: Partial<RoadmapNode> & { node_id: string }): RoadmapNode {
  seq += 1

  return {
    version: 2,
    title: partial.node_id,
    kind: 'task',
    state: 'planned',
    progress: null,
    owner_agent: null,
    parent_node_id: null,
    description: null,
    block_reason: null,
    created_at: null,
    ...partial
  }
}

function rel(partial: Partial<RoadmapRelation> & { relation_id: string; from_node_id: string; to_node_id: string }): RoadmapRelation {
  return { version: 2, kind: 'depends_on', state: 'active', reason: null, ...partial }
}

function version(nodes: RoadmapNode[], relations: RoadmapRelation[] = []): RoadmapVersion {
  return { version: 2, state: 'active', source: null, reason: null, created_at: null, nodes, relations, todos: [] }
}

describe('rpcError', () => {
  it('extracts a numeric code from a `.code`-carrying error', () => {
    expect(rpcError(Object.assign(new Error('x'), { code: 5064 })).code).toBe(5064)
  })

  it('parses the REST bridge shape `Error("409: {detail:{code}}")`', () => {
    expect(rpcError(new Error('409: {"detail":{"code":5067,"message":"conflict"}}')).code).toBe(5067)
  })

  it('treats a string code as numeric', () => {
    expect(rpcError(new Error('422: {"detail":{"code":"5063"}}')).code).toBe(5063)
  })

  it('returns null for a plain error without a structured code', () => {
    expect(rpcError(new Error('network down')).code).toBeNull()
    expect(rpcError({}).code).toBeNull()
  })
})

describe('threadNodes', () => {
  it('keeps only actionable states, blocked first', () => {
    const nodes = [
      node({ node_id: 'b', state: 'blocked' }),
      node({ node_id: 'r', state: 'ready' }),
      node({ node_id: 'i', state: 'in_progress' }),
      node({ node_id: 'done', state: 'completed' }),
      node({ node_id: 'planned', state: 'planned' })
    ]

    expect(threadNodes(version(nodes)).map((n) => n.node_id)).toEqual(['b', 'i', 'r'])
  })
})

describe('dependencies', () => {
  it('depsSatisfied is true when every target is done', () => {
    const a = node({ node_id: 'a', state: 'ready' })
    const b = node({ node_id: 'b', state: 'completed' })
    const v = version([a, b], [rel({ relation_id: 'r1', from_node_id: 'a', to_node_id: 'b' })])
    expect(depsSatisfied(a, v)).toBe(true)
    expect(nodeDepsInfo(a, v).satisfied).toBe(1)
  })

  it('a missing target counts as satisfied', () => {
    const a = node({ node_id: 'a', state: 'ready' })
    const v = version([a], [rel({ relation_id: 'r1', from_node_id: 'a', to_node_id: 'ghost' })])
    expect(depsSatisfied(a, v)).toBe(true)
  })
})

describe('copilotSections', () => {
  it('buckets nodes by state + dependency satisfaction', () => {
    const a = node({ node_id: 'a', state: 'ready' }) // no deps → now
    const b = node({ node_id: 'b', state: 'ready' }) // depends on pending → waiting
    const c = node({ node_id: 'c', state: 'in_progress' })
    const d = node({ node_id: 'd', state: 'blocked', block_reason: 'x' })
    const pending = node({ node_id: 'p', state: 'planned' })
    const v = version([a, b, c, d, pending], [rel({ relation_id: 'r1', from_node_id: 'b', to_node_id: 'p' })])
    const s = copilotSections(v)!
    expect(s.now.map((n) => n.node_id)).toEqual(['a'])
    expect(s.waiting.map((n) => n.node_id)).toEqual(['b'])
    expect(s.inflight.map((n) => n.node_id)).toEqual(['c'])
    expect(s.blocked.map((n) => n.node_id)).toEqual(['d'])
  })
})

describe('nextAction', () => {
  it('prioritizes an unblockable blocked node (tier 0)', () => {
    const b = node({ node_id: 'b', state: 'blocked', block_reason: 'x' })
    const r = node({ node_id: 'r', state: 'ready' })
    const a = nextAction(version([b, r]))!
    expect(a.kind).toBe('unblock')
    expect(a.node.node_id).toBe('b')
  })

  it('claims a ready node with satisfied deps and no owner (tier 1)', () => {
    const r = node({ node_id: 'r', state: 'ready' })
    const a = nextAction(version([r]))!
    expect(a.kind).toBe('claim')
    expect(a.node.node_id).toBe('r')
  })
})

describe('criticalChain', () => {
  it('returns the longest depends_on chain', () => {
    const a = node({ node_id: 'a', state: 'ready' })
    const b = node({ node_id: 'b', state: 'ready' })
    const c = node({ node_id: 'c', state: 'ready' })

    const v = version(
      [a, b, c],
      [
        rel({ relation_id: 'r1', from_node_id: 'a', to_node_id: 'b' }),
        rel({ relation_id: 'r2', from_node_id: 'b', to_node_id: 'c' })
      ]
    )

    expect(criticalChain(v)).toEqual(['a', 'b', 'c'])
  })
})

describe('map + milestones', () => {
  it('mapRelations keeps canonical active relations with resolved endpoints', () => {
    const a = node({ node_id: 'a', state: 'ready' })
    const b = node({ node_id: 'b', state: 'ready' })

    const v = version(
      [a, b],
      [
        rel({ relation_id: 'r1', from_node_id: 'a', to_node_id: 'b' }),
        rel({ relation_id: 'r2', from_node_id: 'a', to_node_id: 'b', kind: 'blocks' }),
        rel({ relation_id: 'r3', from_node_id: 'a', to_node_id: 'b', state: 'inactive' })
      ]
    )

    expect(mapRelations(v).map((r) => r.relation_id)).toEqual(['r1', 'r2'])
  })

  it('milestoneNodes + groupMilestones filter and group', () => {
    const parent = node({ node_id: 'parent', kind: 'milestone', state: 'in_progress' })
    const child = node({ node_id: 'child', kind: 'objective', state: 'ready', parent_node_id: 'parent' })
    const task = node({ node_id: 'task', kind: 'task', state: 'ready' })
    const v = version([parent, child, task])
    expect(milestoneNodes(v).map((n) => n.node_id)).toEqual(['child', 'parent'])
    // The parented objective groups under its parent; the parent itself is flat.
    const groups = groupMilestones(v)
    expect(groups.length).toBe(2)
    expect(groups[0].label).toBe('parent')
    expect(groups[0].nodes.map((n) => n.node_id)).toEqual(['child'])
    expect(groups[1].label).toBeNull()
  })
})

describe('validators', () => {
  it('isValidIdentifier rejects empty/control/oversized', () => {
    expect(isValidIdentifier('ok')).toBe(true)
    expect(isValidIdentifier('')).toBe(false)
    expect(isValidIdentifier('  ')).toBe(false)
    expect(isValidIdentifier('a\u0000b')).toBe(false)
    expect(isValidIdentifier('x'.repeat(129))).toBe(false)
  })

  it('validateRoadmapTitle mirrors the 200-char contract', () => {
    expect(validateRoadmapTitle('ok')).toBe(true)
    expect(validateRoadmapTitle('  ok  ')).toBe(true)
    expect(validateRoadmapTitle('')).toBe(false)
    expect(validateRoadmapTitle('y'.repeat(201))).toBe(false)
  })

  it('validateProgress is integer 0..100 only', () => {
    expect(validateProgress(0)).toBe(true)
    expect(validateProgress(100)).toBe(true)
    expect(validateProgress(50.5)).toBe(false)
    expect(validateProgress(-1)).toBe(false)
    expect(validateProgress('50')).toBe(false)
  })
})

describe('labels + parsing', () => {
  it('nodeLabel prefers title then id', () => {
    expect(nodeLabel(node({ node_id: 'n1', title: 'Title' }))).toBe('Title')
    expect(nodeLabel(node({ node_id: 'n2' }))).toBe('n2')
    expect(nodeLabel(null)).toBe('?')
  })

  it('plural handles singular/plural', () => {
    expect(plural(1, 'node')).toBe('1 node')
    expect(plural(3, 'node')).toBe('3 nodes')
  })

  it('extractPlanJsonBlock parses the last complete fence', () => {
    const text = '```json\n{"title":"a","nodes":[{"node_id":"n","title":"N","kind":"task"}]}\n```'
    expect(extractPlanJsonBlock(text)).toEqual({ title: 'a', nodes: [{ node_id: 'n', title: 'N', kind: 'task' }] })
    expect(extractPlanJsonBlock('no fence')).toBeNull()
  })

  it('planPreviewFromJson + validatePlanPayload round-trip', () => {
    const obj = { title: 'P', nodes: [{ node_id: 'n', title: 'N', kind: 'task' }], relations: [], todos: [] }
    const preview = planPreviewFromJson(obj)!
    expect(preview.counts.nodes).toBe(1)
    expect(preview.kinds).toEqual(['task'])
    const payload = validatePlanPayload(preview)
    expect(payload.nodes).toHaveLength(1)
  })
})

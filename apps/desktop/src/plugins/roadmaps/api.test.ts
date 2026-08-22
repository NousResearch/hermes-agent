/**
 * Roadmaps plugin — data layer (api.ts) tests: query-key shapes and REST
 * path/method/body mapping through the plugin's `ctx.rest` door. The REST
 * door is injected via `bindApi`, so no network is touched.
 */

import type { PluginStorage } from '@hermes/plugin-sdk'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  activatePlan,
  archiveRoadmap,
  bindApi,
  claimNode,
  createPlan,
  createRoadmap,
  fetchPlanningRules,
  fetchPlans,
  fetchRoadmaps,
  fetchSnapshot,
  planningRulesKey,
  roadmapPlansKey,
  roadmapsListKey,
  roadmapSnapshotKey,
  scopeQuery,
  updateProgress,
  updateRoadmap,
  updateTodo
} from './api'

const mockRest = vi.fn()

const mockStorage: PluginStorage = {
  get: <T>(_key: string, fallback: T): T => fallback,
  set: () => {},
  remove: () => {}
}

beforeAll(() => {
  bindApi(mockRest, mockStorage)
})

beforeEach(() => {
  mockRest.mockReset()
  mockRest.mockResolvedValue({})
})

afterEach(() => {
  vi.clearAllMocks()
})

describe('scopeQuery', () => {
  it('builds profile + project query params', () => {
    expect(scopeQuery('default', 'p1')).toBe('?profile=default&project_id=p1')
  })

  it('URL-encodes special characters', () => {
    expect(scopeQuery('my profile', 'p/1')).toBe('?profile=my+profile&project_id=p%2F1')
  })
})

describe('query keys', () => {
  it('scopes every key by profile + project (+ roadmap)', () => {
    expect(roadmapsListKey('default', 'p1')).toEqual(['roadmaps', 'list', 'default', 'p1'])
    expect(roadmapSnapshotKey('default', 'p1', 'r1')).toEqual(['roadmaps', 'steer', 'default', 'p1', 'r1'])
    expect(roadmapPlansKey('default', 'p1', 'r1')).toEqual(['roadmaps', 'plans', 'default', 'p1', 'r1'])
    expect(planningRulesKey).toEqual(['roadmaps', 'planning-rules'])
  })
})

describe('rest mapping', () => {
  it('GET /roadmaps with scope', () => {
    fetchRoadmaps('default', 'p1')
    expect(mockRest).toHaveBeenCalledWith('/roadmaps?profile=default&project_id=p1', undefined)
  })

  it('GET /roadmaps/{id}/snapshot', () => {
    fetchSnapshot('default', 'p1', 'r1')
    expect(mockRest).toHaveBeenCalledWith('/roadmaps/r1/snapshot?profile=default&project_id=p1', undefined)
  })

  it('GET /roadmaps/{id}/plans', () => {
    fetchPlans('default', 'p1', 'r1')
    expect(mockRest).toHaveBeenCalledWith('/roadmaps/r1/plans?profile=default&project_id=p1', undefined)
  })

  it('GET /planning-rules has no scope', () => {
    fetchPlanningRules()
    expect(mockRest).toHaveBeenCalledWith('/planning-rules', undefined)
  })

  it('POST /roadmaps (create)', () => {
    createRoadmap('default', 'p1', { actor: 'user', title: 'T' })
    expect(mockRest).toHaveBeenCalledWith('/roadmaps?profile=default&project_id=p1', {
      method: 'POST',
      body: { actor: 'user', title: 'T' }
    })
  })

  it('PATCH /roadmaps/{id} (update)', () => {
    updateRoadmap('default', 'p1', 'r1', { actor: 'user', expected_version: 2, title: 'T2' })
    expect(mockRest).toHaveBeenCalledWith('/roadmaps/r1?profile=default&project_id=p1', {
      method: 'PATCH',
      body: { actor: 'user', expected_version: 2, title: 'T2' }
    })
  })

  it('POST /roadmaps/{id}/archive', () => {
    archiveRoadmap('default', 'p1', 'r1', { actor: 'user', expected_version: 0 })
    expect(mockRest).toHaveBeenCalledWith('/roadmaps/r1/archive?profile=default&project_id=p1', {
      method: 'POST',
      body: { actor: 'user', expected_version: 0 }
    })
  })

  it('POST /roadmaps/{id}/plans (create plan)', () => {
    createPlan('default', 'p1', 'r1', { actor: 'user', nodes: [] })
    expect(mockRest).toHaveBeenCalledWith('/roadmaps/r1/plans?profile=default&project_id=p1', {
      method: 'POST',
      body: { actor: 'user', nodes: [] }
    })
  })

  it('node + todo mutations map to the right paths', () => {
    claimNode('default', 'p1', 'r1', 'n1', { actor: 'user', expected_version: 1 })
    expect(mockRest).toHaveBeenCalledWith('/roadmaps/r1/nodes/n1/claim?profile=default&project_id=p1', {
      method: 'POST',
      body: { actor: 'user', expected_version: 1 }
    })

    updateProgress('default', 'p1', 'r1', 'n1', { actor: 'user', expected_version: 1, progress: 50 })
    expect(mockRest).toHaveBeenCalledWith('/roadmaps/r1/nodes/n1/progress?profile=default&project_id=p1', {
      method: 'POST',
      body: { actor: 'user', expected_version: 1, progress: 50 }
    })

    updateTodo('default', 'p1', 'r1', 't1', { actor: 'user', state: 'done', expected_version: 1 })
    expect(mockRest).toHaveBeenCalledWith('/roadmaps/r1/todos/t1?profile=default&project_id=p1', {
      method: 'POST',
      body: { actor: 'user', state: 'done', expected_version: 1 }
    })
  })

  it('plan transitions carry the version in the path', () => {
    activatePlan('default', 'p1', 'r1', 3, { actor: 'user', expected_version: 0 })
    expect(mockRest).toHaveBeenCalledWith('/roadmaps/r1/plans/3/activate?profile=default&project_id=p1', {
      method: 'POST',
      body: { actor: 'user', expected_version: 0 }
    })
  })
})

/**
 * The agent-rows contract: plugins report agents they run outside
 * `delegate_task` through `AGENT_ROWS_AREA`; the host validates, caps and
 * namespaces them, and they join the Agents count and page. These drive the
 * REAL plugin context and registry: what a plugin can register, and what the
 * host will accept.
 */
import { AGENT_ROWS_AREA, AGENT_ROWS_LIMITS, type PluginAgentRow } from '@hermes/plugin-sdk'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { activeSubagentCount, buildSubagentTree } from '@/store/subagents'

import { resolvePluginAgentRows } from './agent-rows'
import { createPluginContext } from './plugin'
import { registry } from './registry'

const NOW = 1_800_000_000_000

const loaded: Array<() => void> = []

/** Load a plugin the way the loader does and register its rows; returns its
 *  context's register (to update the rows) and its unload. */
function loadPlugin(id: string, rows: readonly unknown[], localId = 'agents') {
  const disposers: Array<() => void> = []
  const ctx = createPluginContext(id, dispose => disposers.push(dispose))

  const publish = (next: readonly unknown[]) =>
    ctx.register({ area: AGENT_ROWS_AREA, data: { rows: next }, id: localId })

  publish(rows)
  const unload = () => disposers.splice(0).forEach(dispose => dispose())
  loaded.push(unload)

  return { publish, unload }
}

const rows = () => resolvePluginAgentRows(registry.getArea(AGENT_ROWS_AREA), NOW)

const row = (extra: Partial<PluginAgentRow> = {}): PluginAgentRow => ({
  goal: 'research · job 1',
  id: 'job-1:research',
  status: 'running',
  ...extra
})

afterEach(() => {
  loaded.splice(0).forEach(unload => unload())
  vi.restoreAllMocks()
})

describe('agent rows: what a plugin reports', () => {
  it('lists a row in the page shape, namespaced by its plugin and stamped with its provenance', () => {
    loadPlugin('routing', [
      row({ costUsd: 0.0034, inputTokens: 1200, model: 'deepseek/deepseek-flash', startedAt: NOW - 42_000 })
    ])

    expect(rows()).toEqual([
      expect.objectContaining({
        costUsd: 0.0034,
        goal: 'research · job 1',
        id: 'plugin:routing#job-1:research',
        inputTokens: 1200,
        model: 'deepseek/deepseek-flash',
        parentId: null,
        source: 'plugin:routing',
        startedAt: NOW - 42_000,
        status: 'running',
        taskCount: 1,
        updatedAt: NOW - 42_000
      })
    ])
  })

  it('counts only running and queued rows toward the indicator', () => {
    loadPlugin('routing', [
      row({ id: 'a' }),
      row({ id: 'b', status: 'queued' }),
      row({ id: 'c', status: 'completed' }),
      row({ id: 'd', status: 'failed' })
    ])

    expect(activeSubagentCount(rows())).toBe(2)
  })

  it('replaces the rows when the plugin registers the same id again, and drops them on unload', () => {
    const { publish, unload } = loadPlugin('routing', [row()])
    publish([row({ id: 'job-2:review', status: 'queued' })])

    expect(rows().map(r => r.id)).toEqual(['plugin:routing#job-2:review'])
    unload()
    expect(rows()).toEqual([])
  })
})

describe('agent rows: isolation between plugins and from Hermes', () => {
  it('keeps two plugins with the same row id apart', () => {
    loadPlugin('zeta', [row({ id: 'x' })])
    loadPlugin('alpha', [row({ id: 'x' })])

    expect(rows().map(r => r.id)).toEqual(['plugin:alpha#x', 'plugin:zeta#x'])
  })

  it('nests a row only under a row of its own plugin', () => {
    loadPlugin('alpha', [row({ id: 'lead' })])
    loadPlugin('routing', [
      row({ id: 'step' }),
      row({ goal: 'subagent', id: 'sub', parentId: 'step' }),
      row({ goal: 'stray', id: 'stray', parentId: 'lead' }),
      row({ goal: 'self', id: 'self', parentId: 'self' })
    ])

    const byId = Object.fromEntries(rows().map(r => [r.id, r.parentId]))
    expect(byId['plugin:routing#sub']).toBe('plugin:routing#step')
    expect(byId['plugin:routing#stray']).toBeNull()
    expect(byId['plugin:routing#self']).toBeNull()

    const tree = buildSubagentTree(rows())
    const step = tree.find(node => node.id === 'plugin:routing#step')
    expect(step?.children.map(child => child.id)).toEqual(['plugin:routing#sub'])
  })

  it('caps each plugin on its own and keeps the first of duplicate ids', () => {
    const many = Array.from({ length: AGENT_ROWS_LIMITS.perPlugin + 5 }, (_, n) => row({ id: `r${n}` }))
    loadPlugin('busy', many)
    loadPlugin('quiet', [row({ goal: 'first', id: 'same' }), row({ goal: 'second', id: 'same' })])

    const listed = rows()
    expect(listed.filter(r => r.source === 'plugin:busy')).toHaveLength(AGENT_ROWS_LIMITS.perPlugin)
    expect(listed.filter(r => r.source === 'plugin:quiet').map(r => r.goal)).toEqual(['first'])
  })
})

describe('agent rows: what the host accepts', () => {
  it('drops rows without an id, a goal or a known status', () => {
    loadPlugin('routing', [
      row({ id: '' }),
      row({ goal: '   ' }),
      row({ status: 'exploded' as never }),
      'not a row',
      null,
      row({ id: 'kept' })
    ])

    expect(rows().map(r => r.id)).toEqual(['plugin:routing#kept'])
  })

  it('cleans text, shortens it and ignores numbers that are not finite and positive', () => {
    loadPlugin('routing', [
      row({
        costUsd: -1,
        currentTool: 'web_search',
        goal: `  research‮ evil \n${'x'.repeat(400)}`,
        inputTokens: Number.NaN,
        outputTokens: Number.POSITIVE_INFINITY,
        toolCount: 3
      })
    ])

    const [listed] = rows()
    expect(listed.goal.startsWith('research evil x')).toBe(true)
    expect([...listed.goal]).toHaveLength(AGENT_ROWS_LIMITS.goalChars)
    expect(listed.goal.endsWith('…')).toBe(true)
    expect([listed.costUsd, listed.inputTokens, listed.outputTokens, listed.toolCount]).toEqual([
      undefined,
      undefined,
      undefined,
      3
    ])
    expect(listed.currentTool).toBe('web_search')
  })

  it('shows a current tool only while the row runs, and keeps the latest stream lines of known kinds', () => {
    const stream = [
      ...Array.from({ length: AGENT_ROWS_LIMITS.stream + 3 }, (_, n) => ({
        at: NOW + n,
        kind: 'tool',
        text: `step ${n}`
      })),
      { at: NOW, kind: 'sound', text: 'beep' }
    ]

    loadPlugin('routing', [row({ currentTool: 'read_file', id: 'done', status: 'completed', stream: stream as never })])

    const [listed] = rows()
    expect(listed.currentTool).toBeUndefined()
    expect(listed.stream.every(entry => entry.kind === 'tool')).toBe(true)
    expect(listed.stream.at(-1)?.text).toBe(`step ${AGENT_ROWS_LIMITS.stream + 2}`)
    expect(listed.stream.length).toBeLessThanOrEqual(AGENT_ROWS_LIMITS.stream)
  })

  it('trims file lists and skips entries that are not text', () => {
    const files = [...Array.from({ length: AGENT_ROWS_LIMITS.files + 10 }, (_, n) => `src/${n}.ts`), 7, null]
    loadPlugin('routing', [row({ filesRead: files as never, filesWritten: ['out.md', 3 as never] })])

    const [listed] = rows()
    expect(listed.filesRead).toHaveLength(AGENT_ROWS_LIMITS.files)
    expect(listed.filesWritten).toEqual(['out.md'])
  })

  it('skips a contribution whose rows cannot be read and keeps the others', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const disposers: Array<() => void> = []
    const ctx = createPluginContext('broken', dispose => disposers.push(dispose))
    loaded.push(() => disposers.splice(0).forEach(dispose => dispose()))

    ctx.register({
      area: AGENT_ROWS_AREA,
      data: {
        get rows() {
          throw new Error('boom')
        }
      },
      id: 'agents'
    })
    loadPlugin('routing', [row()])

    expect(rows().map(r => r.source)).toEqual(['plugin:routing'])
    expect(warn).toHaveBeenCalledTimes(1)
  })
})

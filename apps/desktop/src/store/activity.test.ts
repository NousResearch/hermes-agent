import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { buildRailTasks } from './activity'

const session = (id: string, title: string, lastActive: number): SessionInfo => ({
  ended_at: null,
  id,
  input_tokens: 0,
  is_active: false,
  last_active: lastActive,
  message_count: 0,
  model: null,
  output_tokens: 0,
  preview: null,
  source: null,
  started_at: lastActive,
  title,
  tool_call_count: 0
})

describe('buildRailTasks', () => {
  it('builds actionable running and finished session work without duplicates', () => {
    const tasks = buildRailTasks(
      ['running'],
      ['finished', 'running'],
      [session('running', 'Ship phone handoff', 2_000), session('finished', 'Audit session recovery', 1_000)],
      null,
      {}
    )

    expect(tasks).toEqual([
      {
        id: 'session:running',
        kind: 'session',
        label: 'Ship phone handoff',
        sessionId: 'running',
        status: 'running',
        updatedAt: 2_000_000
      },
      {
        id: 'session:finished',
        kind: 'session',
        label: 'Audit session recovery',
        sessionId: 'finished',
        status: 'success',
        updatedAt: 1_000_000
      }
    ])
  })

  it('keeps preview and desktop actions in the same time-ordered feed', () => {
    const tasks = buildRailTasks(
      [],
      [],
      [],
      { message: 'Waiting for Vite', status: 'running', taskId: 'preview-1', url: 'http://localhost:5174' },
      {
        lint: {
          status: { exit_code: 1, lines: [], name: 'Lint desktop', pid: null, running: false },
          updatedAt: 500
        }
      },
      1_000
    )

    expect(tasks).toEqual([
      {
        detail: 'Waiting for Vite',
        id: 'preview:preview-1',
        kind: 'preview',
        label: 'Preview restart',
        status: 'running',
        updatedAt: 1_000
      },
      {
        detail: 'Exit 1',
        id: 'action:Lint desktop',
        kind: 'action',
        label: 'Lint desktop',
        status: 'error',
        updatedAt: 500
      }
    ])
  })
})

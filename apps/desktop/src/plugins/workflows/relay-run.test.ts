import { describe, expect, it } from 'vitest'

import { mergeRelay, protoEvents, type RelayEvent } from './relay-run'

const RUN = 'run-1-abc123'
let clock = 0

const at = () => new Date(Date.UTC(2026, 0, 1, 0, 0, 0, clock++)).toISOString()

const session = (uuid: string, sid: string, parent: string | null): RelayEvent => ({
  kind: 'scope',
  scope_category: 'start',
  category: 'agent',
  name: 'hermes.session',
  uuid,
  parent_uuid: parent,
  timestamp: at(),
  metadata: { 'hermes.session_id': sid }
})

const mark = (type: string, payload: Record<string, unknown>, seq: number): RelayEvent => ({
  kind: 'mark',
  name: `hermes.workflow.${type}`,
  uuid: `m${seq}`,
  parent_uuid: 'run',
  timestamp: at(),
  data: { runId: RUN, seq, payload }
})

const tool = (uuid: string, parent: string, name: string, args: unknown): RelayEvent => ({
  kind: 'scope',
  scope_category: 'start',
  category: 'tool',
  name,
  uuid,
  parent_uuid: parent,
  timestamp: at(),
  data: args
})

describe('a run folded from its Relay trace', () => {
  it("keeps the runner's marks and names each step's tool calls from its child session", () => {
    const trace = [
      session('run', `wf-${RUN}`, null),
      mark('RunStarted', { scenario: 'w' }, 0),
      mark('NodeStarted', { nodeId: 'build', iteration: 1, input: 'go', maxIters: 20 }, 1),
      session('child', `wf-${RUN}-build`, 'run'),
      { ...session('turn', 'ignored', 'child'), name: 'hermes.turn', metadata: {} },
      tool('t1', 'turn', 'write_file', { path: 'src/app.tsx' }),
      tool('t2', 'elsewhere', 'terminal', { command: 'rm -rf' }),
      mark('NodeFinished', { nodeId: 'build', iteration: 1 }, 2)
    ]

    const events = protoEvents(RUN, trace)

    expect(events.map(e => e.type)).toEqual(['RunStarted', 'NodeStarted', 'AgentTraceEvent', 'NodeFinished'])
    expect(events[2].payload).toEqual({
      iteration: 1,
      nodeId: 'build',
      tool: { arg: 'src/app.tsx', name: 'write_file' }
    })
    expect(events.map(e => e.seq)).toEqual([0, 1, 2, 3])
  })

  it('takes a live event once even when a fetch already had it', () => {
    const first = [mark('RunStarted', {}, 0)]
    const merged = mergeRelay(first, [first[0], mark('RunFinished', { state: 'succeeded' }, 1)])

    expect(merged).toHaveLength(2)
    expect(mergeRelay(merged, [merged[1]])).toBe(merged)
  })
})

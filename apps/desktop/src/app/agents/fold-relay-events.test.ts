import { describe, expect, it } from 'vitest'

import type { RelayEvent } from '@/store/trace'

import { foldRelayEvents, traceTurns, turnTrace } from './fold-relay-events'

const LABELS = { llmCall: 'LLM call', session: 'Session', subagent: 'Subagent', turn: 'Turn' }
const at = (seconds: number) => new Date(Date.UTC(2026, 8, 30, 12, 0, seconds)).toISOString()

function scope(
  phase: 'end' | 'start',
  uuid: string,
  parent: null | string,
  name: string,
  category: string,
  seconds: number,
  extra: Partial<RelayEvent> = {}
): RelayEvent {
  return {
    category,
    kind: 'scope',
    name,
    parent_uuid: parent,
    scope_category: phase,
    timestamp: at(seconds),
    uuid,
    ...extra
  }
}

// A turn that delegates: the child's session scope hangs off the parent turn in Relay,
// while the delegate_task call that spawned it is still open.
const RECORDED: RelayEvent[] = [
  scope('start', 's', 'root', 'hermes.session', 'agent', 0, { metadata: { 'hermes.session_id': 'S' } }),
  scope('start', 't', 's', 'hermes.turn', 'function', 1),
  {
    kind: 'mark',
    name: 'hermes.turn.input',
    parent_uuid: 't',
    timestamp: at(1),
    uuid: 'm',
    data: { preview: 'ship it' }
  },
  scope('start', 'd', 't', 'delegate_task', 'tool', 2),
  scope('start', 'c', 't', 'hermes.session', 'agent', 3, {
    metadata: { 'hermes.parent_session_id': 'S', 'hermes.session_id': 'C' }
  }),
  scope('start', 'ct', 'c', 'hermes.turn', 'function', 3),
  scope('end', 'ct', 'c', 'hermes.turn', 'function', 8, { data: { outcome: 'success' } }),
  scope('end', 'c', 't', 'hermes.session', 'agent', 8)
]

const mark = (
  uuid: string,
  parent: string,
  name: string,
  seconds: number,
  data: Record<string, unknown>
): RelayEvent => ({
  data,
  kind: 'mark',
  name,
  parent_uuid: parent,
  timestamp: at(seconds),
  uuid
})

describe('foldRelayEvents', () => {
  it('hangs a subagent under the exact call that spawned it, even with two delegations open', () => {
    const doc = foldRelayEvents(
      [
        scope('start', 's', 'root', 'hermes.session', 'agent', 0, { metadata: { 'hermes.session_id': 'S' } }),
        scope('start', 't', 's', 'hermes.turn', 'function', 1),
        scope('start', 'd1', 't', 'delegate_task', 'tool', 2, { category_profile: { tool_call_id: 'call-1' } }),
        scope('start', 'd2', 't', 'delegate_task', 'tool', 2, { category_profile: { tool_call_id: 'call-2' } }),
        scope('start', 'c', 't', 'hermes.session', 'agent', 3, {
          metadata: {
            'hermes.parent_session_id': 'S',
            'hermes.session_id': 'C',
            'hermes.spawned_by_tool_call_id': 'call-2'
          }
        })
      ],
      LABELS,
      Date.parse(at(9)) / 1000
    )

    expect(doc.spans.find(span => span.id === 'c')?.parentId).toBe('d2')
  })

  it('draws a model call as waiting, then reasoning, then text, with the text streamed so far', () => {
    // A managed attempt: marks sit on the logical call scope and name only the request.
    const doc = foldRelayEvents(
      [
        scope('start', 's', 'root', 'hermes.session', 'agent', 0, { metadata: { 'hermes.session_id': 'S' } }),
        scope('start', 't', 's', 'hermes.turn', 'function', 0),
        scope('start', 'l', 't', 'hermes.logical_llm_call', 'function', 0),
        scope('start', 'a', 'l', 'openai.chat_completions', 'llm', 0),
        mark('m1', 'l', 'hermes.llm.first_token', 2, { api_request_id: 'r1', kind: 'reasoning' }),
        mark('m2', 'l', 'hermes.llm.stream', 5, { api_request_id: 'r1', kind: 'reasoning', text: 'hmm' }),
        mark('m3', 'l', 'hermes.llm.stream', 6, { api_request_id: 'r1', kind: 'text', text: 'Hi' })
      ],
      LABELS,
      Date.parse(at(8)) / 1000
    )

    const call = doc.spans.find(span => span.id === 'a')!
    const t0 = Date.parse(at(0)) / 1000

    expect(call.phases?.map(p => [p.kind, p.start - t0, p.end - t0])).toEqual([
      ['wait', 0, 2],
      ['reasoning', 2, 5],
      ['text', 5, 8]
    ])
    expect(call.attributes).toMatchObject({
      'llm.first_token_s': 2,
      'llm.reasoning.value': 'hmm',
      'output.value': 'Hi'
    })
  })

  it('draws a delegated session under the delegate_task call that was open when it started', () => {
    const doc = foldRelayEvents(RECORDED, LABELS, Date.parse(at(9)) / 1000)
    const child = doc.spans.find(span => span.id === 'c')

    expect(child?.parentId).toBe('d')
    expect(child?.name).toBe('Subagent')
    expect(doc.spans.find(span => span.id === 'ct')?.sessionId).toBe('C')
  })

  it('keeps one span identity while a turn runs and after it settles', () => {
    const running = foldRelayEvents(RECORDED, LABELS, Date.parse(at(9)) / 1000)

    const settled = foldRelayEvents(
      [
        ...RECORDED,
        scope('end', 'd', 't', 'delegate_task', 'tool', 9),
        scope('end', 't', 's', 'hermes.turn', 'function', 10, { data: { outcome: 'success' } })
      ],
      LABELS,
      Date.parse(at(60)) / 1000
    )

    const [liveTurn] = traceTurns(running)
    const [doneTurn] = traceTurns(settled)

    expect(liveTurn).toMatchObject({ id: 't', label: 'ship it', running: true })
    expect(doneTurn).toMatchObject({ id: 't', label: 'ship it', running: false })
    expect(
      turnTrace(running, 't')
        .spans.map(span => span.id)
        .sort()
    ).toEqual(
      turnTrace(settled, 't')
        .spans.map(span => span.id)
        .sort()
    )
    // Between turns the open session scope ends with its last child, not at "now".
    expect(settled.spans.find(span => span.id === 's')?.end).toBe(Date.parse(at(10)) / 1000)
  })
})

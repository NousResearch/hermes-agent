import { describe, expect, it } from 'vitest'

import { withoutCoveredAssistantPrefix } from './coverage'
import { assistantTextPart, reasoningPart } from './parts'
import { upsertToolPart } from './tool-parts'
import type { ChatMessage, ChatMessagePart } from './types'

const assistant = (parts: ChatMessagePart[], id = 'live'): ChatMessage => ({ id, parts, role: 'assistant' })

const tools = (phase: 'complete' | 'running') =>
  ['call-one', 'call-two'].map(tool_id => upsertToolPart([], { name: 'search_files', tool_id }, phase, 2)[0])

describe('withoutCoveredAssistantPrefix', () => {
  // A window that attached mid-turn never saw `reasoning.delta`, so its live
  // bubble holds the answer text without the narration the durable row kept.
  // Narration is not an occurrence: matching on it left the answer bubble on
  // screen as a duplicate of the row that already carried it.
  it('subtracts a live answer the durable row already holds when the durable row leads with narration', () => {
    const durable = [
      assistant([reasoningPart('Checking the folder first.'), assistantTextPart('same reply'), ...tools('running')])
    ]

    const local = [
      assistant([assistantTextPart('same reply')], 'live-prefix'),
      assistant([...tools('complete'), assistantTextPart('more')], 'live-tail')
    ]

    const remaining = withoutCoveredAssistantPrefix(durable, local)

    expect(remaining.map(message => message.id)).toEqual(['live-tail'])
    expect(remaining[0].parts).toEqual([assistantTextPart('more')])
  })

  // The mirror asymmetry: `appendReasoningDelta`'s replace branch attaches
  // reasoning as a live row's FIRST part when no visible text has arrived yet,
  // and settling does not strip it. The same stall must not happen from the
  // live side of the walk.
  it('subtracts a live answer the durable row already holds when the live row leads with narration', () => {
    const durable = [assistant([assistantTextPart('same reply'), ...tools('running')])]

    const local = [
      assistant([reasoningPart('Need user input.'), assistantTextPart('same reply'), ...tools('complete')])
    ]

    expect(withoutCoveredAssistantPrefix(durable, local)).toEqual([])
  })

  it('never folds a live row that is narration only', () => {
    const durable = [assistant([assistantTextPart('same reply'), ...tools('running')])]
    const local = [assistant([reasoningPart('Still thinking.')])]

    expect(withoutCoveredAssistantPrefix(durable, local)).toBe(local)
  })

  it('keeps every row when no tool occurrence anchors the match', () => {
    const durable = [assistant([assistantTextPart('same reply')])]
    const local = [assistant([assistantTextPart('same reply')])]

    expect(withoutCoveredAssistantPrefix(durable, local)).toBe(local)
  })

  it('keeps a live row whose text the durable row never held', () => {
    const durable = [assistant([assistantTextPart('one'), ...tools('running')])]
    const local = [assistant([assistantTextPart('one')]), assistant([...tools('complete'), assistantTextPart('two')])]

    const remaining = withoutCoveredAssistantPrefix(durable, local)

    expect(remaining).toHaveLength(1)
    expect(remaining[0].parts).toEqual([assistantTextPart('two')])
  })

  // Field capture (2026-10-12): a turn that streamed while the pane was away
  // lost the second tool of a parallel pair to the frame watermarks — the live
  // bubbles held `call_one … call_three` while the durable fold kept
  // `call_one, call_two … call_three`. Requiring an every-position match
  // stalled the walk on `call_two` and left all three covered narrations on
  // screen beside the fold that already carried them.
  it('folds past a stored tool occurrence the live window never rendered', () => {
    const call = (tool_id: string) => upsertToolPart([], { name: 'terminal', tool_id }, 'complete', 2)[0]

    const durable = [
      assistant([
        assistantTextPart('lead'),
        call('call_one'),
        call('call_two'),
        assistantTextPart('mid'),
        call('call_three'),
        assistantTextPart('tail')
      ])
    ]

    const local = [
      assistant([assistantTextPart('lead')], 'live-lead'),
      assistant([call('call_one'), assistantTextPart('mid')], 'live-mid'),
      assistant([call('call_three'), assistantTextPart('tail')], 'live-tail')
    ]

    expect(withoutCoveredAssistantPrefix(durable, local)).toEqual([])
  })

  it('never skips stored text while aligning the run', () => {
    const call = (tool_id: string) => upsertToolPart([], { name: 'terminal', tool_id }, 'complete', 2)[0]

    const durable = [
      assistant([
        assistantTextPart('lead'),
        assistantTextPart('prose the run never streamed'),
        call('call_one'),
        assistantTextPart('tail')
      ])
    ]

    const local = [assistant([assistantTextPart('lead'), call('call_one'), assistantTextPart('tail')])]

    expect(withoutCoveredAssistantPrefix(durable, local)).toBe(local)
  })
})

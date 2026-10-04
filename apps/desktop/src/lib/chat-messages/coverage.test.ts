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
})

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $clarifyRequest,
  $clarifyRequests,
  type ClarifyRequest,
  clarifyToolCallAlias,
  clearClarifyRequest,
  hasClarifyRequest,
  normalizeChoices,
  normalizeQuestions,
  noteClarifyToolCall,
  rebindClarifyRequest,
  setClarifyRequest,
  settleClarifyRequest,
  skipClarifyRequest
} from './clarify'
import { $gateway } from './gateway'
import { $activeSessionId } from './session'

function clarify(sessionId: string | null, requestId: string): ClarifyRequest {
  return {
    requestId,
    question: `question-${requestId}`,
    choices: null,
    multiSelect: false,
    sessionId
  }
}

describe('clarify store', () => {
  beforeEach(() => {
    $clarifyRequests.set({})
    $activeSessionId.set(null)
  })

  afterEach(() => {
    $clarifyRequests.set({})
    $activeSessionId.set(null)
  })

  it('keeps clarify requests from concurrent sessions independent', () => {
    setClarifyRequest(clarify('session-a', 'req-a'))
    setClarifyRequest(clarify('session-b', 'req-b'))

    expect($clarifyRequests.get()['session-a']?.requestId).toBe('req-a')
    expect($clarifyRequests.get()['session-b']?.requestId).toBe('req-b')
  })

  it('exposes only the active session via the focus-scoped view', () => {
    setClarifyRequest(clarify('session-a', 'req-a'))
    setClarifyRequest(clarify('session-b', 'req-b'))

    $activeSessionId.set('session-a')
    expect($clarifyRequest.get()?.requestId).toBe('req-a')

    $activeSessionId.set('session-b')
    expect($clarifyRequest.get()?.requestId).toBe('req-b')

    $activeSessionId.set('session-c')
    expect($clarifyRequest.get()).toBeNull()
  })

  it('clears only the targeted session, leaving the other pending', () => {
    setClarifyRequest(clarify('session-a', 'req-a'))
    setClarifyRequest(clarify('session-b', 'req-b'))

    clearClarifyRequest('req-a', 'session-a')

    expect($clarifyRequests.get()['session-a']).toBeUndefined()
    expect($clarifyRequests.get()['session-b']?.requestId).toBe('req-b')
  })

  it('ignores a stale clear whose request id no longer matches', () => {
    setClarifyRequest(clarify('session-a', 'req-a2'))

    clearClarifyRequest('req-a1', 'session-a')

    expect($clarifyRequests.get()['session-a']?.requestId).toBe('req-a2')
  })

  it('clears by request id across sessions when no session hint is given', () => {
    setClarifyRequest(clarify('session-a', 'shared'))
    setClarifyRequest(clarify('session-b', 'other'))

    clearClarifyRequest('shared')

    expect($clarifyRequests.get()['session-a']).toBeUndefined()
    expect($clarifyRequests.get()['session-b']?.requestId).toBe('other')
  })
})

describe('skipClarifyRequest', () => {
  const request = vi.fn(async () => ({ ok: true }))

  beforeEach(() => {
    $clarifyRequests.set({})
    request.mockClear()
    $gateway.set({ request } as unknown as ReturnType<typeof $gateway.get>)
  })

  afterEach(() => {
    $clarifyRequests.set({})
    $gateway.set(null)
  })

  it('answers the session\u2019s clarify with an empty answer and drops it', async () => {
    setClarifyRequest(clarify('session-a', 'req-a'))
    setClarifyRequest(clarify('session-b', 'req-b'))

    await expect(skipClarifyRequest('session-a')).resolves.toBe(true)

    expect(request).toHaveBeenCalledWith('clarify.respond', { request_id: 'req-a', answer: '' })
    expect(hasClarifyRequest('session-a')).toBe(false)
    // A background session's question is untouched — only the one being typed
    // over is skipped.
    expect(hasClarifyRequest('session-b')).toBe(true)
  })

  it('is a no-op when the session has no clarify parked', async () => {
    await expect(skipClarifyRequest('session-a')).resolves.toBe(false)
    expect(request).not.toHaveBeenCalled()
  })

  it('still reports the skip when the respond RPC fails', async () => {
    setClarifyRequest(clarify('session-a', 'req-a'))
    request.mockRejectedValueOnce(new Error('socket closed'))

    await expect(skipClarifyRequest('session-a')).resolves.toBe(true)
    expect(hasClarifyRequest('session-a')).toBe(true)
  })

  it('does not resurrect a superseded request after a failed skip', async () => {
    setClarifyRequest(clarify('session-race', 'req-old'))

    let release = (): void => undefined

    const gate = new Promise<void>(resolve => {
      release = resolve
    })

    request.mockImplementation(async () => {
      await gate
      throw new Error('socket closed')
    })

    const skipping = skipClarifyRequest('session-race')
    setClarifyRequest(clarify('session-race', 'req-new'))
    release()
    await skipping

    expect($clarifyRequests.get()['session-race']?.requestId).toBe('req-new')
  })
})

describe('normalizeChoices', () => {
  it('returns empty array for null/undefined', () => {
    expect(normalizeChoices(null)).toEqual([])
    expect(normalizeChoices(undefined)).toEqual([])
  })

  it('returns empty array for non-array input', () => {
    expect(normalizeChoices('hello')).toEqual([])
    expect(normalizeChoices(42)).toEqual([])
    expect(normalizeChoices({})).toEqual([])
  })

  it('filters out non-string items', () => {
    expect(normalizeChoices(['a', 42, 'b', null, 'c'])).toEqual(['a', 'b', 'c'])
  })

  it('drops blank and whitespace-only strings', () => {
    expect(normalizeChoices(['a', '', 'b', '   ', 'c'])).toEqual(['a', 'b', 'c'])
  })

  it('drops strings with newlines', () => {
    expect(normalizeChoices(['a', 'b\nc', 'd'])).toEqual(['a', 'd'])
  })

  it('drops strings over 200 chars', () => {
    const long = 'x'.repeat(201)
    const ok = 'y'.repeat(200)
    expect(normalizeChoices(['a', long, ok])).toEqual(['a', ok])
  })

  it('drops empty items and keeps valid ones', () => {
    expect(normalizeChoices(['valid', '  ', '', 'also valid'])).toEqual(['valid', 'also valid'])
  })

  it('returns empty array when nothing survives', () => {
    expect(normalizeChoices(['', '  ', null, undefined])).toEqual([])
    expect(normalizeChoices([])).toEqual([])
  })
})

describe('normalizeQuestions', () => {
  it('returns empty array for non-array input', () => {
    expect(normalizeQuestions(null)).toEqual([])
    expect(normalizeQuestions('x')).toEqual([])
    expect(normalizeQuestions({})).toEqual([])
  })

  it('normalizes a valid batch and keys by qid', () => {
    const result = normalizeQuestions([
      { choices: ['a', 'b'], qid: 'q0', question: 'One?' },
      { qid: 'q1', question: 'Two?' }
    ])

    expect(result).toEqual([
      { choices: ['a', 'b'], multiSelect: false, qid: 'q0', question: 'One?' },
      { choices: null, multiSelect: false, qid: 'q1', question: 'Two?' }
    ])
  })

  it('drops entries missing qid or question text', () => {
    const result = normalizeQuestions([
      { qid: '', question: 'no qid' },
      { qid: 'q1', question: '   ' },
      'not-an-object',
      { qid: 'q2', question: 'kept' }
    ])

    expect(result.map(q => q.qid)).toEqual(['q2'])
  })

  it('degrades all-blank choices to open-ended per question', () => {
    const result = normalizeQuestions([{ choices: ['', '  '], qid: 'q0', question: 'Q?' }])

    expect(result[0]?.choices).toBeNull()
  })

  it('only honors multi_select when choices survive', () => {
    const result = normalizeQuestions([
      { choices: ['a', 'b'], multi_select: true, qid: 'q0', question: 'A?' },
      { multi_select: true, qid: 'q1', question: 'B?' }
    ])

    expect(result[0]?.multiSelect).toBe(true)
    expect(result[1]?.multiSelect).toBe(false)
  })
})

describe('clarify same-epoch tool-call alias', () => {
  beforeEach(() => {
    $clarifyRequests.set({})
    noteClarifyToolCall('session-alias', null)
  })

  afterEach(() => {
    $clarifyRequests.set({})
    noteClarifyToolCall('session-alias', null)
  })

  const aliased = (sessionId: string, requestId: string, toolCallId: string): void => {
    setClarifyRequest({
      choices: null,
      multiSelect: false,
      question: 'Proceed?',
      requestId,
      sessionId,
      toolCallId
    })
  }

  it('binds the started clarify tool-call id to the request that follows it', () => {
    noteClarifyToolCall('session-alias', { args: { question: 'Proceed?' }, toolCallId: 'call-live' })

    expect(clarifyToolCallAlias('session-alias', 'req-live', { question: 'Proceed?' })).toBe('call-live')
  })

  it('refuses to bind a started tool call whose question is a different one', () => {
    noteClarifyToolCall('session-alias', { args: { question: 'Something else?' }, toolCallId: 'call-other' })

    expect(clarifyToolCallAlias('session-alias', 'req-live', { question: 'Proceed?' })).toBeUndefined()
  })

  it('keeps the alias already bound to this request when the gateway replays it', () => {
    aliased('session-alias', 'req-replay', 'call-replay')

    expect(clarifyToolCallAlias('session-alias', 'req-replay', { question: 'Proceed?' })).toBe('call-replay')
  })

  it('settles the request when the completion carries the bound model tool-call id', () => {
    aliased('session-alias', 'req-remote', 'call-remote')

    expect(
      settleClarifyRequest('session-alias', {
        question: 'Proceed?',
        requestId: 'call-remote',
        toolName: 'clarify'
      })
    ).toBe(true)
    expect(hasClarifyRequest('session-alias')).toBe(false)
  })

  it('does not settle an older epoch’s model tool-call id with identical wording', () => {
    aliased('session-alias', 'req-new-epoch', 'call-new-epoch')

    expect(
      settleClarifyRequest('session-alias', {
        question: 'Proceed?',
        requestId: 'call-old-epoch',
        toolName: 'clarify'
      })
    ).toBe(false)
    expect($clarifyRequests.get()['session-alias']?.requestId).toBe('req-new-epoch')
  })

  it('does not settle from an unrelated tool call in the current epoch', () => {
    aliased('session-alias', 'req-unrelated', 'call-clarify')

    expect(settleClarifyRequest('session-alias', { requestId: 'call-read-1', toolName: 'read_file' })).toBe(false)
    expect(
      settleClarifyRequest('session-alias', { question: 'Proceed?', requestId: 'call-sibling', toolName: 'clarify' })
    ).toBe(false)
    expect(hasClarifyRequest('session-alias')).toBe(true)
  })

  it('never lets a non-clarify completion settle by the alias', () => {
    aliased('session-alias', 'req-guard', 'call-guard')

    expect(settleClarifyRequest('session-alias', { requestId: 'call-guard', toolName: 'read_file' })).toBe(false)
    expect(hasClarifyRequest('session-alias')).toBe(true)
  })

  it('binds a BATCH clarify by its joined question list', () => {
    noteClarifyToolCall('session-alias', {
      args: { questions: [{ question: 'Drink?' }, { question: 'Productive when?' }] },
      toolCallId: 'call-batch'
    })

    expect(
      clarifyToolCallAlias('session-alias', 'req-batch', {
        questions: [
          { choices: null, multiSelect: false, qid: 'q0', question: 'Drink?' },
          { choices: null, multiSelect: false, qid: 'q1', question: 'Productive when?' }
        ]
      })
    ).toBe('call-batch')
  })

  it('consumes the started tool call so one start can alias only one request', () => {
    noteClarifyToolCall('session-alias', { args: { question: 'Proceed?' }, toolCallId: 'call-once' })

    expect(clarifyToolCallAlias('session-alias', 'req-first', { question: 'Proceed?' })).toBe('call-once')
    expect(clarifyToolCallAlias('session-alias', 'req-second', { question: 'Proceed?' })).toBeUndefined()
  })

  it('does not settle a malformed clarify completion with no request id and no usable question', () => {
    aliased('session-alias', 'req-live', 'call-live')

    expect(settleClarifyRequest('session-alias', { toolName: 'clarify' })).toBe(false)
    expect($clarifyRequests.get()['session-alias']?.requestId).toBe('req-live')
    expect(hasClarifyRequest('session-alias')).toBe(true)
  })
})

describe('identity-absent matching-text batch fallback', () => {
  beforeEach(() => {
    $clarifyRequests.set({})
  })

  afterEach(() => {
    $clarifyRequests.set({})
  })

  const batch = (): ClarifyRequest => ({
    choices: null,
    multiSelect: false,
    question: '',
    questions: [
      { choices: null, multiSelect: false, qid: 'q0', question: 'Drink?' },
      { choices: null, multiSelect: false, qid: 'q1', question: 'Productive when?' }
    ],
    requestId: 'req-batch',
    sessionId: 'session-batch'
  })

  it('settles an identity-absent clarify completion whose ordered question list matches', () => {
    setClarifyRequest(batch())

    expect(
      settleClarifyRequest('session-batch', {
        questions: [{ question: 'Drink?' }, { question: 'Productive when?' }],
        toolName: 'clarify'
      })
    ).toBe(true)
    expect(hasClarifyRequest('session-batch')).toBe(false)
  })

  it('does not settle a missing, empty, reordered, or changed question list', () => {
    setClarifyRequest(batch())
    expect(settleClarifyRequest('session-batch', { toolName: 'clarify' })).toBe(false)
    expect(settleClarifyRequest('session-batch', { questions: [], toolName: 'clarify' })).toBe(false)
    expect(
      settleClarifyRequest('session-batch', {
        questions: [{ question: 'Productive when?' }, { question: 'Drink?' }],
        toolName: 'clarify'
      })
    ).toBe(false)
    expect(
      settleClarifyRequest('session-batch', {
        questions: [{ question: 'Drink?' }, { question: 'Changed?' }],
        toolName: 'clarify'
      })
    ).toBe(false)
    expect(hasClarifyRequest('session-batch')).toBe(true)
  })

  it('does not settle an unbound present id, sibling alias, or non-clarify tool', () => {
    setClarifyRequest({ ...batch(), toolCallId: 'call-batch' })
    expect(settleClarifyRequest('session-batch', { requestId: 'unbound', toolName: 'clarify' })).toBe(false)
    expect(settleClarifyRequest('session-batch', { requestId: 'call-other', toolName: 'clarify' })).toBe(false)
    expect(
      settleClarifyRequest('session-batch', {
        questions: [{ question: 'Drink?' }, { question: 'Productive when?' }],
        toolName: 'read_file'
      })
    ).toBe(false)
    expect(hasClarifyRequest('session-batch')).toBe(true)
  })
})

describe('rebind current-map and newer-target precedence', () => {
  beforeEach(() => {
    $clarifyRequests.set({})
  })

  afterEach(() => {
    $clarifyRequests.set({})
  })

  it('keeps the target request when it is a newer epoch and drops the old key', () => {
    setClarifyRequest({
      choices: null,
      multiSelect: false,
      question: 'Old?',
      requestId: 'req-old',
      sessionId: 'rt-old',
      toolCallId: 'call-old'
    })
    setClarifyRequest({
      choices: null,
      multiSelect: false,
      question: 'New?',
      requestId: 'req-new',
      sessionId: 'rt-new',
      toolCallId: 'call-new'
    })

    expect(rebindClarifyRequest('rt-old', 'rt-new')).toBe(true)
    expect($clarifyRequests.get()['rt-old']).toBeUndefined()
    expect($clarifyRequests.get()['rt-new']?.requestId).toBe('req-new')
    expect($clarifyRequests.get()['rt-new']?.toolCallId).toBe('call-new')
  })

  it('is idempotent after the old key is already gone', () => {
    setClarifyRequest({
      choices: null,
      multiSelect: false,
      question: 'Live?',
      requestId: 'req-live',
      sessionId: 'rt-new',
      toolCallId: 'call-live'
    })

    expect(rebindClarifyRequest('rt-old', 'rt-new')).toBe(true)
    expect(rebindClarifyRequest('rt-old', 'rt-new')).toBe(true)
    expect($clarifyRequests.get()['rt-new']?.requestId).toBe('req-live')
    expect(Object.keys($clarifyRequests.get())).toEqual(['rt-new'])
  })
})

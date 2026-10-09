import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $clarifyRequest,
  $clarifyRequests,
  $setupChooseStages,
  answerSetupCard,
  type ClarifyRequest,
  clearClarifyRequest,
  hasClarifyRequest,
  normalizeChoices,
  normalizeQuestions,
  setClarifyRequest,
  setupChooseStage,
  skipClarifyRequest,
  stageSetupChoose
} from './clarify'
import { $gateway } from './gateway'
import { rememberServerRequest, resetServerRequestsForTests } from './server-requests'
import { $activeSessionId } from './session'

function clarify(sessionId: string | null, requestId: string): ClarifyRequest {
  return {
    questions: [{ choices: null, multiSelect: false, qid: 'q0', question: `question-${requestId}` }],
    requestId,
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
    resetServerRequestsForTests()
    request.mockClear()
    $gateway.set({ request } as unknown as ReturnType<typeof $gateway.get>)
  })

  afterEach(() => {
    $clarifyRequests.set({})
    $gateway.set(null)
  })

  it('cancels the session\u2019s clarify with an empty response and drops it', async () => {
    const respond = vi.fn()

    rememberServerRequest({ fail: vi.fn(), id: 'req-a', method: 'clarify', params: {}, respond })
    setClarifyRequest(clarify('session-a', 'req-a'))
    setClarifyRequest(clarify('session-b', 'req-b'))

    await expect(skipClarifyRequest('session-a')).resolves.toBe(true)

    expect(respond).toHaveBeenCalledWith({})
    expect(hasClarifyRequest('session-a')).toBe(false)
    // A background session's question is untouched — only the one being typed
    // over is skipped.
    expect(hasClarifyRequest('session-b')).toBe(true)
  })

  it('is a no-op when the session has no clarify parked', async () => {
    await expect(skipClarifyRequest('session-a')).resolves.toBe(false)
    expect(request).not.toHaveBeenCalled()
  })

  it('still reports the skip when the server request is already gone (expired / other window answered)', async () => {
    setClarifyRequest(clarify('session-a', 'req-a'))

    await expect(skipClarifyRequest('session-a')).resolves.toBe(true)
    expect(hasClarifyRequest('session-a')).toBe(false)
  })
})

describe('answerSetupCard', () => {
  const ACCENTS = { '#0000ff': 'Blue', '#ff0000': 'Red' }
  let look: string

  // The card's pick: snapshot the look once, then apply the row (as setup-pending's `stage` does).
  function mountAccentCard(requestId: string) {
    stageSetupChoose(requestId, {
      labels: ACCENTS,
      preview: id => {
        const before = look
        stageSetupChoose(requestId, {
          picked: [id],
          revert: setupChooseStage(requestId).revert ?? (() => (look = before))
        })
        look = id
      }
    })
  }

  beforeEach(() => {
    look = 'original'
    $clarifyRequests.set({})
    $setupChooseStages.set({})
    resetServerRequestsForTests()
  })

  afterEach(() => {
    $clarifyRequests.set({})
    $setupChooseStages.set({})
  })

  it('applies a typed row over the row previewed on the card, and keeps it after the card clears', () => {
    const respond = vi.fn()

    rememberServerRequest({ fail: vi.fn(), id: 'req-a', method: 'setup_choose', params: {}, respond })
    setClarifyRequest({
      ...clarify('session-a', 'req-a'),
      setup: { kind: 'accent', multiSelect: false, options: null, preselected: [] }
    })
    mountAccentCard('req-a')
    setupChooseStage('req-a').preview?.('#ff0000')

    expect(answerSetupCard('session-a', 'blue')).toBe(true)

    expect(respond).toHaveBeenCalledWith({ label: 'Blue', picked: '#0000ff' })
    expect(hasClarifyRequest('session-a')).toBe(false)
    expect(look).toBe('#0000ff')
  })

  it('does not notify subscribers when a staged patch changes nothing', () => {
    const preview = () => undefined
    const listener = vi.fn()

    stageSetupChoose('req-a', { labels: ACCENTS, preview })
    const before = $setupChooseStages.get()
    const unlisten = $setupChooseStages.listen(listener)

    stageSetupChoose('req-a', { labels: ACCENTS, preview })
    unlisten()

    expect($setupChooseStages.get()).toBe(before)
    expect(listener).not.toHaveBeenCalled()
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

  it('keeps strings with newlines so option reasons can wrap', () => {
    expect(normalizeChoices(['a', 'b\nc', 'd'])).toEqual(['a', 'b\nc', 'd'])
  })

  it('keeps long strings and only drops them past the abuse cap', () => {
    const long = 'x'.repeat(1500)
    const atCap = 'y'.repeat(8000)
    const over = 'z'.repeat(8001)
    expect(normalizeChoices(['a', long, atCap, over])).toEqual(['a', long, atCap])
  })

  it('measures the cap on the bare text, discounting the (Recommended) label', () => {
    const atCap = 'x'.repeat(8000) + ' (Recommended)'
    expect(normalizeChoices([atCap])).toEqual([atCap])
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

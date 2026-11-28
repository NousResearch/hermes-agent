import type { GatewayEvent } from '@hermes/shared'
import { afterEach, expect, it } from 'vitest'

import { _resetItemIndexForTests, ownerIndex, ownerItems } from '@/store/item-index'
import { $shellChoice } from '@/store/shell-variant'

import { recordStreamedItems } from './library-items'
import type { GatewayEventContext } from './types'

function context(event: Partial<GatewayEvent>, payload: Record<string, unknown>): GatewayEventContext {
  const sessionStates = new Map([['runtime-1', { storedSessionId: 'stored-1' }]])

  return {
    deps: { sessionStateByRuntimeIdRef: { current: sessionStates } },
    event: { type: 'tool.complete', ...event },
    payload,
    sessionId: 'runtime-1',
    occurredAt: 1_700_000_000
  } as unknown as GatewayEventContext
}

const recorded = () => ownerItems(ownerIndex('local').snapshot)

afterEach(() => {
  $shellChoice.set(null)
  window.localStorage.clear()
  _resetItemIndexForTests()
})

it('records a finished tool call and reply under the stored chat, in the redesign', () => {
  $shellChoice.set(true)

  recordStreamedItems(
    context({ type: 'tool.complete' }, { name: 'image_generate', result: { image_path: '/out/cat.png' } })
  )
  recordStreamedItems(context({ type: 'message.complete' }, { text: 'Read https://example.com/guide' }))

  expect(recorded().map(item => [item.kind, item.value, item.sessionId])).toEqual(
    expect.arrayContaining([
      ['image', '/out/cat.png', 'stored-1'],
      ['link', 'https://example.com/guide', 'stored-1']
    ])
  )
})

it('records nothing in classic, which does not read the index', () => {
  $shellChoice.set(false)

  recordStreamedItems(context({ type: 'message.complete' }, { text: 'Read https://example.com/guide' }))

  expect(recorded()).toEqual([])
})

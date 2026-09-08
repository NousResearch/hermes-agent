import { act, cleanup } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { renderMessageStream } from './test-harness'

afterEach(() => { cleanup(); vi.restoreAllMocks() })

it('reuses stable indexes through the real tool event and image-dedupe producer', async () => {
  const stream = renderMessageStream('index-session')
  await act(async () => { await Promise.resolve() })
  const original = Map.prototype.set
  let indexedRows = 0
  vi.spyOn(Map.prototype, 'set').mockImplementation(function (this: Map<unknown, unknown>, key, value) {
    if (typeof key === 'string' && key.startsWith('index-probe-') && typeof value === 'number') {indexedRows += 1}

    return original.call(this, key, value)
  })
  act(() => {
    for (let i = 0; i < 160; i += 1) {
      stream.handleEvent({ session_id: 'index-session', type: 'tool.start', payload: { name: 'read_file', tool_id: `index-probe-${i}` } })
    }

    for (let i = 0; i < 160; i += 1) {
      stream.handleEvent({ session_id: 'index-session', type: 'tool.complete', payload: { name: 'read_file', tool_id: `index-probe-${i}`, result: 'ok' } })
    }
  })
  const parts = stream.state()?.messages.at(-1)?.parts
  expect(parts).toHaveLength(160)
  expect(parts?.every(part => part.type === 'tool-call' && part.result !== undefined)).toBe(true)
  expect(indexedRows).toBeGreaterThan(0)
  expect(indexedRows).toBeLessThanOrEqual(160 * 4)
})

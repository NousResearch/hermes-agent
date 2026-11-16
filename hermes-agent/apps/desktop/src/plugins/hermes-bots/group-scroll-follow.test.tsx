import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

vi.mock('@hermes/plugin-sdk', async () => (await import('./group-test-utils')).pluginSdkMock({}))

import { useStickToBottom } from './canonical-group-details'

function Transcript({ item }: { item: string }) {
  const ref = useStickToBottom()

  return <div ref={ref} role="log"><div>{item}</div></div>
}

afterEach(() => { cleanup(); vi.unstubAllGlobals() })

it('follows added messages, approvals and viewport changes, but preserves a reader’s place', () => {
  let resize!: () => void
  vi.stubGlobal('ResizeObserver', class {
    constructor(callback: () => void) { resize = callback }
    observe() {}
    disconnect() {}
  })
  const view = render(<Transcript item="History" />)
  const log = screen.getByRole('log')
  let height = 500
  let viewport = 100
  let top = 0
  Object.defineProperties(log, {
    scrollHeight: { get: () => height },
    clientHeight: { get: () => viewport },
    scrollTop: { get: () => top, set: value => { top = Math.min(value, height - viewport) } }
  })
  act(resize)
  expect(top).toBe(400)

  for (const item of ['New sent message', 'Approval card']) {
    height += 400
    view.rerender(<Transcript item={item} />)
    // Layout can dispatch scroll before ResizeObserver while scrollTop stays put.
    fireEvent.scroll(log)
    act(resize)
    expect(top).toBe(height - viewport)
  }

  log.scrollTop = 200
  fireEvent.scroll(log)
  height += 300
  act(resize)
  expect(top).toBe(200)

  log.scrollTop = height - viewport
  fireEvent.scroll(log)
  viewport = 50
  fireEvent.scroll(log)
  act(resize)
  expect(top).toBe(height - viewport)
})

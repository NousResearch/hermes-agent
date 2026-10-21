import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { LogSearchField, useLogSearch } from './log-search'
import { LogTail } from './log-tail'

afterEach(cleanup)

function Harness({ lines }: { lines: string[] }) {
  const [query, setQuery] = useState('')
  const search = useLogSearch(lines, query)

  return (
    <I18nProvider configClient={null}>
      <LogSearchField onChange={setQuery} placeholder="Search logs" search={search} value={query} />
      <LogTail emptyLabel="No logs" lines={lines} search={search} />
    </I18nProvider>
  )
}

const input = () => screen.getByRole('textbox', { name: 'Search logs' })

const marks = (container: HTMLElement, state?: 'active') =>
  container.querySelectorAll(state ? '[data-log-hit="active"]' : '[data-log-hit]')

describe('LogTail search', () => {
  beforeEach(() => {
    Element.prototype.scrollIntoView = vi.fn()
  })

  it('highlights every hit in place and steps one active hit through all of them', () => {
    const { container } = render(<Harness lines={['Docker docker', 'nothing', 'DOCKER']} />)

    fireEvent.change(input(), { target: { value: 'docker' } })

    expect(marks(container)).toHaveLength(3)
    expect(container.textContent).toContain('nothing')

    const visited = new Set<string>()

    for (let step = 0; step < 3; step += 1) {
      expect(marks(container, 'active')).toHaveLength(1)
      visited.add(`${[...marks(container)].indexOf(marks(container, 'active')[0])}`)
      fireEvent.keyDown(input(), { key: 'Enter' })
    }

    expect(visited.size).toBe(3)
  })

  it('returns to the live tail when the search is cleared', () => {
    let height = 400

    vi.spyOn(HTMLElement.prototype, 'scrollHeight', 'get').mockImplementation(() => height)

    const lines = ['boot', 'needle here', 'ready']
    const { container, rerender } = render(<Harness lines={lines} />)
    const scroller = container.querySelector<HTMLElement>('[data-selectable-text]')!

    fireEvent.change(input(), { target: { value: 'needle' } })
    // Parking on a hit releases the follow, as a real scroll event would.
    scroller.scrollTop = 0
    fireEvent.scroll(scroller)

    fireEvent.change(input(), { target: { value: '' } })
    height = 900
    act(() => rerender(<Harness lines={[...lines, 'new line']} />))

    expect(scroller.scrollTop).toBe(900)
  })
})

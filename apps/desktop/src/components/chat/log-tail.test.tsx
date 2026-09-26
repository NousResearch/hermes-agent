import { cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { LogTail } from './log-tail'

afterEach(cleanup)

describe('LogTail search', () => {
  beforeEach(() => {
    Element.prototype.scrollIntoView = vi.fn()
  })

  it('shows the total number of matches and highlights every hit', () => {
    const onMatchCountChange = vi.fn()
    const { container } = render(
      <LogTail
        emptyLabel="No logs"
        lines={['Docker docker', 'nothing', 'DOCKER']}
        onMatchCountChange={onMatchCountChange}
        query="docker"
      />
    )

    expect(onMatchCountChange).toHaveBeenLastCalledWith(3)
    expect(container.querySelectorAll('mark')).toHaveLength(3)
  })

  it('scrolls the first matching line into view when a query opens', () => {
    const scrollIntoView = vi.fn()
    Element.prototype.scrollIntoView = scrollIntoView

    render(
      <LogTail
        emptyLabel="No logs"
        lines={['alpha', 'needle first', 'needle second']}
        query="needle"
      />
    )

    expect(scrollIntoView).toHaveBeenCalledWith({ block: 'center' })
  })

  it('keeps zero-match searches visible without scrolling', () => {
    const scrollIntoView = vi.fn()
    Element.prototype.scrollIntoView = scrollIntoView

    const onMatchCountChange = vi.fn()

    render(
      <LogTail
        emptyLabel="No logs"
        lines={['alpha', 'beta']}
        onMatchCountChange={onMatchCountChange}
        query="needle"
      />
    )

    expect(onMatchCountChange).toHaveBeenLastCalledWith(0)
    expect(scrollIntoView).not.toHaveBeenCalled()
  })
})

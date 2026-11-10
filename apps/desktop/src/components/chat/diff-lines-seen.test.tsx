import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('./syntax-diff', () => ({ default: () => <span data-testid="shiki-diff" /> }))

import { FileDiffPanel } from './diff-lines'

const DIFF = ['--- a/file.ts', '+++ b/file.ts', '@@ -1,2 +1,2 @@', ' const a = 1', '-const b = 2', '+const b = 3'].join(
  '\n'
)

let intersect: (() => void) | null = null
const realObserver = globalThis.IntersectionObserver

beforeEach(() => {
  intersect = null
  globalThis.IntersectionObserver = class {
    constructor(callback: IntersectionObserverCallback) {
      intersect = () => callback([{ isIntersecting: true } as IntersectionObserverEntry], this as never)
    }
    observe() {}
    unobserve() {}
    disconnect() {}
    takeRecords() {
      return []
    }
  } as unknown as typeof IntersectionObserver
})

afterEach(() => {
  cleanup()
  globalThis.IntersectionObserver = realObserver
})

describe('compact FileDiffPanel highlights only once it is on screen', () => {
  it('keeps the plain colored rows off screen, then highlights on first intersection', async () => {
    const { container } = render(<FileDiffPanel diff={DIFF} path="file.ts" />)

    await act(() => new Promise(resolve => setTimeout(resolve, 50)))
    expect(screen.queryByTestId('shiki-diff')).toBeNull()
    expect(container.textContent).toContain('const b = 3')

    await act(async () => intersect?.())
    expect(await screen.findByTestId('shiki-diff')).toBeTruthy()
  })
})

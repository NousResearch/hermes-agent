import { act, cleanup, render, screen } from '@testing-library/react'
import { Suspense, useState } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { lazyView } from './lazy-view'

afterEach(cleanup)

function Page({ label }: { label: string }) {
  const [count, setCount] = useState(0)

  return (
    <button onClick={() => setCount(value => value + 1)} type="button">
      {label} {count}
    </button>
  )
}

const shell = (node: React.ReactNode) => <Suspense fallback={<p>fallback</p>}>{node}</Suspense>

describe('lazyView', () => {
  it('draws a preloaded page in the navigation commit itself, never through the Suspense fallback', async () => {
    const View = lazyView(async () => Page)
    await View.preload()

    render(shell(<View label="Capabilities" />))

    // No act/await: the first synchronous commit already holds the page.
    expect(screen.getByRole('button').textContent).toBe('Capabilities 0')
    expect(screen.queryByText('fallback')).toBeNull()
  })

  it('suspends a page whose chunk is not in yet, then shows it', async () => {
    let resolve!: (component: typeof Page) => void
    const View = lazyView(() => new Promise<typeof Page>(done => (resolve = done)))

    render(shell(<View label="Library" />))
    expect(screen.getByText('fallback')).toBeTruthy()

    await act(async () => resolve(Page))

    expect((await screen.findByRole('button')).textContent).toBe('Library 0')
  })

  it('loads the chunk once however many mounts and preloads ask for it', async () => {
    const load = vi.fn(async () => Page)
    const View = lazyView(load)

    await Promise.all([View.preload(), View.preload()])
    render(shell(<View label="a" />))
    render(shell(<View label="b" />))

    expect(load).toHaveBeenCalledTimes(1)
  })

  it('keeps page state when its chunk lands mid-session (one component identity)', async () => {
    let resolve!: (component: typeof Page) => void
    const View = lazyView(() => new Promise<typeof Page>(done => (resolve = done)))
    const { rerender } = render(shell(<View label="Agents" />))

    await act(async () => resolve(Page))
    const button = await screen.findByRole('button')
    act(() => button.click())
    rerender(shell(<View label="Agents" />))

    expect(screen.getByRole('button').textContent).toBe('Agents 1')
  })

  it('retries a chunk whose fetch failed instead of failing every later visit', async () => {
    const load = vi.fn<() => Promise<typeof Page>>().mockRejectedValueOnce(new Error('offline')).mockResolvedValue(Page)
    const View = lazyView(load)

    await expect(View.preload()).rejects.toThrow('offline')
    await View.preload()
    render(shell(<View label="Schedules" />))

    expect(screen.getByRole('button').textContent).toBe('Schedules 0')
    expect(load).toHaveBeenCalledTimes(2)
  })
})

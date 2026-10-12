import { cleanup, render, screen } from '@testing-library/react'
import { createRef, type ReactElement, type ReactNode, type Ref } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { Tip, Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './tooltip'

const { contentProps } = vi.hoisted(() => ({ contentProps: vi.fn() }))

vi.mock('radix-ui', async () => {
  const { cloneElement, createElement, forwardRef } = await import('react')
  const passthrough = ({ children }: { children: ReactNode }) => children

  return {
    Tooltip: {
      Arrow: passthrough,
      Content: ({ children, ...props }: { children: ReactNode }) => {
        contentProps(props)

        return createElement('div', { role: 'tooltip' }, children)
      },
      Portal: passthrough,
      Provider: passthrough,
      Root: passthrough,
      Trigger: forwardRef<HTMLButtonElement, { asChild?: boolean; children: ReactNode }>(
        ({ asChild, children, ...props }, ref) =>
          asChild
            ? cloneElement(children as ReactElement<{ ref?: Ref<HTMLButtonElement> }>, { ...props, ref })
            : createElement('button', { ...props, ref }, children)
      )
    }
  }
})

afterEach(() => {
  cleanup()
  contentProps.mockClear()
  vi.restoreAllMocks()
})

const latestContent = () => contentProps.mock.calls.at(-1)?.[0]

describe('tooltip placement', () => {
  it('uses the owning pane for a control without changing its trigger', () => {
    render(
      <div data-testid="pane" data-tree-group="test-pane">
        <Tip label="Details">
          <button>Trigger</button>
        </Tip>
      </div>
    )

    expect(latestContent().collisionBoundary).toBe(screen.getByTestId('pane'))
    expect(screen.getByRole('button').parentElement).toBe(screen.getByTestId('pane'))
  })

  it('skips a tree-group host without layout and clips against the enclosing pane', () => {
    // The floating-composer host carries its own data-tree-group while being
    // `display: contents`; a zero-rect boundary would hide every composer tip.
    const rect = (width: number, height: number) =>
      ({ top: 0, left: 0, right: width, bottom: height, width, height, x: 0, y: 0, toJSON: () => ({}) }) as DOMRect

    vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function (this: Element) {
      return (this as HTMLElement).dataset.treeGroup === 'floating-host' ? rect(0, 0) : rect(40, 30)
    })

    render(
      <div data-testid="pane" data-tree-group="test-pane">
        <div data-tree-group="floating-host">
          <Tip label="Details">
            <button>Trigger</button>
          </Tip>
        </div>
      </div>
    )

    expect(latestContent().collisionBoundary).toBe(screen.getByTestId('pane'))
  })

  it.each(['row', 'left-rail', 'right-rail'] as const)('lets %s escape the pane', placement => {
    render(
      <div data-tree-group="test-pane">
        <Tip label="Details" placement={placement}>
          <button>Trigger</button>
        </Tip>
      </div>
    )

    expect(latestContent().collisionBoundary).toBeUndefined()
  })

  it('allows a control to opt out of its pane boundary', () => {
    render(
      <div data-tree-group="test-pane">
        <Tip boundary="viewport" label="Details">
          <button>Trigger</button>
        </Tip>
      </div>
    )

    expect(latestContent().collisionBoundary).toBeUndefined()
  })

  it('forwards object refs and clears them on unmount', () => {
    const ref = createRef<HTMLButtonElement>()

    const { unmount } = render(
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger ref={ref}>Trigger</TooltipTrigger>
          <TooltipContent>Details</TooltipContent>
        </Tooltip>
      </TooltipProvider>
    )

    expect(ref.current).toBe(screen.getByRole('button'))
    unmount()
    expect(ref.current).toBeNull()
  })

  it('honors callback-ref cleanup', () => {
    const dispose = vi.fn()
    const ref = vi.fn(() => dispose)

    const { unmount } = render(
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger ref={ref}>Trigger</TooltipTrigger>
          <TooltipContent>Details</TooltipContent>
        </Tooltip>
      </TooltipProvider>
    )

    expect(ref).toHaveBeenCalledWith(screen.getByRole('button'))
    unmount()
    expect(dispose).toHaveBeenCalledOnce()
    expect(ref).not.toHaveBeenCalledWith(null)
  })
})

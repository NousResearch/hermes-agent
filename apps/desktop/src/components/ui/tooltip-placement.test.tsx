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
})

const latestContent = () => contentProps.mock.calls.at(-1)?.[0]

describe('tooltip placement', () => {
  it.each([
    ['control', 'top'],
    ['toolbar', 'bottom'],
    ['row', 'right'],
    ['left-rail', 'right'],
    ['right-rail', 'left']
  ] as const)('prefers %s tooltips on the %s', (placement, side) => {
    render(
      <Tip label="Details" placement={placement}>
        <button>Trigger</button>
      </Tip>
    )

    expect(latestContent()).toMatchObject({ side, align: 'center', hideWhenDetached: true, collisionPadding: 12 })
    expect(latestContent().avoidCollisions).not.toBe(false)
  })

  it('keeps explicit overrides', () => {
    render(
      <Tip align="end" collisionPadding={20} label="Details" placement="row" side="bottom">
        <button>Trigger</button>
      </Tip>
    )

    expect(latestContent()).toMatchObject({ side: 'bottom', align: 'end', collisionPadding: 20 })
  })

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

  it('preserves block content within one label surface', () => {
    render(
      <Tip
        label={
          <div>
            Long description
            <br />
            Second line
          </div>
        }
      >
        <button>Trigger</button>
      </Tip>
    )

    expect(screen.getByRole('tooltip').querySelector('[data-slot="tooltip-label"]')?.textContent).toBe(
      'Long descriptionSecond line'
    )
    expect(screen.getByRole('tooltip').querySelector('[data-slot="tooltip-arrow"]')).not.toBeNull()
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

import { act, cleanup, render } from '@testing-library/react'
import type * as RadixUi from 'radix-ui'
import { cloneElement, createElement, forwardRef, type ReactElement, type ReactNode, type Ref } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $shellChoice } from '@/store/shell-variant'

import { TooltipIconButton } from '../assistant-ui/tooltip-icon-button'

import { CopyButton } from './copy-button'

const { contentProps } = vi.hoisted(() => ({ contentProps: vi.fn() }))

vi.mock('radix-ui', async importOriginal => {
  const actual = await importOriginal<typeof RadixUi>()
  const passthrough = ({ children }: { children: ReactNode }) => children

  return {
    ...actual,
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
  act(() => $shellChoice.set(null))
})

const sides = () => contentProps.mock.calls.map(([props]) => (props as { side?: string }).side)

function renderActionRow() {
  render(
    <div>
      <CopyButton appearance="icon" label="Copy" text="hello" />
      <TooltipIconButton tooltip="Refresh">↻</TooltipIconButton>
    </div>
  )
}

describe('CopyButton tooltip side', () => {
  it('opens on the same side as the icon buttons beside it in the redesign', () => {
    act(() => $shellChoice.set(true))
    renderActionRow()

    const [copy, neighbour] = sides()

    expect(copy).toBe('top')
    expect(copy).toBe(neighbour)
  })

  it('keeps classic opening below', () => {
    act(() => $shellChoice.set(false))
    renderActionRow()

    expect(sides()[0]).toBe('bottom')
  })

  it('honours an explicit side in both shells', () => {
    act(() => $shellChoice.set(true))
    render(<CopyButton appearance="icon" side="left" text="hello" />)

    expect(sides()[0]).toBe('left')
  })
})

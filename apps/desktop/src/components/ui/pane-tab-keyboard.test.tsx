import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { type KeyboardEvent, type KeyboardEventHandler, useRef, useState } from 'react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { stubMenuDomApis } from '@/test/jsdom'

import { ActionsContextMenu } from './actions-menu'
import { PaneTab, PaneTabLabel } from './pane-tab'
import { usePaneTabKeyboard } from './pane-tab-keyboard'

beforeAll(stubMenuDomApis)
afterEach(cleanup)

function TabList({
  initialIds = ['one', 'two', 'three'],
  vertical = false,
  wrapped = false,
  onKeyDown
}: {
  initialIds?: string[]
  vertical?: boolean
  wrapped?: boolean
  onKeyDown?: KeyboardEventHandler<HTMLDivElement>
}) {
  const [ids, setIds] = useState(initialIds)
  const [active, setActive] = useState(initialIds[0]!)
  const [collapsed, setCollapsed] = useState(vertical)
  const ref = useRef<HTMLDivElement>(null)
  const tabProps = usePaneTabKeyboard(ids, active, ref)

  return (
    <div ref={ref}>
      <div
        aria-label="Sessions"
        aria-orientation={collapsed ? 'vertical' : 'horizontal'}
        key={String(collapsed)}
        role="tablist"
      >
        {ids.map(id => {
          const tab = (
            <PaneTab
              {...tabProps(id)}
              active={id === active}
              aria-label={id}
              aria-selected={id === active}
              key={id}
              onActivate={() => {
                setActive(id)
                setCollapsed(false)
              }}
              onClose={() => setIds(current => current.filter(value => value !== id))}
              onKeyDown={event => {
                onKeyDown?.(event)

                if (!event.defaultPrevented) {
                  tabProps(id).onKeyDown(event)
                }
              }}
              role="tab"
              vertical={collapsed}
            >
              <PaneTabLabel>{id}</PaneTabLabel>
            </PaneTab>
          )

          return wrapped ? (
            <ActionsContextMenu items={kit => <kit.Item>Session action</kit.Item>} key={id}>
              {tab}
            </ActionsContextMenu>
          ) : (
            tab
          )
        })}
        <button type="button">New tab</button>
      </div>
      <button type="button">After tabs</button>
    </div>
  )
}

const focus = (element: HTMLElement) => act(() => element.focus())
const tab = (name: string) => screen.getByRole('tab', { name })

describe('pane tab keyboard navigation', () => {
  it('has one entry point, focuses siblings without selecting, and activates with Enter/Space', () => {
    render(<TabList wrapped />)
    const one = tab('one')
    const two = tab('two')
    expect(screen.getAllByRole('tab').map(element => element.tabIndex)).toEqual([0, -1, -1])
    focus(one)
    fireEvent.keyDown(one, { key: 'ArrowRight' })
    expect(globalThis.document.activeElement).toBe(two)
    expect(one.getAttribute('aria-selected')).toBe('true')
    expect(two.getAttribute('aria-selected')).toBe('false')
    expect(screen.getAllByRole('tab').map(element => element.tabIndex)).toEqual([-1, 0, -1])
    fireEvent.keyDown(two, { key: 'Enter' })
    expect(two.getAttribute('aria-selected')).toBe('true')
    expect(globalThis.document.activeElement).toBe(two)
    fireEvent.keyDown(two, { key: 'ArrowRight' })
    fireEvent.keyDown(tab('three'), { key: ' ' })
    expect(tab('three').getAttribute('aria-selected')).toBe('true')
    expect(globalThis.document.activeElement).toBe(tab('three'))
    expect(within(two).getByRole('button', { name: 'Close' }).tabIndex).toBe(-1)
    fireEvent.keyDown(tab('three'), { key: 'ArrowLeft' })
    focus(screen.getByRole('button', { name: 'New tab' }))
    expect(screen.getAllByRole('tab').map(element => element.tabIndex)).toEqual([-1, -1, 0])
  })

  it('wraps siblings, supports Home/End, ignores the other axis and leaves Tab unclaimed', () => {
    render(
      <>
        <TabList />
        <TabList initialIds={['other']} />
      </>
    )
    focus(tab('one'))
    fireEvent.keyDown(tab('one'), { key: 'ArrowLeft' })
    expect(globalThis.document.activeElement).toBe(tab('three'))
    fireEvent.keyDown(tab('three'), { key: 'Home' })
    expect(globalThis.document.activeElement).toBe(tab('one'))
    fireEvent.keyDown(tab('one'), { key: 'End' })
    expect(globalThis.document.activeElement).toBe(tab('three'))
    expect(fireEvent.keyDown(tab('three'), { key: 'ArrowDown' })).toBe(true)
    expect(globalThis.document.activeElement).toBe(tab('three'))
    expect(fireEvent.keyDown(tab('three'), { key: 'Tab' })).toBe(true)
    expect(fireEvent.keyDown(tab('three'), { key: 'Tab', shiftKey: true })).toBe(true)
    expect(fireEvent.keyDown(tab('three'), { key: 'ArrowLeft', metaKey: true })).toBe(true)
    expect(globalThis.document.activeElement).toBe(tab('three'))
  })

  it('uses vertical arrows and preserves the activated tab through a restore remount', () => {
    render(<TabList vertical />)
    focus(tab('one'))
    expect(fireEvent.keyDown(tab('one'), { key: 'ArrowRight' })).toBe(true)
    fireEvent.keyDown(tab('one'), { key: 'ArrowDown' })
    expect(globalThis.document.activeElement).toBe(tab('two'))
    fireEvent.keyDown(tab('two'), { key: 'ArrowUp' })
    expect(globalThis.document.activeElement).toBe(tab('one'))
    fireEvent.keyDown(tab('one'), { key: 'End' })
    fireEvent.keyDown(tab('three'), { key: 'Enter' })
    expect(screen.getByRole('tablist').getAttribute('aria-orientation')).toBe('horizontal')
    expect(globalThis.document.activeElement).toBe(tab('three'))
    expect(tab('three').getAttribute('aria-selected')).toBe('true')
  })

  it('closes with Delete and focuses the next sibling, or the previous last sibling', () => {
    render(<TabList />)
    focus(tab('two'))
    fireEvent.keyDown(tab('two'), { key: 'Delete' })
    expect(screen.queryByRole('tab', { name: 'two' })).toBeNull()
    expect(globalThis.document.activeElement).toBe(tab('three'))
    fireEvent.keyDown(tab('three'), { key: 'Delete' })
    expect(globalThis.document.activeElement).toBe(tab('one'))
    expect(tab('one').tabIndex).toBe(0)
  })

  it('composes caller cancellation and context-menu handlers without activating the tab', () => {
    const handler = vi.fn((event: KeyboardEvent<HTMLDivElement>) => event.preventDefault())
    render(<TabList onKeyDown={handler} wrapped />)
    const one = tab('one')
    focus(tab('two'))
    fireEvent.keyDown(tab('two'), { key: 'Enter' })
    expect(handler).toHaveBeenCalledTimes(1)
    expect(one.getAttribute('aria-selected')).toBe('true')
    fireEvent.contextMenu(tab('two'), { clientX: 10, clientY: 10 })
    expect(screen.getByRole('menuitem', { name: 'Session action' })).toBeTruthy()
    expect(one.getAttribute('aria-selected')).toBe('true')
  })
})

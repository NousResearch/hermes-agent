import { act, renderHook } from '@testing-library/react'
import type { ReactNode } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $boardSlug } from './api'
import { parseKanbanDeepLink, useKanbanDeepLink } from './deep-link'

afterEach(() => {
  $boardSlug.set('')
})

describe('parseKanbanDeepLink', () => {
  it('reads board and task from the query string', () => {
    expect(parseKanbanDeepLink('?board=default&task=t_c3626291')).toEqual({
      board: 'default',
      task: 't_c3626291'
    })
  })

  it('defaults absent params to empty strings', () => {
    expect(parseKanbanDeepLink('')).toEqual({ board: '', task: '' })
    expect(parseKanbanDeepLink('?board=default')).toEqual({ board: 'default', task: '' })
    expect(parseKanbanDeepLink('?task=t_c3626291')).toEqual({ board: '', task: 't_c3626291' })
  })

  it('trims whitespace so a stray space never becomes a bogus slug/id', () => {
    expect(parseKanbanDeepLink('?board=%20default%20&task=%20t_c3626291%20')).toEqual({
      board: 'default',
      task: 't_c3626291'
    })
  })
})

function wrapper(route: string) {
  return function Wrapper({ children }: { children: ReactNode }) {
    return <MemoryRouter initialEntries={[route]}>{children}</MemoryRouter>
  }
}

describe('useKanbanDeepLink', () => {
  it('opens the task drawer for the given id and sets the board', () => {
    const onOpenTask = vi.fn()

    renderHook(() => useKanbanDeepLink(onOpenTask), {
      wrapper: wrapper('/kanban?board=default&task=t_c3626291')
    })

    expect($boardSlug.get()).toBe('default')
    expect(onOpenTask).toHaveBeenCalledWith('t_c3626291')
    expect(onOpenTask).toHaveBeenCalledTimes(1)
  })

  it('leaves the board untouched when the param matches the current selection', () => {
    $boardSlug.set('default')
    const onOpenTask = vi.fn()
    const setSpy = vi.spyOn($boardSlug, 'set')

    renderHook(() => useKanbanDeepLink(onOpenTask), {
      wrapper: wrapper('/kanban?board=default&task=t_c3626291')
    })

    expect(setSpy).not.toHaveBeenCalled()
  })

  it('does nothing when the URL carries no deep-link params (plain /kanban nav)', () => {
    $boardSlug.set('shipping')
    const onOpenTask = vi.fn()

    renderHook(() => useKanbanDeepLink(onOpenTask), { wrapper: wrapper('/kanban') })

    expect($boardSlug.get()).toBe('shipping')
    expect(onOpenTask).not.toHaveBeenCalled()
  })

  it('applies only once per mount, even if the hook re-renders', () => {
    const onOpenTask = vi.fn()

    const { rerender } = renderHook(() => useKanbanDeepLink(onOpenTask), {
      wrapper: wrapper('/kanban?board=default&task=t_c3626291')
    })

    act(() => {
      $boardSlug.set('shipping')
    })
    rerender()

    expect(onOpenTask).toHaveBeenCalledTimes(1)
    // A later, unrelated board switch (board-switcher, another deep link
    // navigated to since) is not clobbered back to the stale URL value.
    expect($boardSlug.get()).toBe('shipping')
  })
})

import { fireEvent } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { createDropClickSwallow, findSessionRow } from './reorderable-list'

/**
 * The regression the swallow exists for: freezing the rows during a project drag (so a nest target
 * cannot slide out from under the pointer) leaves the pointer over the target's row at release, and
 * the browser delivers a `click` to it. On a project row that click IS the row's activation, so a
 * nest release also entered the project and switched the view.
 *
 * Tested directly rather than through a simulated drag: jsdom gives dnd-kit no layout to measure, so a
 * pointer drag there never reaches `handleDragEnd` and would assert nothing about the listener. The
 * behaviour under test is entirely the listener.
 */
describe('createDropClickSwallow', () => {
  afterEach(() => {
    document.body.replaceChildren()
    vi.restoreAllMocks()
  })

  /** A row shaped like the real ones: a wrapper carrying `blockAttr`, holding a row carrying `rowAttr`. */
  const projectRow = (blockAttr: string, rowAttr: string, id: string) => {
    const block = document.createElement('div')
    const row = document.createElement('button')
    const onActivate = vi.fn()

    block.setAttribute(blockAttr, id)
    row.setAttribute(rowAttr, id)
    row.addEventListener('click', onActivate)
    block.append(row)
    document.body.append(block)

    return { block, onActivate, row }
  }

  const draggedSession = () => {
    const wrap = document.createElement('div')
    const onActivate = vi.fn()

    wrap.setAttribute('data-session-row', 's1')
    wrap.addEventListener('click', onActivate)
    document.body.append(wrap)

    return { onActivate, row: wrap }
  }

  it('eats the click that lands on the dragged row', () => {
    const swallow = createDropClickSwallow()
    const dragged = draggedSession()

    swallow.arm(dragged.row, null)
    fireEvent.click(dragged.row)

    expect(dragged.onActivate).not.toHaveBeenCalled()
  })

  it('eats the click that lands on the nest target row, which is a different row', () => {
    const swallow = createDropClickSwallow()
    const dragged = draggedSession()
    const target = projectRow('data-sessions-project', 'data-project-row', 'p2')

    swallow.arm(dragged.row, target.row)
    fireEvent.click(target.row)

    // The release click would otherwise "enter project 2" the instant the nest landed.
    expect(target.onActivate).not.toHaveBeenCalled()
  })

  it('leaves a click on any other row alone — one drag must not eat the next click', () => {
    const swallow = createDropClickSwallow()
    const dragged = draggedSession()
    const target = projectRow('data-sessions-project', 'data-project-row', 'p2')
    const unrelated = projectRow('data-sessions-project', 'data-project-row', 'p3')

    swallow.arm(dragged.row, target.row)
    fireEvent.click(unrelated.row)

    expect(unrelated.onActivate).toHaveBeenCalledTimes(1)
  })

  it('does not arm at all when a drag ended with neither a row nor a target', () => {
    const swallow = createDropClickSwallow()
    const other = projectRow('data-sessions-project', 'data-project-row', 'p9')

    swallow.arm(null, null)
    fireEvent.click(other.row)

    expect(other.onActivate).toHaveBeenCalledTimes(1)
  })

  it('re-arms for the new target on the next drag, never a stale one', () => {
    const swallow = createDropClickSwallow()
    const first = projectRow('data-sessions-project', 'data-project-row', 'p2')
    const second = projectRow('data-sessions-project', 'data-project-row', 'p3')

    // Two drags in a row, the first one's click never delivered (a release the browser did not turn
    // into a click). The second drag's swallow must cover the second target, not resurrect the first.
    swallow.arm(draggedSession().row, first.row)
    swallow.arm(draggedSession().row, second.row)

    fireEvent.click(second.row)

    expect(second.onActivate).not.toHaveBeenCalled()

    fireEvent.click(first.row)

    expect(first.onActivate).toHaveBeenCalledTimes(1)
  })
})

describe('findSessionRow', () => {
  afterEach(() => {
    document.body.replaceChildren()
  })

  it('finds a row whose id is not selector-safe, instead of throwing on it', () => {
    // The gateway and profile groups sort by `JSON.stringify`ed arrays, so their ids carry quotes.
    const id = JSON.stringify(['gateway', 'local'])
    const row = document.createElement('div')

    row.setAttribute('data-session-row', id)
    document.body.append(row)

    expect(findSessionRow(id)).toBe(row)
  })

  it('returns null when no row carries the id', () => {
    expect(findSessionRow('s1')).toBeNull()
  })
})
import { fireEvent } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { createDropClickSwallow, findSessionRow } from './reorderable-list'

/**
 * The regression the swallow exists for: a nest release leaves the pointer over the region it landed
 * in, and the browser delivers a `click` to whatever is under it — the target's own row, one of its
 * session rows, a nested project. Each of those has a press of its own (entering the project, opening
 * the session), so a nest release used to also do that.
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

  /** A project shaped like the real ones: a wrapper carrying `blockAttr` (the area a drop lands in),
   *  holding the project's own row, plus one of its session rows. */
  const projectRow = (blockAttr: string, id: string) => {
    const block = document.createElement('div')
    const row = document.createElement('button')
    const onActivate = vi.fn()

    block.setAttribute(blockAttr, id)
    row.addEventListener('click', onActivate)

    const session = document.createElement('button')
    const onOpenSession = vi.fn()

    session.setAttribute('data-session-row', `${id}-session`)
    session.addEventListener('click', onOpenSession)
    block.append(row, session)
    document.body.append(block)

    return { block, onActivate, onOpenSession, row, session }
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

  it('eats the click that lands anywhere in the region the nest took', () => {
    const swallow = createDropClickSwallow()
    const dragged = draggedSession()
    const target = projectRow('data-sessions-project', 'p2')

    swallow.arm(dragged.row, target.block)
    fireEvent.click(target.row)

    // The release click would otherwise "enter project 2" the instant the nest landed.
    expect(target.onActivate).not.toHaveBeenCalled()

    // The same release a few pixels lower lands on one of the target's session rows: opening that
    // session is just as wrong, and it is the same drag.
    swallow.arm(draggedSession().row, target.block)
    fireEvent.click(target.session)

    expect(target.onOpenSession).not.toHaveBeenCalled()
  })

  it('leaves a click on any other row alone — one drag must not eat the next click', () => {
    const swallow = createDropClickSwallow()
    const dragged = draggedSession()
    const target = projectRow('data-sessions-project', 'p2')
    const unrelated = projectRow('data-sessions-project', 'p3')

    swallow.arm(dragged.row, target.block)
    fireEvent.click(unrelated.row)

    expect(unrelated.onActivate).toHaveBeenCalledTimes(1)
  })

  it('does not arm at all when a drag ended with neither a row nor a target', () => {
    const swallow = createDropClickSwallow()
    const other = projectRow('data-sessions-project', 'p9')

    swallow.arm(null, null)
    fireEvent.click(other.row)

    expect(other.onActivate).toHaveBeenCalledTimes(1)
  })

  it('re-arms for the new target on the next drag, never a stale one', () => {
    const swallow = createDropClickSwallow()
    const first = projectRow('data-sessions-project', 'p2')
    const second = projectRow('data-sessions-project', 'p3')

    // Two drags in a row, the first one's click never delivered (a release the browser did not turn
    // into a click). The second drag's swallow must cover the second target, not resurrect the first.
    swallow.arm(draggedSession().row, first.block)
    swallow.arm(draggedSession().row, second.block)

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
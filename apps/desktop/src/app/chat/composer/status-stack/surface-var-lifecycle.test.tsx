import { act, cleanup, render } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { STATUS_STACK_VAR } from '@/app/chat/surface-vars'
import { I18nProvider } from '@/i18n'
import { $goalsBySession, type SessionGoal } from '@/store/goals'

import { ComposerStatusStack } from './index'

// The stack measures itself into a surface var — jsdom has no ResizeObserver.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}

vi.stubGlobal('ResizeObserver', ResizeObserverStub)

const SID = 'sess-height-1'

const goal = (): SessionGoal => ({ status: 'active', title: 'ship the feature', updatedAt: Date.now() })

/**
 * Regression: when the stack collapses (its last item finishes), React removes
 * the stack div BEFORE the layout-effect cleanup runs. Resolving the surface
 * from the ref at cleanup time then walks a DETACHED node, misses
 * [data-chat-surface], and clears the document root instead — the stale height
 * stays on the surface and keeps inflating the thread's bottom clearance
 * (`--thread-last-message-clearance`) until the next publish. The effect must
 * capture its surface root while the node is still attached.
 */
describe('ComposerStatusStack surface-var lifecycle', () => {
  beforeEach(() => {
    $goalsBySession.set({})
  })

  afterEach(() => {
    cleanup()
    $goalsBySession.set({})
    document.documentElement.style.removeProperty(STATUS_STACK_VAR)
  })

  function renderOnSurface() {
    const surface = document.createElement('div')
    surface.setAttribute('data-chat-surface', '')
    document.body.append(surface)

    const view = render(
      <MemoryRouter>
        <I18nProvider configClient={null} initialLocale="en">
          <ComposerStatusStack queue={null} sessionId={SID} />
        </I18nProvider>
      </MemoryRouter>,
      { container: surface }
    )

    return { surface, view }
  }

  it('publishes its measured height onto the owning surface while visible', () => {
    $goalsBySession.set({ [SID]: goal() })

    const { surface } = renderOnSurface()

    // jsdom measures 0 — the value is irrelevant, the target element is not.
    expect(surface.style.getPropertyValue(STATUS_STACK_VAR)).toBe('0px')
    expect(document.documentElement.style.getPropertyValue(STATUS_STACK_VAR)).toBe('')
  })

  it('clears the surface var when the stack collapses to nothing', () => {
    $goalsBySession.set({ [SID]: goal() })

    const { surface } = renderOnSurface()
    expect(surface.style.getPropertyValue(STATUS_STACK_VAR)).toBe('0px')

    // Last status item goes away → the component renders null and React
    // detaches the stack div before the cleanup runs.
    act(() => $goalsBySession.set({}))

    expect(surface.style.getPropertyValue(STATUS_STACK_VAR)).toBe('')
  })

  it('clears the surface var on unmount', () => {
    $goalsBySession.set({ [SID]: goal() })

    const { surface, view } = renderOnSurface()
    expect(surface.style.getPropertyValue(STATUS_STACK_VAR)).toBe('0px')

    view.unmount()

    expect(surface.style.getPropertyValue(STATUS_STACK_VAR)).toBe('')
  })
})

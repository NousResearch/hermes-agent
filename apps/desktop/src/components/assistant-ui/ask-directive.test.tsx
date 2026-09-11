// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterAll, afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { markActiveComposer, onComposerSubmitRequest } from '@/app/chat/composer/focus'
import { PRIMARY_SESSION_VIEW, SessionViewProvider } from '@/app/chat/session-view'

const identity = vi.hoisted(() => ({ message: 'message-1' }))
vi.mock('@assistant-ui/react', () => ({
  useAuiState: <T,>(select: (state: { message: { id: string } }) => T) => select({ message: { id: identity.message } })
}))

const submit = vi.fn()
const stop = onComposerSubmitRequest(submit)

import { AskDirective } from '@/components/assistant-ui/ask-directive'

describe('::ask directive', () => {
  beforeEach(() => {
    const surface = document.createElement('div')
    surface.dataset.composerTarget = 'main'
    surface.dataset.composerSurfaceId = 'main-surface'
    document.body.append(surface)
  })

  afterAll(stop)
  afterEach(() => {
    cleanup()
    document.querySelectorAll('[data-composer-target]').forEach(node => node.remove())
    submit.mockClear()
  })

  it('renders option pills and submits the pick as a visible turn', () => {
    render(
      <AskDirective
        attrs={{ options: 'Lead story|Exclusive|Embargoed brief', question: 'Which angle leads?' }}
        streaming={false}
      />
    )

    expect(screen.getByText('Which angle leads?')).toBeTruthy()
    fireEvent.click(screen.getByText('Exclusive'))
    expect(submit).toHaveBeenCalledWith(expect.objectContaining({ text: 'Exclusive', target: 'main' }))

    // Settled: pills disable, no double submit.
    fireEvent.click(screen.getByText('Lead story'))
    expect(submit).toHaveBeenCalledTimes(1)
  })

  it('renders the question but no input row when input is requested — the composer takes typed answers', () => {
    const { container } = render(
      <AskDirective attrs={{ input: 'true', placeholder: 'e.g. 24 months', question: 'Runway?' }} streaming={false} />
    )

    expect(screen.getByText('Runway?')).toBeTruthy()
    expect(container.querySelector('input')).toBeNull()
    expect(container.querySelector('form')).toBeNull()
  })

  it('renders nothing without a question or any affordance', () => {
    const { container } = render(<AskDirective attrs={{ question: 'Orphan?' }} streaming={false} />)

    expect(container.textContent).toBe('')
  })

  it('stays inert while streaming', () => {
    render(<AskDirective attrs={{ options: 'A|B', question: 'Pick' }} streaming={true} />)

    fireEvent.click(screen.getByText('A'))
    expect(submit).not.toHaveBeenCalled()
  })
  it('routes the answer to its own view despite focus in another composer', () => {
    const view = { ...PRIMARY_SESSION_VIEW, kind: 'tile' as const, $storedId: atom<string | null>('view-b') }

    const { container } = render(
      <SessionViewProvider value={view}>
        <div data-composer-surface-id="b-surface" data-composer-target="tile:view-b">
          <AskDirective attrs={{ question: 'Which view?', options: 'This one|Another' }} streaming={false} />
        </div>
      </SessionViewProvider>
    )

    markActiveComposer('main')
    fireEvent.click(within(container).getByText('This one'))
    expect(submit).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({ text: 'This one', target: 'tile:view-b', surfaceId: 'b-surface' })
    )
  })

  it('settles only this session and message, retaining that choice across remounts', () => {
    const attrs = { question: 'What do you want next?', options: 'Continue|Stop' }
    const view = { ...PRIMARY_SESSION_VIEW, $storedId: atom<string | null>('settlement-a') }

    const mount = () =>
      render(
        <SessionViewProvider value={view}>
          <AskDirective attrs={attrs} streaming={false} />
        </SessionViewProvider>
      )

    identity.message = 'checkpoint-1'
    let card = mount()
    fireEvent.click(screen.getByText('Continue'))
    card.unmount()
    card = mount()
    expect(screen.getByRole('button', { name: 'Continue' }).hasAttribute('disabled')).toBe(true)
    card.unmount()
    identity.message = 'checkpoint-2'
    card = mount()
    fireEvent.click(screen.getByText('Continue'))
    card.unmount()
    view.$storedId.set('settlement-b')
    mount()
    fireEvent.click(screen.getByText('Continue'))
    expect(submit).toHaveBeenCalledTimes(3)
  })
})

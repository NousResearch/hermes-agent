// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const submit = vi.fn((_text: string) => true)

vi.mock('@/app/chat/composer/focus', () => ({
  requestComposerSubmit: (text: string) => submit(text)
}))

import { AskDirective } from '@/components/assistant-ui/ask-directive'

describe('::ask directive', () => {
  afterEach(() => {
    cleanup()
    submit.mockClear()
  })

  it('renders option pills and submits the pick as a visible turn', () => {
    render(
      <AskDirective attrs={{ options: 'Lead story|Exclusive|Embargoed brief', question: 'Which angle leads?' }} streaming={false} />
    )

    expect(screen.getByText('Which angle leads?')).toBeTruthy()
    fireEvent.click(screen.getByText('Exclusive'))
    expect(submit).toHaveBeenCalledWith('Exclusive')

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
})

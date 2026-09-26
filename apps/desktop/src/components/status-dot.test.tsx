import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { StatusDot, type StatusTone } from './status-dot'

afterEach(cleanup)

function mark(tone: StatusTone): HTMLElement {
  const { container } = render(<StatusDot tone={tone} />)
  const element = container.firstElementChild

  if (!(element instanceof HTMLElement)) {
    throw new Error('status mark did not render')
  }

  return element
}

describe('StatusDot', () => {
  it('uses a distinct non-color shape for each semantic tone', () => {
    expect(mark('good').className).toContain('rounded-full')
    expect(mark('warn').className).toContain('rotate-45')
    expect(mark('bad').className).toContain('rounded-[1px]')
    expect(mark('muted').className).toContain('border')
  })

  it('exposes the semantic tone for diagnostics without adding spoken noise', () => {
    const element = mark('warn')

    expect(element.dataset.statusTone).toBe('warn')
    expect(element.getAttribute('aria-hidden')).toBe('true')
  })
})

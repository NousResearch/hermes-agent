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
    const good = mark('good').className
    const warn = mark('warn').className
    const bad = mark('bad').className
    const muted = mark('muted').className

    expect(good).toContain('rounded-full')
    expect(good).not.toContain('border')
    expect(warn).toContain('rotate-45')
    expect(bad).toContain('rounded-[1px]')
    expect(bad).not.toContain('rotate-45')
    expect(muted).toContain('rounded-full')
    expect(muted).toContain('border')
  })

  it('exposes the semantic tone for diagnostics without adding spoken noise', () => {
    const element = mark('warn')

    expect(element.dataset.statusTone).toBe('warn')
    expect(element.getAttribute('aria-hidden')).toBe('true')
  })
})

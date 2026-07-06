import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import WireframeRenderer from './wireframe-embed'

afterEach(cleanup)

describe('WireframeRenderer', () => {
  it('renders sanitized wireframe html', () => {
    render(<WireframeRenderer code='<section class="wf-card"><h3>Decision panel</h3><button class="wf-button">Approve</button></section>' />)

    expect(screen.getByText('Wireframe')).toBeTruthy()
    expect(screen.getByText('Decision panel')).toBeTruthy()
    expect(screen.getByText('Approve')).toBeTruthy()
  })

  it('shows a safe empty state for empty input', () => {
    render(<WireframeRenderer code='   ' />)

    expect(screen.getByText('Empty wireframe block')).toBeTruthy()
  })

  it('strips executable attacker html', () => {
    const { container } = render(
      <WireframeRenderer code='<img src="x" onerror="alert(1)"><script>alert(1)</script><a href="javascript:alert(1)">Link</a>' />
    )

    const href = container.querySelector('a')?.getAttribute('href')

    expect(container.querySelector('script')).toBeNull()
    expect(container.querySelector('img')?.getAttribute('onerror')).toBeNull()
    expect(href ?? '').not.toMatch(/^javascript:/i)
  })

  it('strips inline style attributes that can escape the wireframe frame', () => {
    const { container } = render(
      <WireframeRenderer code='<div style="position:fixed;inset:0;z-index:999999;background:url(javascript:alert(1))">Overlay</div>' />
    )

    expect(screen.getByText('Overlay')).toBeTruthy()
    expect(container.querySelector('[style]')).toBeNull()
  })

  it('strips global utility classes while preserving wireframe classes', () => {
    const { container } = render(
      <WireframeRenderer code='<div class="fixed inset-0 z-50 pointer-events-auto wf-card">Overlay<button class="absolute wf-button">Approve</button></div>' />
    )

    const overlay = screen.getByText('Overlay').closest('div')
    const approveButton = screen.getByText('Approve')

    expect(overlay?.getAttribute('class')).toBe('wf-card')
    expect(approveButton.getAttribute('class')).toBe('wf-button')
    expect(container.querySelector('.fixed')).toBeNull()
    expect(container.querySelector('.inset-0')).toBeNull()
    expect(container.querySelector('.z-50')).toBeNull()
    expect(container.querySelector('.pointer-events-auto')).toBeNull()
    expect(container.querySelector('.absolute')).toBeNull()
  })
})

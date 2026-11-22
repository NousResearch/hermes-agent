import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $shellChoice } from '@/store/shell-variant'

import { GatewayModeSection } from './gateway-mode-section'

afterEach(() => {
  $shellChoice.set(null)
  cleanup()
})

const cards = () => screen.getAllByRole('button').filter(button => button.classList.contains('gateway-mode-card'))

describe('GatewayModeSection in the redesign', () => {
  it('rings the saved mode and hands a pick to the connection form', () => {
    $shellChoice.set(true)
    const onSelect = vi.fn()
    render(<GatewayModeSection disabled={false} mode="ssh" onSelect={onSelect} />)

    expect(cards().map(card => [card.dataset.mode, card.getAttribute('aria-pressed')])).toEqual([
      ['cloud', 'false'],
      ['remote', 'false'],
      ['ssh', 'true'],
      ['local', 'false']
    ])

    fireEvent.click(cards()[0])
    expect(onSelect).toHaveBeenCalledWith('cloud')
  })

  it('offers no pick while the launch environment pins the connection', () => {
    $shellChoice.set(true)
    const onSelect = vi.fn()
    render(<GatewayModeSection disabled mode="remote" onSelect={onSelect} />)

    for (const card of cards()) {
      expect((card as HTMLButtonElement).disabled).toBe(true)
    }

    fireEvent.click(cards()[3])
    expect(onSelect).not.toHaveBeenCalled()
  })
})

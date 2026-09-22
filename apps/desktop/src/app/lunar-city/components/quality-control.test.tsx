// @vitest-environment jsdom
import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { QualityControl } from './quality-control'

describe('QualityControl', () => {
  it('starts at efficient, dispatches the selected quality tier, and announces the truthful choice', () => {
    const onTierChange = vi.fn()
    const { rerender } = render(<QualityControl onTierChange={onTierChange} />)

    const select = screen.getByRole('combobox', { name: /3D quality/i })
    expect((select as HTMLSelectElement).value).toBe('efficient')
    expect(screen.getByRole('status').textContent).toMatch(/3D quality: Efficient/i)

    fireEvent.change(select, { target: { value: 'balanced' } })
    expect(onTierChange).toHaveBeenCalledWith('balanced')

    rerender(<QualityControl onTierChange={onTierChange} tier="balanced" />)
    expect((select as HTMLSelectElement).value).toBe('balanced')
    expect(screen.getByRole('status').textContent).toMatch(/3D quality: Balanced/i)
  })

  it('keeps semantic controls available while reporting degraded renderer status', () => {
    render(<QualityControl rendererStatus="unavailable" />)

    expect((screen.getByRole('combobox', { name: /3D quality/i }) as HTMLSelectElement).disabled).toBe(false)
    expect(screen.getByRole('status').textContent).toMatch(/3D renderer unavailable/i)
    expect((screen.getByRole('combobox') as HTMLSelectElement).options[0]?.textContent).toMatch(/Efficient/i)
  })
})

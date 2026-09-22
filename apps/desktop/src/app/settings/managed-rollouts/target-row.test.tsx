import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { targetIdentity, TargetRow } from './target-row'

describe('managed rollout target identity', () => {
  it('keeps aliases display-only and keys selection by machine plus install identity', () => {
    expect(targetIdentity({ installId: 'install-a', machineId: 'machine-a', label: 'Alias', alias: 'old-name', supported: true, sharedMachine: false })).toBe(
      JSON.stringify(['machine-a', 'install-a'])
    )
    expect(targetIdentity({ installId: 'install-a', machineId: 'machine-a', label: 'Alias', alias: 'new-name', supported: true, sharedMachine: false })).toBe(
      JSON.stringify(['machine-a', 'install-a'])
    )
  })

  it('keeps the target action keyboard reachable without submitting its owning form', () => {
    const onToggle = vi.fn()
    const onSubmit = vi.fn(event => event.preventDefault())

    render(
      <form onSubmit={onSubmit}>
        <TargetRow
          onToggle={onToggle}
          selected={false}
          target={{
            installId: 'install-a',
            label: 'A very long target label that must remain readable',
            machineId: 'machine-a',
            supported: true,
            sharedMachine: false
          }}
        />
      </form>
    )

    const select = screen.getByRole('button', { name: 'Select' })
    expect(select.getAttribute('type')).toBe('button')
    expect(select.className).toContain('motion-reduce:transition-none')

    select.focus()
    fireEvent.click(select)

    expect(document.activeElement).toBe(select)
    expect(onToggle).toHaveBeenCalledWith(JSON.stringify(['machine-a', 'install-a']))
    expect(onSubmit).not.toHaveBeenCalled()
  })
})

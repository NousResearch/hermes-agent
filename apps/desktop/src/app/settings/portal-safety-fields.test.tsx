import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $shellChoice } from '@/store/shell-variant'

import { PortalSafetyFields, PortalSafetySettings } from './portal-safety-fields'

// The redesign is the only shell that renders these fields.
beforeEach(() => $shellChoice.set(true))
afterEach(() => {
  cleanup()
  $shellChoice.set(null)
})

it('changes only the chosen approval field and keeps advanced controls available to deep links', () => {
  const onChange = vi.fn()
  const manage = vi.fn()
  const config = { approvals: { mode: 'smart', timeout: 300 }, security: { redact_secrets: true }, other: 'kept' }

  const props = {
    config,
    onChange,
    onManagePasswords: manage,
    fields: [
      ['approvals.mode', { type: 'select', options: ['smart', 'manual', 'off'] }],
      ['approvals.timeout', { type: 'number' }],
      ['security.redact_secrets', { type: 'boolean' }]
    ] as Parameters<typeof PortalSafetyFields>[0]['fields']
  }

  const view = render(<PortalSafetyFields {...props} advancedOpen={false} />)
  fireEvent.click(screen.getByRole('radio', { name: /^Manual/ }))
  expect(onChange).toHaveBeenCalledWith({ ...config, approvals: { mode: 'manual', timeout: 300 } })
  const details = view.container.querySelector('details')!
  expect(details.open).toBe(false)
  view.rerender(<PortalSafetyFields {...props} advancedOpen />)
  expect(details.open).toBe(true)
  view.rerender(<PortalSafetyFields {...props} advancedOpen={false} />)
  expect(details.open).toBe(true)
  fireEvent.click(screen.getByRole('button', { name: /Manage: Passwords/i }))
  expect(manage).toHaveBeenCalledOnce()
})

it('keeps advanced deep links visible and preserves profile scope when opening passwords', () => {
  function Location() {
    return <output aria-label="Location">{useLocation().search}</output>
  }

  const view = render(
    <MemoryRouter initialEntries={['/?tab=safety&page=approvals&field=approvals.timeout&profile=research']}>
      <PortalSafetySettings
        config={{ approvals: { timeout: 300 } }}
        fields={[['approvals.timeout', { type: 'number' }]]}
        onChange={vi.fn()}
        subpage="approvals"
      />
      <Location />
    </MemoryRouter>
  )

  expect(view.container.querySelector('details')!.open).toBe(true)
  fireEvent.click(screen.getByRole('button', { name: /Manage: Passwords/i }))
  const query = new URLSearchParams(screen.getByLabelText('Location').textContent!)
  expect(Object.fromEntries(query)).toEqual({ tab: 'vault', profile: 'research' })
})

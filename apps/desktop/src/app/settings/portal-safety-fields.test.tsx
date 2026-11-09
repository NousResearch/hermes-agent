import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { PortalSafetyFields } from './portal-safety-fields'

afterEach(cleanup)

it('changes only the chosen approval field and keeps advanced controls available to deep links', () => {
  const onChange = vi.fn()
  const manage = vi.fn()
  const config = { approvals: { mode: 'smart', timeout: 300 }, security: { redact_secrets: true }, other: 'kept' }

  const props = { config, onChange, onManagePasswords: manage, fields: [
    ['approvals.mode', { type: 'select', options: ['smart', 'manual', 'off'] }],
    ['approvals.timeout', { type: 'number' }],
    ['security.redact_secrets', { type: 'boolean' }]
  ] as Parameters<typeof PortalSafetyFields>[0]['fields'] }

  const view = render(<PortalSafetyFields {...props} advancedOpen={false} />)
  fireEvent.click(screen.getByRole('radio', { name: /^Manual/ }))
  expect(onChange).toHaveBeenCalledWith({ ...config, approvals: { mode: 'manual', timeout: 300 } })
  const details = view.container.querySelector('details')!
  expect(details.open).toBe(false)
  view.rerender(<PortalSafetyFields {...props} advancedOpen />)
  expect(details.open).toBe(true)
  view.rerender(<PortalSafetyFields {...props} advancedOpen={false} />)
  expect(details.open).toBe(true)
  fireEvent.click(screen.getByRole('button', { name: /Passwords/i }))
  expect(manage).toHaveBeenCalledOnce()
})

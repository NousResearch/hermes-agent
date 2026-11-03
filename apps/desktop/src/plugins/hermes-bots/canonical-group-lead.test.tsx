import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ComponentProps, ReactNode } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

const request = vi.hoisted(() => vi.fn())

// A native <select> stands in for the Radix one: same value/onValueChange contract.
vi.mock('@hermes/plugin-sdk', () => ({
  Button: (props: ComponentProps<'button'>) => <button {...props} />,
  Dialog: ({ children, open }: { children: ReactNode; open: boolean }) => (open ? <div>{children}</div> : null),
  DialogContent: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  DialogFooter: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  DialogHeader: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  DialogTitle: ({ children }: { children: ReactNode }) => <h2>{children}</h2>,
  host: {},
  Input: (props: ComponentProps<'input'>) => <input {...props} />,
  randomUuid: () => 'uuid-1',
  Select: ({ children, onValueChange, value, disabled }: { children: ReactNode; onValueChange: (v: string) => void; value: string; disabled?: boolean }) =>
    <select aria-label="routing" disabled={disabled} onChange={event => onValueChange(event.target.value)} value={value}>{children}</select>,
  SelectContent: ({ children }: { children: ReactNode }) => <>{children}</>,
  SelectItem: ({ children, value }: { children: ReactNode; value: string }) => <option value={value}>{children}</option>,
  SelectTrigger: () => null,
  SelectValue: () => null,
  Textarea: (props: ComponentProps<'textarea'>) => <textarea {...props} />
}))
vi.mock('./canonical-groups', async importOriginal => ({ ...(await importOriginal<object>()), canonicalGroupRequest: request }))
vi.mock('./canonical-group-labels', async () => {
  const { CANONICAL_GROUP_LOCALES } = await import('./canonical-group-locales')

  return { useCanonicalGroupLabels: () => ({ ...CANONICAL_GROUP_LOCALES.en, cancel: 'Cancel', save: 'Save' }) }
})

import { CanonicalGroupAssignmentRow, markRoomLead } from './canonical-group-lead'
import { CanonicalGroupRoomSettings } from './canonical-group-room-settings'

const members = [
  { member_id: 'm-ada', handle: 'ada', display_name: 'Ada', profile: 'ada' },
  { member_id: 'm-linus', handle: 'linus', display_name: 'Linus', profile: 'linus' },
  { member_id: 'm-grace', handle: 'grace', profile: 'grace' }
]

const binding = { roomId: 'room-1', profile: 'ada', connectionId: 'local' }
const assignment = (payload: Record<string, unknown>) => ({ seq: 4, event_id: 'dassign:x', room_id: 'room-1', kind: 'room.assignment', payload })

afterEach(() => {
  cleanup()
  request.mockReset()
})

it('names who the lead gave the message to, by their room names', () => {
  render(<CanonicalGroupAssignmentRow event={assignment({ lead_member_id: 'm-ada', member_ids: ['m-linus', 'm-grace'], fallback: false })} members={members} />)
  expect(screen.getByTestId('room-assignment').textContent).toBe('Ada gave this to Linus, grace')
})

it('says the lead kept a message it could not assign', () => {
  render(<CanonicalGroupAssignmentRow event={assignment({ lead_member_id: 'm-ada', member_ids: ['m-ada'], fallback: true })} members={members} />)
  expect(screen.getByTestId('room-assignment').textContent).toBe('Ada is taking this')
})

it('flags only the lead of a lead-routed room for the member chips', () => {
  expect(markRoomLead(members, { purpose: '', notes: '', routing: 'lead', lead_member_id: 'm-linus' }).map(m => m.lead)).toEqual([false, true, false])
  expect(markRoomLead(members, { purpose: '', notes: '', routing: 'everyone', lead_member_id: '' }).some(m => m.lead)).toBe(false)
})

it('saves the lead picked in Room settings, and Everyone clears it', async () => {
  request.mockResolvedValue({})
  render(<CanonicalGroupRoomSettings binding={binding} context={{ purpose: 'Kernel', notes: '', routing: 'everyone', lead_member_id: '' }} members={members} />)
  fireEvent.click(screen.getByRole('button', { name: 'Room settings' }))
  fireEvent.change(screen.getByLabelText('routing'), { target: { value: 'm-ada' } })
  fireEvent.click(screen.getByText('Save'))

  await waitFor(() => expect(request).toHaveBeenCalledWith(binding, 'groups.context', {
    room_id: 'room-1', event_id: 'uuid-1', purpose: 'Kernel', notes: '', routing: 'lead', lead_member_id: 'm-ada' }))
})

it('offers no routing choice to a runtime that does not report one', () => {
  render(<CanonicalGroupRoomSettings binding={binding} context={{ purpose: '', notes: '' }} members={members} />)
  fireEvent.click(screen.getByRole('button', { name: 'Room settings' }))
  expect(screen.queryByLabelText('routing')).toBeNull()
})

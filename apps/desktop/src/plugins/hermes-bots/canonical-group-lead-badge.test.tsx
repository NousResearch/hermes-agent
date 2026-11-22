import { useStore } from '@nanostores/react'
import { cleanup, render, screen, within } from '@testing-library/react'
import { atom } from 'nanostores'
import type { ComponentProps } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

const request = vi.hoisted(() => vi.fn())
vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock, createGroupGateway } = await import('./group-test-utils')
  const gateway = createGroupGateway()
  const { en } = await import('@/i18n/en')
  const { CANONICAL_GROUP_LOCALES } = await import('./canonical-group-locales')
  const { ConfirmDialog } = await import('@/components/ui/confirm-dialog')

  return {
    ...(await pluginSdkMock(gateway.host)),
    atom,
    useValue: useStore,
    ConfirmDialog,
    useI18n: () => ({ t: en }),
    usePluginI18n: () => (key: string) =>
      CANONICAL_GROUP_LOCALES.en[key.replace('canonical.', '') as keyof typeof CANONICAL_GROUP_LOCALES.en] ?? key,
    Button: (p: ComponentProps<'button'>) => <button {...p} />,
    Input: (p: ComponentProps<'input'>) => <input {...p} />,
    RowButton: (p: ComponentProps<'button'>) => <button type="button" {...p} />,
    host: { ...gateway.host, requestProfile: request }
  }
})

import { CanonicalGroupWorkspace } from './canonical-group-workspace'

const originalDesktop = window.hermesDesktop
beforeEach(() => {
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, writable: true, value: undefined })
})
afterEach(() => {
  cleanup()
  request.mockReset()
  localStorage.clear()
  window.hermesDesktop = originalDesktop
})

it('badges the room lead on its header chip when the room routes through a lead', async () => {
  const binding = { connectionId: 'local', profile: 'team', roomId: 'lead-room' }
  const members = [
    { member_id: 'ada', profile: 'ada', handle: 'ada', display_name: 'Ada' },
    { member_id: 'linus', profile: 'linus', handle: 'linus', display_name: 'Linus' }
  ]

  request.mockImplementation(async (_route, method) =>
    method === 'groups.state'
      ? {
          room: { name: 'Lead room', members },
          context: { purpose: '', notes: '', routing: 'lead', lead_member_id: 'ada' },
          driver_status: {}
        }
      : method === 'groups.log'
        ? { events: [] }
        : {}
  )
  render(<CanonicalGroupWorkspace binding={binding} />)

  const roster = await screen.findByRole('list', { name: 'Members' })
  const chips = within(roster).getAllByRole('listitem')

  expect(chips).toHaveLength(2)
  expect(chips[0].textContent).toContain('Lead')
  expect(chips[1].textContent).not.toContain('Lead')
})

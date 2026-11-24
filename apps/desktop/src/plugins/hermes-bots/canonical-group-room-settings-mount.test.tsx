import { useStore } from '@nanostores/react'
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
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

  return { ...await pluginSdkMock(gateway.host), atom, useValue: useStore, ConfirmDialog,
    useI18n: () => ({ t: en }),
    usePluginI18n: () => (key: string) => CANONICAL_GROUP_LOCALES.en[key.replace('canonical.', '') as keyof typeof CANONICAL_GROUP_LOCALES.en] ?? key,
    Button: (p: ComponentProps<'button'>) => <button {...p} />,
    Input: (p: ComponentProps<'input'>) => <input {...p} />,
    RowButton: (p: ComponentProps<'button'>) => <button type="button" {...p} />,
    host: { ...gateway.host, requestProfile: request } }
})

import { CanonicalGroupWorkspace } from './canonical-group-workspace'

const originalDesktop = window.hermesDesktop
beforeEach(() => { Object.defineProperty(window, 'hermesDesktop', { configurable: true, writable: true, value: undefined }) })
afterEach(() => { cleanup(); request.mockReset(); localStorage.clear(); window.hermesDesktop = originalDesktop })

it('shows the room purpose and its Room settings control in the Group details panel', async () => {
  const binding = { connectionId: 'local', profile: 'team', roomId: 'with-purpose' }
  const members = [{ member_id: 'ops', profile: 'ops', handle: 'ops', display_name: 'Ops' }]

  request.mockImplementation(async (_route, method) => method === 'groups.state'
    ? { room: { name: 'Release room', members }, context: { purpose: 'Ship the release', notes: '' }, driver_status: {} }
    : method === 'groups.log' ? { events: [] } : {})
  render(<CanonicalGroupWorkspace binding={binding} />)

  fireEvent.click(await screen.findByRole('button', { name: 'Group details' }))
  const details = await screen.findByRole('complementary', { name: 'Group details' })

  expect(within(details).getByText('Ship the release')).toBeTruthy()
  expect((within(details).getByRole('button', { name: 'Room settings' }) as HTMLButtonElement).disabled).toBe(false)
})

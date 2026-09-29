import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { setApiRequestConnection } from '@/api/client'
import { deleteProfile } from '@/hermes'
import { retireLocalProfileGateways } from '@/store/gateway'
import { dropTilesForProfile } from '@/store/session-states'

import { DeleteProfileDialog } from './delete-profile-dialog'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof import('@/hermes')>()),
  deleteProfile: vi.fn(async () => ({ ok: true }))
}))
vi.mock('@/store/gateway', () => ({ retireLocalProfileGateways: vi.fn() }))
vi.mock('@/store/session-states', () => ({ dropTilesForProfile: vi.fn() }))

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  setApiRequestConnection(null)
})

it.each([
  { connectionId: 'remote-a', profile: 'worker' },
  { connectionId: 'remote-a', profile: null },
  'worker'
])('removes local UI state only from the remote owner that accepted deletion (%j)', async scope => {
  setApiRequestConnection('remote-a')

  render(
    <DeleteProfileDialog
      onClose={vi.fn()}
      open
      profile={{ name: 'worker', path: '/fixture/worker' }}
      scope={scope}
    />
  )
  fireEvent.click(screen.getByRole('button', { name: /^delete$/i }))

  await waitFor(() =>
    expect(dropTilesForProfile).toHaveBeenCalledWith('worker', { connectionId: 'remote-a', profile: 'worker' })
  )
  expect(deleteProfile).toHaveBeenCalledWith('worker', scope)
  expect(retireLocalProfileGateways).not.toHaveBeenCalled()
})

import { QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { PluginRecord } from '@/contrib/plugins-store'
import { queryClient } from '@/lib/query-client'

import { PluginFolderAccess } from './plugin-folder-access'

vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))

const grants = [{ grantedAt: 'x', id: 'f1', name: 'Vault', path: '/home/me/Vault' }]

const door = {
  grants: vi.fn(async () => [...grants]),
  op: vi.fn(),
  request: vi.fn(),
  revoke: vi.fn(async () => {
    grants.length = 0

    return { ok: true, removed: 1 }
  })
}

afterEach(() => {
  cleanup()
  queryClient.clear()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('PluginFolderAccess', () => {
  it('shows each granted folder and revokes it for that plugin', async () => {
    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = { pluginFiles: door }
    render(
      <QueryClientProvider client={queryClient}>
        <PluginFolderAccess name="Plexus" plugin={{ id: 'plexus' } as PluginRecord} />
      </QueryClientProvider>
    )

    fireEvent.click(await screen.findByRole('button', { name: 'Revoke access to /home/me/Vault' }))

    expect(door.revoke).toHaveBeenCalledWith('plexus', 'f1')
    await waitFor(() => expect(screen.queryByText('/home/me/Vault')).toBeNull())
  })

  it('a failed read says so and can be retried, instead of looking like no access', async () => {
    grants.splice(0, grants.length, { grantedAt: 'x', id: 'f1', name: 'Vault', path: '/home/me/Vault' })
    door.grants.mockRejectedValueOnce(new Error('EACCES: grants.json'))
    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = { pluginFiles: door }
    render(
      <QueryClientProvider client={queryClient}>
        <PluginFolderAccess name="Plexus" plugin={{ id: 'plexus' } as PluginRecord} />
      </QueryClientProvider>
    )

    expect(await screen.findByText('Could not load the folders shared with Plexus.')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))

    expect(await screen.findByText('/home/me/Vault')).toBeTruthy()
  })
})

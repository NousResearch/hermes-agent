/**
 * The Bot Mode delete confirmation must describe what the action will actually do: a row whose
 * delete reaches the default profile (the default bot, or an alias bot targeting it) is confirmed
 * as a RESET, never as "permanently delete the bot <name>" (PR #141671 review).
 */

import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { RosterRow } from './types'

const { deleteBot, notify } = vi.hoisted(() => ({ deleteBot: vi.fn(), notify: vi.fn() }))

vi.mock('@hermes/plugin-sdk', async () => {
  const { atom } = await import('nanostores')

  return {
    atom,
    ConfirmDialog: ({
      confirmLabel,
      description,
      onConfirm,
      open,
      title
    }: {
      confirmLabel: string
      description: ReactNode
      onConfirm: () => Promise<void>
      open: boolean
      title: string
    }) =>
      open ? (
        <div>
          <h2>{title}</h2>
          <div>{description}</div>
          <button onClick={() => void onConfirm()}>{confirmLabel}</button>
        </div>
      ) : null,
    host: {
      notify,
      request: vi.fn(),
      requestProfile: vi.fn(),
      state: { connectionId: { get: vi.fn(() => 'local') }, profile: { get: vi.fn(() => 'default') } }
    },
    queryClient: undefined,
    useQuery: vi.fn(),
    useValue: vi.fn()
  }
})

vi.mock('./shared', () => ({ getPluginCtx: () => null, ID: 'hermes-bots' }))
vi.mock('./profile-ops', () => ({ deleteBot }))
vi.mock('./create-dialog', () => ({
  CreateAgentDialog: () => null,
  CreateGroupChatDialog: () => null,
  GroupDialog: () => null
}))
vi.mock('./edit-profile-dialog', () => ({ EditProfileDialog: () => null }))
vi.mock('./group-chat-view', () => ({ disbandGroupChat: vi.fn(), openGroupChat: vi.fn() }))
vi.mock('./user-sections-ui', () => ({ SectionNameDialog: () => null }))

const b = {
  bot: {
    deleteTitle: 'Delete bot?',
    resetAction: 'Reset',
    resetDescription: 'Resetting erases the default bot.',
    resetDone: 'Reset',
    resetTitle: 'Reset the default bot?',
    resetToast: (name: string) => `Reset ${name}`,
    resetting: 'Resetting…'
  }
}

const t = { common: { delete: 'Delete' } }

async function renderDialog(deleting: RosterRow & { path?: string }) {
  const { DeleteBotDialog } = await import('./roster-pane-dialogs')

  render(
    <DeleteBotDialog
      b={b as never}
      deleting={deleting}
      onClose={vi.fn()}
      refetch={vi.fn(async () => undefined) as never}
      t={t as never}
    />
  )
}

beforeEach(() => {
  vi.clearAllMocks()
})

describe('the delete confirmation matches the action', () => {
  it('confirms an alias bot that targets the default profile as a reset', async () => {
    const alias = {
      name: 'worker',
      path: '/home/u/.hermes/profiles/worker',
      route: { connectionId: 'source-a', mode: 'remote', profile: 'worker', targetProfile: 'default' },
      sourceScoped: true
    } as unknown as RosterRow

    await renderDialog(alias)

    expect(screen.getByRole('heading').textContent).toBe('Reset the default bot?')
    expect(screen.queryByText(/permanently delete the bot/i)).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Reset' }))

    await waitFor(() => expect(deleteBot).toHaveBeenCalledWith(alias))
    await waitFor(() => expect(notify).toHaveBeenCalledWith({ kind: 'success', message: 'Reset worker' }))
  })

  it('confirms the default bot itself as a reset', async () => {
    await renderDialog({ name: 'default' } as RosterRow)

    expect(screen.getByRole('heading').textContent).toBe('Reset the default bot?')
  })

  it('still confirms an ordinary bot as a deletion', async () => {
    await renderDialog({ name: 'ops', path: '/home/u/.hermes/profiles/ops' } as RosterRow & { path: string })

    expect(screen.getByRole('heading').textContent).toBe('Delete bot?')
    expect(screen.getByText(/permanently delete the bot/i)).toBeTruthy()
  })
})

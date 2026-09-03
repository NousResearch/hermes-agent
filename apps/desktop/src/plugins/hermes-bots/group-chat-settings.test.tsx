import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { translateBots } from './i18n-test-helper'

const { compressGroupChatSessions, host } = vi.hoisted(() => ({
  compressGroupChatSessions: vi.fn(),
  host: { notify: vi.fn() }
}))

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')
  const base = await pluginSdkMock(host)
  const Div = ({ children }: React.PropsWithChildren) => <div>{children}</div>

  return {
    ...base,
    Button: ({ children, size: _size, variant: _variant, ...props }: React.ComponentProps<'button'> & { size?: string; variant?: string }) => (
      <button type="button" {...props}>
        {children}
      </button>
    ),
    Codicon: () => null,
    Dialog: ({ children, open }: React.PropsWithChildren<{ open?: boolean }>) => (open ? <div>{children}</div> : null),
    DialogContent: Div,
    DialogDescription: Div,
    DialogFooter: Div,
    DialogHeader: Div,
    DialogTitle: Div,
    Input: (props: React.ComponentProps<'input'>) => <input {...props} />,
    useI18n: () => ({ t: { common: { cancel: 'Cancel', save: 'Save' } } }),
    usePluginI18n: () => translateBots
  }
})

vi.mock('./group-chat-parts', () => ({
  GroupClarifyCard: () => null,
  GroupImageControls: () => null,
  GroupMentionInput: () => null
}))

vi.mock('./group-turns', () => ({
  clearGroupClarify: vi.fn(),
  compressGroupChatSessions
}))

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('group settings maintenance', () => {
  it('makes hidden member-session compression reachable from the room', async () => {
    compressGroupChatSessions.mockResolvedValue([
      { member: 'builder', status: 'compressed' },
      { member: 'critic', status: 'pending' }
    ])
    const { GroupChatSettingsDialog } = await import('./group-chat-view')
    const members = [{ name: 'builder' }, { name: 'critic' }]

    render(<GroupChatSettingsDialog group="Core" members={members} onClose={() => undefined} open />)
    fireEvent.click(screen.getByRole('button', { name: 'Compress member histories' }))

    await waitFor(() => expect(compressGroupChatSessions).toHaveBeenCalledWith('Core', members))
    expect(host.notify).toHaveBeenCalledWith({
      kind: 'success',
      message: 'Compressed 1 member history; 1 still running.'
    })
  })

  it.each([
    ['aborted', 'No compression provider is configured.'],
    ['lock-held', 'Another compression may still be running; try again shortly.']
  ])('reports %s compression as an actionable failure', async (_case, error) => {
    compressGroupChatSessions.mockResolvedValue([{ error, member: 'builder', status: 'failed' }])
    const { GroupChatSettingsDialog } = await import('./group-chat-view')

    render(<GroupChatSettingsDialog group="Core" members={[{ name: 'builder' }]} onClose={() => undefined} open />)
    fireEvent.click(screen.getByRole('button', { name: 'Compress member histories' }))

    await waitFor(() =>
      expect(host.notify).toHaveBeenCalledWith({
        kind: 'error',
        message: `Could not compress 1 of 1 member histories. builder: ${error}`
      })
    )
    expect(host.notify).not.toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'success', message: 'No existing member histories need compression.' })
    )
  })
})
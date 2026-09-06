import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { ComponentProps, ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { translateBots } from './i18n-test-helper'
import type { GroupMember } from './types'

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')
  const base = await pluginSdkMock(host)

  return {
    ...base,
    Button: (props: ComponentProps<'button'>) => <button type="button" {...props} />,
    cn: (...values: unknown[]) => values.filter(Boolean).join(' '),
    Codicon: ({ name }: { name: string }) => <span aria-hidden data-icon={name} />,
    ConfirmDialog: ({
      open,
      title,
      description,
      confirmLabel
    }: {
      open: boolean
      title: ReactNode
      description: ReactNode
      confirmLabel: string
    }) =>
      open ? (
        <div role="dialog">
          <h2>{title}</h2>
          <div>{description}</div>
          <button>{confirmLabel}</button>
        </div>
      ) : null,
    CopyButton: () => null,
    Dialog: () => null,
    DialogContent: ({ children }: { children?: ReactNode }) => <>{children}</>,
    DialogDescription: ({ children }: { children?: ReactNode }) => <>{children}</>,
    DialogFooter: ({ children }: { children?: ReactNode }) => <>{children}</>,
    DialogHeader: ({ children }: { children?: ReactNode }) => <>{children}</>,
    DialogTitle: ({ children }: { children?: ReactNode }) => <>{children}</>,
    Input: (props: ComponentProps<'input'>) => <input {...props} />,
    relativeTime: () => 'now',
    RowButton: (props: ComponentProps<'button'>) => <button type="button" {...props} />,
    Tip: ({ children }: { children?: ReactNode }) => <>{children}</>,
    useI18n: () => ({ t: { common: { cancel: 'Cancel', save: 'Save' } } }),
    usePluginI18n: () => translateBots
  }
})

vi.mock('./group-chat-parts', () => ({
  GroupClarifyCard: () => null,
  GroupImageControls: () => null,
  GroupMentionInput: (props: { 'aria-label'?: string; value?: string }) => (
    <textarea aria-label={props['aria-label']} readOnly value={props.value} />
  )
}))

const MEMBERS: GroupMember[] = [
  { connectionId: 'gateway-a', name: 'writer', sourceScoped: true, targetProfile: 'writer' },
  { connectionId: 'gateway-a', name: 'reviewer', sourceScoped: true, targetProfile: 'reviewer' }
]

beforeEach(() => {
  vi.resetModules()
  Object.assign(host, { notify: vi.fn() })
  Object.defineProperty(Element.prototype, 'scrollIntoView', { configurable: true, value: vi.fn() })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('Group Chat removal copy', () => {
  it.each(['classic', 'ready', 'unavailable', 'deleted'] as const)(
    'describes the removal scope for %s',
    async state => {
      const [{ GroupChatWorkspace }, chat] = await Promise.all([import('./group-chat-view'), import('./group-chat')])
      chat.$groupChats.set({
        Core: {
          continuityMode: state === 'classic' ? 'desktop' : 'gateway',
          hosted: state === 'classic' ? null : 'install:home',
          hostedConnectionId: 'gateway-a',
          hostedStatus: state === 'classic' ? null : { state, label: state },
          log: [],
          members: MEMBERS,
          roomId: 'room-1',
          watermarks: {}
        }
      })
      const view = render(<GroupChatWorkspace group="Core" members={MEMBERS} />)
      const remove = view.container.querySelector('[data-icon="trash"]')?.closest('button')
      expect(remove).not.toBeNull()
      fireEvent.click(remove!)
      const dialog = screen.getByRole('dialog')

      if (state === 'deleted') {
        expect(dialog.textContent).toContain(translateBots('group.hostedDeleteLocally'))
        expect(dialog.textContent).not.toContain('shared room log')
        expect(dialog.textContent).not.toContain('2 bots')
        expect(screen.getByRole('heading').textContent).toBe(translateBots('group.hostedDeleteLocalTitle'))
        expect(screen.getByRole('button', { name: 'Delete' })).toBeTruthy()
        expect(remove?.getAttribute('aria-label')).toBe(translateBots('group.hostedDeleteLocally'))
      } else {
        expect(dialog.textContent).toContain(translateBots('group.disbandDescSuffix', MEMBERS.length))
        expect(dialog.textContent).not.toContain(translateBots('group.hostedDeleteLocally'))
        expect(screen.getByRole('heading').textContent).toBe(translateBots('group.disbandTitle'))
        expect(screen.getByRole('button', { name: 'Disband' })).toBeTruthy()
      }
    }
  )
})

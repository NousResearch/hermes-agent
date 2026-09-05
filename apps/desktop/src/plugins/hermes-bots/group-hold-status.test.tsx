import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { translateBots } from './i18n-test-helper'
import type { GroupMember } from './types'

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')
  const base = await pluginSdkMock(host)

  return {
    ...base,
    Button: (props: React.ComponentProps<'button'>) => <button type="button" {...props} />,
    cn: (...values: unknown[]) => values.filter(Boolean).join(' '),
    Codicon: ({ name }: { name: string }) => <span aria-hidden data-icon={name} />,
    ConfirmDialog: () => null,
    CopyButton: () => null,
    Dialog: () => null,
    DialogContent: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
    DialogDescription: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
    DialogFooter: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
    DialogHeader: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
    DialogTitle: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
    Input: (props: React.ComponentProps<'input'>) => <input {...props} />,
    relativeTime: () => 'now',
    RowButton: (props: React.ComponentProps<'button'>) => <button type="button" {...props} />,
    Tip: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
    useI18n: () => ({ t: { common: { cancel: 'Cancel', save: 'Save' } } }),
    usePluginI18n: () => translateBots
  }
})

vi.mock('./group-chat-parts', () => ({
  GroupClarifyCard: () => null,
  GroupImageControls: () => null,
  GroupMentionInput: (props: { 'aria-label'?: string }) => <textarea aria-label={props['aria-label']} />
}))

const MEMBERS: GroupMember[] = [
  { name: 'research', title: 'Research' },
  { name: 'builder', title: 'Builder' }
]

beforeEach(() => {
  vi.resetModules()
  Object.defineProperty(Element.prototype, 'scrollIntoView', {
    configurable: true,
    value: vi.fn()
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('durable group holds', () => {
  it('shows an accessible all-members status without relying on the activity feed', async () => {
    const { GroupHoldStatus } = await import('./group-hold-status')

    render(
      <GroupHoldStatus
        holds={{ builder: { at: 2 }, research: { at: 1 } }}
        memberLabel={member => member.title || member.name}
        members={MEMBERS}
      />
    )

    const status = screen.getByRole('status')

    expect(status.textContent).toContain('All 2 bots are paused')
    expect(status.textContent).toContain('Mention a paused bot or send @all resume to release them.')
    expect(status.querySelector('[data-icon="debug-pause"]')).not.toBeNull()
  })

  it('keeps unmatched holds visible instead of reporting all current members held', async () => {
    const { GroupHoldStatus } = await import('./group-hold-status')

    render(
      <GroupHoldStatus
        holds={{ builder: { at: 1 }, 'mac-mini::research': { at: 2 } }}
        memberLabel={member => member.title || member.name}
        members={[{ name: 'builder', title: 'Builder' }]}
      />
    )

    const status = screen.getByRole('status')

    expect(status.textContent).toContain('Paused:')
    expect(status.textContent).toContain('Builder')
    expect(status.textContent).toContain('research')
    expect(status.textContent).not.toContain('All 1 bots are paused')
  })

  it('names a partial hold and disappears after the production resume directive clears it', async () => {
    const [{ GroupHoldStatus }, { applyGroupHoldDirective }] = await Promise.all([
      import('./group-hold-status'),
      import('./group-rounds')
    ])

    const held = { builder: { at: 2 } }

    const view = render(
      <GroupHoldStatus holds={held} memberLabel={member => member.title || member.name} members={MEMBERS} />
    )

    expect(screen.getByRole('status').textContent).toContain('Paused: Builder')

    const released = applyGroupHoldDirective(held, { everyone: true, mentioned: [] }, '@all resume', {})
    view.rerender(
      <GroupHoldStatus holds={released} memberLabel={member => member.title || member.name} members={MEMBERS} />
    )

    expect(screen.queryByRole('status')).toBeNull()
  })

  it('keeps a persisted source-scoped hold visible while its roster row is unavailable', async () => {
    const { GroupHoldStatus } = await import('./group-hold-status')

    render(
      <GroupHoldStatus
        holds={{ 'mac-mini::builder': { at: 2 } }}
        memberLabel={member => member.title || member.name}
        members={[]}
      />
    )

    expect(screen.getByRole('status').textContent).toContain('Paused: builder')
  })

  it('source-qualifies same-named holds whose roster rows are unavailable', async () => {
    const { GroupHoldStatus } = await import('./group-hold-status')

    render(
      <GroupHoldStatus
        holds={{ 'mac-mini::builder': { at: 1 }, 'laptop::builder': { at: 2 } }}
        memberLabel={member => member.title || member.name}
        members={[]}
      />
    )

    const status = screen.getByRole('status')

    expect(status.textContent).toContain('builder (mac-mini)')
    expect(status.textContent).toContain('builder (laptop)')
  })

  it('projects hydrated holds into the real group workspace', async () => {
    const [{ GroupChatWorkspace }, chat] = await Promise.all([import('./group-chat-view'), import('./group-chat')])

    chat.$groupChats.set({
      Core: {
        holds: { research: { at: 1 } },
        log: [],
        members: MEMBERS,
        watermarks: {}
      }
    })

    render(<GroupChatWorkspace group="Core" members={MEMBERS} />)

    expect(screen.getByRole('status').textContent).toContain('Paused: Research')
  })
})

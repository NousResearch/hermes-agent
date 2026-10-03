import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import type { DesktopAgentRoster } from '@/global'
import { $sessionTabAgentNames } from '@/store/session-tab-agent-names'
import type { ProfileInfo } from '@/types/hermes'

import { SessionTabLabel, sessionTabOwnerLabel, sessionTabText } from './session-tab-label'

afterEach(() => {
  cleanup()
  $sessionTabAgentNames.set(false)
})

const profile = (name: string, display_name?: string): ProfileInfo =>
  ({ bot_title: undefined, display_name, name }) as ProfileInfo

const sources = (
  profiles: ProfileInfo[],
  roster: DesktopAgentRoster | null = null,
  profilesByConnection = new Map<string, ProfileInfo[]>()
) => ({ profiles, profilesByConnection, roster, workspaceOwnerLabels: {} })

describe('session tab identity', () => {
  it('uses the friendly agent name and keeps duplicate profiles distinguishable by connection', () => {
    const roster: DesktopAgentRoster = {
      agents: [
        {
          connectionId: 'cloud',
          connectionKind: 'remote',
          connectionLabel: 'Cloud',
          handle: '@default-cloud',
          profile: 'default'
        },
        {
          connectionId: 'studio',
          connectionKind: 'remote',
          connectionLabel: 'Studio',
          handle: '@default-studio',
          profile: 'default'
        }
      ],
      sources: []
    }

    const profilesByConnection = new Map([['cloud', [profile('default', 'Miranda')]]])

    expect(
      sessionTabOwnerLabel(
        { connection_id: 'cloud', profile: 'default' },
        undefined,
        sources([], roster, profilesByConnection)
      )
    ).toBe('Miranda (Cloud)')
  })

  it('preserves the plain session title when the preference is off', () => {
    render(<SessionTabLabel session={{ profile: 'default' }} title="Session title" />)

    expect(screen.queryByLabelText('default · Session title')).toBeNull()
    expect(screen.getByText('Session title').children).toHaveLength(0)
  })

  it('renders a bold inherited-color owner before a muted separator and dominant title when enabled', () => {
    $sessionTabAgentNames.set(true)
    render(<SessionTabLabel session={{ profile: 'default' }} title="Session title" />)

    const label = screen.getByLabelText('default · Session title')
    const [owner, separator, title] = [...label.children]

    expect(owner.textContent).toBe('default')
    expect(owner.className).toContain('font-bold')
    expect(owner.className).not.toMatch(/(?:^|\s)text-/)
    expect(separator.textContent).toBe('·')
    expect(separator.className).toContain('text-(--ui-text-quaternary)')
    expect(title.textContent).toBe('Session title')
    expect(title.className).toContain('text-inherit')
  })

  it('degrades without null text or an empty separator when no owner is known', () => {
    expect(sessionTabText(null, 'New session')).toBe('New session')
    expect(sessionTabText('', 'Session title')).toBe('Session title')
    expect(sessionTabOwnerLabel({})).toBeNull()

    const { container } = render(<SessionTabLabel session={null} title="New session" />)

    expect(container.textContent).toBe('New session')
    expect(container.textContent).not.toContain('null')
    expect(container.textContent).not.toContain('·')
  })
})

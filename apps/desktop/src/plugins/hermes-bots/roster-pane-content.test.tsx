import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { renderRosterContent } from './roster-pane-content'

vi.mock('@hermes/plugin-sdk', () => ({
  Button: 'button',
  Codicon: () => null,
  DisclosureCaret: () => null,
  GlyphSpinner: () => <span aria-hidden="true" />,
  PanelEmpty: ({ description, title }: { description?: string; title?: string }) => <div>{description || title}</div>,
  RowButton: 'button'
}))
vi.mock('./hidden-bots', () => ({ $showHiddenBots: { set: vi.fn() } }))

type ContentProps = Parameters<typeof renderRosterContent>[0]

function rosterProps(overrides: Partial<ContentProps> = {}): ContentProps {
  return {
    b: {
      roster: {
        emptyDesc: 'Create your first bot',
        emptyTitle: 'No bots yet',
        loadingGatewayBots: 'Loading gateway bots',
        noMatchFilters: 'No bots match these filters'
      }
    } as ContentProps['b'],
    staleNotice: null,
    isLoading: false,
    rosterDiscoveryPending: false,
    loadingLabel: 'Loading',
    roster: [{ name: 'local-bot' }],
    error: null,
    gatewayUp: true,
    refetch: vi.fn(),
    allBotsHidden: false,
    hiddenExpanded: false,
    rosterRows: [{ kind: 'bot', bot: { name: 'local-bot' }, active: false, activity: 0, pinned: false }],
    matchingHiddenBots: [],
    query: '',
    selectedGateway: undefined,
    showGatewaySections: false,
    sortedGroupRows: [],
    gatewaySections: { sectioned: false, sections: [] },
    showHiddenSection: false,
    hiddenSectionRef: { current: null },
    hasRosterConstraint: false,
    hiddenBots: [],
    showHiddenRows: false,
    hiddenGatewaySections: { sectioned: false, sections: [] },
    renderBotRow: () => null,
    renderGroupChatSection: () => <></>,
    renderGatewaySection: () => <></>,
    renderUserSections: () => [<div key="local-bot">Local bot</div>],
    renderHiddenGatewaySection: () => <></>,
    ...overrides
  }
}

describe('gateway roster loading feedback', () => {
  it('keeps local bots usable and announces pending gateway discovery until it settles', () => {
    const { rerender } = render(renderRosterContent(rosterProps({ rosterDiscoveryPending: true })))

    expect(screen.getByText('Local bot')).not.toBeNull()
    expect(screen.getByRole('status').textContent).toBe('Loading gateway bots')

    rerender(renderRosterContent(rosterProps()))

    expect(screen.getByText('Local bot')).not.toBeNull()
    expect(screen.queryByText('Loading gateway bots')).toBeNull()
  })

  it('waits for discovery before declaring that a filtered roster has no matches', () => {
    const filtered = { rosterRows: [], hasRosterConstraint: true }
    const { rerender } = render(renderRosterContent(rosterProps({ ...filtered, rosterDiscoveryPending: true })))

    expect(screen.getByText('Loading gateway bots')).not.toBeNull()
    expect(screen.queryByText('No bots match these filters')).toBeNull()

    rerender(renderRosterContent(rosterProps(filtered)))

    expect(screen.getByText('No bots match these filters')).not.toBeNull()
    expect(screen.queryByText('Loading gateway bots')).toBeNull()
  })

  it('retains the initial loading state without a duplicate gateway indicator', () => {
    render(renderRosterContent(rosterProps({ roster: [], rosterRows: [], rosterDiscoveryPending: true })))

    expect(screen.getByRole('status').textContent).toBe('Loading')
    expect(screen.queryByText('Loading gateway bots')).toBeNull()
    expect(screen.queryByText('No bots yet')).toBeNull()
  })
})

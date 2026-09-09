import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { botsText } from './i18n'
import { RosterGroupRowView } from './roster-pane-groups'
import type { GroupPrompt } from './types'

vi.mock('./bot-row', () => ({
  GroupRow: ({ needsYou }: { needsYou: boolean }) => <div data-testid="attention">{String(needsYou)}</div>
}))

afterEach(cleanup)

it('keeps hosted, mention and pending-prompt attention independent in the extracted roster row', () => {
  const prompt: GroupPrompt = {
    group: 'Core',
    memberKey: 'member',
    member: 'member',
    kind: 'approval',
    at: 1,
    multiSelect: false,
    requestId: 'approval-1',
    sessionId: 'session-1',
    question: 'Allow?',
    choices: [],
    questions: null
  }
  const props = {
    active: false,
    b: botsText(),
    group: 'Core',
    members: [],
    onDisband: vi.fn(),
    onOpen: vi.fn(),
    groupRooms: {},
    sortedGroupRows: [],
    groupNeedsYou: {},
    groupHostedNeedsYou: {},
    groupClarify: {}
  }
  const view = render(<RosterGroupRowView {...props} groupHostedNeedsYou={{ Core: true }} />)
  expect(screen.getByTestId('attention').textContent).toBe('true')
  view.rerender(<RosterGroupRowView {...props} groupClarify={{ 'Core::member': prompt }} />)
  expect(screen.getByTestId('attention').textContent).toBe('true')
  view.rerender(<RosterGroupRowView {...props} groupNeedsYou={{ Core: true }} />)
  expect(screen.getByTestId('attention').textContent).toBe('true')
  view.rerender(
    <RosterGroupRowView
      {...props}
      groupClarify={{ 'Other::member': { ...prompt, group: 'Other' } }}
      groupHostedNeedsYou={{ Other: true }}
    />
  )
  expect(screen.getByTestId('attention').textContent).toBe('false')
})

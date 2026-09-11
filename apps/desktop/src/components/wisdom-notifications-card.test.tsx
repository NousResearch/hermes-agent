// @vitest-environment jsdom
import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { WisdomNotificationsCard } from './wisdom-notifications-card'

const openExternalLink = vi.fn()

vi.mock('@/lib/external-link', () => ({
  openExternalLink: (href: string) => openExternalLink(href)
}))

describe('WisdomNotificationsCard', () => {
  it('renders skill names and versions with update and Portal actions', async () => {
    const markAllRead = vi.fn().mockResolvedValue(undefined)
    const planAction = vi.fn().mockResolvedValue(undefined)

    render(
      <WisdomNotificationsCard
        events={[
          {
            category: 'update_available',
            event_id: 'event-1',
            kind: 'updated',
            occurred_at: '2026-08-28T00:00:00Z',
            portal_url: 'https://portal.example/orgs/team/wisdom/skills/skill-1?version=3',
            skill_id: 'skill-1',
            skill_name: 'incident-handoff',
            source_event_ids: ['event-1'],
            version: 3
          }
        ]}
        onMarkAllRead={markAllRead}
        onPlanAction={planAction}
      />
    )

    expect(screen.getByText('incident-handoff v3 is available to update.')).toBeTruthy()
    expect(screen.queryByText(/skill-1/)).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Review update' }))
    expect(planAction).toHaveBeenCalledWith('update', expect.objectContaining({ skill_id: 'skill-1', version: 3 }))

    fireEvent.click(screen.getByRole('button', { name: 'View skill' }))
    expect(openExternalLink).toHaveBeenCalledWith('https://portal.example/orgs/team/wisdom/skills/skill-1?version=3')
  })

  it('offers install for newly shared skills', () => {
    const planAction = vi.fn()

    render(
      <WisdomNotificationsCard
        events={[
          {
            category: 'new_skill',
            editorial_description: 'Coordinate incidents with a repeatable team workflow.',
            editorial_name: 'Team Incident Runbook',
            event_id: 'event-new',
            kind: 'new',
            skill_id: 'skill-new',
            skill_name: 'team-runbook',
            source_event_ids: ['event-new'],
            version: 1
          }
        ]}
        onMarkAllRead={vi.fn()}
        onPlanAction={planAction}
      />
    )

    expect(screen.getByText('Team Incident Runbook v1 was shared with your collective.')).toBeTruthy()
    expect(screen.getByText('Coordinate incidents with a repeatable team workflow.')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Install…' }))
    expect(planAction).toHaveBeenCalledWith('install', expect.objectContaining({ skill_id: 'skill-new' }))
  })

  it('offers install with update copy for a revised uninstalled shared skill', () => {
    const planAction = vi.fn()

    render(
      <WisdomNotificationsCard
        events={[
          {
            category: 'new_skill',
            event_id: 'event-shared-update',
            kind: 'updated',
            skill_id: 'skill-shared',
            skill_name: 'team-runbook',
            source_event_ids: ['event-shared-update'],
            version: 2
          }
        ]}
        onMarkAllRead={vi.fn()}
        onPlanAction={planAction}
      />
    )

    expect(screen.getByText('team-runbook v2 is available to update.')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Install…' }))
    expect(planAction).toHaveBeenCalledWith('install', expect.objectContaining({ skill_id: 'skill-shared' }))
  })
})

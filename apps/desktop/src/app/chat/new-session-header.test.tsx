import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { NewSessionHeader } from './new-session-header'

afterEach(cleanup)

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      sidebar: {
        nav: { 'new-session': 'New session' },
        noProject: 'No project'
      }
    }
  })
}))

vi.mock('./profile-tag', () => ({
  ProfileTag: ({ profile }: { profile: null | string }) => <span data-testid="profile-tag">{profile}</span>
}))

const tipTrigger = (el: HTMLElement) => el.closest('[data-slot="tooltip-trigger"]')

describe('NewSessionHeader', () => {
  it('shows a named project with the full cwd on a focusable Tip trigger', async () => {
    render(
      <NewSessionHeader
        cwd="/repos/hermes-agent/worktrees/feature"
        profile="default"
        projectName="Hermes"
        showProfileTag={false}
      />
    )

    expect(screen.getByText('New session')).toBeTruthy()
    const project = screen.getByText('Hermes')
    expect(project.getAttribute('tabindex')).toBe('0')
    expect(tipTrigger(project)).toBeTruthy()

    fireEvent.pointerMove(project, { pointerType: 'mouse' })
    expect((await screen.findByRole('tooltip')).textContent).toContain('/repos/hermes-agent/worktrees/feature')
  })

  it('falls back to the cwd leaf when there is no named project', () => {
    render(<NewSessionHeader cwd="C:\\work\\standalone" profile="default" projectName={null} showProfileTag={false} />)

    expect(screen.getByText('standalone')).toBeTruthy()
  })

  it('shows an explicit no-project state when the draft is detached', () => {
    render(<NewSessionHeader cwd="" profile="default" projectName={null} showProfileTag={false} />)

    expect(screen.getByText('No project')).toBeTruthy()
  })

  it('shows the active profile identity in a multi-profile setup', () => {
    render(<NewSessionHeader cwd="/repos/hermes" profile="work" projectName="Hermes" showProfileTag />)

    expect(screen.getByTestId('profile-tag').textContent).toBe('work')
  })
})

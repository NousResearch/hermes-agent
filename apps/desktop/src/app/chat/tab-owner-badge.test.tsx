// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $profiles } from '@/store/profile'
import type { ProfileInfo } from '@/types/hermes'

import { TabOwnerBadge } from './tab-owner-badge'

// The tab strip is the only header a pane has in the tiled layout, so the tab's
// owner mark is the tiled-layout answer to "which agent answers this chat?". It
// must stay silent when it would answer nothing: one profile on the gateway, or
// a session on the default profile — the same rule the sidebar row follows.

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      sidebar: {
        row: { ownedByProfile: (name: string) => `Profile: ${name}` }
      }
    }
  })
}))

const profile = (name: string): ProfileInfo => ({
  has_env: true,
  is_default: name === 'default',
  model: null,
  name,
  path: `/profiles/${name}`,
  provider: null,
  skill_count: 0
})

const previousProfiles = $profiles.value

beforeEach(() => {
  $profiles.set([profile('default'), profile('work')])
})

afterEach(() => {
  cleanup()
  $profiles.set(previousProfiles)
})

describe('TabOwnerBadge', () => {
  it('marks a tab whose session belongs to a named profile', () => {
    render(<TabOwnerBadge profile="work" />)

    expect(screen.getByRole('img', { name: 'Profile: work' })).toBeTruthy()
  })

  it('stays silent for the default profile', () => {
    const { container } = render(<TabOwnerBadge profile="default" />)

    expect(container.querySelector('[role="img"]')).toBeNull()
  })

  it('stays silent while the session has no profile yet', () => {
    const { container } = render(<TabOwnerBadge profile={null} />)

    expect(container.querySelector('[role="img"]')).toBeNull()
  })

  it('stays silent on a single-profile gateway', () => {
    $profiles.set([profile('default')])
    const { container } = render(<TabOwnerBadge profile="work" />)

    expect(container.querySelector('[role="img"]')).toBeNull()
  })
})

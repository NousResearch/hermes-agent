import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $attentionSessions } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { SidebarSessionRow } from './session-row'

const row = (profile: string): SessionInfo =>
  ({ id: 'same-id', last_active: 1, profile, started_at: 1, title: `${profile} row` }) as SessionInfo

function renderRow(profile: string) {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <SidebarSessionRow
        isPinned={false}
        isSelected={false}
        isWorking
        onArchive={() => undefined}
        onDelete={() => undefined}
        onPin={() => undefined}
        onResume={() => undefined}
        session={row(profile)}
      />
    </I18nProvider>
  )
}

describe('SidebarSessionRow profile attention', () => {
  afterEach(() => {
    cleanup()
    $attentionSessions.set([])
  })

  it('does not show another profile same-id attention state', () => {
    $attentionSessions.set([{ profile: 'default', sessionId: 'same-id' }])
    const { container } = renderRow('work')

    expect(container.querySelector('.bg-amber-500')).toBeNull()
    expect(screen.getByLabelText('Session running')).toBeTruthy()
  })

  it('shows attention for the matching profile row', () => {
    $attentionSessions.set([{ profile: 'work', sessionId: 'same-id' }])
    const { container } = renderRow('work')

    expect(container.querySelector('.bg-amber-500')).toBeTruthy()
  })
})

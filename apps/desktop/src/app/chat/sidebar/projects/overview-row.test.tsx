import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ProjectOverviewRow } from './overview-row'

describe('ProjectOverviewRow new session target', () => {
  afterEach(() => {
    cleanup()
  })

  it('marks the synthetic no-project row as an explicit no-workspace target', () => {
    const onNewSession = vi.fn()

    render(
      <ProjectOverviewRow
        onNewSession={onNewSession}
        project={{
          id: '__no_project__',
          isNoProject: true,
          label: 'No project',
          path: null,
          repos: [],
          sessionCount: 0
        }}
      />
    )

    fireEvent.click(screen.getByRole('button', { name: /new session/i }))

    expect(onNewSession).toHaveBeenCalledWith(null, true)
  })
})

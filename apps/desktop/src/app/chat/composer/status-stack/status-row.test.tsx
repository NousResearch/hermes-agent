import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import type { ComposerStatusItem } from '@/store/composer-status'

import { StatusItemRow } from './status-row'

const background = (fields: Partial<ComposerStatusItem>): ComposerStatusItem => ({
  id: 'proc_preview',
  state: 'running',
  title: 'python -m http.server 8731',
  type: 'background',
  ...fields
})

describe('StatusItemRow background activity', () => {
  afterEach(() => {
    cleanup()
  })

  it('flags a running process that has been silent past the threshold', () => {
    render(<StatusItemRow item={background({ quietMinutes: 164, runtimeMinutes: 165 })} />)

    expect(screen.getByText('no output for 2h 44m')).toBeTruthy()
  })

  it('shows uptime, not a warning, while the process keeps printing', () => {
    render(<StatusItemRow item={background({ quietMinutes: 1, runtimeMinutes: 42 })} />)

    expect(screen.getByText('up 42m')).toBeTruthy()
    expect(screen.queryByText(/no output/)).toBeNull()
  })
})

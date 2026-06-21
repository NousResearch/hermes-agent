import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopUpdateCommit } from '@/global'
import {
  $backendUpdateApply,
  $backendUpdateChecking,
  $backendUpdateStatus,
  $updateApply,
  $updateChecking,
  $updateOverlayOpen,
  $updateOverlayTarget,
  $updateStatus
} from '@/store/updates'

import { UpdatesOverlay } from './updates-overlay'

function slot(root: ParentNode, name: string): HTMLElement {
  const element = root.querySelector(`[data-slot="${name}"]`)

  if (!(element instanceof HTMLElement)) {
    throw new Error(`Missing ${name}`)
  }

  return element
}

function commits(count: number): DesktopUpdateCommit[] {
  return Array.from({ length: count }, (_, index) => ({
    at: Date.now() - index,
    author: 'Hermes',
    sha: `commit-${index}`,
    summary: `fix(desktop): keep update dialog action ${index + 1} reachable`
  }))
}

function resetUpdateStores() {
  const idleApply = {
    applying: false,
    command: null,
    error: null,
    log: [],
    message: '',
    percent: null,
    stage: 'idle' as const
  }

  $updateOverlayOpen.set(false)
  $updateOverlayTarget.set('client')
  $updateStatus.set(null)
  $updateChecking.set(false)
  $updateApply.set(idleApply)
  $backendUpdateStatus.set(null)
  $backendUpdateChecking.set(false)
  $backendUpdateApply.set(idleApply)
}

describe('UpdatesOverlay', () => {
  beforeEach(() => {
    resetUpdateStores()
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    resetUpdateStores()
  })

  it('keeps update actions outside the scrollable release notes area', () => {
    $updateStatus.set({
      behind: 30,
      commits: commits(24),
      fetchedAt: Date.now(),
      supported: true,
      targetSha: 'target-sha'
    })
    $updateOverlayOpen.set(true)

    render(<UpdatesOverlay />)

    expect(screen.getByText('New update available')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Update now' })).toBeTruthy()

    const notes = slot(document.body, 'updates-overlay-notes')
    const actions = slot(document.body, 'updates-overlay-actions')

    expect(slot(document.body, 'updates-overlay-idle').className).toContain('max-h-[85dvh]')
    expect(notes.className).toContain('min-h-0')
    expect(notes.className).toContain('flex-1')
    expect(notes.className).toContain('overflow-y-auto')
    expect(actions.className).toContain('shrink-0')
    expect(notes.contains(screen.getByRole('button', { name: 'Update now' }))).toBe(false)
  })
})

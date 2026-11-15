import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { leaveWorkspacePageForChat, runSharedNewSessionOrLegacy } from '@/app/chat/new-session-tab'
import { $workspaceIsPage, sessionRoute } from '@/app/routes'
import { $newSessionTabAction } from '@/components/pane-shell/tree/store'

describe('leaveWorkspacePageForChat', () => {
  beforeEach(() => {
    $workspaceIsPage.set(false)
  })

  afterEach(() => {
    // The atom is module state; leave it clean for the next test file.
    $workspaceIsPage.set(false)
  })

  it('does nothing off a workspace page (the minted tab is already visible)', () => {
    const navigate = vi.fn()

    leaveWorkspacePageForChat(navigate, () => 'stored-1')

    expect(navigate).not.toHaveBeenCalled()
  })

  it('leaves a workspace page for the selected session route — the primary chat survives', () => {
    $workspaceIsPage.set(true)
    const navigate = vi.fn()

    leaveWorkspacePageForChat(navigate, () => 'stored-1')

    expect(navigate).toHaveBeenCalledWith(sessionRoute('stored-1'))
  })

  it('falls back to the new-chat route when the primary holds no session', () => {
    $workspaceIsPage.set(true)
    const navigate = vi.fn()

    leaveWorkspacePageForChat(navigate, () => null)

    expect(navigate).toHaveBeenCalledWith('/')
  })
})

describe('runSharedNewSessionOrLegacy (the palette row)', () => {
  afterEach(() => {
    // Module state; leave it clean for the next test file.
    $newSessionTabAction.set(null)
  })

  it('runs the shared action wiring publishes — one creation path for the palette too', () => {
    const shared = vi.fn()
    const fallback = vi.fn()
    $newSessionTabAction.set(shared)

    runSharedNewSessionOrLegacy(fallback)

    expect(shared).toHaveBeenCalledOnce()
    expect(fallback).not.toHaveBeenCalled()
  })

  it('keeps the legacy new-chat navigation when the shared action is not wired', () => {
    const fallback = vi.fn()
    $newSessionTabAction.set(null)

    runSharedNewSessionOrLegacy(fallback)

    expect(fallback).toHaveBeenCalledOnce()
  })
})

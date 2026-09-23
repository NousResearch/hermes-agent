import { describe, expect, it } from 'vitest'

import type { BackendUpdateCheckResponse } from '@/types/hermes'

import { mapBackendCheck } from './updates'

const response = (over: Partial<BackendUpdateCheckResponse> = {}): BackendUpdateCheckResponse => ({
  install_method: 'git',
  current_version: '9.9.9',
  behind: 0,
  update_available: false,
  can_apply: true,
  update_command: null,
  message: null,
  ...over
})

describe('mapBackendCheck', () => {
  it('surfaces a failed check instead of claiming the backend is up to date', () => {
    // The endpoint answers `behind: null` + a message when it could not run the check
    // (GitHub unreachable, rate limited, offline). Folding that to 0 reported "you're on
    // the latest version" for a check that never happened.
    const status = mapBackendCheck(
      response({ behind: null, message: "Couldn't reach the update source — try again later." })
    )

    expect(status.error).toBe('check-failed')
    expect(status.updateAvailable).not.toBe(true)
    expect(status.message).toContain("Couldn't reach the update source")
    expect(status.targetSha).toBeUndefined()
  })

  it('still reports an up-to-date backend when the check ran and found no gap', () => {
    const status = mapBackendCheck(response({ behind: 0, message: "You're on the latest version." }))

    expect(status.error).toBeUndefined()
    expect(status.behind).toBe(0)
    expect(status.updateAvailable).not.toBe(true)
  })

  it('keeps a real gap intact', () => {
    const status = mapBackendCheck(response({ behind: 3, update_available: true }))

    expect(status.error).toBeUndefined()
    expect(status.behind).toBe(3)
    expect(status.updateAvailable).toBe(true)
    expect(status.targetSha).toBe('backend:9.9.9')
  })

  it('leaves a backend that cannot self-update to the unsupported branch', () => {
    // pip/nix and managed runtimes also answer `behind: null`, but `can_apply: false` renders
    // the "not available" copy first — those must not be turned into failures.
    const status = mapBackendCheck(
      response({ can_apply: false, behind: null, message: 'pip install -U hermes-agent' })
    )

    expect(status.supported).toBe(false)
    expect(status.error).toBeUndefined()
    expect(status.message).toBe('pip install -U hermes-agent')
  })
})

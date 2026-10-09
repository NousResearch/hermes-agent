import { render } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/app/chat/session-view', () => ({ useSessionView: () => ({ kind: 'primary' }) }))
vi.mock('./scope', () => ({ useComposerScope: () => ({ target: 'main' }) }))

describe('LocalSetupCard', () => {
  beforeEach(() => {
    vi.resetModules()
    // A relaunch: the card was shown last run, and the backend connection is not there yet.
    localStorage.setItem('hermes.desktop.offers.local-setup.v1', JSON.stringify({ state: 'shown' }))
  })

  it('settles on one answer while there is no connection instead of re-reading in a loop', async () => {
    const { $localModelsEnabled } = await import('@/store/local-models-flag')
    const { $localSetupEligibility } = await import('@/store/local-setup-offer')
    const { LocalSetupCard } = await import('./local-setup-card')
    $localModelsEnabled.set(true)

    expect(() => render(<LocalSetupCard busy={false} guidedChat={false} />)).not.toThrow()
    expect($localSetupEligibility.get()?.reason).toBe('connection not established yet')
  })
})

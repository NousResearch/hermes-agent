import { cleanup, render, screen } from '@testing-library/react'
import { useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $composerPopoutGesturesEnabled, $composerPopoutZones } from '@/store/composer-popout'

import { useComposerPopout } from './use-composer-popout'

vi.mock('@/components/pane-shell/pane-visibility', () => ({
  usePaneGroup: () => 'test-zone',
  usePaneVisible: () => true
}))

vi.mock('@/hooks/use-resize-observer', () => ({ useResizeObserver: () => undefined }))
vi.mock('@/store/windows', () => ({ isSecondaryWindow: () => false }))

function PopoutAffordanceHarness() {
  const composerRef = useRef<HTMLFormElement>(null)
  const { popoutAllowed } = useComposerPopout({ composerRef })

  return (
    <form ref={composerRef}>{popoutAllowed && <div data-slot="composer-drag-region" data-testid="drag-region" />}</form>
  )
}

describe('useComposerPopout', () => {
  beforeEach(() => {
    $composerPopoutZones.set({})
    $composerPopoutGesturesEnabled.set(true)
  })

  afterEach(() => {
    cleanup()
    $composerPopoutZones.set({})
    $composerPopoutGesturesEnabled.set(true)
  })

  it('removes the pop-out affordance when gestures are disabled', () => {
    $composerPopoutGesturesEnabled.set(false)

    render(<PopoutAffordanceHarness />)

    expect(screen.queryByTestId('drag-region')).toBeNull()
  })
})

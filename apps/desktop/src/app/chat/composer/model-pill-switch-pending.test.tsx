import { useStore } from '@nanostores/react'
import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { PRIMARY_SESSION_VIEW } from '@/app/chat/session-view'
import { useModelControls } from '@/app/session/hooks/use-model-controls'
import { formatModelPillLabel } from '@/lib/model-status-label'
import { $modelSwitchPending } from '@/store/model-switch-pending'
import {
  $activeSessionId,
  $currentModel,
  setCurrentModel,
  setCurrentModelSource,
  setCurrentProvider
} from '@/store/session'

import { deferred } from '../../../test/deferred'

import { ModelPill } from './model-pill'

vi.mock('@/hermes', () => ({
  getGlobalModelInfo: vi.fn(),
  setApiRequestProfile: vi.fn(),
  setGlobalModel: vi.fn()
}))

/**
 * The composer as it is actually wired: one `useModelControls` instance owns
 * the `config.set model` round trip, the pill renders the primary session
 * view's model. This is the user-visible symptom — picking a model leaves the
 * pill showing the model being replaced, with no sign anything is happening.
 */
function PendingPill({
  requestGateway
}: {
  requestGateway: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
}) {
  const { selectModel } = useModelControls({ queryClient: new QueryClient(), requestGateway })
  const model = useStore(PRIMARY_SESSION_VIEW.$model)
  const provider = useStore(PRIMARY_SESSION_VIEW.$provider)

  return (
    <>
      <button onClick={() => void selectModel({ model: 'new-model', provider: 'anthropic' })} type="button">
        Pick new model
      </button>
      <ModelPill disabled={false} model={{ canSwitch: true, model, provider }} />
    </>
  )
}

describe('model pill while a model switch is pending', () => {
  beforeEach(() => {
    $activeSessionId.set(null)
    setCurrentModel('')
    setCurrentModelSource('')
    setCurrentProvider('')
    $modelSwitchPending.set({})
  })

  afterEach(() => {
    cleanup()
    $activeSessionId.set(null)
    setCurrentModel('')
    setCurrentModelSource('')
    setCurrentProvider('')
    $modelSwitchPending.set({})
  })

  it('shows a spinner instead of a model name until the gateway answers', async () => {
    $activeSessionId.set('runtime-1')
    setCurrentModel('old-model')
    setCurrentProvider('xai')

    const answer = deferred<{ key: string; value: string }>()
    const requestGateway = vi.fn(() => answer.promise as never)

    render(<PendingPill requestGateway={requestGateway} />)

    expect(screen.getByText(formatModelPillLabel('old-model'))).toBeTruthy()

    fireEvent.click(screen.getByText('Pick new model'))

    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith('config.set', expect.objectContaining({ session_id: 'runtime-1' }))
    )

    // Neither name is on screen: the switch has not landed, so the pill says
    // "working" rather than presenting a model that may still roll back.
    expect(screen.queryByText(formatModelPillLabel('old-model'))).toBeNull()
    expect(screen.queryByText(formatModelPillLabel('new-model'))).toBeNull()
    expect(screen.getByRole('status')).toBeTruthy()

    await act(async () => {
      answer.resolve({ key: 'model', value: 'new-model' })
      await Promise.resolve()
    })

    await waitFor(() => expect(screen.queryByRole('status')).toBeNull())
    expect(screen.getByText(formatModelPillLabel('new-model'))).toBeTruthy()
  })

  it('stops spinning and names the kept model when the switch fails', async () => {
    $activeSessionId.set('runtime-1')
    setCurrentModel('old-model')
    setCurrentProvider('xai')

    const answer = deferred<{ key: string; value: string }>()
    const requestGateway = vi.fn(() => answer.promise as never)

    render(<PendingPill requestGateway={requestGateway} />)

    fireEvent.click(screen.getByText('Pick new model'))

    await waitFor(() => expect(screen.getByRole('status')).toBeTruthy())

    await act(async () => {
      answer.reject(new Error('no such model'))
      await Promise.resolve()
    })

    // A failed pick must not leave the composer spinning forever.
    await waitFor(() => expect(screen.queryByRole('status')).toBeNull())
    expect(screen.getByText(formatModelPillLabel('old-model'))).toBeTruthy()
    expect($currentModel.get()).toBe('old-model')
  })
})

import { cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  fetchVoiceLiveStatus: vi.fn()
}))

vi.mock('@/lib/voice-live', () => ({
  fetchVoiceLiveStatus: mocks.fetchVoiceLiveStatus
}))

const { DropdownMenu, DropdownMenuContent } = await import('@/components/ui/dropdown-menu')
const { $voiceLiveStatus } = await import('@/store/voice-live')
const { VoiceEngineRows } = await import('./voice-engine-rows')

afterEach(() => {
  cleanup()
  mocks.fetchVoiceLiveStatus.mockReset()
  $voiceLiveStatus.set(null)
})

describe('VoiceEngineRows readiness recovery', () => {
  it('refreshes stale unavailable readiness when the engine picker mounts', async () => {
    $voiceLiveStatus.set({
      mode: 'chained',
      available: false,
      auth: 'subscription',
      model: 'gpt-live-1-codex',
      voice: 'cove',
      reason: 'subscription credentials temporarily unavailable'
    })
    mocks.fetchVoiceLiveStatus.mockResolvedValue({
      mode: 'gpt-live',
      available: true,
      auth: 'subscription',
      model: 'gpt-live-1-codex',
      voice: 'cove',
      reason: null
    })

    render(
      <DropdownMenu open>
        <DropdownMenuContent>
          <VoiceEngineRows disabled={false} />
        </DropdownMenuContent>
      </DropdownMenu>
    )

    await waitFor(() => expect(mocks.fetchVoiceLiveStatus).toHaveBeenCalledTimes(1))
    await waitFor(() => expect($voiceLiveStatus.get()?.available).toBe(true))
  })
})

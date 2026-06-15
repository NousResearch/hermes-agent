import { cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { loadCloudChannelParticipants, loadSessionCloudStatus } from '@/lib/cloud-share'
import { $localDeviceName, $sessionParticipants, setSessionParticipants } from '@/store/session'

import { ParticipantChips } from './participant-chips'

vi.mock('@/lib/cloud-share', () => ({
  loadCloudChannelParticipants: vi.fn(),
  loadSessionCloudStatus: vi.fn()
}))

const mockLoadSessionCloudStatus = vi.mocked(loadSessionCloudStatus)
const mockLoadCloudChannelParticipants = vi.mocked(loadCloudChannelParticipants)

describe('ParticipantChips', () => {
  beforeEach(() => {
    $sessionParticipants.set({})
    $localDeviceName.set('')
    mockLoadSessionCloudStatus.mockResolvedValue({ configured: false, shared: false })
    mockLoadCloudChannelParticipants.mockResolvedValue({ participants: [] })
  })

  afterEach(() => {
    cleanup()
    $sessionParticipants.set({})
    $localDeviceName.set('')
    vi.clearAllMocks()
  })

  it('renders nothing when only this device is viewing', () => {
    $localDeviceName.set("Omar's MacBook")
    setSessionParticipants('s1', [{ count: 1, device: "Omar's MacBook" }])

    const { queryByTestId } = render(<ParticipantChips sessionId="s1" />)

    // A solo local session shows no chip row — presence only matters with peers.
    expect(queryByTestId('session-participants')).toBeNull()
  })

  it('renders a chip for each OTHER viewing device and filters out this one', () => {
    $localDeviceName.set("Omar's MacBook")
    setSessionParticipants('s1', [
      { count: 1, device: "Omar's MacBook" },
      { count: 1, device: 'omar-iphone' },
      { count: 1, device: 'studio-linux' }
    ])

    const { getByText, queryByText } = render(<ParticipantChips sessionId="s1" />)

    expect(getByText('omar-iphone')).toBeTruthy()
    expect(getByText('studio-linux')).toBeTruthy()
    expect(queryByText("Omar's MacBook")).toBeNull()
  })

  it('shows a count badge when one device watches from multiple windows', () => {
    $localDeviceName.set('mac')
    setSessionParticipants('s1', [{ count: 2, device: 'omar-iphone' }])

    const { getByText } = render(<ParticipantChips sessionId="s1" />)

    expect(getByText('omar-iphone')).toBeTruthy()
    expect(getByText('×2')).toBeTruthy()
  })

  it('renders cloud roster viewers for a shared session', async () => {
    mockLoadSessionCloudStatus.mockResolvedValue({ channel_id: 'ch_cloud', configured: true, shared: true })
    mockLoadCloudChannelParticipants.mockResolvedValue({
      participants: [{ count: 1, device: 'remote-browser' }]
    })

    const { findByText } = render(<ParticipantChips sessionId="s1" />)

    expect(await findByText('remote-browser')).toBeTruthy()
    expect(mockLoadCloudChannelParticipants).toHaveBeenCalledWith('ch_cloud', { quiet: true })
  })

  it('renders nothing for a session with no roster', () => {
    $localDeviceName.set('mac')

    const { queryByTestId } = render(<ParticipantChips sessionId="unknown" />)

    expect(queryByTestId('session-participants')).toBeNull()
  })
})

describe('setSessionParticipants', () => {
  beforeEach(() => $sessionParticipants.set({}))
  afterEach(() => $sessionParticipants.set({}))

  it('stores a roster and replaces it on update', () => {
    setSessionParticipants('s1', [{ count: 1, device: 'a' }])
    expect($sessionParticipants.get().s1).toEqual([{ count: 1, device: 'a' }])

    setSessionParticipants('s1', [{ count: 1, device: 'b' }])
    expect($sessionParticipants.get().s1).toEqual([{ count: 1, device: 'b' }])
  })

  it('drops the key on an empty roster so the map does not accumulate', () => {
    setSessionParticipants('s1', [{ count: 1, device: 'a' }])
    setSessionParticipants('s1', [])

    expect('s1' in $sessionParticipants.get()).toBe(false)
  })

  it('ignores an empty session id', () => {
    setSessionParticipants('', [{ count: 1, device: 'a' }])

    expect($sessionParticipants.get()).toEqual({})
  })
})

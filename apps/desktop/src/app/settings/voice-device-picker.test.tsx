import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'

import { VoiceDevicePicker } from './voice-device-picker'

// Radix's Select content needs browser APIs jsdom lacks (ResizeObserver, pointer capture); without
// them it throws on open, which presents as "the options never appear".
stubResizeObserver()
stubMenuDomApis()

const DEVICES = [
  { deviceId: 'mic-a', kind: 'audioinput', label: 'MacBook Pro Microphone' },
  { deviceId: 'mic-b', kind: 'audioinput', label: 'AirPods Pro' },
  { deviceId: 'spk-a', kind: 'audiooutput', label: 'MacBook Pro Speakers' }
] as MediaDeviceInfo[]

function stubDevices(devices: MediaDeviceInfo[] = DEVICES) {
  Object.defineProperty(navigator, 'mediaDevices', {
    configurable: true,
    value: {
      addEventListener: vi.fn(),
      enumerateDevices: vi.fn().mockResolvedValue(devices),
      removeEventListener: vi.fn()
    }
  })
}

afterEach(() => cleanup())

describe('VoiceDevicePicker', () => {
  it('lists the microphones plus a system-default entry, and reports the choice', async () => {
    stubDevices()
    const onChange = vi.fn()

    render(<VoiceDevicePicker onChange={onChange} schemaKey="voice.mic_device_id" value="" />)

    // the picker seeds from an async enumeration; the closed trigger shows the selection only,
    // so the device options are asserted with the listbox open
    await waitFor(() => expect(screen.getByText('System default')).toBeTruthy())

    fireEvent.click(screen.getByRole('combobox'))
    expect(await screen.findByRole('option', { name: 'MacBook Pro Microphone' })).toBeTruthy()
    expect(within(screen.getByRole('listbox')).getByText('System default')).toBeTruthy()
    expect(screen.getByRole('option', { name: 'AirPods Pro' })).toBeTruthy()
    // an output device must not be offered as an input
    expect(screen.queryByRole('option', { name: 'MacBook Pro Speakers' })).toBeNull()

    fireEvent.click(screen.getByRole('option', { name: 'AirPods Pro' }))

    await waitFor(() => expect(onChange).toHaveBeenCalledWith('mic-b'))
  })

  it('offers speakers for the output row', async () => {
    stubDevices()

    render(<VoiceDevicePicker onChange={vi.fn()} schemaKey="voice.speaker_device_id" value="" />)

    await waitFor(() => expect(screen.getByText('System default')).toBeTruthy())

    fireEvent.click(screen.getByRole('combobox'))
    expect(await screen.findByRole('option', { name: 'MacBook Pro Speakers' })).toBeTruthy()
    expect(screen.queryByRole('option', { name: 'AirPods Pro' })).toBeNull()
  })

  it('flags a configured device that is no longer connected', async () => {
    stubDevices()

    render(<VoiceDevicePicker onChange={vi.fn()} schemaKey="voice.mic_device_id" value="mic-gone" />)

    expect(await screen.findByText(/not connected/)).toBeTruthy()
  })

  it('stays quiet when the configured device is present', async () => {
    stubDevices()

    render(<VoiceDevicePicker onChange={vi.fn()} schemaKey="voice.mic_device_id" value="mic-a" />)

    await waitFor(() => expect(screen.getByText('MacBook Pro Microphone')).toBeTruthy())
    expect(screen.queryByText(/not connected/)).toBeNull()
  })

  it('explains an empty device list (permissions not granted yet)', async () => {
    stubDevices([])

    render(<VoiceDevicePicker onChange={vi.fn()} schemaKey="voice.mic_device_id" value="" />)

    expect(await screen.findByText(/No devices found/)).toBeTruthy()
  })
})

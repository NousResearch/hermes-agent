import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { HermesConfigRecord } from '@/types/hermes'

import { VoiceCommandProviderAction, VoiceCommandProviderDialog } from './voice-command-provider-dialog'

afterEach(cleanup)

const config: HermesConfigRecord = {
  stt: { enabled: false, provider: 'local', echo_transcripts: true }
}

describe('VoiceCommandProviderDialog', () => {
  it('opens from the Voice settings action', () => {
    render(<VoiceCommandProviderAction config={config} onApply={vi.fn()} />)

    fireEvent.click(screen.getByRole('button', { name: 'Add local STT provider' }))

    expect(screen.getByRole('dialog', { name: 'Add local STT provider' })).toBeTruthy()
  })

  it('creates and activates a local STT command provider', () => {
    const onApply = vi.fn()
    const onOpenChange = vi.fn()

    render(<VoiceCommandProviderDialog config={config} onApply={onApply} onOpenChange={onOpenChange} open />)

    fireEvent.change(screen.getByLabelText('Provider name'), { target: { value: 'sensevoice' } })
    fireEvent.change(screen.getByLabelText('Command'), {
      target: { value: 'sensevoice-cli {input_path} --json | tee {output_path}' }
    })
    fireEvent.change(screen.getByLabelText('Language'), { target: { value: 'zh' } })
    fireEvent.change(screen.getByLabelText('Model'), { target: { value: 'iic/SenseVoiceSmall' } })
    fireEvent.click(screen.getByRole('button', { name: 'Add provider' }))

    expect(onApply).toHaveBeenCalledWith({
      stt: {
        enabled: true,
        provider: 'sensevoice',
        echo_transcripts: true,
        providers: {
          sensevoice: {
            type: 'command',
            command: 'sensevoice-cli {input_path} --json | tee {output_path}',
            format: 'txt',
            language: 'zh',
            model: 'iic/SenseVoiceSmall'
          }
        }
      }
    })
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })

  it('keeps the dialog open and shows actionable placeholder errors', () => {
    const onApply = vi.fn()
    const onOpenChange = vi.fn()

    render(<VoiceCommandProviderDialog config={config} onApply={onApply} onOpenChange={onOpenChange} open />)

    fireEvent.change(screen.getByLabelText('Provider name'), { target: { value: 'sensevoice' } })
    fireEvent.change(screen.getByLabelText('Command'), { target: { value: 'sensevoice-cli audio.wav' } })
    fireEvent.click(screen.getByRole('button', { name: 'Add provider' }))

    expect(screen.getByText('The command must include {input_path}.')).toBeTruthy()
    expect(onApply).not.toHaveBeenCalled()
    expect(onOpenChange).not.toHaveBeenCalledWith(false)
  })
})

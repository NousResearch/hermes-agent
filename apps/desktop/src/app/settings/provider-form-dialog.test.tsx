import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { ProviderFormDialog } from './provider-form-dialog'
import type { CustomProviderEntry } from '@/lib/custom-provider-config'

const noop = () => {}

describe('ProviderFormDialog', () => {
  it('submits a parsed entry in add mode, storing the generated id as the name', () => {
    const onSave = vi.fn().mockResolvedValue(undefined)
    render(<ProviderFormDialog open onClose={noop} onSave={onSave} />)

    fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'My Provider' } })
    fireEvent.change(screen.getByLabelText('Base URL'), { target: { value: 'https://my.host/v1' } })

    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    expect(onSave).toHaveBeenCalledTimes(1)
    const entry: CustomProviderEntry = onSave.mock.calls[0][0]
    expect(entry.name).toBe('my-provider')
    expect(entry.base_url).toBe('https://my.host/v1')
    expect(entry.models).toEqual([])
  })

  it('shows a read-only Provider ID preview derived from the friendly name', () => {
    render(<ProviderFormDialog open onClose={noop} onSave={noop} />)

    // Hidden until a name is entered.
    expect(screen.queryByLabelText('Provider ID')).toBeNull()

    fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'My Provider' } })

    const preview = screen.getByLabelText('Provider ID')
    expect(preview.textContent).toBe('my-provider')
  })

  it('uniquifies the generated id against existing providers', () => {
    const onSave = vi.fn().mockResolvedValue(undefined)
    render(<ProviderFormDialog existingNames={['my-provider']} open onClose={noop} onSave={onSave} />)

    fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'My Provider' } })
    fireEvent.change(screen.getByLabelText('Base URL'), { target: { value: 'https://x/v1' } })

    expect(screen.getByLabelText('Provider ID').textContent).toBe('my-provider-2')

    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    expect(onSave).toHaveBeenCalledTimes(1)
    expect(onSave.mock.calls[0][0].name).toBe('my-provider-2')
  })

  it('disables Save for an invalid base URL', () => {
    const onSave = vi.fn()
    render(<ProviderFormDialog open onClose={noop} onSave={onSave} />)

    fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'P' } })
    fireEvent.change(screen.getByLabelText('Base URL'), { target: { value: 'not-a-url' } })

    const save = screen.getByRole('button', { name: 'Save' })
    expect((save as HTMLButtonElement).disabled).toBe(true)

    fireEvent.click(save)

    expect(onSave).not.toHaveBeenCalled()
  })

  it('prefills and deletes in edit mode', () => {
    const onDelete = vi.fn().mockResolvedValue(undefined)
    const onSave = vi.fn()
    const initial: CustomProviderEntry = {
      name: 'Lab',
      base_url: 'https://lab/v1',
      api_mode: 'chat_completions',
      models: [{ id: 'a' }, { id: 'b' }]
    }

    render(<ProviderFormDialog initial={initial} open onClose={noop} onDelete={onDelete} onSave={onSave} />)

    expect((screen.getByLabelText('Name') as HTMLInputElement).value).toBe('Lab')
    expect((screen.getByLabelText('Base URL') as HTMLInputElement).value).toBe('https://lab/v1')

    // Open the delete confirmation, then confirm.
    fireEvent.click(screen.getByRole('button', { name: 'Delete' }))
    fireEvent.click(screen.getByRole('button', { name: 'Delete' }))

    expect(onDelete).toHaveBeenCalledWith('Lab')
  })
})

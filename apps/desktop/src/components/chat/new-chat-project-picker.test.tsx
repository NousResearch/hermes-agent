import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { NewChatProjectPicker } from './new-chat-project-picker'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  Reflect.deleteProperty(window, 'hermesDesktop')
})

describe('NewChatProjectPicker', () => {
  it('shows the selected workspace and branch', () => {
    render(<NewChatProjectPicker branch="main" cwd="/Users/me/projects/hermes" onChangeCwd={vi.fn()} />)

    expect(screen.getByText('Project folder')).toBeTruthy()
    expect(screen.getByText('/Users/me/projects/hermes')).toBeTruthy()
    expect(screen.getByText('Branch main')).toBeTruthy()
    expect(screen.getByText('Workspace: hermes')).toBeTruthy()
  })

  it('opens the directory picker and forwards the selected folder', async () => {
    const selectPaths = vi.fn().mockResolvedValue(['/tmp/hermes'])
    const onChangeCwd = vi.fn().mockResolvedValue(undefined)

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        selectPaths
      }
    })

    render(<NewChatProjectPicker cwd="/tmp" onChangeCwd={onChangeCwd} />)
    fireEvent.click(screen.getByRole('button', { name: /change/i }))

    await waitFor(() => expect(onChangeCwd).toHaveBeenCalledWith('/tmp/hermes'))
    expect(selectPaths).toHaveBeenCalledWith({
      defaultPath: '/tmp',
      directories: true,
      multiple: false,
      title: 'Choose project folder'
    })
  })

  it('does not change cwd when the picker is cancelled', async () => {
    const selectPaths = vi.fn().mockResolvedValue([])
    const onChangeCwd = vi.fn()

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        selectPaths
      }
    })

    render(<NewChatProjectPicker cwd="" onChangeCwd={onChangeCwd} />)
    fireEvent.click(screen.getByRole('button', { name: /choose folder/i }))

    await waitFor(() => expect(selectPaths).toHaveBeenCalled())
    expect(onChangeCwd).not.toHaveBeenCalled()
  })
})
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $commandPaletteOpen } from '@/store/command-palette'

import { SidebarBlankState } from './section-states'

const onNewChat = vi.fn<() => void>()
const onNewProject = vi.fn<() => void>()

beforeEach(() => {
  $commandPaletteOpen.set(false)
  onNewChat.mockClear()
  onNewProject.mockClear()
})

afterEach(cleanup)

// The blank state used to dead-end: one "New project" button and nothing else,
// even though the sidebar's primary first action is simply starting a chat.
describe('SidebarBlankState starter actions', () => {
  it('renders the no-sessions line with all three starter rows', () => {
    render(<SidebarBlankState onNewChat={onNewChat} onNewProject={onNewProject} />)

    expect(screen.getByText('No sessions yet')).toBeTruthy()
    expect(screen.getByTestId('blank-start-chat').textContent).toContain('Start a new chat')
    expect(screen.getByTestId('blank-discover-commands').textContent).toContain('Discover commands')
    expect(screen.getByRole('button', { name: /new project/i })).toBeTruthy()
  })

  it('fires the new-chat seam when "Start a new chat" is tapped', () => {
    render(<SidebarBlankState onNewChat={onNewChat} onNewProject={onNewProject} />)

    fireEvent.click(screen.getByTestId('blank-start-chat'))

    expect(onNewChat).toHaveBeenCalledTimes(1)
    expect(onNewProject).not.toHaveBeenCalled()
  })

  it('opens the command palette from the default discover row', () => {
    render(<SidebarBlankState onNewChat={onNewChat} onNewProject={onNewProject} />)

    fireEvent.click(screen.getByTestId('blank-discover-commands'))

    expect($commandPaletteOpen.get()).toBe(true)
  })

  it('honours an injected discover handler', () => {
    const onDiscover = vi.fn<() => void>()
    render(<SidebarBlankState onDiscoverCommands={onDiscover} onNewChat={onNewChat} onNewProject={onNewProject} />)

    fireEvent.click(screen.getByTestId('blank-discover-commands'))

    expect(onDiscover).toHaveBeenCalledTimes(1)
    expect($commandPaletteOpen.get()).toBe(false)
  })

  it('keeps the New project button firing its own seam', () => {
    render(<SidebarBlankState onNewChat={onNewChat} onNewProject={onNewProject} />)

    fireEvent.click(screen.getByRole('button', { name: /new project/i }))

    expect(onNewProject).toHaveBeenCalledTimes(1)
    expect(onNewChat).not.toHaveBeenCalled()
  })
})

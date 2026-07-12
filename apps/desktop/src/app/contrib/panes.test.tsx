import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { setActiveSessionId } from '@/store/session'

import { FilesPane } from './panes'

vi.mock('@/app/right-sidebar', () => ({
  RightSidebarPane: ({ onStartProjectFromFolder }: { onStartProjectFromFolder?: () => void }) =>
    onStartProjectFromFolder ? (
      <button onClick={onStartProjectFromFolder} type="button">
        Open folder
      </button>
    ) : (
      <span>No project action</span>
    )
}))

describe('FilesPane project entry', () => {
  afterEach(() => {
    cleanup()
    setActiveSessionId(null)
  })

  it('forwards the Project-owned folder action for a detached draft', () => {
    setActiveSessionId(null)
    const onStartProjectFromFolder = vi.fn()

    render(<FilesPane onStartProjectFromFolder={onStartProjectFromFolder} />)

    fireEvent.click(screen.getByRole('button', { name: 'Open folder' }))
    expect(onStartProjectFromFolder).toHaveBeenCalledOnce()
  })

  it('withholds the folder action while a session is active', () => {
    setActiveSessionId('runtime-session')

    render(<FilesPane onStartProjectFromFolder={vi.fn()} />)

    expect(screen.queryByRole('button', { name: 'Open folder' })).toBeNull()
  })
})

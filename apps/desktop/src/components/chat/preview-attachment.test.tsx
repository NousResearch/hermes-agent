import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import { PreviewAttachment } from './preview-attachment'

// A delivered file's path arrives in the transcript as model-supplied text, so
// this card is the boundary that decides what may reach the OS.
const LOCAL_FILE = '/home/user/Documents/report.docx'
const UNC_FILE = '\\\\fileserver\\share\\report.docx'
const REVEAL_LABEL = 'Open containing folder'
// The card labels copy 'Copy path' for a local file and plain 'Copy' when the
// path is remote, so match the control by purpose rather than one string.
const COPY_LABEL = /copy/i

describe('PreviewAttachment path actions', () => {
  const revealPath = vi.fn(async () => true)

  beforeEach(() => {
    $connection.set(null)
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { revealPath }
    })
  })

  afterEach(() => {
    cleanup()
    $connection.set(null)
    revealPath.mockClear()
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('reveals a local file in the OS file manager, with copy alongside it', () => {
    render(<PreviewAttachment target={LOCAL_FILE} />)

    fireEvent.click(screen.getByRole('button', { name: REVEAL_LABEL }))

    expect(revealPath).toHaveBeenCalledWith(LOCAL_FILE)
    expect(screen.getByRole('button', { name: COPY_LABEL })).toBeTruthy()
  })

  it('hides reveal — but keeps copy — when the path is not on this machine', () => {
    // Remote gateway: the path lives on the gateway's disk, not this one.
    $connection.set({ mode: 'remote' } as never)

    const remote = render(<PreviewAttachment target={LOCAL_FILE} />)

    expect(screen.queryByRole('button', { name: REVEAL_LABEL })).toBeNull()
    expect(screen.getByRole('button', { name: COPY_LABEL })).toBeTruthy()
    remote.unmount()

    // UNC: resolving it makes Windows dial that host and offer NTLM credentials.
    $connection.set(null)

    render(<PreviewAttachment target={UNC_FILE} />)

    expect(screen.queryByRole('button', { name: REVEAL_LABEL })).toBeNull()
  })
})

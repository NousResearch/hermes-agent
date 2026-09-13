/**
 * Focused tests for the kanban comment file-path linkifier.
 *
 * The @hermes/plugin-sdk host is mocked so revealFileInTree calls are
 * asserted, never really executed against a store.
 */

import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { isAbsoluteFilePath, LinkifiedFilePath } from './filepath-links'

const { hostMock } = vi.hoisted(() => ({
  hostMock: { revealFileInTree: vi.fn() }
}))

vi.mock('@hermes/plugin-sdk', () => ({
  host: hostMock
}))

/** Render the component and return its accessible text content. */
function renderText(text: string): string {
  const { container } = render(<LinkifiedFilePath text={text} />)
  return container.textContent ?? ''
}

/** Click every rendered file-path button and return the paths passed to the host. */
function linkLabels(text: string): string[] {
  render(<LinkifiedFilePath text={text} />)
  return screen
    .queryAllByRole('button')
    .map(button => button.getAttribute('aria-label') ?? '')
    .filter(label => label.startsWith('Reveal '))
    .map(label => label.replace(/^Reveal /, '').replace(/ in file tree$/, ''))
}

describe('isAbsoluteFilePath', () => {
  it('accepts POSIX absolute paths', () => {
    expect(isAbsoluteFilePath('/opt/data/skills/x/SKILL.md')).toBe(true)
    expect(isAbsoluteFilePath('/opt/hermes/agent.log')).toBe(true)
    expect(isAbsoluteFilePath('/opt/data/receipt-processing/')).toBe(true)
    expect(isAbsoluteFilePath('/')).toBe(false)
  })

  it('accepts Windows absolute paths', () => {
    expect(isAbsoluteFilePath('C:/Users/me/src/a.ts')).toBe(true)
    expect(isAbsoluteFilePath('C:\\Users\\me\\src\\a.ts')).toBe(true)
    expect(isAbsoluteFilePath('\\\\server\\share\\file.md')).toBe(true)
  })

  it('rejects relative paths and bare words', () => {
    expect(isAbsoluteFilePath('src/a.ts')).toBe(false)
    expect(isAbsoluteFilePath('SKILL.md')).toBe(false)
    expect(isAbsoluteFilePath('Updated')).toBe(false)
  })
})

describe('LinkifiedFilePath', () => {
  beforeEach(() => {
    hostMock.revealFileInTree.mockClear()
  })

  afterEach(() => {
    cleanup()
  })

  it('renders plain text unchanged when there are no absolute paths', () => {
    expect(renderText('Updated the worker to v1.4.1')).toBe('Updated the worker to v1.4.1')
  })

  it('wraps an absolute POSIX path in a reveal link', () => {
    const labels = linkLabels('Updated `/opt/data/skills/x/SKILL.md` to v1.4.1')
    expect(labels).toContain('/opt/data/skills/x/SKILL.md')
  })

  it('does not linkify a relative path', () => {
    const labels = linkLabels('see skills/x/SKILL.md for details')
    expect(labels).not.toContain('skills/x/SKILL.md')
  })

  it('renders every file-path link as a clickable button', () => {
    render(<LinkifiedFilePath text="Updated `/opt/data/skills/x/SKILL.md`" />)
    expect(screen.getByRole('button')).toBeTruthy()
  })

  it('does not call revealFileInTree at render time', () => {
    render(<LinkifiedFilePath text="Updated `/opt/data/skills/x/SKILL.md`" />)
    expect(hostMock.revealFileInTree).not.toHaveBeenCalled()
  })

  it('stops the path at sentence punctuation, not mid-filename', () => {
    const labels = linkLabels('see /opt/data/file.sh! and /opt/data/file.tar.gz.')
    expect(labels).toContain('/opt/data/file.sh')
    expect(labels).toContain('/opt/data/file.tar.gz')
  })

  it('links a directory reference ending in a slash', () => {
    const labels = linkLabels('Everything in /opt/data/receipt-processing/ now')
    expect(labels).toContain('/opt/data/receipt-processing/')
  })

  it('links Windows drive and UNC paths', () => {
    const labels = linkLabels('edit C:/Users/me/src/a.ts and \\\\server\\share\\file.md')
    expect(labels).toContain('C:/Users/me/src/a.ts')
    expect(labels).toContain('\\\\server\\share\\file.md')
  })

  it('does not linkify a bare slash or a leading double-slash', () => {
    expect(linkLabels('/ alone')).toEqual([])
    expect(linkLabels('see //share/x.ts')).toEqual([])
  })
})

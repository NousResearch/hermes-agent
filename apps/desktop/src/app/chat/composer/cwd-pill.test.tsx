import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { CwdPill, folderName } from './cwd-pill'

describe('folderName', () => {
  it('returns the last path segment of a POSIX path', () => {
    expect(folderName('/Users/yates/code/hermes-agent')).toBe('hermes-agent')
  })

  it('returns the last path segment of a Windows path', () => {
    expect(folderName('C:\\Users\\yates\\projects\\app')).toBe('app')
  })

  it('returns the segment for a relative path', () => {
    expect(folderName('code/hermes-agent')).toBe('hermes-agent')
  })

  it('returns the input verbatim when there is no separator', () => {
    expect(folderName('my-folder')).toBe('my-folder')
  })

  it('returns empty string for empty / whitespace-only input', () => {
    expect(folderName('')).toBe('')
    expect(folderName('   ')).toBe('')
  })

  it('returns the last non-empty segment for trailing slashes', () => {
    expect(folderName('/Users/yates/code/hermes-agent/')).toBe('hermes-agent')
  })
})

describe('CwdPill', () => {
  it('renders the folder basename', () => {
    render(<CwdPill cwd="/Users/yates/code/hermes-agent" />)

    expect(screen.getByText('hermes-agent')).toBeTruthy()
  })

  it('renders nothing when cwd is null', () => {
    const { container } = render(<CwdPill cwd={null} />)

    expect(container.textContent).toBe('')
  })

  it('renders nothing when cwd is undefined', () => {
    const { container } = render(<CwdPill cwd={undefined} />)

    expect(container.textContent).toBe('')
  })

  it('renders nothing when cwd is an empty string', () => {
    const { container } = render(<CwdPill cwd="" />)

    expect(container.textContent).toBe('')
  })

  it('renders the folder icon', () => {
    const { container } = render(<CwdPill cwd="/home/user/project" />)

    expect(container.querySelector('svg')).toBeTruthy()
  })
})

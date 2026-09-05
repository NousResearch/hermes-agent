import { beforeEach, describe, expect, it } from 'vitest'

import { consumeWorkspaceChange, notifyWorkspaceChanged, notifyWorkspaceDirectoryChanged } from './workspace-events'

beforeEach(() => {
  consumeWorkspaceChange()
})

describe('notifyWorkspaceChanged', () => {
  it('preserves native Windows separators in targeted refresh directories', () => {
    notifyWorkspaceChanged('C:\\repo\\docs\\child')

    expect(consumeWorkspaceChange()).toEqual({ dirs: ['C:\\repo\\docs'], full: false })
  })

  it('preserves POSIX separators in targeted refresh directories', () => {
    notifyWorkspaceChanged('/repo/docs/child')

    expect(consumeWorkspaceChange()).toEqual({ dirs: ['/repo/docs'], full: false })
  })

  it.each([
    ['/child', '/'],
    ['C:\\child', 'C:\\']
  ])('keeps filesystem roots intact for %s', (changedPath, expectedDirectory) => {
    notifyWorkspaceChanged(changedPath)

    expect(consumeWorkspaceChange()).toEqual({ dirs: [expectedDirectory], full: false })
  })

  it('queues a known WSL display directory without deriving its parent', () => {
    notifyWorkspaceDirectoryChanged('/home/alex/repo/')

    expect(consumeWorkspaceChange()).toEqual({ dirs: ['/home/alex/repo'], full: false })
  })

  it('preserves valid trailing spaces in a known POSIX directory', () => {
    notifyWorkspaceDirectoryChanged('/repo/trailing ')

    expect(consumeWorkspaceChange()).toEqual({ dirs: ['/repo/trailing '], full: false })
  })
})

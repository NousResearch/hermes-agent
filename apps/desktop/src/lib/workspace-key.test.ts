import { describe, expect, it } from 'vitest'

import { normalizeWorkspacePath, workspaceKey } from './workspace-key'

describe('workspaceKey', () => {
  it('groups equivalent folder paths', () => {
    expect(workspaceKey('/home/user/project/')).toBe(workspaceKey('/home/user/project'))
    expect(workspaceKey('C:\\Users\\Dev\\project\\')).toBe(workspaceKey('c:/Users/Dev/project'))
  })

  it('preserves case-sensitive POSIX paths', () => {
    expect(normalizeWorkspacePath('/work/Project')).not.toBe(normalizeWorkspacePath('/work/project'))
  })
})

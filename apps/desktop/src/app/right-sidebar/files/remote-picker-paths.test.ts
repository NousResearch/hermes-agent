import { describe, expect, it } from 'vitest'

import { buildRemotePathCrumbs } from './remote-picker-paths'

describe('buildRemotePathCrumbs', () => {
  it('keeps the leading slash for POSIX absolute paths', () => {
    expect(buildRemotePathCrumbs('/home/me/proj')).toEqual([
      { label: '/', path: '/' },
      { label: 'home', path: '/home' },
      { label: 'me', path: '/home/me' },
      { label: 'proj', path: '/home/me/proj' }
    ])
  })

  it('keeps drive roots and UNC prefixes navigable', () => {
    expect(buildRemotePathCrumbs('C:\\Users\\me')).toEqual([
      { label: 'C:\\', path: 'C:\\' },
      { label: 'Users', path: 'C:\\Users' },
      { label: 'me', path: 'C:\\Users\\me' }
    ])
    expect(buildRemotePathCrumbs('\\\\server\\share\\folder').at(2)).toEqual({
      label: 'share',
      path: '\\\\server\\share'
    })
  })
})

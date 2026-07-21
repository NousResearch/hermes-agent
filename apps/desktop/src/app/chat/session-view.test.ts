import { describe, expect, it } from 'vitest'

import { PRIMARY_SESSION_VIEW } from './session-view'

describe('PRIMARY_SESSION_VIEW', () => {
  it('identifies the permanent chat pane by pane id', () => {
    expect(PRIMARY_SESSION_VIEW.paneId).toBe('workspace')
  })
})

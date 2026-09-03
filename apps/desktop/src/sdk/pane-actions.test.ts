import { describe, expect, it, vi } from 'vitest'

import * as tree from '@/components/pane-shell/tree/store'

import { host } from './index'

describe('host pane actions', () => {
  it('reveals a plugin pane by contribution-scoped id', () => {
    const spy = vi.spyOn(tree, 'revealTreePane')
    host.revealPane('browser-annotator:browser-annotator')
    expect(spy).toHaveBeenCalledWith('browser-annotator:browser-annotator')
  })
})

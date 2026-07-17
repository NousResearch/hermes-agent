import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/hermes'
import { sessionActivityKey } from '@/store/session-activity'

import { isSidebarSessionSelected, isSidebarSessionWorking, sidebarSessionScopeKey } from './session-row-state'

const session = (profile: string, lineageRoot?: string): SessionInfo =>
  ({ _lineage_root_id: lineageRoot, id: 'same', profile }) as SessionInfo

describe('sidebar session row scope', () => {
  it('keeps selected and working state isolated when raw ids collide', () => {
    const alpha = session('alpha')
    const beta = session('beta')
    const working = new Set([sessionActivityKey('beta', 'same')])

    expect(sidebarSessionScopeKey(alpha)).not.toBe(sidebarSessionScopeKey(beta))
    expect(isSidebarSessionSelected(alpha, 'same', 'alpha')).toBe(true)
    expect(isSidebarSessionSelected(beta, 'same', 'alpha')).toBe(false)
    expect(isSidebarSessionWorking(alpha, working)).toBe(false)
    expect(isSidebarSessionWorking(beta, working)).toBe(true)
  })

  it('projects lineage-root work only within the owning profile', () => {
    const alpha = session('alpha', 'root')
    const beta = session('beta', 'root')
    const working = new Set([sessionActivityKey('alpha', 'root')])

    expect(isSidebarSessionWorking(alpha, working)).toBe(true)
    expect(isSidebarSessionWorking(beta, working)).toBe(false)
  })
})

import type { BrowserWindow } from 'electron'
import { describe, expect, it, vi } from 'vitest'

import type { PlayerAgentOperation, PlayerAgentScope, PlayerMediaTarget } from '../../src/lib/player-context-contract'

import type { PlayerContextRegistry } from './context-capabilities'
import { PlayerManager, type PlayerMediaJobHost } from './manager'
import type { AwarenessLink, Slot } from './manager-state'

vi.mock('electron', () => ({
  app: { getLocale: () => 'en-US' },
  components: undefined,
  BrowserWindow: class {},
  dialog: {},
  Menu: {},
  screen: {},
  session: {},
  WebContentsView: class {}
}))

const scope: PlayerAgentScope = { connectionId: 'local', profileId: 'default', sessionId: 'session-1' }

const target: PlayerMediaTarget = {
  source: 'youtube',
  videoId: 'abcdefghijk',
  canonicalUrl: 'https://www.youtube.com/watch?v=abcdefghijk',
  title: null,
  capturedTimeSeconds: 12,
  adStatus: 'unknown'
}

describe('PlayerManager media job routing', () => {
  it.each([
    { type: 'transcript.start', generation: 1, stale: true },
    { type: 'download.start', generation: 1, stale: true },
    { type: 'transcript.start', generation: 2, stale: false },
    { type: 'download.start', generation: 2, stale: false }
  ] as const)(
    '$type with captured generation $generation (stale: $stale)',
    async ({ type, generation, stale }) => {
      const start = vi.fn(async () => ({ id: 'job_1' }))

      const mediaJobs = {
        capabilities: () => ({ available: true, reason: null }),
        start
      } as unknown as PlayerMediaJobHost

      const manager = new PlayerManager({ mediaJobs })
      const owner = { id: 7, isDestroyed: () => false } as unknown as BrowserWindow
      const link: AwarenessLink = { ownerId: 7, linkId: 'active-link', scope }

      const slot = {
        ownerId: 7,
        status: 'ready',
        navigationGeneration: 2,
        guest: { webContents: { isDestroyed: () => false } }
      } as unknown as Slot

      const internals = manager as unknown as {
        slot: Slot
        awarenessLink: AwarenessLink
        capturedContexts: PlayerContextRegistry
      }

      internals.slot = slot
      internals.awarenessLink = link

      const { token } = internals.capturedContexts.issue({
        ownerId: owner.id,
        linkId: link.linkId,
        scope,
        generation,
        target
      })

      const operation: PlayerAgentOperation =
        type === 'transcript.start'
          ? { type, contextToken: token, language: 'en' }
          : { type, contextToken: token, quality: 'best' }

      const result = await manager.agentRequest(owner, { scope, operation })

      if (stale) {
        expect(result.error?.code).toBe('player-context-stale')
        expect(start).not.toHaveBeenCalled()
      } else {
        expect(result.job?.id).toBe('job_1')
        expect(start).toHaveBeenCalledOnce()
        expect(start.mock.calls[0]).toEqual([
          owner, scope, target,
          type === 'transcript.start'
            ? { kind: 'transcript', language: 'en', range: undefined }
            : { kind: 'download', quality: 'best' }
        ])
      }
    }
  )
})

import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { FederatedSessionIndexStore } from './federated-session-index'
import { createRegistrySessionEnumeration } from './registry-session-enumeration'
import { ENUMERATION_INITIAL_DELAY_MS, STAGGER_MS } from './registry-session-enumerator'

const NOW_SECONDS = Math.floor(Date.now() / 1000)

const connections = [
  { id: 'local', kind: 'local' },
  { id: 'homelab', kind: 'ssh' },
  { id: 'cloud', kind: 'cloud' }
]

const sessionRow = (id: string, profile = 'default') => ({
  id,
  last_active: NOW_SECONDS,
  message_count: 3,
  profile,
  source: 'cli',
  started_at: NOW_SECONDS
})

describe('createRegistrySessionEnumeration', () => {
  let dir: string

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'hermes-enumeration-'))
  })

  afterEach(() => {
    vi.useRealTimers()
    rmSync(dir, { recursive: true, force: true })
  })

  function build(overrides: Partial<Parameters<typeof createRegistrySessionEnumeration>[0]> = {}) {
    const dial = vi.fn(async (connectionId: string) => ({ connectionId }))
    const getJson = vi.fn<(descriptor: unknown, path: string, timeoutMs: number) => Promise<unknown>>()

    const enumeration = createRegistrySessionEnumeration({
      dial,
      getJson,
      readRegistry: () => ({ connections, primary: 'local' }),
      userDataDir: () => dir,
      ...overrides
    })

    return { dial, enumeration, getJson }
  }

  describe('fetchSessions', () => {
    it('reads a shared host once across every profile and tags each row with its connection', async () => {
      const { dial, enumeration, getJson } = build()

      getJson.mockResolvedValue({
        profile_totals: { default: 1, omer: 1 },
        sessions: [sessionRow('a'), sessionRow('b', 'omer')],
        total: 2
      })

      const result = await enumeration.fetchSessions('cloud')

      expect(dial).toHaveBeenCalledWith('cloud')
      expect(String(getJson.mock.calls[0][1])).toMatch(/^\/api\/profiles\/sessions\?/)
      expect(String(getJson.mock.calls[0][1])).toContain('profile=all')
      expect(String(getJson.mock.calls[0][1])).toContain('archived=exclude')
      expect(result?.sessions).toEqual([
        expect.objectContaining({ connection_id: 'cloud', id: 'a', profile: 'default' }),
        expect.objectContaining({ connection_id: 'cloud', id: 'b', profile: 'omer' })
      ])
    })

    it('reads an ssh connection from its own default-profile backend', async () => {
      const { enumeration, getJson } = build()

      getJson.mockResolvedValue({ sessions: [sessionRow('a')], total: 1 })

      const result = await enumeration.fetchSessions('homelab')

      expect(String(getJson.mock.calls[0][1])).toMatch(/^\/api\/sessions\?/)
      expect(result?.sessions).toEqual([expect.objectContaining({ connection_id: 'homelab', profile: 'default' })])
    })

    it('treats an empty read that reports a total as a successful read (the host really is empty)', async () => {
      const { enumeration, getJson } = build()

      getJson.mockResolvedValue({ profile_totals: { default: 0 }, sessions: [], total: 0 })

      expect(await enumeration.fetchSessions('cloud')).toEqual({ sessions: [] })
    })

    it('treats an empty read with no total as a failure, so it can never clear remembered rows', async () => {
      const { enumeration, getJson } = build()

      getJson.mockResolvedValue({ sessions: [] })

      expect(await enumeration.fetchSessions('cloud')).toBeNull()
    })

    it('returns null when the backend read fails', async () => {
      const { enumeration, getJson } = build()

      getJson.mockRejectedValue(new Error('502'))

      expect(await enumeration.fetchSessions('cloud')).toBeNull()
    })

    it('returns null when the dial fails', async () => {
      const { enumeration, getJson } = build({
        dial: async () => {
          throw new Error('tunnel down')
        }
      })

      expect(await enumeration.fetchSessions('homelab')).toBeNull()
      expect(getJson).not.toHaveBeenCalled()
    })

    it('does not dial a connection the registry does not hold', async () => {
      const { dial, enumeration } = build()

      expect(await enumeration.fetchSessions('ghost')).toBeNull()
      expect(dial).not.toHaveBeenCalled()
    })
  })

  describe('coldRows', () => {
    const seed = () => {
      const store = new FederatedSessionIndexStore(dir)

      store.remember('homelab', { sessions: [{ ...sessionRow('h1'), connection_id: 'homelab' }] })
      store.remember('cloud', { sessions: [{ ...sessionRow('c1'), connection_id: 'cloud' }] })
    }

    it('backfills only connections with no readable backend', () => {
      seed()
      const { enumeration } = build()

      const rows = enumeration.coldRows(new URLSearchParams(), new Set(['cloud']))

      expect(rows.map(row => row.id)).toEqual(['h1'])
    })

    it('backfills nothing while every remote connection is readable', () => {
      seed()
      const { enumeration } = build()

      expect(enumeration.coldRows(new URLSearchParams(), new Set(['homelab', 'cloud']))).toEqual([])
    })

    it('never backfills the registry primary: the primary aggregate carries its rows', () => {
      seed()
      const { enumeration } = build({ readRegistry: () => ({ connections, primary: 'homelab' }) })

      expect(enumeration.coldRows(new URLSearchParams(), new Set()).map(row => row.id)).toEqual(['c1'])
    })

    it('applies the request filters', () => {
      seed()
      const { enumeration } = build()

      expect(enumeration.coldRows(new URLSearchParams('source=cron'), new Set())).toEqual([])
    })

    it('is empty before anything has been indexed', () => {
      const { enumeration } = build()

      expect(enumeration.coldRows(new URLSearchParams(), new Set())).toEqual([])
    })
  })

  it('a started enumerator indexes a cold connection after its initial delay, and that survives a restart', async () => {
    vi.useFakeTimers()

    const { enumeration, getJson } = build()

    getJson.mockImplementation(async () => ({ sessions: [sessionRow('c1')], total: 1 }))
    enumeration.start()

    await vi.advanceTimersByTimeAsync(ENUMERATION_INITIAL_DELAY_MS - 1)
    expect(getJson).not.toHaveBeenCalled()

    await vi.advanceTimersByTimeAsync(1)
    expect(getJson).toHaveBeenCalled()

    // The next connection's pass follows one stagger later.
    await vi.advanceTimersByTimeAsync(STAGGER_MS)
    expect(getJson).toHaveBeenCalledTimes(2)
    enumeration.stop()

    // A fresh process (new enumeration object) reads the same file.
    const restarted = build().enumeration

    expect(
      restarted
        .coldRows(new URLSearchParams(), new Set())
        .map(row => `${row.connection_id}:${row.id}`)
        .sort()
    ).toEqual(['cloud:c1', 'homelab:c1'])
  })

  it('stop() prevents any further pass', async () => {
    vi.useFakeTimers()

    const { enumeration, getJson } = build()

    enumeration.start()
    enumeration.stop()
    await vi.advanceTimersByTimeAsync(ENUMERATION_INITIAL_DELAY_MS * 4)

    expect(getJson).not.toHaveBeenCalled()
  })
})

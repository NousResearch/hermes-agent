import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const recordImprint = vi.fn(async () => ({ ok: true, recorded: true }))
const listImprints = vi.fn(async () => ({ enabled: true, imprints: [] as { message_id: string; valence: 'up' | 'down' }[] }))
const notifyError = vi.fn()

vi.mock('@/hermes', () => ({ recordImprint, listImprints }))
vi.mock('@/store/notifications', () => ({ notifyError }))

const {
  $imprints,
  $imprintsEnabled,
  applyImprintsEnabledFromConfig,
  hydrateImprints,
  imprintFor,
  toggleImprint
} = await import('./imprints')

beforeEach(() => {
  $imprints.set({})
  $imprintsEnabled.set(true)
  recordImprint.mockClear()
  recordImprint.mockResolvedValue({ ok: true, recorded: true })
  listImprints.mockClear()
  notifyError.mockClear()
})

afterEach(() => {
  $imprints.set({})
})

describe('imprints store', () => {
  it('reads the enabled flag from config, defaulting on when absent', () => {
    applyImprintsEnabledFromConfig({ memory: { imprints_enabled: false } })
    expect($imprintsEnabled.get()).toBe(false)

    applyImprintsEnabledFromConfig({ memory: {} })
    expect($imprintsEnabled.get()).toBe(true)

    applyImprintsEnabledFromConfig(null)
    expect($imprintsEnabled.get()).toBe(true)
  })

  it('records a thumb optimistically and sends the valence', async () => {
    await toggleImprint('m1', 'up', { excerpt: 'a tight answer' })

    expect(imprintFor('m1')).toBe('up')
    expect(recordImprint).toHaveBeenCalledWith({
      messageId: 'm1',
      valence: 'up',
      excerpt: 'a tight answer',
      sessionId: undefined
    })
  })

  it('tapping the active thumb again clears it', async () => {
    await toggleImprint('m1', 'up')
    await toggleImprint('m1', 'up')

    expect(imprintFor('m1')).toBeUndefined()
    expect(recordImprint).toHaveBeenLastCalledWith(expect.objectContaining({ valence: 'none' }))
  })

  it('flips from up to down', async () => {
    await toggleImprint('m1', 'up')
    await toggleImprint('m1', 'down')

    expect(imprintFor('m1')).toBe('down')
    expect(recordImprint).toHaveBeenLastCalledWith(expect.objectContaining({ valence: 'down' }))
  })

  it('reverts the optimistic update when the write fails', async () => {
    recordImprint.mockRejectedValueOnce(new Error('offline'))
    await toggleImprint('m1', 'up')

    expect(imprintFor('m1')).toBeUndefined()
    expect(notifyError).toHaveBeenCalled()
  })

  it('hydrates the map and enabled flag from the backend', async () => {
    listImprints.mockResolvedValueOnce({
      enabled: false,
      imprints: [
        { message_id: 'a', valence: 'up' },
        { message_id: 'b', valence: 'down' }
      ]
    })

    await hydrateImprints()

    expect($imprintsEnabled.get()).toBe(false)
    expect($imprints.get()).toEqual({ a: 'up', b: 'down' })
  })
})

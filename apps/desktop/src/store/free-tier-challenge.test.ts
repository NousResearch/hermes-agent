import { afterEach, describe, expect, it, vi } from 'vitest'

import { runFreeTierChallenge } from './free-tier-challenge'

const challenge = {
  type: 'browser',
  url: 'https://portal.nousresearch.com/challenge?code=abc',
  required: true,
  expires_in: 600,
  message: 'A quick check first.'
}

function installBridge(run: (request: unknown) => Promise<string>) {
  ;(window as unknown as { hermesDesktop?: unknown }).hermesDesktop = { freeTierChallenge: { run } }
}

afterEach(() => {
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('runFreeTierChallenge', () => {
  it('reports the matching attempt outcome without treating the host as an auth authority', async () => {
    const report = vi.fn().mockResolvedValue({ accepted: true })
    const run = vi.fn().mockResolvedValue('error')

    installBridge(run)
    await expect(runFreeTierChallenge({ ...challenge, attempt: 3 }, report)).resolves.toBe('error')
    expect(run).toHaveBeenCalledWith(expect.objectContaining({ attempt: 3 }))
    expect(report).toHaveBeenCalledWith('free_tier.challenge_result', { url: challenge.url, attempt: 3, outcome: 'error' })
  })
  it('hands a browser challenge to the main process', async () => {
    const run = vi.fn().mockResolvedValue('done')

    installBridge(run)

    await expect(runFreeTierChallenge(challenge)).resolves.toBe('done')
    expect(run).toHaveBeenCalledWith({ url: challenge.url, required: true, expiresIn: 600, attempt: 0 })
  })

  it('runs one window per URL while it is in flight, reports it once, then allows it again', async () => {
    let settle: (outcome: string) => void = () => {}
    const run = vi.fn().mockImplementation(() => new Promise<string>(resolve => (settle = resolve)))
    const report = vi.fn().mockResolvedValue({ accepted: true })

    installBridge(run)

    const first = runFreeTierChallenge(challenge, report)
    const second = runFreeTierChallenge(challenge, report)

    expect(run).toHaveBeenCalledTimes(1)
    settle('done')
    await expect(Promise.all([first, second])).resolves.toEqual(['done', 'done'])
    // The event and the status read both asked; the backend hears about it once.
    expect(report).toHaveBeenCalledTimes(1)

    void runFreeTierChallenge(challenge)
    expect(run).toHaveBeenCalledTimes(2)
  })

  it('a backend that rejects the report (older gateway, dropped socket) does not change the outcome', async () => {
    installBridge(vi.fn().mockResolvedValue('done'))
    const report = vi.fn().mockRejectedValue(new Error('Method not found'))

    await expect(runFreeTierChallenge({ ...challenge, url: `${challenge.url}5` }, report)).resolves.toBe('done')
  })

  it('passes an optional challenge through as not-required', async () => {
    const run = vi.fn().mockResolvedValue('failed')

    installBridge(run)
    await runFreeTierChallenge({ ...challenge, url: `${challenge.url}2`, required: false })
    expect(run).toHaveBeenCalledWith(expect.objectContaining({ required: false }))
  })

  it('ignores anything that is not a browser challenge', () => {
    installBridge(vi.fn())
    expect(runFreeTierChallenge(undefined)).toBeNull()
    expect(runFreeTierChallenge(null)).toBeNull()
    expect(runFreeTierChallenge({ type: 'attestation', url: 'https://x.test' })).toBeNull()
    expect(runFreeTierChallenge({ type: 'browser' })).toBeNull()
  })

  it('reports unsupported when no shell can host the page, and an IPC failure as an error', async () => {
    await expect(runFreeTierChallenge({ ...challenge, url: `${challenge.url}3` })).resolves.toBe('unsupported')

    installBridge(vi.fn().mockRejectedValue(new Error('ipc down')))
    await expect(runFreeTierChallenge({ ...challenge, url: `${challenge.url}4` })).resolves.toBe('error')
  })
})

/**
 * The agent-to-agent messaging protocol section every SOUL has to keep.
 *
 * #16: a CUSTOM SOUL silently dropped the handoff protocol, which broke
 * @mentions for every customized bot. The protocol is therefore appended
 * idempotently — at create time, on save, and as a one-shot backfill for the
 * pre-Bot-Mode personas (`default` above all) that never went through
 * `composeSoul`.
 *
 * The escape hatch is the backend capability: once `profiles.list` reports
 * `bot_mode_protocol`, the gateway injects the protocol into every session's
 * system prompt and Bot Mode must stop writing to user SOUL files entirely —
 * a second copy in SOUL.md is duplicated prompt the user did not ask for.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { RosterRow } from './types'

const { hostMock, serverInjects } = vi.hoisted(() => ({
  hostMock: { request: vi.fn() },
  serverInjects: { value: false }
}))

vi.mock('@hermes/plugin-sdk', () => ({ host: hostMock }))

vi.mock('./data', () => ({
  botHandle: (name: string) => (String(name).toLowerCase() === 'default' ? 'hermes' : name),
  // A live binding in the real module — useRoster flips it from the roster
  // response, so the mock has to be a getter too.
  get serverInjectsProtocol() {
    return serverInjects.value
  }
}))

const EXISTING_SOUL = `# Main agent

I am the default profile on this machine.

## Notes
- Execute directly.
`

const roster = [
  { description: 'main agent', name: 'default' },
  { description: 'research specialist', name: 'researcher' }
] as RosterRow[]

const sectionCount = (soul: string) => soul.split('## Messaging other agents').length - 1

async function loadSoul() {
  vi.resetModules()

  return import('./soul')
}

beforeEach(() => {
  vi.clearAllMocks()
  serverInjects.value = false
})

describe('appending the protocol to an existing SOUL', () => {
  it('appends once and never duplicates', async () => {
    const { ensureMessagingProtocol } = await loadSoul()
    const once = ensureMessagingProtocol(EXISTING_SOUL, 'default', roster)

    expect(sectionCount(once)).toBe(1)
    // Identity text is never overwritten.
    expect(once).toMatch(/I am the default profile on this machine/)
    expect(once).toMatch(/`researcher` — research specialist/)
    // The primary profile addresses itself by its callable alias.
    expect(once).toMatch(/@hermes/)
    expect(once).not.toMatch(/@default/)

    const twice = ensureMessagingProtocol(once, 'default', roster)

    expect(twice).toBe(once.trim())
    expect(sectionCount(twice)).toBe(1)
  })

  it('points at the real CLI verb, not the plural that does not exist', async () => {
    const { ensureMessagingProtocol } = await loadSoul()
    const soul = ensureMessagingProtocol('', 'default', roster)

    expect(soul).toMatch(/run `hermes profile list` for the LIVE/)
    expect(soul).not.toMatch(/hermes profiles list/)
  })

  it('seeds an empty SOUL with the section alone', async () => {
    const { ensureMessagingProtocol } = await loadSoul()

    expect(ensureMessagingProtocol('', 'ops', [])).toMatch(/^## Messaging other agents/)
    expect(ensureMessagingProtocol('', 'ops', [])).toMatch(/- \(none yet\)/)
  })
})

describe('composeSoul', () => {
  it('does not double-append when the custom SOUL already carries the protocol', async () => {
    const { composeSoul } = await loadSoul()

    const withProtocol = composeSoul({
      customSoul: '',
      description: 'literature review',
      name: 'researcher',
      roster,
      title: 'Researcher'
    })

    const cloned = composeSoul({ customSoul: withProtocol, name: 'researcher', roster })

    expect(sectionCount(cloned)).toBe(1)
  })
})

describe('one-shot backfill for profiles that predate the protocol', () => {
  it('describes every bot but only configures the ones missing the section', async () => {
    const calls: Array<{ method: string; payload: Record<string, string> }> = []

    hostMock.request.mockImplementation(async (method: string, payload: Record<string, string>) => {
      calls.push({ method, payload })

      if (method === 'profiles.describe') {
        return payload.name === 'researcher'
          ? { soul: '# Researcher\n\n## Messaging other agents\n' }
          : { soul: EXISTING_SOUL }
      }

      return { ok: true }
    })

    const { backfillMessagingProtocol } = await loadSoul()

    backfillMessagingProtocol(roster)
    await vi.waitFor(() => expect(calls.some(call => call.method === 'profiles.configure')).toBe(true))

    expect(
      calls
        .filter(call => call.method === 'profiles.describe')
        .map(call => call.payload.name)
        .sort()
    ).toEqual(['default', 'researcher'])

    const configures = calls.filter(call => call.method === 'profiles.configure')

    expect(configures).toHaveLength(1)
    expect(configures[0].payload.name).toBe('default')
    expect(configures[0].payload.soul).toMatch(/## Messaging other agents/)
  })

  it('does not hammer a gateway that fails the describe', async () => {
    hostMock.request.mockRejectedValue(new Error('older gateway'))

    const { backfillMessagingProtocol } = await loadSoul()

    backfillMessagingProtocol(roster)
    await vi.waitFor(() => expect(hostMock.request).toHaveBeenCalledTimes(2))

    backfillMessagingProtocol(roster)
    await vi.waitFor(() => expect(hostMock.request).toHaveBeenCalledTimes(2))
  })
})

describe('the bot_mode_protocol backend capability suppresses every SOUL write', () => {
  it('turns ensureMessagingProtocol into identity', async () => {
    serverInjects.value = true

    const { ensureMessagingProtocol } = await loadSoul()

    expect(ensureMessagingProtocol(EXISTING_SOUL, 'default', [])).toBe(EXISTING_SOUL.trim())
  })

  it('issues no RPC at all from the backfill', async () => {
    serverInjects.value = true

    const { backfillMessagingProtocol } = await loadSoul()

    backfillMessagingProtocol(roster)
    await Promise.resolve()

    expect(hostMock.request).not.toHaveBeenCalled()
  })

  it('leaves the section out of a generated identity SOUL', async () => {
    serverInjects.value = true

    const { composeSoul } = await loadSoul()

    expect(composeSoul({ customSoul: '', description: 'D', name: 'newbot', roster: [], title: 'T' })).not.toMatch(
      /## Messaging other agents/
    )
  })
})

import { beforeEach, describe, expect, it } from 'vitest'

import { $connection } from '@/store/session'

import { agentMachineLabel, isAgentOnAnotherMachine, withAgentLocality } from './agent-locality'

const connection = (extra: Record<string, unknown>) => ({ baseUrl: '', token: '', wsUrl: '', ...extra }) as never

const WINDOW = { window: { app: 'Figma', title: '' }, platform: 'darwin' }

beforeEach(() => $connection.set(null))

describe('isAgentOnAnotherMachine', () => {
  it('is false on a local backend and with no connection yet', () => {
    expect(isAgentOnAnotherMachine(null)).toBe(false)
    expect(isAgentOnAnotherMachine(connection({ mode: 'local' }))).toBe(false)
  })

  it('is true for every remote shape, including a tunnelled SSH loopback URL', () => {
    for (const remoteKind of ['ssh', 'url', 'cloud'] as const) {
      expect(
        isAgentOnAnotherMachine(connection({ baseUrl: 'http://127.0.0.1:41001', mode: 'remote', remoteKind }))
      ).toBe(true)
    }
  })
})

describe('agentMachineLabel', () => {
  it('is empty when the agent is already on this machine', () => {
    expect(agentMachineLabel(connection({ mode: 'local', remoteHost: 'ignored' }))).toBe('')
  })

  it('prefers the stable SSH identity over the forwarded loopback port', () => {
    const label = agentMachineLabel(
      connection({
        baseUrl: 'http://127.0.0.1:41001',
        mode: 'remote',
        remoteHost: 'remote-box',
        remoteIdentity: 'operator@remote-box',
        remoteKind: 'ssh'
      })
    )

    expect(label).toBe('operator@remote-box')
  })

  it('names cloud and URL backends', () => {
    expect(agentMachineLabel(connection({ mode: 'remote', remoteKind: 'cloud' }))).toBe('Hermes Cloud')
    expect(
      agentMachineLabel(connection({ baseUrl: 'https://nas.local:9119', mode: 'remote', remoteKind: 'url' }))
    ).toBe('nas.local:9119')
  })

  it('always says something, even for a remote with no identifying fields', () => {
    expect(agentMachineLabel(connection({ mode: 'remote' }))).toBe('the connected backend')
  })
})

describe('withAgentLocality', () => {
  it('adds nothing on a local session, so the common case costs no tokens', () => {
    expect(withAgentLocality(WINDOW, connection({ mode: 'local' }))).toEqual(WINDOW)
  })

  it('flags the gap on a remote session without leaking where the backend is', () => {
    const located = withAgentLocality(
      WINDOW,
      connection({ mode: 'remote', remoteHost: 'remote-box', remoteIdentity: 'operator@remote-box', remoteKind: 'ssh' })
    )

    expect(located).toEqual({ ...WINDOW, agent_on_this_machine: false })
    expect(JSON.stringify(located)).not.toContain('remote-box')
  })

  it('leaves an unavailable answer alone so the tool still reports enumeration failure', () => {
    const remote = connection({ mode: 'remote' })

    expect(withAgentLocality(null, remote)).toBeNull()
    expect(withAgentLocality('', remote)).toBe('')
  })

  it('reads the live connection when none is passed', () => {
    $connection.set(connection({ mode: 'remote' }))

    expect(withAgentLocality(WINDOW)).toEqual({ ...WINDOW, agent_on_this_machine: false })
  })
})

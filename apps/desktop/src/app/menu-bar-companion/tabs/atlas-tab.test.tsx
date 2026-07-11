// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const stores = vi.hoisted(() => ({ activeProfile: null as any }))
const broadcastDesktopStateChange = vi.fn()
const listAllProfileSessions = vi.fn()
const getSkills = vi.fn()
const toggleSkill = vi.fn()
const listMcpServers = vi.fn()
const setMcpServerEnabled = vi.fn()
const getCronJobs = vi.fn()
const pauseCronJob = vi.fn()
const resumeCronJob = vi.fn()

let skills: Array<Record<string, any>>
let mcp: Array<Record<string, any>>
let cron: Array<Record<string, any>>

vi.mock('@/store/profile', async () => {
  const { atom } = await import('nanostores')
  stores.activeProfile = atom('work')

  return {
    $activeGatewayProfile: stores.activeProfile,
    normalizeProfileKey: (value: string) => value || 'default'
  }
})

vi.mock('@/lib/desktop-state-sync', () => ({
  broadcastDesktopStateChange: (domain: string, options?: unknown) => broadcastDesktopStateChange(domain, options)
}))

vi.mock('@/hermes', () => ({
  listAllProfileSessions: (...args: unknown[]) => listAllProfileSessions(...args),
  getSkills: () => getSkills(),
  toggleSkill: (name: string, enabled: boolean) => toggleSkill(name, enabled),
  listMcpServers: () => listMcpServers(),
  setMcpServerEnabled: (name: string, enabled: boolean) => setMcpServerEnabled(name, enabled),
  getCronJobs: () => getCronJobs(),
  pauseCronJob: (id: string) => pauseCronJob(id),
  resumeCronJob: (id: string) => resumeCronJob(id)
}))

import { AtlasTab } from './atlas-tab'

beforeEach(() => {
  stores.activeProfile.set('work')
  skills = [{ name: 'research', description: 'Research', category: 'work', enabled: true }]
  mcp = [{ name: 'filesystem', transport: 'stdio', enabled: true }]
  cron = [{ id: 'job-1', name: 'Daily brief', enabled: true, state: 'active' }]

  listAllProfileSessions.mockResolvedValue({
    sessions: [
      {
        id: 'session-1',
        title: 'Companion QA',
        is_active: false,
        ended_at: null,
        input_tokens: 0,
        last_active: 1,
        message_count: 1,
        model: null,
        output_tokens: 0,
        preview: null,
        source: 'desktop',
        started_at: 1,
        tool_call_count: 0,
        profile: 'work'
      }
    ],
    total: 1,
    offset: 0
  })
  getSkills.mockImplementation(async () => structuredClone(skills))
  toggleSkill.mockImplementation(async (name, enabled) => {
    skills = skills.map(skill => (skill.name === name ? { ...skill, enabled } : skill))

    return { ok: true, name, enabled }
  })
  listMcpServers.mockImplementation(async () => ({ servers: structuredClone(mcp) }))
  setMcpServerEnabled.mockImplementation(async (name, enabled) => {
    mcp = mcp.map(server => (server.name === name ? { ...server, enabled } : server))

    return { ok: true }
  })
  getCronJobs.mockImplementation(async () => structuredClone(cron))
  pauseCronJob.mockImplementation(async id => {
    cron = cron.map(job => (job.id === id ? { ...job, enabled: false, state: 'paused' } : job))

    return cron.find(job => job.id === id)
  })
  resumeCronJob.mockImplementation(async id => {
    cron = cron.map(job => (job.id === id ? { ...job, enabled: true, state: 'active' } : job))

    return cron.find(job => job.id === id)
  })
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      openExternal: vi.fn().mockResolvedValue(undefined),
      openSessionWindow: vi.fn().mockResolvedValue({ ok: true })
    }
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('AtlasTab Desktop integration', () => {
  it('uses the active profile and verifies every mutable Desktop map', async () => {
    render(<AtlasTab />)
    await screen.findByRole('heading', { name: 'Recent sessions (1)' })
    expect(listAllProfileSessions).toHaveBeenCalledWith(40, 0, 'exclude', 'recent', 'work')
    expect(screen.queryByRole('heading', { name: /Plugins/i })).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Companion QA' }))
    await waitFor(() => expect(window.hermesDesktop.openSessionWindow).toHaveBeenCalledWith('session-1'))

    fireEvent.click(screen.getByRole('checkbox', { name: /research/i }))
    await waitFor(() => expect(toggleSkill).toHaveBeenCalledWith('research', false))
    expect(broadcastDesktopStateChange).toHaveBeenCalledWith('skills', {
      profile: 'work',
      value: { enabled: false, name: 'research' }
    })

    fireEvent.click(screen.getByRole('checkbox', { name: /filesystem/i }))
    await waitFor(() => expect(setMcpServerEnabled).toHaveBeenCalledWith('filesystem', false))
    expect(broadcastDesktopStateChange).toHaveBeenCalledWith('mcp', {
      profile: 'work',
      value: { enabled: false, name: 'filesystem' }
    })

    fireEvent.click(screen.getByRole('checkbox', { name: /Daily brief/i }))
    await waitFor(() => expect(pauseCronJob).toHaveBeenCalledWith('job-1'))
    expect(broadcastDesktopStateChange).toHaveBeenCalledWith('cron', {
      profile: 'work',
      value: expect.objectContaining({ enabled: false, id: 'job-1', state: 'paused' })
    })

    fireEvent.click(screen.getAllByText('Official')[0].closest('button')!)
    await waitFor(() => expect(window.hermesDesktop.openExternal).toHaveBeenCalled())
  })
})

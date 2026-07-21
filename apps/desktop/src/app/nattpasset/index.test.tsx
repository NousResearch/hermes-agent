import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ProjectInfo } from '@/types/hermes'

import { buildNattpassMission, launchNattpassMission, NattpassetView, nattpassLaunchContextMatches } from '.'

const project: ProjectInfo = {
  archived: false,
  board_slug: null,
  color: null,
  created_at: 1,
  description: null,
  folders: [],
  icon: null,
  id: 'p_hermes',
  name: 'Hermes Agent',
  primary_path: '/work/hermes-agent',
  slug: 'hermes-agent'
}

afterEach(cleanup)

describe('buildNattpassMission', () => {
  it('makes project identity verification and evidence-classified delivery mandatory', () => {
    const prompt = buildNattpassMission({
      acceptanceCriteria: ['Focused tests pass', 'The launch state is honest'],
      goal: 'Ship the first Nattpasset slice',
      project
    })

    expect(prompt).toContain('Project id: p_hermes')
    expect(prompt).toContain('Expected workspace: /work/hermes-agent')
    expect(prompt).toMatch(/verify.*cwd.*repository.*remote.*branch.*status/is)
    expect(prompt).toMatch(/stop.*decision/is)
    expect(prompt).toContain('Focused tests pass')
    expect(prompt).toContain('KLART')
    expect(prompt).toContain('VERIFIERAT')
    expect(prompt).toContain('KVAR')
    expect(prompt).toContain('BESLUT KRÄVS')
    expect(prompt).toMatch(/existing.*todo.*delegation.*Kanban.*cron/is)
  })
})

describe('launchNattpassMission', () => {
  it('captures the verified workspace in a fresh session before submitting to the backend', async () => {
    const trace: string[] = []

    const accepted = await launchNattpassMission(
      { project, prompt: 'mission' },
      path => trace.push(`draft:${path}`),
      async () => {
        trace.push('ready')
      },
      async prompt => {
        trace.push(`submit:${prompt}`)

        return true
      },
      () => true
    )

    expect(accepted).toBe(true)
    expect(trace).toEqual(['draft:/work/hermes-agent', 'ready', 'submit:mission'])
  })

  it.each(['project', 'profile', 'session'])('fails closed when the %s identity changes while routing', async field => {
    const expected = {
      activeProjectId: 'p_hermes',
      profile: 'default',
      projectId: project.id,
      workspace: '/work/hermes-agent'
    }

    const current: Parameters<typeof nattpassLaunchContextMatches>[1] = {
      activeProjectId: 'p_hermes',
      cwd: '/work/hermes-agent',
      freshDraftReady: true,
      onFreshDraftRoute: true,
      profile: 'default',
      project,
      runtimeSessionId: null,
      storedSessionId: null
    }

    let releaseRoute!: () => void

    const routeReady = new Promise<void>(resolve => {
      releaseRoute = resolve
    })

    const submitPrompt = vi.fn(async () => true)

    const pending = launchNattpassMission(
      { project, prompt: 'mission' },
      () => undefined,
      () => routeReady,
      submitPrompt,
      () => nattpassLaunchContextMatches(expected, current)
    )

    if (field === 'project') {
      current.activeProjectId = 'p_other'
    } else if (field === 'profile') {
      current.profile = 'other'
    } else {
      current.runtimeSessionId = 'runtime-other'
    }

    releaseRoute()

    await expect(pending).resolves.toBe(false)
    expect(submitPrompt).not.toHaveBeenCalled()
  })
})

describe('NattpassetView', () => {
  it('launches a real mission only after goal, acceptance criteria, and scope confirmation', () => {
    const onLaunch = vi.fn(async () => true)

    render(<NattpassetView activeProjectId="p_hermes" onClose={vi.fn()} onLaunch={onLaunch} projects={[project]} />)

    expect(screen.getByText(/live Hermes session/i)).toBeTruthy()
    expect(screen.getByText(/not yet durable across app or runtime restarts/i)).toBeTruthy()

    const launch = screen.getByRole('button', { name: 'Start Nattpasset' }) as HTMLButtonElement
    expect(launch.disabled).toBe(true)

    fireEvent.change(screen.getByLabelText('What should be different by morning?'), {
      target: { value: 'Ship the first Nattpasset slice' }
    })
    fireEvent.change(screen.getByLabelText('How will we know it is done?'), {
      target: { value: 'Focused tests pass\nThe launch state is honest' }
    })

    expect(launch.disabled).toBe(true)
    fireEvent.click(screen.getByRole('checkbox', { name: /Hermes Agent.*\/work\/hermes-agent/i }))
    expect(launch.disabled).toBe(false)

    fireEvent.click(launch)

    expect(onLaunch).toHaveBeenCalledTimes(1)
    expect(onLaunch).toHaveBeenCalledWith(
      expect.objectContaining({
        project,
        prompt: expect.stringContaining('Ship the first Nattpasset slice')
      }),
      expect.any(Function)
    )
  })

  it('fails closed when no project with a workspace is available', () => {
    render(<NattpassetView activeProjectId={null} onClose={vi.fn()} onLaunch={vi.fn()} projects={[]} />)

    expect(screen.getByText(/Create or select a project with a workspace/i)).toBeTruthy()
    expect((screen.getByRole('button', { name: 'Start Nattpasset' }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('ignores an older failed launch after a newer launch succeeds', async () => {
    const secondProject = { ...project, id: 'p_two', name: 'Second Project', primary_path: '/work/two' }
    let resolveFirst!: (accepted: boolean) => void

    const first = new Promise<boolean>(resolve => {
      resolveFirst = resolve
    })

    const onLaunch = vi.fn().mockReturnValueOnce(first).mockResolvedValueOnce(true)

    render(
      <NattpassetView
        activeProjectId="p_hermes"
        onClose={vi.fn()}
        onLaunch={onLaunch}
        projects={[project, secondProject]}
      />
    )
    fireEvent.change(screen.getByLabelText('What should be different by morning?'), { target: { value: 'First goal' } })
    fireEvent.change(screen.getByLabelText('How will we know it is done?'), { target: { value: 'Done' } })
    fireEvent.click(screen.getByRole('checkbox', { name: /Hermes Agent.*\/work\/hermes-agent/i }))
    fireEvent.click(screen.getByRole('button', { name: 'Start Nattpasset' }))

    fireEvent.click(screen.getByRole('button', { name: /Second Project/i }))
    fireEvent.click(screen.getByRole('checkbox', { name: /Second Project.*\/work\/two/i }))
    fireEvent.click(screen.getByRole('button', { name: 'Start Nattpasset' }))
    await act(async () => undefined)

    expect(onLaunch).toHaveBeenCalledTimes(2)
    expect(screen.getByRole('button', { name: 'Starting…' })).toBeTruthy()

    await act(async () => resolveFirst(false))

    expect(screen.getByRole('button', { name: 'Starting…' })).toBeTruthy()
  })
})

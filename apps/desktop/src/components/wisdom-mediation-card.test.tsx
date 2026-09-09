// @vitest-environment jsdom
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { WisdomMediationCard } from './wisdom-mediation-card'

const { read, resolve, prepare } = vi.hoisted(() => ({ read: vi.fn(), resolve: vi.fn(), prepare: vi.fn() }))
vi.mock('@/hermes', () => ({
  getWisdomMediation: read,
  resolveWisdomConsent: resolve,
  prepareWisdomConsentPublication: prepare
}))
vi.mock('@/components/wisdom-publication-review', () => ({
  WisdomPublicationReview: ({
    draftId,
    consent
  }: {
    draftId: string
    consent: { interaction_id: string; session_id: string }
  }) => (
    <div>
      Local package {draftId} for {consent.interaction_id} in {consent.session_id}
    </div>
  )
}))

const interaction = {
  id: 'consent',
  assessment_id: 'event',
  state: 'pending',
  operation: 'update',
  expires_at: Date.now() / 1000 + 86400,
  actions: ['defer', 'inspect', 'confirm'],
  facts: { slug: 'Team Runbook', version: 2, compatibility: { outcome: 'compatible' } }
}

const activity = {
  mode: 'agent',
  assessments: [
    {
      id: 'event',
      owner_session: 'session',
      state: 'delivered',
      advice: { title: 'An updated runbook', explanation: 'It may help your current task.', relevance: 'recommend' }
    }
  ],
  interactions: [interaction]
}

describe('WisdomMediationCard', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    read.mockResolvedValue(activity)
  })
  afterEach(() => vi.useRealTimers())

  it.each(['prerequisite', 'setup', 'verify'])(
    'preserves fully reviewed %s approval with an older backend',
    async phase => {
      const step = {
        ...interaction,
        operation: 'setup',
        facts: {
          ...interaction.facts,
          step: { phase, index: 0, command: phase === 'prerequisite' ? '' : 'echo reviewed' },
          setup_instruction: 'Installed setup instructions',
          setup_explanation: 'Why this step is proposed'
        }
      }

      read.mockResolvedValue({ ...activity, interactions: [step] })
      const { rerender } = render(<WisdomMediationCard sessionId="session" />)
      await screen.findByText(/Installed setup instructions/)
      expect(screen.queryByRole('button', { name: 'Update' })).toBeNull()
      expect(resolve).not.toHaveBeenCalled()

      if (phase !== 'prerequisite') {
        expect(screen.getByText('echo reviewed').tagName).toBe('CODE')
      }
      resolve.mockResolvedValue({ ...step, state: 'completed', actions: ['inspect'] })
      fireEvent.click(
        screen.getByRole('button', { name: phase === 'prerequisite' ? 'Confirm prerequisite' : 'Run this step' })
      )
      await waitFor(() => expect(resolve).toHaveBeenCalledWith('consent', 'session', 'confirm', undefined))
      read.mockResolvedValue({
        ...activity,
        assessments: activity.assessments.map(entry => ({ ...entry, owner_session: 'new' })),
        interactions: [{ ...step, facts: { step: step.facts.step } }]
      })
      rerender(<WisdomMediationCard sessionId="new" />)
      await screen.findByText('Collective Wisdom is unavailable.')
      expect(screen.queryByRole('button', { name: 'Run this step' })).toBeNull()
      expect(screen.queryByRole('button', { name: 'Confirm prerequisite' })).toBeNull()
    }
  )

  it.each(['agent', 'fixed'])(
    'shows requested advice and native controls in %s mode without automatic apply',
    async mode => {
      read.mockResolvedValue({ ...activity, mode })
      render(<WisdomMediationCard sessionId="session" />)
      await screen.findByText('An updated runbook')
      expect(screen.getAllByRole('button').map(button => button.textContent)).toEqual([
        'Not Now',
        'Review first',
        'Update'
      ])
      expect(resolve).not.toHaveBeenCalled()
      resolve.mockResolvedValue({ ...interaction, state: 'completed', actions: ['inspect'] })
      fireEvent.click(screen.getByRole('button', { name: 'Update' }))
      await waitFor(() => expect(resolve).toHaveBeenCalledWith('consent', 'session', 'confirm', undefined))
      await waitFor(() => expect(screen.queryByRole('button', { name: 'Update' })).toBeNull())
    }
  )

  it.each(['agent', 'fixed'])('keeps other surfaces passive in %s mode', async mode => {
    read.mockResolvedValue({ ...activity, mode })
    render(<WisdomMediationCard passive sessionId="other" />)
    await screen.findByText('An updated runbook')
    expect(screen.queryByRole('button', { name: 'Update' })).toBeNull()
  })

  it.each(['share', 'publish'])('opens full local review for %s without authorising upload', async operation => {
    read.mockResolvedValue({ ...activity, interactions: [{ ...interaction, operation }] })
    prepare.mockResolvedValue({ draft_id: 'local:draft' })
    render(<WisdomMediationCard sessionId="session" />)
    fireEvent.click(await screen.findByRole('button', { name: 'Review first' }))
    expect(await screen.findByText('Local package local:draft for consent in session')).toBeTruthy()
    expect(prepare).toHaveBeenCalledWith('consent', 'session', undefined)
    expect(resolve).not.toHaveBeenCalled()
  })

  it('retains advice through a polling failure', async () => {
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] })
    render(<WisdomMediationCard sessionId="session" />)
    await screen.findByText('An updated runbook')
    read.mockRejectedValue(new Error('offline'))
    await act(async () => {
      vi.advanceTimersByTime(10_000)
    })
    expect(screen.getByText('An updated runbook')).toBeTruthy()
  })

  it('honors durable surface-local defer but retains passive access', async () => {
    read.mockResolvedValue({ ...activity, interactions: [{ ...interaction, deferred_surfaces: ['local'] }] })
    const { rerender } = render(<WisdomMediationCard sessionId="session" />)
    await waitFor(() => expect(read).toHaveBeenCalled())
    expect(screen.queryByText('An updated runbook')).toBeNull()
    rerender(<WisdomMediationCard passive sessionId="session" />)
    await screen.findByText('An updated runbook')
  })

  it.each(['agent', 'fixed'])(
    'follows the exact setup control through status, recovery and recheck in %s mode',
    async mode => {
      vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] })

      const completed = {
        ...interaction,
        state: 'completed',
        setup_review: {
          summary: 'Files updated',
          detail: 'Setup is queued.',
          command: '',
          actions: [{ action: 'setup.status', label: 'Check setup', primary: false }]
        }
      }

      read.mockResolvedValue({ ...activity, mode, interactions: [completed] })
      render(<WisdomMediationCard sessionId="session" />)
      expect(resolve).not.toHaveBeenCalled()

      const step = {
        ...interaction,
        id: 'setup-control',
        assessment_id: 'setup-event',
        operation: 'setup',
        setup_review: {
          summary: 'Review setup step',
          detail: 'Inspect the installed guide. Only this step is authorized.',
          command: 'python -c "print(123)"',
          actions: [
            { action: 'defer', label: 'Not Now', primary: false },
            { action: 'confirm', label: 'Run this step', primary: true }
          ]
        }
      }

      resolve.mockResolvedValue(step)
      fireEvent.click(await screen.findByRole('button', { name: 'Check setup' }))
      await screen.findByText(step.setup_review.detail)
      expect(screen.getByText(step.setup_review.command).tagName).toBe('CODE')
      expect(screen.queryByRole('button', { name: 'Update' })).toBeNull()

      const unknown = {
        ...step,
        state: 'needs_review',
        setup_review: {
          ...step.setup_review,
          summary: 'Setup outcome unknown',
          actions: [
            { action: 'inspect', label: 'Check progress', primary: false },
            { action: 'setup.recover', label: 'Review interruption', primary: false }
          ]
        }
      }

      resolve.mockResolvedValue(unknown)
      fireEvent.click(screen.getByRole('button', { name: 'Run this step' }))
      await screen.findByRole('button', { name: 'Review interruption' })
      expect(resolve).toHaveBeenLastCalledWith('setup-control', 'session', 'confirm', undefined)

      const recovery = {
        ...unknown,
        setup_review: {
          ...unknown.setup_review,
          detail: 'Check that the command and child processes have stopped. Clearing does not undo changes.',
          actions: [
            { action: 'inspect', label: 'Back', primary: false },
            { action: 'setup.clear', label: 'Confirmed stopped; clear record', primary: true }
          ]
        }
      }

      resolve.mockResolvedValue(recovery)
      fireEvent.click(screen.getByRole('button', { name: 'Review interruption' }))
      await screen.findByText(recovery.setup_review.detail)
      read.mockResolvedValue({
        ...activity,
        mode,
        interactions: [completed, unknown],
        assessments: [...activity.assessments, { ...activity.assessments[0], id: 'setup-event' }]
      })
      await act(async () => {
        vi.advanceTimersByTime(10_000)
      })
      expect(screen.getAllByText(recovery.setup_review.detail)).toHaveLength(1)

      const cleared = {
        ...step,
        state: 'needs_review',
        setup_review: {
          ...step.setup_review,
          summary: 'Interrupted step cleared',
          actions: [{ action: 'recheck', label: 'Recheck', primary: false }]
        }
      }

      resolve.mockResolvedValue(cleared)
      fireEvent.click(screen.getByRole('button', { name: 'Confirmed stopped; clear record' }))
      await screen.findByRole('button', { name: 'Recheck' })
      expect(resolve).toHaveBeenLastCalledWith('setup-control', 'session', 'setup.clear', undefined)
      resolve.mockResolvedValue({ ...step, id: 'fresh-control' })
      fireEvent.click(screen.getByRole('button', { name: 'Recheck' }))
      await screen.findByRole('button', { name: 'Run this step' })
      expect(resolve).toHaveBeenLastCalledWith('setup-control', 'session', 'recheck', undefined)
      expect(resolve).toHaveBeenCalledTimes(5)
    }
  )

  it.each(['session', 'other'])(
    'keeps setup passive and drops an old action response after session changes from %s',
    async sessionId => {
      const step = {
        ...interaction,
        operation: 'setup',
        setup_review: {
          summary: 'Review setup step',
          detail: 'Exact installed instructions',
          command: 'echo reviewed',
          actions: [{ action: 'confirm', label: 'Run this step', primary: true }]
        }
      }

      read.mockResolvedValue({ ...activity, interactions: [step] })
      const { rerender } = render(<WisdomMediationCard passive sessionId={sessionId} />)
      await screen.findByText('Exact installed instructions')
      expect(screen.queryByRole('button')).toBeNull()
      rerender(<WisdomMediationCard sessionId="session" />)
      let finish!: (value: unknown) => void
      resolve.mockReturnValue(
        new Promise(done => {
          finish = done
        })
      )
      fireEvent.click(await screen.findByRole('button', { name: 'Run this step' }))
      read.mockResolvedValue({ mode: 'agent', assessments: [], interactions: [] })
      rerender(<WisdomMediationCard sessionId="new-session" />)
      await act(async () => {
        finish(step)
      })
      expect(screen.queryByText('Exact installed instructions')).toBeNull()
      expect(screen.queryByRole('button')).toBeNull()
    }
  )
})

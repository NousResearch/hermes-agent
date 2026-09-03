// @vitest-environment jsdom
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'

const getWisdomEvents = vi.fn()
const getWisdomCandidates = vi.fn()
const approveWisdomCandidate = vi.fn()
const deferWisdomCandidate = vi.fn()
const suggestWisdomSkill = vi.fn()
const saveWisdomPreparedDraft = vi.fn()
const reviewWisdomDraft = vi.fn()
const reviseWisdomDraft = vi.fn()
const decideWisdomDraft = vi.fn()

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  approveWisdomCandidate,
  decideWisdomDraft,
  deferWisdomCandidate,
  getWisdomCandidates,
  getWisdomEvents,
  reviewWisdomDraft,
  reviseWisdomDraft,
  saveWisdomPreparedDraft,
  suggestWisdomSkill
}))

vi.mock('@/store/notifications', () => ({ notifyError: vi.fn() }))

const candidate = {
  id: 'event-1',
  kind: 'wisdom.candidate',
  session_id: 'session-1',
  task_id: 'task-1',
  content_hash: 'sha256:local',
  qualification_sequence: 1,
  notice_variant: 'first',
  organization_name: 'Nous Research',
  payload: {
    skill_name: 'safe-skill',
    editorial_name: 'Safe Skill',
    editorial_description: 'Share a dependable workflow with your team.',
    qualification: 'meaningful_refinements',
    local_reasons: { meaningful_refinements: 3 },
    consent_required: true,
    networked: false
  }
}

const systemSpecification = {
  hermes: { minimum_version: '0.20.5' },
  platforms: ['macOS'],
  architectures: ['arm64'],
  model: { capabilities: [], minimum_context_window: null },
  tools: [],
  plugins: [],
  credentials: [],
  connections: [],
  filesystem: { read: [], write: [] },
  network: { destinations: [] },
  runtime: { shell: false, browser: false, code: false, sandbox: true },
  hardware: [],
  known_limitations: []
}

const manifest = `${JSON.stringify({ schema_version: 1, name: 'safe-skill', requirements: systemSpecification })}\n`

const localScan = {
  guard: { allowed: true, findings: [] },
  skill_evaluator: { status: 'available', findings: [] }
}

const prepared = (skill = '# Safe\n') => ({
  network_submission: false as const,
  local_draft_id: 'local-1',
  overlay_path: '/private/overlay',
  drafted_description: 'Owner copy',
  system_specification: systemSpecification,
  files: [
    { path: 'SKILL.md', mode: 'file' as const, hash: 'sha256:local-skill', content_utf8: skill },
    { path: 'skill.manifest.json', mode: 'file' as const, hash: 'sha256:local-manifest', content_utf8: manifest }
  ],
  local_scan: localScan,
  next_step: 'review'
})

const exactReview = (id = 'draft-1', skill = '# Server reviewed\n') => ({
  draft: {
    id,
    slug: 'safe-skill',
    state: 'ready',
    updatedAt: `revision-${id}`,
    authorDescription: 'Owner-authored claim',
    scanVerdict: 'pass',
    scan: { verdict: 'pass' },
    explanation: 'Server facts only',
    systemSpec: systemSpecification
  },
  effective_policy: {},
  files: [
    { path: 'SKILL.md', mode: 'file' as const, hash: `sha256:${id}-skill`, content_utf8: skill },
    { path: 'skill.manifest.json', mode: 'file' as const, hash: `sha256:${id}-manifest`, content_utf8: manifest }
  ],
  hashes: {
    content: `sha256:${id}-content`,
    author_description: `sha256:${id}-copy`,
    package_manifest: `sha256:${id}-manifest`
  },
  receipt: null
})

async function renderCard() {
  const { WisdomCandidateCard } = await import('./wisdom-candidate-card')

  return render(
    <div data-slot="aui_thread-viewport">
      <WisdomCandidateCard profile="research" sessionId="session-1" />
    </div>
  )
}

beforeEach(() => {
  vi.resetAllMocks()
  window.location.hash = '#/session-1'
  getWisdomEvents.mockResolvedValue({ events: [candidate] })
  getWisdomCandidates.mockResolvedValue({
    candidates: [
      {
        local_skill_id: 'local-skill-1',
        name: 'safe-skill',
        content_hash: 'sha256:local',
        eligibility: 'eligible',
        reason: null,
        qualification: 'meaningful_refinements',
        contribution_state: 'new'
      }
    ]
  })
  suggestWisdomSkill.mockResolvedValue(prepared())
})

afterEach(() => {
  vi.clearAllMocks()
  vi.restoreAllMocks()
})

describe('WisdomCandidateCard', () => {
  it('auto-prepares a compact card and reveals minimal then detailed editing on demand', async () => {
    const scrollIntoView = vi.fn()
    Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', { configurable: true, value: scrollIntoView })

    await renderCard()

    expect(await screen.findByRole('button', { name: 'Review & edit' })).toBeTruthy()
    expect(screen.getByText(/Your organisation \(Nous Research\) has enabled Collective Wisdom/)).toBeTruthy()
    expect(screen.getByText(/Congratulations! Hermes detected a skill/)).toBeTruthy()
    expect(screen.getByText('Safe Skill')).toBeTruthy()
    expect(screen.getByText('Share a dependable workflow with your team.')).toBeTruthy()
    expect(screen.queryByDisplayValue('Owner copy')).toBeNull()
    expect(screen.queryByLabelText('Edit SKILL.md')).toBeNull()
    expect(screen.queryByText('Minimum Hermes version')).toBeNull()
    expect(screen.getByText('Would you like to share?')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Review first' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Not Now' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Yes' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Review & edit' }))
    expect(screen.getByDisplayValue('Owner copy')).toBeTruthy()
    expect(screen.getByLabelText('Edit SKILL.md')).toBeTruthy()
    expect(screen.getByDisplayValue('safe-skill')).toBeTruthy()
    expect(screen.queryByText('Minimum Hermes version')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Edit detailed requirements' }))
    expect(screen.getByText('Minimum Hermes version')).toBeTruthy()
    expect(screen.getByText('Why Hermes suggested this skill')).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Prepare exact package' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Open Collective' })).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Open full review' }))
    expect(window.location.hash).toBe('#/skills?tab=collective')
    expect(screen.queryByText(/expires/i)).toBeNull()
    expect(screen.queryByText(/publisher device|from device/i)).toBeNull()
    expect(suggestWisdomSkill.mock.calls[0][3]).toBe('local-skill-1')
    await waitFor(() => expect(scrollIntoView).toHaveBeenCalledWith({ block: 'nearest' }))
  }, 30_000)

  it('saves local edits before upload, saves a rescanned server revision, then requires a fresh receipt', async () => {
    const setIntervalSpy = vi.spyOn(window, 'setInterval')
    saveWisdomPreparedDraft.mockResolvedValue(prepared('# Locally edited\n'))
    suggestWisdomSkill
      .mockResolvedValueOnce(prepared())
      .mockResolvedValueOnce({ draft: { id: 'draft-1' }, local_scan: localScan, notice: 'owner private' })
    reviewWisdomDraft
      .mockResolvedValueOnce(exactReview('draft-1'))
      .mockResolvedValueOnce(exactReview('draft-2', '# Rescanned edit\n'))
      .mockResolvedValueOnce({ ...exactReview('draft-2', '# Rescanned edit\n'), receipt: 'receipt-1' })
    reviseWisdomDraft.mockResolvedValue({
      draft: { id: 'draft-2' },
      local_scan: localScan,
      notice: 'rescanned'
    })
    decideWisdomDraft.mockResolvedValue({ ok: true })

    await renderCard()
    fireEvent.click(await screen.findByRole('button', { name: 'Review & edit' }))
    const localEditor = await screen.findByLabelText('Edit SKILL.md')
    fireEvent.change(screen.getByDisplayValue('safe-skill'), { target: { value: 'safer-skill' } })
    fireEvent.change(localEditor, { target: { value: '# Locally edited\n' } })
    expect((screen.getByRole('button', { name: 'Review first' }) as HTMLButtonElement).disabled).toBe(true)
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))
    await waitFor(() =>
      expect(saveWisdomPreparedDraft).toHaveBeenCalledWith(
        'local-1',
        'Owner copy',
        expect.arrayContaining([
          expect.objectContaining({ path: 'SKILL.md', content_utf8: '# Locally edited\n' }),
          expect.objectContaining({ path: 'skill.manifest.json', content_utf8: expect.stringContaining('safer-skill') })
        ]),
        'research'
      )
    )

    await waitFor(() =>
      expect((screen.getByRole('button', { name: 'Review first' }) as HTMLButtonElement).disabled).toBe(false)
    )
    fireEvent.click(screen.getByRole('button', { name: 'Review first' }))
    expect(await screen.findByText(/sha256:draft-1-content/)).toBeTruthy()
    expect(suggestWisdomSkill.mock.calls[1][3]).toBe('local-skill-1')

    fireEvent.change(screen.getByLabelText('Edit SKILL.md'), { target: { value: '# Rescanned edit\n' } })
    expect((screen.getByRole('button', { name: 'Yes' }) as HTMLButtonElement).disabled).toBe(true)
    fireEvent.click(screen.getByRole('button', { name: 'Save changes & rescan' }))
    await waitFor(() =>
      expect(reviseWisdomDraft).toHaveBeenCalledWith(
        'draft-1',
        'Owner-authored claim',
        expect.arrayContaining([expect.objectContaining({ path: 'SKILL.md', content_utf8: '# Rescanned edit\n' })]),
        expect.objectContaining({ content: 'sha256:draft-1-content' }),
        'research'
      )
    )

    fireEvent.click(await screen.findByRole('button', { name: 'Yes' }))
    await waitFor(() => expect(decideWisdomDraft).toHaveBeenCalledWith('draft-2', 'approve', 'research'))
    expect(reviewWisdomDraft.mock.calls[2].slice(0, 2)).toEqual(['draft-2', true])
    expect(screen.queryByText('safe-skill')).toBeNull()

    getWisdomEvents.mockClear()
    const poll = setIntervalSpy.mock.calls.find(([, delay]) => delay === 10_000)?.[0]
    expect(poll).toBeTypeOf('function')
    act(() => (poll as () => void)())
    await waitFor(() => expect(getWisdomEvents).toHaveBeenCalledTimes(1))
    expect(screen.queryByText('safe-skill')).toBeNull()
  })

  it('defers this notification without declining the qualified candidate', async () => {
    deferWisdomCandidate.mockResolvedValue({ event_id: 'event-1', state: 'deferred' })
    await renderCard()
    fireEvent.click(await screen.findByRole('button', { name: 'Not Now' }))
    await waitFor(() => expect(deferWisdomCandidate).toHaveBeenCalledWith('event-1', 'research'))
    expect(screen.queryByText('safe-skill')).toBeNull()
  })

  it('publishes directly from Yes through the candidate approval boundary', async () => {
    approveWisdomCandidate.mockResolvedValue({ publication_state: 'pending_moderation' })
    await renderCard()

    fireEvent.click(await screen.findByRole('button', { name: 'Yes' }))

    await waitFor(() => expect(approveWisdomCandidate).toHaveBeenCalledWith('event-1', 'research'))
    expect(decideWisdomDraft).not.toHaveBeenCalled()
    expect(screen.queryByText('safe-skill')).toBeNull()
  })

  it('shows the notification mute placeholder as a local visual toggle', async () => {
    await renderCard()

    const mute = await screen.findByRole('button', { name: 'Mute notifications (coming soon)' })
    expect(mute.getAttribute('aria-pressed')).toBe('false')
    fireEvent.click(mute)
    const unmute = screen.getByRole('button', { name: 'Unmute notifications (coming soon)' })
    expect(unmute.getAttribute('aria-pressed')).toBe('true')
    expect(deferWisdomCandidate).not.toHaveBeenCalled()
    expect(approveWisdomCandidate).not.toHaveBeenCalled()
  })
})

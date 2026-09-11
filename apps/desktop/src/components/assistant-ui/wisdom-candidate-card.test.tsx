// @vitest-environment jsdom
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'

const getWisdomEvents = vi.fn()
const prepareWisdomCandidate = vi.fn()
const approveWisdomCandidate = vi.fn()
const deferWisdomCandidate = vi.fn()
const suggestWisdomSkill = vi.fn()
const saveWisdomPreparedDraft = vi.fn()
const reviewWisdomDraft = vi.fn()
const reviseWisdomDraft = vi.fn()
const decideWisdomDraft = vi.fn()
const reviewWisdomPublication = vi.fn()
const submitWisdomPublication = vi.fn()

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  reviewWisdomPublication,
  submitWisdomPublication,
  approveWisdomCandidate,
  decideWisdomDraft,
  deferWisdomCandidate,
  getWisdomEvents,
  prepareWisdomCandidate,
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
  prepareWisdomCandidate.mockResolvedValue({
    stage: 'prepared',
    prepared: prepared(),
    local_skill_id: 'local-skill-1',
    skill_name: 'safe-skill'
  })
  suggestWisdomSkill.mockResolvedValue(prepared())
  reviewWisdomPublication.mockResolvedValue({
    ...exactReview('local-1'),
    draft: { ...exactReview('local-1').draft, state: 'prepared' },
    publication_mode: 'moderated'
  })
  submitWisdomPublication.mockResolvedValue({
    draft_id: 'draft-1',
    publication_state: 'pending_moderation',
    portal_url: 'https://portal.example/review/draft-1'
  })
})

afterEach(() => {
  vi.clearAllMocks()
  vi.restoreAllMocks()
})

describe('WisdomCandidateCard', () => {
  it('opens the entire local package before one final moderation submission', async () => {
    await renderCard()
    expect(await screen.findByText(/Your organization \(Nous Research\) has enabled Collective Wisdom/)).toBeTruthy()
    expect(screen.getByText(/Congratulations! Hermes detected a skill/)).toBeTruthy()
    expect(screen.getByText('Skill name: Safe Skill')).toBeTruthy()
    expect(screen.getByText('What it does: Share a dependable workflow with your team.')).toBeTruthy()
    expect(screen.queryByText(/Nothing is shared without your approval/)).toBeNull()
    expect(screen.queryByText(/Reusable skill ready to review/)).toBeNull()
    expect(screen.queryByDisplayValue('Owner copy')).toBeNull()
    expect(screen.queryByLabelText('Edit SKILL.md')).toBeNull()
    expect(screen.queryByText('Minimum Hermes version')).toBeNull()
    expect(screen.getByText('Would you like to share it?')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Review first' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Not Now' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Share' })).toBeTruthy()
    expect(submitWisdomPublication).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Review first' }))
    const submit = await screen.findByRole('button', { name: 'Submit for approval' })
    expect(screen.getByLabelText('Edit SKILL.md')).toBeTruthy()
    expect(screen.getByText('Minimum Hermes version')).toBeTruthy()
    await waitFor(() => expect(submit).toHaveProperty('disabled', false))
    expect(prepareWisdomCandidate).toHaveBeenCalledWith('event-1', 'research')
    fireEvent.click(submit)
    expect(await screen.findByText('Waiting for collective administrator approval')).toBeTruthy()
    expect(submitWisdomPublication).toHaveBeenCalledWith(
      expect.objectContaining({
        hashes: exactReview('local-1').hashes,
        publication_mode: 'moderated'
      }),
      'research',
      undefined
    )
    expect(suggestWisdomSkill).not.toHaveBeenCalled()
    expect(approveWisdomCandidate).not.toHaveBeenCalled()
  }, 30_000)

  it('blocks unsaved edits, rescans locally, then submits the new hashes', async () => {
    await renderCard()
    fireEvent.click(await screen.findByRole('button', { name: 'Review first' }))
    fireEvent.change(await screen.findByLabelText('Edit SKILL.md'), { target: { value: '# Edited' } })
    expect(screen.getByRole('button', { name: 'Submit for approval' })).toHaveProperty('disabled', true)
    saveWisdomPreparedDraft.mockResolvedValue(prepared())

    const updated = {
      ...exactReview('saved', '# Edited'),
      draft: { ...exactReview('saved').draft, state: 'prepared' },
      publication_mode: 'moderated'
    }

    reviewWisdomPublication.mockResolvedValue(updated)
    fireEvent.click(screen.getByRole('button', { name: 'Save changes & rescan' }))
    await waitFor(() => expect(saveWisdomPreparedDraft).toHaveBeenCalledTimes(1))
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Submit for approval' })).toHaveProperty('disabled', false)
    )
    fireEvent.click(screen.getByRole('button', { name: 'Submit for approval' }))
    await waitFor(() => expect(submitWisdomPublication).toHaveBeenCalledWith(updated, 'research', undefined))
  })

  it('uses the open-policy label and keeps failures available for retry', async () => {
    reviewWisdomPublication.mockResolvedValue({ ...exactReview(), publication_mode: 'open' })
    submitWisdomPublication.mockRejectedValueOnce(new Error('Package changed; reload review'))
    await renderCard()
    fireEvent.click(await screen.findByRole('button', { name: 'Review first' }))
    const button = await screen.findByRole('button', { name: 'Publish to team' })
    fireEvent.click(button)
    expect(await screen.findByRole('alert')).toHaveProperty('textContent', 'Package changed; reload review')
    expect(screen.getByRole('button', { name: 'Reload review' })).toBeTruthy()
    expect(button).toHaveProperty('disabled', false)
    expect(screen.queryByRole('link', { name: 'View in Portal' })).toBeNull()
  })

  it('defers this notification without declining the qualified candidate', async () => {
    deferWisdomCandidate.mockResolvedValue({ event_id: 'event-1', state: 'deferred' })
    await renderCard()
    fireEvent.click(await screen.findByRole('button', { name: 'Not Now' }))
    await waitFor(() => expect(deferWisdomCandidate).toHaveBeenCalledWith('event-1', 'research'))
    expect(screen.queryByText('safe-skill')).toBeNull()
  })

  it('Share opens local review and never approves an unseen package', async () => {
    await renderCard()
    fireEvent.click(await screen.findByRole('button', { name: 'Share' }))
    await screen.findByRole('button', { name: 'Submit for approval' })
    expect(prepareWisdomCandidate).toHaveBeenCalledWith('event-1', 'research')
    expect(approveWisdomCandidate).not.toHaveBeenCalled()
    expect(submitWisdomPublication).not.toHaveBeenCalled()
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

  it('keeps the qualification card actionable when exact-package preparation fails', async () => {
    prepareWisdomCandidate.mockRejectedValueOnce(new Error('Gateway temporarily unavailable'))

    await renderCard()
    fireEvent.click(await screen.findByRole('button', { name: 'Review first' }))

    expect((await screen.findByRole('alert')).textContent).toContain('Gateway temporarily unavailable')
    expect((screen.getByRole('button', { name: 'Not Now' }) as HTMLButtonElement).disabled).toBe(false)
    expect((screen.getByRole('button', { name: 'Review first' }) as HTMLButtonElement).disabled).toBe(false)
    expect((screen.getByRole('button', { name: 'Share' }) as HTMLButtonElement).disabled).toBe(false)

    fireEvent.click(screen.getByRole('button', { name: 'Review first' }))
    expect(await screen.findByRole('button', { name: 'Submit for approval' })).toBeTruthy()
    expect(prepareWisdomCandidate).toHaveBeenCalledTimes(2)
  })
})

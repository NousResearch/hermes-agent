// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'

const getWisdomStatus = vi.fn()
const getWisdomDiscovery = vi.fn()
const getWisdomCandidates = vi.fn()
const getWisdomDrafts = vi.fn()
const getWisdomSkill = vi.fn()
const getWisdomInstallations = vi.fn()
const getWisdomVersionContent = vi.fn()
const suggestWisdomSkill = vi.fn()
const reviewWisdomDraft = vi.fn()
const reviseWisdomDraft = vi.fn()
const decideWisdomDraft = vi.fn()
const planWisdomInstall = vi.fn()
const applyWisdomInstall = vi.fn()
const planWisdomUpdate = vi.fn()
const applyWisdomUpdate = vi.fn()
const checkWisdom = vi.fn()
const setupWisdom = vi.fn()
const getActionStatus = vi.fn()

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  decideWisdomDraft,
  getWisdomCandidates,
  getWisdomDiscovery,
  getWisdomDrafts,
  getWisdomSkill,
  getWisdomInstallations,
  getWisdomVersionContent,
  getWisdomStatus,
  reviewWisdomDraft,
  reviseWisdomDraft,
  suggestWisdomSkill,
  planWisdomInstall,
  applyWisdomInstall,
  planWisdomUpdate,
  applyWisdomUpdate,
  checkWisdom,
  setupWisdom,
  getActionStatus
}))

vi.mock('@/store/notifications', () => ({ notifyError: vi.fn() }))

const scope = { connectionId: 'gateway-a', profile: 'research' }
const originalScrollIntoView = Element.prototype.scrollIntoView

beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
})

afterAll(() => {
  Element.prototype.scrollIntoView = originalScrollIntoView
})

beforeEach(() => {
  checkWisdom.mockResolvedValue({ installations: [] })
})

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

async function renderTab(initialEntry = '/skills?tab=collective') {
  const { CollectiveTab } = await import('./collective-tab')
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <QueryClientProvider client={client}>
        <CollectiveTab profile={scope} query="" />
      </QueryClientProvider>
    </MemoryRouter>
  )
}

afterEach(() => vi.clearAllMocks())

function mockInstallations() {
  getWisdomInstallations.mockResolvedValue({ installations: [], notifications: [] })
}

describe('CollectiveTab', () => {
  it('keeps collective reads disabled until disclosure setup is accepted', async () => {
    getWisdomStatus
      .mockResolvedValueOnce({ configured: false, verified_org_id: null })
      .mockResolvedValueOnce({ configured: true, verified_org_id: 'org-1' })
    getWisdomDiscovery.mockResolvedValue({ next_cursor: null, skills: [] })
    getWisdomCandidates.mockResolvedValue({ candidates: [] })
    getWisdomDrafts.mockResolvedValue({ drafts: [] })
    mockInstallations()
    setupWisdom.mockResolvedValue({ ok: true, name: 'wisdom-setup', pid: 1 })
    getActionStatus.mockResolvedValue({
      name: 'wisdom-setup',
      running: false,
      exit_code: 0,
      pid: 1,
      lines: []
    })

    await renderTab()
    expect(await screen.findByText(/Candidate qualification stays on this profile/)).toBeTruthy()
    expect(getWisdomDiscovery).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: /set up this profile/ }))

    await waitFor(() => expect(setupWisdom).toHaveBeenCalledWith(scope))
    expect(getActionStatus).toHaveBeenCalledWith('wisdom-setup', 80, scope)
  }, 30_000)

  it('scopes reads to the selected connection/profile and renders hostile text as text', async () => {
    mockInstallations()
    getWisdomStatus.mockResolvedValue({ configured: true, verified_org_id: 'org-1' })
    getWisdomCandidates.mockResolvedValue({ candidates: [] })
    getWisdomDrafts.mockResolvedValue({ drafts: [] })
    getWisdomDiscovery.mockResolvedValue({
      next_cursor: null,
      skills: [
        {
          id: 'skill-1',
          slug: '<img src=x onerror=alert(1)>',
          author_description: '<script>window.pwned=true</script>',
          install_count: 0,
          latest_version: 1,
          state: 'active'
        }
      ]
    })

    await renderTab()

    expect(await screen.findByText('<img src=x onerror=alert(1)>')).toBeTruthy()
    expect(globalThis.document.querySelector('script')).toBeNull()
    expect(getWisdomDiscovery).toHaveBeenCalledWith(scope)
    expect(getWisdomCandidates).toHaveBeenCalledWith(scope)
  })

  it('refreshes shared-skill discovery on demand', async () => {
    mockInstallations()
    getWisdomStatus.mockResolvedValue({ configured: true, verified_org_id: 'org-1' })
    getWisdomCandidates.mockResolvedValue({ candidates: [] })
    getWisdomDrafts.mockResolvedValue({ drafts: [] })
    getWisdomDiscovery
      .mockResolvedValueOnce({
        next_cursor: null,
        skills: [
          {
            id: 'skill-1',
            slug: 'existing-skill',
            author_description: 'Already visible',
            install_count: 0,
            latest_version: 1,
            state: 'active'
          }
        ]
      })
      .mockResolvedValue({
        next_cursor: null,
        skills: [
          {
            id: 'skill-1',
            slug: 'existing-skill',
            author_description: 'Already visible',
            install_count: 0,
            latest_version: 1,
            state: 'active'
          },
          {
            id: 'skill-2',
            slug: 'newly-shared-skill',
            author_description: 'Published after the screen opened',
            install_count: 0,
            latest_version: 1,
            state: 'active'
          }
        ]
      })

    await renderTab()
    expect(await screen.findByText('existing-skill')).toBeTruthy()
    expect(screen.queryByText('newly-shared-skill')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Refresh shared skills' }))

    expect(await screen.findByText('newly-shared-skill')).toBeTruthy()
    expect(getWisdomDiscovery).toHaveBeenCalledTimes(2)
  })

  it('refreshes registry discovery while checking managed updates', async () => {
    mockInstallations()
    checkWisdom.mockResolvedValue({ installations: [] })
    getWisdomStatus.mockResolvedValue({ configured: true, verified_org_id: 'org-1' })
    getWisdomCandidates.mockResolvedValue({ candidates: [] })
    getWisdomDrafts.mockResolvedValue({ drafts: [] })
    getWisdomDiscovery.mockResolvedValueOnce({ next_cursor: null, skills: [] }).mockResolvedValue({
      next_cursor: null,
      skills: [
        {
          id: 'skill-2',
          slug: 'discovered-during-update-check',
          author_description: 'Newly shared',
          install_count: 0,
          latest_version: 1,
          state: 'active'
        }
      ]
    })

    await renderTab()
    fireEvent.click(await screen.findByRole('button', { name: 'Check updates' }))

    expect(await screen.findByText('discovered-during-update-check')).toBeTruthy()
    expect(checkWisdom).toHaveBeenCalledWith(scope)
    expect(getWisdomDiscovery).toHaveBeenCalledTimes(2)
  })

  it('checks on load and marks installed skills with a pending target version', async () => {
    getWisdomStatus.mockResolvedValue({ configured: true, verified_org_id: 'org-1' })
    getWisdomCandidates.mockResolvedValue({ candidates: [] })
    getWisdomDrafts.mockResolvedValue({ drafts: [] })
    getWisdomDiscovery.mockResolvedValue({
      next_cursor: null,
      skills: [
        {
          id: 'skill-1',
          slug: 'gateway-pull-canary',
          author_description: 'Managed canary',
          install_count: 1,
          latest_version: 2,
          state: 'active'
        }
      ]
    })
    getWisdomInstallations.mockResolvedValue({
      installations: [
        {
          skill_id: 'skill-1',
          slug: 'gateway-pull-canary',
          version: 1,
          update_mode: 'MANUAL',
          state: 'active',
          target_path: '/managed/gateway-pull-canary'
        }
      ],
      notifications: []
    })
    checkWisdom.mockResolvedValue({
      installations: [
        {
          skill_id: 'skill-1',
          state: 'update_available',
          plan: { skill_id: 'skill-1', version: 2, receipt: 'wup_1' }
        }
      ]
    })
    getWisdomSkill.mockResolvedValue({
      skill: { id: 'skill-1', slug: 'gateway-pull-canary' },
      versions: [{ version: 2 }]
    })
    getWisdomVersionContent.mockResolvedValue({ commit: 'sha256:commit', content_hash: 'sha256:content', files: [] })

    await renderTab()

    await waitFor(() => expect(checkWisdom).toHaveBeenCalledWith(scope))
    expect(await screen.findByText('v2 update available')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Check updates (1)' })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: /gateway-pull-canary/ }))
    expect(await screen.findByRole('button', { name: 'Review update' })).toBeTruthy()
  })

  it('prepares locally, accepts explicit owner fields, then submits without local evidence', async () => {
    mockInstallations()
    getWisdomStatus.mockResolvedValue({ configured: true, verified_org_id: 'org-1' })
    getWisdomDiscovery.mockResolvedValue({ next_cursor: null, skills: [] })
    getWisdomCandidates.mockResolvedValue({
      candidates: [
        {
          local_skill_id: 'local-1',
          name: 'candidate-skill',
          eligibility: 'eligible',
          reason: null,
          qualification: 'manual_selection',
          contribution_state: 'new'
        }
      ]
    })
    getWisdomDrafts.mockResolvedValue({ drafts: [] })
    suggestWisdomSkill
      .mockResolvedValueOnce({
        network_submission: false,
        local_draft_id: 'local:draft',
        overlay_path: '/private/overlay',
        drafted_description: 'Drafted copy',
        system_specification: systemSpecification,
        next_step: 'review'
      })
      .mockResolvedValueOnce({ draft: { id: 'draft-1' } })

    await renderTab()
    fireEvent.click(await screen.findByText('View all local skills (1)'))
    fireEvent.click(await screen.findByRole('button', { name: 'Start contribution' }))
    const description = await screen.findByLabelText('Owner-authored description')
    fireEvent.change(description, { target: { value: 'Approved owner copy' } })
    fireEvent.click(screen.getByRole('button', { name: 'Submit draft' }))

    await waitFor(() => expect(suggestWisdomSkill).toHaveBeenCalledTimes(2))
    const payload = suggestWisdomSkill.mock.calls[1][2]
    expect(payload).toEqual({
      description: 'Approved owner copy',
      systemSpecification
    })
    expect(suggestWisdomSkill.mock.calls[0][3]).toBe('local-1')
    expect(suggestWisdomSkill.mock.calls[1][3]).toBe('local-1')
    expect(JSON.stringify(payload)).not.toMatch(/usage|refinement|candidate|ranking|stability/)
  })

  it('separates qualified suggestions, manual inventory, and submissions waiting on collective approval', async () => {
    mockInstallations()
    getWisdomStatus.mockResolvedValue({ configured: true, verified_org_id: 'org-1' })
    getWisdomDiscovery.mockResolvedValue({ next_cursor: null, skills: [] })
    getWisdomCandidates.mockResolvedValue({
      candidates: [
        {
          local_skill_id: 'qualified-1',
          name: 'qualified-skill',
          editorial_name: 'Qualified Skill',
          editorial_description: 'A useful skill ready for team review.',
          eligibility: 'eligible',
          reason: null,
          qualification: 'high_usage',
          contribution_state: 'new'
        },
        {
          local_skill_id: 'manual-1',
          name: 'manual-only-skill',
          eligibility: 'eligible',
          reason: null,
          qualification: 'manual_selection',
          contribution_state: 'new'
        }
      ]
    })
    getWisdomDrafts.mockResolvedValue({
      drafts: [
        { id: 'draft-ready', slug: 'needs-review', state: 'ready' },
        { id: 'draft-pending', slug: 'waiting-on-admin', state: 'pending_moderation' },
        { id: 'draft-published', slug: 'already-shared', state: 'published' },
        { id: 'draft-invalid', slug: 'old-revision', state: 'invalidated' }
      ]
    })

    await renderTab()

    expect(await screen.findByText('needs-review')).toBeTruthy()
    expect(screen.getByText('Ready for your review')).toBeTruthy()
    expect(screen.getByText('waiting-on-admin')).toBeTruthy()
    expect(screen.getByText('Waiting for collective administrator approval')).toBeTruthy()
    expect(
      screen.getByText('Drafts awaiting your review and submissions waiting for collective approval.')
    ).toBeTruthy()
    expect(screen.getByText('1 qualified suggestion')).toBeTruthy()
    expect(screen.getByText('Qualified Skill')).toBeTruthy()
    expect(screen.getByText('A useful skill ready for team review.')).toBeTruthy()
    expect(screen.getAllByRole('button', { name: 'Start contribution' })).toHaveLength(1)
    expect(screen.getByText('View all local skills (1)')).toBeTruthy()
    expect(screen.queryByText('already-shared')).toBeNull()
    expect(screen.queryByText('old-revision')).toBeNull()
  })

  it('edits owner copy and Markdown through a rescanned successor before approval', async () => {
    mockInstallations()
    getWisdomStatus.mockResolvedValue({ configured: true, verified_org_id: 'org-1' })
    getWisdomDiscovery.mockResolvedValue({ next_cursor: null, skills: [] })
    getWisdomCandidates.mockResolvedValue({ candidates: [] })
    getWisdomDrafts.mockResolvedValue({
      drafts: [{ id: 'draft-1', slug: 'editable-skill', state: 'ready', authorDescription: 'Original copy' }]
    })
    const manifest = `${JSON.stringify({ schema_version: 1, name: 'editable-skill', requirements: systemSpecification })}\n`

    const initialReview = {
      draft: {
        id: 'draft-1',
        slug: 'editable-skill',
        state: 'ready',
        authorDescription: 'Original copy',
        scanVerdict: 'PASS'
      },
      effective_policy: {},
      files: [
        { path: 'SKILL.md', mode: 'file', hash: 'sha256:skill', content_utf8: '# Original\n' },
        { path: 'skill.manifest.json', mode: 'file', hash: 'sha256:manifest', content_utf8: manifest }
      ],
      hashes: {
        content: 'sha256:content',
        author_description: 'sha256:description',
        package_manifest: 'sha256:manifest'
      },
      receipt: null
    }

    const revisedReview = {
      ...initialReview,
      draft: { ...initialReview.draft, id: 'draft-2', authorDescription: 'Revised copy' },
      files: [{ ...initialReview.files[0], content_utf8: '# Revised\n' }, initialReview.files[1]],
      hashes: { ...initialReview.hashes, content: 'sha256:revised' }
    }

    reviewWisdomDraft.mockResolvedValueOnce(initialReview).mockResolvedValueOnce(revisedReview)
    reviseWisdomDraft.mockResolvedValue({ draft: revisedReview.draft, local_scan: {}, notice: 'rescanned' })

    await renderTab()
    fireEvent.click(await screen.findByRole('button', { name: /editable-skill.*View details/ }))
    fireEvent.change(await screen.findByLabelText('Edit owner-authored description'), {
      target: { value: 'Revised copy' }
    })
    fireEvent.change(screen.getByLabelText('Edit SKILL.md'), { target: { value: '# Revised\n' } })
    expect(screen.getByRole('button', { name: 'Approve exact content & publish' })).toHaveProperty('disabled', true)
    fireEvent.click(screen.getByRole('button', { name: 'Save changes & rescan' }))

    await waitFor(() => expect(reviseWisdomDraft).toHaveBeenCalledTimes(1))
    expect(reviseWisdomDraft).toHaveBeenCalledWith(
      'draft-1',
      'Revised copy',
      [
        { path: 'SKILL.md', content_utf8: '# Revised\n' },
        { path: 'skill.manifest.json', content_utf8: manifest }
      ],
      initialReview.hashes,
      scope
    )
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Approve exact content & publish' })).toHaveProperty('disabled', false)
    )
  })

  it('requires a verified plan before applying a managed install', async () => {
    mockInstallations()
    getWisdomStatus.mockResolvedValue({ configured: true, verified_org_id: 'org-1' })
    getWisdomCandidates.mockResolvedValue({ candidates: [] })
    getWisdomDrafts.mockResolvedValue({ drafts: [] })
    getWisdomDiscovery.mockResolvedValue({
      next_cursor: null,
      skills: [
        {
          id: 'skill-1',
          slug: 'managed-skill',
          author_description: 'Does work',
          install_count: 0,
          latest_version: 2,
          state: 'active'
        }
      ]
    })
    getWisdomSkill.mockResolvedValue({ skill: { id: 'skill-1', slug: 'managed-skill' }, versions: [{ version: 2 }] })
    getWisdomVersionContent.mockResolvedValue({ commit: 'sha256:commit', content_hash: 'sha256:content', files: [] })
    planWisdomInstall
      .mockResolvedValueOnce({
        receipt: 'wip_1',
        skill_id: 'skill-1',
        version: 2,
        compatibility: { outcome: 'compatible' }
      })
      .mockResolvedValueOnce({
        receipt: 'wip_auto',
        skill_id: 'skill-1',
        version: 2,
        update_mode: 'AUTO_WITH_NOTICE',
        compatibility: { outcome: 'compatible' }
      })
    applyWisdomInstall.mockResolvedValue({ installed: true })

    await renderTab()
    fireEvent.click(await screen.findByRole('button', { name: /managed-skill/ }))
    fireEvent.click(await screen.findByRole('button', { name: 'Install…' }))
    const preview = await screen.findByRole('region', { name: 'Confirm install' })
    const dialog = screen.getByRole('dialog', { name: 'Verified managed action plan' })

    expect(preview.textContent).toContain('wip_1')
    fireEvent.click(within(dialog).getByRole('combobox', { name: 'Future updates' }))
    fireEvent.click(await screen.findByRole('option', { name: 'Automatic with notice' }))
    await waitFor(() => expect(planWisdomInstall).toHaveBeenLastCalledWith('skill-1', scope, 'AUTO_WITH_NOTICE'))
    expect(preview.textContent).toContain('wip_auto')

    const confirm = screen.getByRole('button', { name: 'Confirm install' })
    expect(preview.contains(confirm)).toBe(false)
    fireEvent.click(confirm)
    await waitFor(() => expect(applyWisdomInstall).toHaveBeenCalledWith('wip_auto', false, scope))
  })

  it('opens the verified update plan from a notification deep link', async () => {
    mockInstallations()
    getWisdomStatus.mockResolvedValue({ configured: true, verified_org_id: 'org-1' })
    getWisdomCandidates.mockResolvedValue({ candidates: [] })
    getWisdomDrafts.mockResolvedValue({ drafts: [] })
    getWisdomDiscovery.mockResolvedValue({ next_cursor: null, skills: [] })
    getWisdomSkill.mockResolvedValue({
      skill: { id: 'skill-1', slug: 'managed-skill' },
      versions: [{ version: 3 }]
    })
    getWisdomVersionContent.mockResolvedValue({ commit: 'sha256:commit', content_hash: 'sha256:content', files: [] })
    planWisdomUpdate.mockResolvedValue({
      receipt: 'wup_notification',
      skill_id: 'skill-1',
      version: 3,
      compatibility: { outcome: 'compatible' }
    })

    await renderTab('/skills?tab=collective&wisdomAction=update&wisdomSkillId=skill-1')

    await waitFor(() => expect(planWisdomUpdate).toHaveBeenCalledWith('skill-1', scope))
    expect((await screen.findByRole('region', { name: 'Confirm update' })).textContent).toContain('wup_notification')
    expect(applyWisdomUpdate).not.toHaveBeenCalled()
  })

  it('plans a pasted Portal install link before allowing an install', async () => {
    const portalLink =
      'http://127.0.0.1:3111/orgs/wisdom-local/wisdom/skills/0a192cc7-486e-426d-a6b4-493119c1c011?version=1'

    mockInstallations()
    getWisdomStatus.mockResolvedValue({ configured: true, verified_org_id: 'org-1' })
    getWisdomCandidates.mockResolvedValue({ candidates: [] })
    getWisdomDrafts.mockResolvedValue({ drafts: [] })
    getWisdomDiscovery.mockResolvedValue({ next_cursor: null, skills: [] })
    getWisdomSkill.mockResolvedValue({
      skill: { id: '0a192cc7-486e-426d-a6b4-493119c1c011', slug: 'gateway-pull-canary' },
      versions: []
    })
    planWisdomInstall.mockResolvedValue({
      receipt: 'wip_from_link',
      skill_id: '0a192cc7-486e-426d-a6b4-493119c1c011',
      version: 1,
      update_mode: 'AUTO_WITH_NOTICE',
      compatibility: { outcome: 'compatible' }
    })
    applyWisdomInstall.mockResolvedValue({ installed: true })

    await renderTab()
    fireEvent.change(await screen.findByLabelText('Install from link or skill ID'), {
      target: { value: portalLink }
    })
    fireEvent.click(screen.getByRole('combobox', { name: 'Future updates' }))
    fireEvent.click(await screen.findByRole('option', { name: 'Automatic with notice' }))
    fireEvent.click(screen.getByRole('button', { name: 'Review install' }))

    await waitFor(() => expect(planWisdomInstall).toHaveBeenCalledWith(portalLink, scope, 'AUTO_WITH_NOTICE'))
    expect(applyWisdomInstall).not.toHaveBeenCalled()
    expect(await screen.findByText(/wip_from_link/)).toBeTruthy()
    const dialog = screen.getByRole('dialog', { name: 'Verified managed action plan' })
    expect(within(dialog).getByRole('combobox', { name: 'Future updates' }).textContent).toContain(
      'Automatic with notice'
    )
    fireEvent.click(screen.getByRole('button', { name: 'Confirm install' }))
    await waitFor(() => expect(applyWisdomInstall).toHaveBeenCalledWith('wip_from_link', false, scope))
  })
})

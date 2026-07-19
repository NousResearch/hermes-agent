// @vitest-environment jsdom
import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import type * as ReactRouterDom from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import { queryClient } from '@/lib/query-client'

const getSkills = vi.fn()
const getSkillFiles = vi.fn()
const getSkillContent = vi.fn()
const updateSkillContent = vi.fn()
const deleteSkillFile = vi.fn()
const getToolsets = vi.fn()
const toggleSkill = vi.fn()
const toggleToolset = vi.fn()
const getToolsetConfig = vi.fn()
const selectToolsetProvider = vi.fn()
const getUsageAnalytics = vi.fn()

// Partial mock: keep the real module (SkillsView pulls in @/store/profile,
// whose import-time subscription calls setApiRequestProfile) and stub only the
// calls we assert on.
vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  getSkills: () => getSkills(),
  getSkillFiles: (name: string) => getSkillFiles(name),
  getSkillContent: (name: string, filePath: string) => getSkillContent(name, filePath),
  updateSkillContent: (name: string, filePath: string, content: string) => updateSkillContent(name, filePath, content),
  deleteSkillFile: (name: string, filePath: string) => deleteSkillFile(name, filePath),
  getToolsets: () => getToolsets(),
  toggleSkill: (name: string, enabled: boolean) => toggleSkill(name, enabled),
  toggleToolset: (name: string, enabled: boolean) => toggleToolset(name, enabled),
  getToolsetConfig: (name: string) => getToolsetConfig(name),
  selectToolsetProvider: (toolset: string, provider: string) => selectToolsetProvider(toolset, provider),
  getUsageAnalytics: (days: number) => getUsageAnalytics(days)
}))

vi.mock('@/components/chat/code-editor', () => ({
  CodeEditor: ({
    disabled,
    filePath,
    initialValue,
    onChange
  }: {
    disabled?: boolean
    filePath: string
    initialValue: string
    onChange: (value: string) => void
  }) => (
    <textarea
      aria-label={`Editor ${filePath}`}
      defaultValue={initialValue}
      disabled={disabled}
      onChange={event => onChange(event.currentTarget.value)}
    />
  )
}))

// Notifications hit nanostores/timers we don't care about here.
vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

// The vision detail navigates to Settings → Models via useNavigate; spy on it
// so the deep-link target is assertable.
const navigateSpy = vi.fn()

vi.mock('react-router-dom', async importOriginal => ({
  ...(await importOriginal<typeof ReactRouterDom>()),
  useNavigate: () => navigateSpy
}))

function toolset(overrides: Record<string, unknown> = {}) {
  return {
    name: 'web',
    label: 'Web Search',
    description: 'web_search, web_extract',
    enabled: true,
    available: true,
    configured: true,
    tools: ['web_search', 'web_extract'],
    ...overrides
  }
}

function skill(overrides: Record<string, unknown> = {}) {
  return {
    name: 'package-skill',
    description: 'A package skill',
    category: 'general',
    enabled: true,
    provenance: 'agent',
    ...overrides
  }
}

async function renderSkills(initialEntry = '/skills?tab=toolsets') {
  const { SkillsView } = await import('./index')
  let result: ReturnType<typeof render>
  await act(async () => {
    result = render(
      // SkillsView reads skills/toolsets via useQuery, so it needs a provider.
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={[initialEntry]}>
          <SkillsView />
        </MemoryRouter>
      </QueryClientProvider>
    )
  })

  return result!
}

beforeEach(() => {
  getSkills.mockResolvedValue([])
  getSkillFiles.mockResolvedValue({ name: 'package-skill', files: [] })
  getSkillContent.mockResolvedValue({
    name: 'package-skill',
    file_path: 'SKILL.md',
    path: '/skills/package-skill/SKILL.md',
    content: '# Skill\n'
  })
  updateSkillContent.mockResolvedValue({ success: true, message: 'updated' })
  deleteSkillFile.mockResolvedValue({ success: true, message: 'deleted' })
  getToolsets.mockResolvedValue([toolset()])
  toggleToolset.mockResolvedValue({ ok: true, name: 'web', enabled: false })
  getToolsetConfig.mockResolvedValue({ has_category: true, active_provider: null, providers: [] })
  getUsageAnalytics.mockResolvedValue({ tools: [] })
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  vi.clearAllMocks()
  // Shared singleton client — drop cached skills/toolsets so each test refetches.
  queryClient.clear()
})

describe('SkillsView toolset management', () => {
  it('renders a switch for each toolset and toggles it off', async () => {
    await renderSkills()

    const sw = await screen.findByRole('switch', { name: 'Toggle Web Search toolset' })
    expect(sw.getAttribute('aria-checked')).toBe('true')

    await act(async () => {
      fireEvent.click(sw)
    })

    await waitFor(() => expect(toggleToolset).toHaveBeenCalledWith('web', false))
  })

  it('renders toolset titles without leading emoji', async () => {
    getToolsets.mockResolvedValue([toolset({ name: 'cronjob', label: '⏰ Cron Jobs', description: 'cron tools' })])

    await renderSkills()

    // The label renders in both the row and the auto-selected detail header, so
    // assert via the switch's (emoji-stripped) accessible name and the absence
    // of the emoji rather than a single-match text lookup.
    await screen.findByRole('switch', { name: 'Toggle Cron Jobs toolset' })
    expect(screen.queryByText(/⏰/)).toBeNull()
  })

  it('renders the provider config panel inline for the selected toolset', async () => {
    // The master-detail UI dropped the resting "Configured" pill and the
    // "Configure" expander: the detail column auto-selects the first toolset
    // and renders its config panel directly, which fetches on mount.
    await renderSkills()

    await screen.findByRole('switch', { name: 'Toggle Web Search toolset' })
    await waitFor(() => expect(getToolsetConfig).toHaveBeenCalledWith('web'))
  })

  it('shows a vision explainer that deep-links to Settings → Models', async () => {
    // Vision has no TOOL_CATEGORIES provider matrix — its model lives in the
    // auxiliary model config, so the detail pane must point there instead of
    // rendering an empty panel.
    getToolsets.mockResolvedValue([
      toolset({
        name: 'vision',
        label: 'Vision / Image Analysis',
        description: 'vision_analyze',
        tools: ['vision_analyze']
      })
    ])
    getToolsetConfig.mockResolvedValue({ has_category: false, active_provider: null, providers: [] })

    await renderSkills()

    expect(await screen.findByText(/auxiliary model configuration/)).toBeTruthy()
    const link = screen.getByRole('button', { name: /Choose vision model in Settings/ })

    await act(async () => {
      fireEvent.click(link)
    })

    // Internal route change into the Models section with the aux slot target —
    // consumed by ModelSettings' deep-link highlight. Never an external URL.
    await waitFor(() => expect(navigateSpy).toHaveBeenCalledWith('/settings?tab=config:model&aux=vision'))
  })
})

describe('SkillsView package files', () => {
  const files = [
    { path: 'SKILL.md', kind: 'skill', is_binary: false },
    { path: 'references/guide.md', kind: 'references', is_binary: false },
    { path: 'references/other.md', kind: 'references', is_binary: false },
    { path: 'assets/logo.png', kind: 'assets', is_binary: true }
  ]

  beforeEach(() => {
    getSkills.mockResolvedValue([skill()])
    getSkillFiles.mockResolvedValue({ name: 'package-skill', files })
    getSkillContent.mockImplementation((_name: string, filePath: string) =>
      Promise.resolve({
        name: 'package-skill',
        file_path: filePath,
        path: `/skills/package-skill/${filePath}`,
        content: filePath === 'SKILL.md' ? '# Skill\n' : '# Guide\n'
      })
    )
  })

  it('opens and saves a support file by its real path', async () => {
    await renderSkills('/skills?tab=skills')

    fireEvent.click(await screen.findByRole('button', { name: 'references/guide.md' }))
    const editor = await screen.findByRole('textbox', { name: 'Editor references/guide.md' })
    fireEvent.change(editor, { target: { value: '# Updated guide\n' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    await waitFor(() =>
      expect(updateSkillContent).toHaveBeenCalledWith('package-skill', 'references/guide.md', '# Updated guide\n')
    )
  })

  it('deletes support files but never offers delete for SKILL.md', async () => {
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    await renderSkills('/skills?tab=skills')

    expect(await screen.findByRole('button', { name: 'SKILL.md' })).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Delete SKILL.md' })).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Delete references/guide.md' }))

    await waitFor(() => expect(deleteSkillFile).toHaveBeenCalledWith('package-skill', 'references/guide.md'))
  })

  it('keeps bundled package files read-only', async () => {
    getSkills.mockResolvedValue([skill({ provenance: 'bundled' })])
    await renderSkills('/skills?tab=skills')

    fireEvent.click(await screen.findByRole('button', { name: 'references/guide.md' }))
    const editor = await screen.findByRole('textbox', { name: 'Editor references/guide.md' })

    expect(editor.hasAttribute('disabled')).toBe(true)
    expect(screen.queryByRole('button', { name: 'Delete references/guide.md' })).toBeNull()
  })

  it('does not open binary assets in the text editor', async () => {
    await renderSkills('/skills?tab=skills')

    fireEvent.click(await screen.findByRole('button', { name: 'assets/logo.png' }))

    expect(await screen.findByText(/assets\/logo\.png is binary/)).toBeTruthy()
    expect(screen.queryByRole('textbox', { name: 'Editor assets/logo.png' })).toBeNull()
    expect(getSkillContent).not.toHaveBeenCalledWith('package-skill', 'assets/logo.png')
  })

  it('guards file switches when the current draft is dirty', async () => {
    vi.spyOn(window, 'confirm').mockReturnValue(false)
    await renderSkills('/skills?tab=skills')

    fireEvent.click(await screen.findByRole('button', { name: 'references/guide.md' }))
    const editor = await screen.findByRole('textbox', { name: 'Editor references/guide.md' })
    fireEvent.change(editor, { target: { value: 'unsaved' } })
    fireEvent.click(screen.getByRole('button', { name: 'references/other.md' }))

    expect(window.confirm).toHaveBeenCalledWith('Discard unsaved changes and switch files?')
    expect(getSkillContent).not.toHaveBeenCalledWith('package-skill', 'references/other.md')
    expect(screen.getByRole('textbox', { name: 'Editor references/guide.md' })).toBeTruthy()
  })
})

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const getSkills = vi.fn()
const getToolsets = vi.fn()
const toggleSkill = vi.fn()
const toggleToolset = vi.fn()
const getToolsetConfig = vi.fn()
const selectToolsetProvider = vi.fn()
const getLearnStatus = vi.fn()
const startLearn = vi.fn()
const pauseLearn = vi.fn()
const resumeLearn = vi.fn()
const stopLearn = vi.fn()
const deleteLearnData = vi.fn()
const reviewLearnSuggestions = vi.fn()
const updateLearnConfig = vi.fn()

vi.mock('@/hermes', () => ({
  getSkills: () => getSkills(),
  getToolsets: () => getToolsets(),
  toggleSkill: (name: string, enabled: boolean) => toggleSkill(name, enabled),
  toggleToolset: (name: string, enabled: boolean) => toggleToolset(name, enabled),
  getToolsetConfig: (name: string) => getToolsetConfig(name),
  selectToolsetProvider: (toolset: string, provider: string) => selectToolsetProvider(toolset, provider),
  getLearnStatus: () => getLearnStatus(),
  startLearn: (mode: string) => startLearn(mode),
  pauseLearn: () => pauseLearn(),
  resumeLearn: () => resumeLearn(),
  stopLearn: () => stopLearn(),
  deleteLearnData: () => deleteLearnData(),
  reviewLearnSuggestions: () => reviewLearnSuggestions(),
  updateLearnConfig: (config: unknown) => updateLearnConfig(config),
  deleteEnvVar: vi.fn(),
  revealEnvVar: vi.fn(),
  setEnvVar: vi.fn()
}))

// Notifications hit nanostores/timers we don't care about here.
vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
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

function learnStatus(overrides: Record<string, unknown> = {}) {
  return {
    mode: 'off',
    state: 'stopped',
    enabled: false,
    running: false,
    paused: false,
    retention_days: 14,
    allowlist: [],
    denylist: [],
    started_at: null,
    paused_at: null,
    resumed_at: null,
    stopped_at: null,
    data_deleted_at: null,
    updated_at: null,
    collected_event_count: 0,
    hermes_home: 'C:/Users/jason/AppData/Local/hermes',
    storage_path: 'C:/Users/jason/AppData/Local/hermes/learn',
    ...overrides
  }
}

function renderSkills() {
  return import('./index').then(({ SkillsView }) =>
    render(
      <MemoryRouter initialEntries={['/skills?tab=toolsets']}>
        <SkillsView />
      </MemoryRouter>
    )
  )
}

beforeEach(() => {
  getSkills.mockResolvedValue([])
  getToolsets.mockResolvedValue([toolset()])
  toggleToolset.mockResolvedValue({ ok: true, name: 'web', enabled: false })
  getToolsetConfig.mockResolvedValue({ has_category: false, active_provider: null, providers: [] })
  getLearnStatus.mockResolvedValue(learnStatus())
  startLearn.mockResolvedValue(learnStatus({ mode: 'learn', state: 'running', enabled: true, running: true }))
  pauseLearn.mockResolvedValue(learnStatus({ mode: 'learn', state: 'paused', enabled: true, paused: true }))
  resumeLearn.mockResolvedValue(learnStatus({ mode: 'learn', state: 'running', enabled: true, running: true }))
  stopLearn.mockResolvedValue(learnStatus({ mode: 'learn', state: 'stopped', enabled: true }))
  deleteLearnData.mockResolvedValue(learnStatus({ mode: 'learn', state: 'stopped', enabled: true }))
  reviewLearnSuggestions.mockResolvedValue({ created_count: 1, suggestions: [{ id: 'learn1', status: 'pending' }] })
  updateLearnConfig.mockResolvedValue(
    learnStatus({ allowlist: ['code.exe', 'chrome.exe'], denylist: ['slack.exe'], retention_days: 21 })
  )
  vi.spyOn(window, 'confirm').mockReturnValue(true)
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('SkillsView toolset management', () => {
  it('renders a switch for each toolset and toggles it off', async () => {
    await renderSkills()

    const sw = await screen.findByRole('switch', { name: 'Toggle Web Search toolset' })
    expect(sw.getAttribute('aria-checked')).toBe('true')

    fireEvent.click(sw)

    await waitFor(() => expect(toggleToolset).toHaveBeenCalledWith('web', false))
  })

  it('renders toolset titles without leading emoji', async () => {
    getToolsets.mockResolvedValue([
      toolset({ name: 'cronjob', label: '⏰ Cron Jobs', description: 'cron tools' })
    ])

    await renderSkills()

    expect(await screen.findByText('Cron Jobs')).toBeTruthy()
    expect(screen.queryByText(/⏰/)).toBeNull()
  })

  it('keeps the configured pill alongside the switch', async () => {
    await renderSkills()

    await screen.findByRole('switch', { name: 'Toggle Web Search toolset' })
    expect(screen.getByText('Configured')).toBeTruthy()
  })

  it('renders the Learn surface without requiring a backend toolset', async () => {
    getToolsets.mockResolvedValue([])

    await renderSkills()

    expect(await screen.findByText('Learn')).toBeTruthy()
    expect(screen.queryByText('No toolsets found')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Configure Learn' }))

    expect(await screen.findByText(/Learn converts approved signals/)).toBeTruthy()
    expect(screen.getByText('Stopped')).toBeTruthy()
  })

  it('shows the live Learn status in the toolsets list', async () => {
    getToolsets.mockResolvedValue([])
    getLearnStatus.mockResolvedValue(learnStatus({ mode: 'learn', state: 'running', enabled: true, running: true }))

    await renderSkills()

    expect(await screen.findByText('Running')).toBeTruthy()
    expect(screen.queryByText('Preview')).toBeNull()
  })

  it('sorts the Learn surface alphabetically with toolset rows', async () => {
    getToolsets.mockResolvedValue([
      toolset({ name: 'browser', label: 'Browser Automation' }),
      toolset({ name: 'zebra', label: 'Zebra Tools' })
    ])

    await renderSkills()

    await screen.findByText('Browser Automation')
    const text = document.body.textContent || ''

    expect(text.indexOf('Browser Automation')).toBeLessThan(text.indexOf('Learn'))
    expect(text.indexOf('Learn')).toBeLessThan(text.indexOf('Zebra Tools'))
  })

  it('expands the provider config panel when the configured pill is clicked', async () => {
    await renderSkills()

    const configureBtn = await screen.findByRole('button', { name: 'Configure Web Search' })
    fireEvent.click(configureBtn)

    await waitFor(() => expect(getToolsetConfig).toHaveBeenCalledWith('web'))
  })

  it('starts Learn mode from the panel and renders the running status', async () => {
    getToolsets.mockResolvedValue([])

    await renderSkills()

    fireEvent.click(await screen.findByRole('button', { name: 'Configure Learn' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Start Learn' }))

    await waitFor(() => expect(startLearn).toHaveBeenCalledWith('learn'))
    await waitFor(() => expect(screen.getAllByText('Running').length).toBeGreaterThanOrEqual(2))
  })

  it('pauses, resumes, stops, and deletes Learn data from the panel', async () => {
    getToolsets.mockResolvedValue([])
    getLearnStatus.mockResolvedValue(learnStatus({ mode: 'learn', state: 'running', enabled: true, running: true }))

    await renderSkills()

    fireEvent.click(await screen.findByRole('button', { name: 'Configure Learn' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Pause Learn' }))
    await waitFor(() => expect(pauseLearn).toHaveBeenCalled())

    fireEvent.click(await screen.findByRole('button', { name: 'Resume Learn' }))
    await waitFor(() => expect(resumeLearn).toHaveBeenCalled())

    fireEvent.click(await screen.findByRole('button', { name: 'Stop Learn' }))
    await waitFor(() => expect(stopLearn).toHaveBeenCalled())

    fireEvent.click(await screen.findByRole('button', { name: 'Review suggestions' }))
    await waitFor(() => expect(reviewLearnSuggestions).toHaveBeenCalled())

    fireEvent.click(await screen.findByRole('button', { name: 'Delete Learn data' }))
    await waitFor(() => expect(deleteLearnData).toHaveBeenCalled())
  })

  it('updates Learn collection controls from the panel', async () => {
    getToolsets.mockResolvedValue([])
    getLearnStatus.mockResolvedValue(
      learnStatus({ allowlist: ['code.exe'], denylist: ['slack.exe'], retention_days: 14 })
    )

    await renderSkills()

    fireEvent.click(await screen.findByRole('button', { name: 'Configure Learn' }))
    await screen.findByDisplayValue('code.exe')
    fireEvent.change(await screen.findByLabelText('Allowed apps or domains'), {
      target: { value: 'code.exe, chrome.exe' }
    })
    fireEvent.change(screen.getByLabelText('Blocked apps or domains'), { target: { value: 'slack.exe' } })
    fireEvent.change(screen.getByLabelText('Retention days'), { target: { value: '21' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save Learn controls' }))

    await waitFor(() =>
      expect(updateLearnConfig).toHaveBeenCalledWith({
        allowlist: ['code.exe', 'chrome.exe'],
        denylist: ['slack.exe'],
        retention_days: 21
      })
    )
  })
})

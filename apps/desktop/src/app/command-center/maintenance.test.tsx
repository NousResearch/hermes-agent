import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import { getActionStatus, getCuratorStatus, setCuratorPaused } from '@/hermes'
import { type BundledLocale, I18nProvider, TRANSLATIONS } from '@/i18n'
import { $desktopActionTasks } from '@/store/activity'

import { MaintenancePanel } from './maintenance'

// The backend spawns each op under one fixed action name ('doctor', 'security-audit',
// 'backup', 'curator-run'), a re-spawn replaces the record under that name, and
// /api/actions/<name>/status reports the latest run. The fake keeps that contract.
const runs: Record<string, number> = {}
const running: Record<string, boolean> = {}
let nextRunStaysRunning = false

function spawn(name: string) {
  runs[name] = (runs[name] ?? 0) + 1
  running[name] = nextRunStaysRunning

  return Promise.resolve({ name, ok: true, pid: 1000 + runs[name] })
}

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  getActionStatus: vi.fn(async (name: string) => ({
    exit_code: running[name] ? null : 0,
    lines: [`${name} run ${runs[name]} output`],
    name,
    pid: 1000 + runs[name],
    running: running[name]
  })),
  getCuratorStatus: vi.fn(() => new Promise(() => {})),
  getMemoryStatus: vi.fn(() => new Promise(() => {})),
  setCuratorPaused: vi.fn(async () => ({ ok: true })),
  runDoctor: vi.fn(() => spawn('doctor')),
  runSecurityAudit: vi.fn(() => spawn('security-audit'))
}))

beforeEach(() => {
  for (const key of Object.keys(runs)) {
    delete runs[key]
    delete running[key]
  }

  nextRunStaysRunning = false
  $desktopActionTasks.set({})
  vi.mocked(getActionStatus).mockClear()
  vi.mocked(getCuratorStatus).mockReset()
  vi.mocked(getCuratorStatus).mockImplementation(() => new Promise(() => {}))
  vi.mocked(setCuratorPaused).mockClear()
})

afterEach(cleanup)

const scopeCopy = {
  en: [
    'Background review that archives stale agent-created skills',
    'Background review that archives stale agent-created and bundled skills'
  ],
  zh: ['后台审查并归档过期的智能体自建技能', '后台审查并归档过期的智能体自建技能和内置技能'],
  de: [
    'Hintergrundprüfung, die veraltete, von Agenten erstellte Skills archiviert',
    'Hintergrundprüfung, die veraltete, von Agenten erstellte und mitgelieferte Skills archiviert'
  ],
  es: [
    'Revisión en segundo plano que archiva skills creados por agentes que ya no se usan',
    'Revisión en segundo plano que archiva skills creados por agentes y skills incluidos que ya no se usan'
  ],
  fr: [
    'Revue en arrière-plan qui archive les skills agent obsolètes',
    'Revue en arrière-plan qui archive les skills obsolètes créés par les agents et les skills intégrés'
  ],
  ru: [
    'Фоновый обзор, архивирующий устаревшие навыки, созданные агентом',
    'Фоновый обзор, архивирующий устаревшие навыки, созданные агентом, и встроенные навыки'
  ]
} as const

function curatorStatus(pruneBuiltins?: boolean): HermesApi.CuratorStatusResponse {
  return {
    enabled: true,
    paused: false,
    interval_hours: 24,
    last_run_at: null,
    min_idle_hours: 24,
    stale_after_days: 30,
    archive_after_days: 60,
    ...(pruneBuiltins === undefined ? {} : { prune_builtins: pruneBuiltins })
  }
}

function localizedPanel(locale: BundledLocale, profile = 'default') {
  return (
    <I18nProvider configClient={null} initialLocale={locale}>
      <MaintenancePanel key={profile} />
    </I18nProvider>
  )
}

describe('MaintenancePanel curator scope', () => {
  const locales = ['en', 'zh', 'de', 'es', 'fr', 'ru', 'ja', 'ar', 'zh-hant'] as const

  it.each(locales.flatMap(locale => [false, true, undefined].map(scope => ({ locale, scope }))))(
    'renders the effective scope for $locale with prune_builtins=$scope',
    async ({ locale, scope }) => {
      const [agentOnly, withBuiltins] = scopeCopy[locale as keyof typeof scopeCopy] ?? scopeCopy.en
      const copy = TRANSLATIONS[locale].commandCenter.maintenance
      expect(copy.curatorDesc).toBe(agentOnly)
      expect(copy.curatorDescWithBuiltins).toBe(withBuiltins)
      vi.mocked(getCuratorStatus).mockResolvedValue(curatorStatus(scope))
      render(localizedPanel(locale))
      const caption = await screen.findByText(text => text.startsWith(scope ? withBuiltins : agentOnly))
      expect(caption.textContent).not.toContain(scope ? agentOnly + ' · ' : withBuiltins)
      await act(async () => void fireEvent.click(button(copy.pause)))
      expect(vi.mocked(setCuratorPaused)).toHaveBeenCalledWith(true)
      expect(caption.textContent).toContain(scope ? withBuiltins : agentOnly)
      expect(screen.getByRole('button', { name: copy.resume })).toBeTruthy()
    }
  )

  it('replaces the caption when remounted for profile A → B → A', async () => {
    vi.mocked(getCuratorStatus)
      .mockResolvedValueOnce(curatorStatus(false))
      .mockResolvedValueOnce(curatorStatus(true))
      .mockResolvedValueOnce(curatorStatus(false))
    const [agentOnly, withBuiltins] = scopeCopy.en
    const view = render(localizedPanel('en', 'profile-a'))
    await screen.findByText(text => text.startsWith(agentOnly + ' · '))
    view.rerender(localizedPanel('en', 'profile-b'))
    await screen.findByText(text => text.startsWith(withBuiltins + ' · '))
    expect(screen.queryByText(text => text.startsWith(agentOnly + ' · '))).toBeNull()
    view.rerender(localizedPanel('en', 'profile-a'))
    await screen.findByText(text => text.startsWith(agentOnly + ' · '))
    expect(screen.queryByText(text => text.startsWith(withBuiltins + ' · '))).toBeNull()
    expect(getCuratorStatus).toHaveBeenCalledTimes(3)
  })
})

const button = (name: string) => screen.getByRole('button', { name }) as HTMLButtonElement

describe('MaintenancePanel action tail', () => {
  it('tails a second run of the same op', async () => {
    render(<MaintenancePanel />)

    await act(async () => void fireEvent.click(button('Run doctor')))
    await screen.findByText('doctor run 1 output')

    nextRunStaysRunning = true
    await act(async () => void fireEvent.click(button('Run doctor')))

    await screen.findByText('doctor run 2 output')
    expect(vi.mocked(getActionStatus)).toHaveBeenCalledTimes(2)
    expect(screen.getByText('Running...')).toBeTruthy()
    expect(button('Run doctor').disabled).toBe(true)
    expect($desktopActionTasks.get().doctor?.status).toMatchObject({ pid: 1002, running: true })
  })

  it('tails a different op launched after the first', async () => {
    render(<MaintenancePanel />)

    await act(async () => void fireEvent.click(button('Run doctor')))
    await screen.findByText('doctor run 1 output')

    nextRunStaysRunning = true
    await act(async () => void fireEvent.click(button('Security audit')))

    await screen.findByText('security-audit run 1 output')
    expect(vi.mocked(getActionStatus)).toHaveBeenLastCalledWith('security-audit', 200)
    expect(button('Security audit').disabled).toBe(true)
  })
})

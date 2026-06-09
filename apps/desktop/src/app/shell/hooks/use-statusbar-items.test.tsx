import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $desktopActionTasks } from '@/store/activity'
import { $previewServerRestart } from '@/store/preview'
import {
  $activeSessionId,
  $approvalMode,
  $busy,
  $connection,
  $currentFastMode,
  $currentModel,
  $currentProvider,
  $currentReasoningEffort,
  $currentUsage,
  $sessionStartedAt,
  $turnStartedAt,
  $workingSessionIds,
  $yoloActive
} from '@/store/session'
import { $subagentsBySession } from '@/store/subagents'
import {
  $backendUpdateApply,
  $backendUpdateStatus,
  $desktopVersion,
  $updateApply,
  $updateStatus
} from '@/store/updates'
import type { StatusResponse } from '@/types/hermes'

import { StatusbarControls } from '../statusbar-controls'

import { useStatusbarItems } from './use-statusbar-items'

type RequestGateway = Parameters<typeof useStatusbarItems>[0]['requestGateway']

const defaultRequestGateway: RequestGateway = async <T = unknown>() => ({}) as T

const idleApply = {
  applying: false,
  command: null,
  error: null,
  log: [],
  message: '',
  percent: null,
  stage: 'idle' as const
}

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (error: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })

  return { promise, reject, resolve }
}

function statusSnapshot(overrides: Partial<StatusResponse> = {}): StatusResponse {
  return {
    active_sessions: 0,
    config_path: '/Users/luigi/.hermes/config.yaml',
    config_version: 1,
    env_path: '/Users/luigi/.hermes/.env',
    gateway_exit_reason: null,
    gateway_health_url: null,
    gateway_pid: 123,
    gateway_platforms: {},
    gateway_running: true,
    gateway_state: 'connected',
    gateway_updated_at: null,
    hermes_home: '/Users/luigi/.hermes',
    latest_config_version: 1,
    release_date: '',
    version: '0.16.0',
    ...overrides
  }
}

function renderStatusbar({
  connectionMode = 'local',
  freshDraftReady = true,
  requestGateway = defaultRequestGateway,
  status = statusSnapshot()
}: {
  connectionMode?: 'local' | 'remote'
  freshDraftReady?: boolean
  requestGateway?: RequestGateway
  status?: StatusResponse | null
} = {}) {
  $connection.set({
    baseUrl: 'http://127.0.0.1:8765',
    isFullscreen: false,
    logs: [],
    mode: connectionMode,
    nativeOverlayWidth: 0,
    token: '',
    windowButtonPosition: null,
    wsUrl: 'ws://127.0.0.1:8765/ws'
  })

  function Probe() {
    const { leftStatusbarItems, statusbarItems } = useStatusbarItems({
      agentsOpen: false,
      commandCenterOpen: false,
      extraLeftItems: [],
      extraRightItems: [],
      freshDraftReady,
      gatewayLogLines: [],
      gatewayState: 'open',
      inferenceStatus: {
        checksDisagree: false,
        ready: true,
        reason: null,
        source: 'runtime_check'
      },
      openAgents: vi.fn(),
      openCommandCenterSection: vi.fn(),
      requestGateway,
      statusSnapshot: status,
      toggleCommandCenter: vi.fn()
    })

    return <StatusbarControls items={statusbarItems} leftItems={leftStatusbarItems} />
  }

  return render(
    <I18nProvider configClient={null}>
      <MemoryRouter>
        <Probe />
      </MemoryRouter>
    </I18nProvider>
  )
}

describe('useStatusbarItems', () => {
  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    $activeSessionId.set(null)
    $busy.set(false)
    $connection.set(null)
    $currentFastMode.set(false)
    $currentModel.set('')
    $currentProvider.set('')
    $currentReasoningEffort.set('')
    $currentUsage.set({ calls: 0, input: 0, output: 0, total: 0 })
    $desktopActionTasks.set({})
    $previewServerRestart.set(null)
    $sessionStartedAt.set(null)
    $subagentsBySession.set({})
    $turnStartedAt.set(null)
    $updateApply.set(idleApply)
    $updateStatus.set(null)
    $backendUpdateApply.set(idleApply)
    $backendUpdateStatus.set(null)
    $desktopVersion.set(null)
    $workingSessionIds.set([])
    $approvalMode.set('manual')
    $yoloActive.set(false)
  })

  it('renders the Codex-like status row with gateway, work, usage, model, and versions', () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-06-09T04:50:45Z'))

    $workingSessionIds.set(['sess-running'])
    $busy.set(true)
    $turnStartedAt.set(Date.now() - 115_000)
    $sessionStartedAt.set(Date.now() - 123_000)
    $currentUsage.set({
      calls: 3,
      context_max: 400_000,
      context_percent: 52,
      context_used: 207_400,
      input: 120_000,
      output: 3_000,
      total: 207_400
    })
    $currentModel.set('gpt-5.5')
    $currentProvider.set('openai-codex')
    $currentFastMode.set(true)
    $currentReasoningEffort.set('xhigh')
    $desktopVersion.set({
      appVersion: '0.16.0',
      electronVersion: '40.9.3',
      hermesRoot: '/Users/luigi/.hermes/hermes-agent',
      nodeVersion: '24.12.2',
      platform: 'darwin'
    })
    $updateStatus.set({
      behind: 49,
      fetchedAt: Date.now(),
      supported: true,
      targetSha: 'abcdef1'
    })
    $backendUpdateStatus.set({
      behind: 32,
      fetchedAt: Date.now(),
      supported: true,
      targetSha: 'backend:0.16.0'
    })

    renderStatusbar({ connectionMode: 'local' })

    expect(screen.getByText('Gateway')).toBeTruthy()
    expect(screen.getByText('ready')).toBeTruthy()
    expect(screen.getByText('Agents')).toBeTruthy()
    expect(screen.getByText('1 running')).toBeTruthy()
    expect(screen.getByText('Cron')).toBeTruthy()
    expect(screen.getByText('Running')).toBeTruthy()
    expect(screen.getByText('1:55')).toBeTruthy()
    expect(screen.getByText('207.4k/400.0k')).toBeTruthy()
    expect(screen.getByText(/\[.*] 52%/)).toBeTruthy()
    expect(screen.getByText('Session')).toBeTruthy()
    expect(screen.getByText('2:03')).toBeTruthy()
    expect(screen.getByText(/GPT-5\.5/)).toBeTruthy()
    expect(screen.getByText(/Fast Max/)).toBeTruthy()
    expect(screen.getByText('Ask every time')).toBeTruthy()
    expect(screen.getByText('client v0.16.0 (+49)')).toBeTruthy()
    expect(screen.getByText('backend v0.16.0 (+32)')).toBeTruthy()

    const rowText = document.querySelector('footer')?.textContent ?? ''
    expect(rowText.indexOf('GPT-5.5 · Fast Max')).toBeLessThan(rowText.indexOf('Ask every time'))
    expect(rowText.indexOf('Ask every time')).toBeLessThan(rowText.indexOf('client v0.16.0 (+49)'))
  })

  it('labels the approval statusbar menu with the current effective mode', () => {
    $currentModel.set('gpt-5.5')
    $currentProvider.set('openai-codex')

    const manual = renderStatusbar()
    expect(screen.getByRole('button', { name: /Ask every time/i })).toBeTruthy()
    manual.unmount()
    cleanup()

    $approvalMode.set('smart')
    const smart = renderStatusbar()
    expect(screen.getByRole('button', { name: /Smart approvals/i })).toBeTruthy()
    smart.unmount()
    cleanup()

    $approvalMode.set('manual')
    $yoloActive.set(true)
    renderStatusbar()
    expect(screen.getByRole('button', { name: /Full access/i })).toBeTruthy()
  })

  it('opens a Codex-like approval mode menu and enables full access globally', async () => {
    $activeSessionId.set('sess-1')
    $currentModel.set('gpt-5.5')
    $currentProvider.set('openai-codex')
    $desktopVersion.set({
      appVersion: '0.16.0',
      electronVersion: '40.9.3',
      hermesRoot: '/Users/luigi/.hermes/hermes-agent',
      nodeVersion: '24.12.2',
      platform: 'darwin'
    })
    const requestGateway = vi.fn(async () => ({ value: '1' })) as RequestGateway

    renderStatusbar({ requestGateway })

    fireEvent.pointerDown(screen.getByRole('button', { name: /Ask every time/i }), {
      button: 0,
      ctrlKey: false
    })

    expect(await screen.findByText('How should Hermes actions be approved?')).toBeTruthy()
    expect(screen.getByText('Always ask to edit external files and use the internet')).toBeTruthy()
    expect(screen.getByText('Only ask for actions detected as potentially unsafe')).toBeTruthy()
    expect(screen.getByText('Unrestricted access to the internet and any file on your computer')).toBeTruthy()

    fireEvent.click(screen.getByRole('menuitemradio', { name: /Full access/i }))

    await waitFor(() => {
      expect(requestGateway).toHaveBeenCalledWith('config.set', {
        key: 'approvals.mode',
        value: 'off'
      })
    })
    expect($yoloActive.get()).toBe(true)
    expect(screen.getByRole('button', { name: /Full access/i })).toBeTruthy()
  })

  it('sets ask-every-time as the manual approval mode', async () => {
    $activeSessionId.set('sess-1')
    $currentModel.set('gpt-5.5')
    $currentProvider.set('openai-codex')
    const requestGateway = vi.fn(async (_method, params) => ({ value: params?.value })) as RequestGateway

    renderStatusbar({ requestGateway })

    fireEvent.pointerDown(screen.getByRole('button', { name: /Ask every time/i }), {
      button: 0,
      ctrlKey: false
    })
    fireEvent.click(await screen.findByRole('menuitemradio', { name: /Ask every time/i }))

    await waitFor(() => {
      expect(requestGateway).toHaveBeenCalledWith('config.set', {
        key: 'approvals.mode',
        value: 'manual'
      })
    })
    expect($yoloActive.get()).toBe(false)
  })

  it('keeps a newer approval mode selection when the initial mode read resolves late', async () => {
    $activeSessionId.set('sess-1')
    $approvalMode.set('smart')
    $currentModel.set('gpt-5.5')
    $currentProvider.set('openai-codex')
    const initialMode = deferred<{ value: string }>()
    const requestGateway = vi.fn(async (method, params) => {
      if (method === 'config.get' && params?.key === 'approvals.mode') {
        return initialMode.promise
      }

      if (method === 'config.set' && params?.key === 'approvals.mode') {
        return { value: params.value }
      }

      if (method === 'config.set' && params?.key === 'yolo') {
        return { value: params.value }
      }

      return {}
    }) as RequestGateway

    renderStatusbar({ requestGateway })

    fireEvent.pointerDown(screen.getByRole('button', { name: /Smart approvals/i }), {
      button: 0,
      ctrlKey: false
    })
    fireEvent.click(await screen.findByRole('menuitemradio', { name: /Full access/i }))

    await waitFor(() => {
      expect($approvalMode.get()).toBe('off')
      expect($yoloActive.get()).toBe(true)
    })
    expect(screen.getByRole('button', { name: /Full access/i })).toBeTruthy()

    await act(async () => {
      initialMode.resolve({ value: 'manual' })
      await initialMode.promise
    })

    expect($approvalMode.get()).toBe('off')
    expect($yoloActive.get()).toBe(true)
  })

  it('keeps the approval mode menu visible between active sessions', () => {
    $activeSessionId.set(null)
    $currentModel.set('gpt-5.5')
    $currentProvider.set('openai-codex')
    $desktopVersion.set({
      appVersion: '0.16.0',
      electronVersion: '40.9.3',
      hermesRoot: '/Users/luigi/.hermes/hermes-agent',
      nodeVersion: '24.12.2',
      platform: 'darwin'
    })

    renderStatusbar({ freshDraftReady: false })

    const rowText = document.querySelector('footer')?.textContent ?? ''
    expect(screen.getByRole('button', { name: /Ask every time/i })).toBeTruthy()
    expect(rowText.indexOf('GPT-5.5 · Med')).toBeLessThan(rowText.indexOf('Ask every time'))
    expect(rowText.indexOf('Ask every time')).toBeLessThan(rowText.indexOf('client v0.16.0'))
  })

  it('renders the backend version in remote mode too', () => {
    $desktopVersion.set({
      appVersion: '0.16.0',
      electronVersion: '40.9.3',
      hermesRoot: '/Users/luigi/.hermes/hermes-agent',
      nodeVersion: '24.12.2',
      platform: 'darwin'
    })

    renderStatusbar({ connectionMode: 'remote' })

    expect(screen.getByText('client v0.16.0')).toBeTruthy()
    expect(screen.getByText('backend v0.16.0')).toBeTruthy()
  })

  it('counts active subagents as running agent work', () => {
    $subagentsBySession.set({
      parent: [
        {
          filesRead: [],
          filesWritten: [],
          goal: 'Research options',
          id: 'subagent-1',
          parentId: null,
          startedAt: Date.now(),
          status: 'running',
          stream: [],
          taskCount: 1,
          taskIndex: 0,
          updatedAt: Date.now()
        },
        {
          filesRead: [],
          filesWritten: [],
          goal: 'Prepare patch',
          id: 'subagent-2',
          parentId: null,
          startedAt: Date.now(),
          status: 'queued',
          stream: [],
          taskCount: 1,
          taskIndex: 0,
          updatedAt: Date.now()
        }
      ]
    })

    renderStatusbar()

    expect(screen.getByText('Agents')).toBeTruthy()
    expect(screen.getByText('2 running')).toBeTruthy()
  })
})

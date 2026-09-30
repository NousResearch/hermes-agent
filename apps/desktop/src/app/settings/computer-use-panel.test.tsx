import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { getActionStatus, getComputerUseStatus, grantComputerUsePermissions } from '@/hermes'
import { I18nProvider } from '@/i18n'
import { notify } from '@/store/notifications'
import type { ComputerUseStatus } from '@/types/hermes'

import { ComputerUsePanel } from './computer-use-panel'

vi.mock('@/hermes', () => ({
  getActionStatus: vi.fn(),
  getComputerUseStatus: vi.fn(),
  grantComputerUsePermissions: vi.fn()
}))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))
vi.mock('@/store/activity', () => ({ upsertDesktopActionTask: vi.fn() }))

const status: ComputerUseStatus = {
  platform: 'darwin',
  platform_supported: true,
  installed: true,
  version: 'cua-driver fixture',
  ready: false,
  can_grant: true,
  checks: [],
  accessibility: false,
  screen_recording: null,
  screen_recording_capturable: null,
  source: null,
  error: null
}

const panel = () =>
  render(
    <I18nProvider configClient={null} initialLocale="ko">
      <ComputerUsePanel />
    </I18nProvider>
  )

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.clearAllMocks()
})

it('renders Korean permission and driver states from backend platform data, preserving diagnostics', async () => {
  const cases: Array<{ patch: Partial<ComputerUseStatus>; expected: string[] }> = [
    { patch: { platform_supported: false, platform: 'future-os' }, expected: ['future-os', '지원하지 않습니다'] },
    { patch: { installed: false }, expected: ['cua-driver', '설치', '손쉬운 사용', '화면 기록'] },
    { patch: {}, expected: ['허용되지 않음', '알 수 없음', 'com.trycua.driver', '권한 요청'] },
    {
      patch: { accessibility: true, screen_recording: true, ready: true },
      expected: ['허용됨', '사용할 준비가 되었습니다']
    },
    { patch: { platform: 'win32', can_grant: false }, expected: ['Windows SmartScreen', '준비되지 않음'] },
    { patch: { platform: 'linux', can_grant: false, ready: null }, expected: ['X11/XWayland', '알 수 없음'] },
    {
      patch: {
        platform: 'linux',
        can_grant: false,
        ready: true,
        checks: [{ label: 'raw-check-ID', status: 'warn', message: 'Original diagnostic' }],
        error: 'Original backend error'
      },
      expected: ['준비 완료', 'raw-check-ID', 'Original diagnostic', 'Original backend error']
    }
  ]

  for (const { patch, expected } of cases) {
    vi.mocked(getComputerUseStatus).mockResolvedValue({ ...status, ...patch })
    panel()
    expect(screen.getByText('컴퓨터 사용 상태 확인 중…')).toBeTruthy()
    await waitFor(() => expect(screen.queryByText('컴퓨터 사용 상태 확인 중…')).toBeNull())

    for (const text of expected) {
      expect(window.document.body.textContent).toContain(text)
    }

    expect(grantComputerUsePermissions).not.toHaveBeenCalled()
    cleanup()
  }
})

it('keeps the explicit permission action and polling contract while translating approval guidance', async () => {
  vi.useFakeTimers()
  vi.mocked(getComputerUseStatus)
    .mockResolvedValueOnce(status)
    .mockResolvedValueOnce({ ...status, ready: true })
  vi.mocked(grantComputerUsePermissions).mockResolvedValue({ ok: true, name: 'fixture-action', pid: 12345 })
  vi.mocked(getActionStatus).mockResolvedValue({ name: 'fixture-action', running: false } as Awaited<
    ReturnType<typeof getActionStatus>
  >)
  await act(async () => {
    panel()
  })
  await act(async () => {
    fireEvent.click(screen.getByRole('button', { name: '권한 요청' }))
  })
  expect(grantComputerUsePermissions).toHaveBeenCalledTimes(1)
  expect(screen.getByRole('button', { name: '승인 대기 중…' })).toHaveProperty('disabled', true)
  expect(notify).toHaveBeenCalledWith(
    expect.objectContaining({ title: '시스템 설정에서 승인하세요', message: expect.stringContaining('CuaDriver') })
  )
  await act(async () => {
    await vi.advanceTimersByTimeAsync(1500)
  })
  expect(getActionStatus).toHaveBeenCalledWith('fixture-action', 200)
  expect(screen.getByText(/컴퓨터를 사용할 준비가 되었습니다/)).toBeTruthy()
})

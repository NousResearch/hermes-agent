import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { PRIMARY_SESSION_VIEW, SessionViewProvider } from '@/app/chat/session-view'
import type { HermesConnection } from '@/global'
import { I18nProvider, type Locale, setRuntimeI18nLocale, useI18n } from '@/i18n'
import { folderMarkdownHref } from '@/lib/folder-links'
import { $connection, _resetSessionOwnerHintsForTests, setSessionOwnerHint } from '@/store/session'

import { MarkdownTextContent } from './markdown-text'

const originalBridge = window.hermesDesktop
const owner = { connectionId: 'local', profile: 'default', mode: 'local' as const }

function setup() {
  const openExistingDirectory = vi.fn(async () => ({ ok: true }))
  window.hermesDesktop = { openExistingDirectory } as unknown as typeof window.hermesDesktop
  $connection.set({ ...owner } as HermesConnection)
  setSessionOwnerHint('folder-session', owner)

  const view = {
    ...PRIMARY_SESSION_VIEW,
    kind: 'tile' as const,
    $storedId: atom<string | null>('folder-session'),
    $runtimeId: atom<string | null>(null)
  }

  return { openExistingDirectory, view }
}

function SwitchLocale({ locale }: { locale: Locale }) {
  const { setLocale } = useI18n()

  return <button onClick={() => void setLocale(locale)}>Switch locale</button>
}

const localizedCopy = [
  {
    locale: 'ru',
    guard: 'Ссылки на папки требуют текущего локального подключения этой сессии.',
    unavailable: 'Открытие папок в системе недоступно.',
    failed: 'Не удалось открыть папку.',
    failedWithMessage: (message: string) => `Не удалось открыть папку: ${message}`
  },
  {
    locale: 'ja',
    guard: 'フォルダーリンクには、このセッションの現在のローカル接続が必要です。',
    unavailable: 'システムでフォルダーを開く機能は利用できません。',
    failed: 'フォルダーを開けませんでした。',
    failedWithMessage: (message: string) => `フォルダーを開けませんでした：${message}`
  }
] as const

afterEach(() => {
  cleanup()
  setRuntimeI18nLocale('en')
  window.document.documentElement.lang = 'en'
  window.document.documentElement.dir = 'ltr'
  vi.restoreAllMocks()
  window.hermesDesktop = originalBridge
  $connection.set(null)
  _resetSessionOwnerHintsForTests({ storage: true })
})

describe('MarkdownTextContent folder links', () => {
  it.each(localizedCopy)('localizes fail-closed guards in $locale and follows locale changes', async copy => {
    const { openExistingDirectory, view } = setup()
    const browserOpen = vi.spyOn(window, 'open')

    const { container } = render(
      <I18nProvider configClient={null} initialLocale={copy.locale}>
        <SwitchLocale locale="en" />
        <SessionViewProvider value={view}>
          <MarkdownTextContent isRunning={false} text={`[Folder](${folderMarkdownHref('C:/Reports')})`} />
        </SessionViewProvider>
      </I18nProvider>
    )

    const button = await screen.findByRole('button', { name: 'Folder' })

    $connection.set(null)
    fireEvent.click(button)
    expect((await screen.findByRole('alert')).textContent).toBe(copy.guard)
    expect(openExistingDirectory).not.toHaveBeenCalled()

    $connection.set({ ...owner } as HermesConnection)

    for (const bridge of [{}, undefined]) {
      window.hermesDesktop = bridge as typeof window.hermesDesktop
      fireEvent.click(button)
      expect((await screen.findByRole('alert')).textContent).toBe(copy.unavailable)
    }

    fireEvent.click(screen.getByRole('button', { name: 'Switch locale' }))
    await waitFor(() => expect(screen.getByRole('alert').textContent).toBe('Native folder opening is unavailable.'))
    expect(openExistingDirectory).not.toHaveBeenCalled()
    expect(browserOpen).not.toHaveBeenCalled()
    expect(container.querySelector('a')).toBeNull()
  })

  it.each(localizedCopy)(
    'localizes native failures in $locale without translating diagnostics or falling back',
    async copy => {
      const { openExistingDirectory, view } = setup()
      const browserOpen = vi.spyOn(window, 'open')
      const diagnostic = 'EACCES: C:/Reports/Проект #1 — access denied <native>'

      const { container } = render(
        <I18nProvider configClient={null} initialLocale={copy.locale}>
          <SwitchLocale locale="en" />
          <SessionViewProvider value={view}>
            <MarkdownTextContent isRunning={false} text={`[Folder](${folderMarkdownHref('C:/Reports')})`} />
          </SessionViewProvider>
        </I18nProvider>
      )

      const button = await screen.findByRole('button', { name: 'Folder' })

      openExistingDirectory.mockResolvedValueOnce({ ok: false })
      fireEvent.click(button)
      expect((await screen.findByRole('alert')).textContent).toBe(copy.failed)

      openExistingDirectory.mockRejectedValueOnce(new Error(''))
      fireEvent.click(button)
      expect((await screen.findByRole('alert')).textContent).toBe(copy.failed)

      openExistingDirectory.mockResolvedValueOnce({ ok: false, error: diagnostic } as { ok: boolean })
      fireEvent.click(button)
      expect((await screen.findByRole('alert')).textContent).toBe(copy.failedWithMessage(diagnostic))

      for (const cause of [new Error(diagnostic), diagnostic]) {
        openExistingDirectory.mockRejectedValueOnce(cause)
        fireEvent.click(button)
        expect((await screen.findByRole('alert')).textContent).toBe(copy.failedWithMessage(diagnostic))
      }

      fireEvent.click(screen.getByRole('button', { name: 'Switch locale' }))
      await waitFor(() => expect(screen.getByRole('alert').textContent).toBe(`Could not open folder: ${diagnostic}`))
      expect(openExistingDirectory).toHaveBeenCalledTimes(5)
      expect(browserOpen).not.toHaveBeenCalled()
      expect(container.querySelector('a')).toBeNull()
    }
  )

  it('fails closed for remote, unknown, stale and other-tile owners at click time', async () => {
    const { openExistingDirectory, view } = setup()
    render(
      <SessionViewProvider value={view}>
        <MarkdownTextContent isRunning={false} text={`[Folder](${folderMarkdownHref('C:/Reports')})`} />
      </SessionViewProvider>
    )
    const button = await screen.findByRole('button', { name: 'Folder' })

    for (const connection of [
      null,
      { ...owner, mode: undefined },
      { ...owner, mode: 'remote', baseUrl: 'http://127.0.0.1:8000' },
      { ...owner, connectionId: 'elsewhere' },
      { ...owner, profile: 'other' }
    ]) {
      $connection.set(connection as HermesConnection | null)
      fireEvent.click(button)
      await screen.findByRole('alert')
      expect(openExistingDirectory).not.toHaveBeenCalled()
    }

    $connection.set({ ...owner } as HermesConnection)

    for (const route of [undefined, { ...owner, mode: 'remote' as const }, { ...owner, connectionId: 'remote-tile' }]) {
      _resetSessionOwnerHintsForTests()

      if (route) {
        setSessionOwnerHint('folder-session', route)
      }

      fireEvent.click(button)
      expect(openExistingDirectory).not.toHaveBeenCalled()
    }
  })

  it('keeps invalid payloads inert and reports native errors without a browser fallback', async () => {
    const { openExistingDirectory, view } = setup()

    const { container } = render(
      <SessionViewProvider value={view}>
        <MarkdownTextContent
          isRunning={false}
          text={'[Bad](#folder/%ZZ) [Relative](#folder/relative) [Folder](#folder/C%3A%2FReports)'}
        />
      </SessionViewProvider>
    )

    await screen.findByText('Bad')
    expect(screen.queryByRole('button', { name: 'Bad' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Relative' })).toBeNull()
    expect(container.querySelector('a')).toBeNull()
    openExistingDirectory.mockResolvedValueOnce({ ok: false, error: 'Denied by native validation' } as { ok: boolean })
    fireEvent.click(screen.getByRole('button', { name: 'Folder' }))
    expect((await screen.findByRole('alert')).textContent).toContain('Denied by native validation')
    openExistingDirectory.mockRejectedValueOnce(new Error('IPC unavailable'))
    fireEvent.click(screen.getByRole('button', { name: 'Folder' }))
    await waitFor(() => expect(screen.getByRole('alert').textContent).toContain('IPC unavailable'))
    expect(container.querySelector('a')).toBeNull()
  })

  it('preserves the label and sends a real click to only the narrow native capability', async () => {
    const { openExistingDirectory, view } = setup()
    const path = 'C:/Reports/Проект #1 50% (final)'

    const { container } = render(
      <SessionViewProvider value={view}>
        <MarkdownTextContent isRunning={false} text={`[Open **folder**](${folderMarkdownHref(path)})`} />
      </SessionViewProvider>
    )

    const button = await screen.findByRole('button', { name: 'Open folder' })
    expect(container.querySelector('a')).toBeNull()
    expect(button.querySelector('[data-streamdown="strong"]')?.textContent).toBe('folder')
    expect(openExistingDirectory).not.toHaveBeenCalled()
    fireEvent.click(button)
    await waitFor(() =>
      expect(openExistingDirectory).toHaveBeenCalledExactlyOnceWith({
        path,
        owner: { connectionId: owner.connectionId, profile: owner.profile }
      })
    )
    expect(screen.queryByRole('button', { name: 'Open preview' })).toBeNull()
  })
})

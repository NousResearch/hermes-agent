import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/api/client'
import type { HermesApiRequest } from '@/global'
import type { HermesConfigRecord } from '@/types/hermes'

import { I18nProvider, useI18n } from './context'
import type { Locale } from './types'

function Probe({ target = 'ko' }: { target?: Locale }) {
  const { locale, isLoadingConfig, isSavingLocale, setLocale } = useI18n()

  return (
    <>
      <span data-testid="locale">{locale}</span>
      <span data-testid="ready">{String(!isLoadingConfig && !isSavingLocale)}</span>
      <button onClick={() => void setLocale(target)} type="button">
        {target === 'ko' ? '한국어' : 'English'}
      </button>
    </>
  )
}

describe('language persistence through the real renderer config API', () => {
  let directory: string
  let pauseRead: (() => Promise<void>) | undefined
  let machineLocale: string | undefined
  const configPath = (connection: string, profile: string) => join(directory, `${connection}-${profile}.json`)

  const readConfig = (connection: string, profile: string): HermesConfigRecord =>
    JSON.parse(readFileSync(configPath(connection, profile), 'utf8'))

  beforeEach(() => {
    directory = mkdtempSync(join(tmpdir(), 'hermes-korean-config-'))
    pauseRead = undefined
    machineLocale = undefined

    for (const connection of ['local', 'remote']) {
      for (const profile of ['default', 'writer']) {
        writeFileSync(
          configPath(connection, profile),
          JSON.stringify({
            display: { language: 'en', skin: `${connection}-${profile}` },
            terminal: { cwd: `/${connection}/${profile}` }
          })
        )
      }
    }

    // Only replace the native transport boundary. The provider, API helpers,
    // scope resolver and config merge run unchanged; persistence is real disk I/O.
    vi.stubGlobal('hermesDesktop', {
      getMachineProfile: async () => ({ locale: machineLocale }),
      api: async (request: HermesApiRequest) => {
        const url = new URL(request.path, 'http://hermes.test')
        expect(url.pathname).toBe('/api/config')
        const connection = request.connectionId || 'local'
        const profile = request.profile || 'default'

        if (request.method === 'PUT') {
          const incoming = structuredClone((request.body as { config: HermesConfigRecord }).config)
          const previous = readConfig(connection, profile)

          // The backend deep-merges PUT fields onto the latest disk contents.
          // These fixtures use a nested display and an otherwise untouched terminal.
          const saved: HermesConfigRecord = {
            ...previous,
            ...incoming,
            display: {
              ...(previous.display as Record<string, unknown>),
              ...(incoming.display as Record<string, unknown>)
            }
          }

          const display = saved.display as Record<string, unknown>
          const previousDisplay = previous.display
          const languageWasExplicit = Object.hasOwn(previousDisplay ?? {}, 'language')

          // Match the backend's sparse writer: existing explicit keys survive,
          // while a first choice of default English needs preserve_language.
          if (
            display?.language === 'en' &&
            !languageWasExplicit &&
            url.searchParams.get('preserve_language') !== 'true'
          ) {
            delete display.language
          }

          writeFileSync(configPath(connection, profile), JSON.stringify(saved))

          return { ok: true }
        }

        const config = readConfig(connection, profile)

        if (url.searchParams.get('include_defaults') !== 'false') {
          config.display = { language: 'en', ...(config.display as Record<string, unknown>) }
        }

        const wait = pauseRead
        pauseRead = undefined
        await wait?.()

        return config
      }
    })
  })

  afterEach(() => {
    cleanup()
    setApiRequestProfile(null)
    setApiRequestConnection(null)
    vi.unstubAllGlobals()
    rmSync(directory, { recursive: true, force: true })
  })

  it.each(['local', 'remote'])('saves and reloads Korean independently for profiles on %s', async connection => {
    setApiRequestConnection(connection)
    setApiRequestProfile('writer')

    const view = render(
      <I18nProvider>
        <Probe />
      </I18nProvider>
    )

    await waitFor(() => expect(screen.getByTestId('ready').textContent).toBe('true'))
    fireEvent.click(screen.getByRole('button', { name: '한국어' }))
    await waitFor(() =>
      expect(readConfig(connection, 'writer').display).toEqual({ language: 'ko', skin: `${connection}-writer` })
    )
    view.unmount()

    const restarted = render(
      <I18nProvider>
        <Probe />
      </I18nProvider>
    )

    await waitFor(() => expect(screen.getByTestId('locale').textContent).toBe('ko'))
    restarted.unmount()
    setApiRequestProfile('default')
    render(
      <I18nProvider>
        <Probe />
      </I18nProvider>
    )
    await waitFor(() => expect(screen.getByTestId('ready').textContent).toBe('true'))
    expect(screen.getByTestId('locale').textContent).toBe('en')
    expect(readConfig(connection, 'writer').terminal).toEqual({ cwd: `/${connection}/writer` })
    expect(readConfig(connection, 'default').display).toEqual({ language: 'en', skin: `${connection}-default` })
  })

  it.each(['local', 'remote'])('keeps settings changed after the language read on %s', async connection => {
    setApiRequestConnection(connection)
    setApiRequestProfile('writer')
    render(
      <I18nProvider>
        <Probe />
      </I18nProvider>
    )
    await waitFor(() => expect(screen.getByTestId('ready').textContent).toBe('true'))
    let releaseRead!: () => void
    pauseRead = () =>
      new Promise<void>(resolve => {
        releaseRead = resolve
      })
    fireEvent.click(screen.getByRole('button', { name: '한국어' }))
    await waitFor(() => expect(releaseRead).toBeTypeOf('function'))
    writeFileSync(
      configPath(connection, 'writer'),
      JSON.stringify({
        display: { language: 'en', skin: 'new-theme' },
        terminal: { cwd: '/changed-elsewhere' }
      })
    )
    await act(async () => {
      releaseRead()
    })
    await waitFor(() => expect(screen.getByTestId('ready').textContent).toBe('true'))
    expect(readConfig(connection, 'writer')).toEqual({
      display: { language: 'ko', skin: 'new-theme' },
      terminal: { cwd: '/changed-elsewhere' }
    })
    expect(readConfig(connection, 'default').display).toEqual({ language: 'en', skin: `${connection}-default` })
  })

  it('reloads a changed scope and keeps an in-flight language save on its original profile', async () => {
    setApiRequestConnection('remote')
    setApiRequestProfile('writer')
    render(
      <I18nProvider>
        <Probe />
      </I18nProvider>
    )
    await waitFor(() => expect(screen.getByTestId('ready').textContent).toBe('true'))
    let releaseRead!: () => void
    pauseRead = () =>
      new Promise<void>(resolve => {
        releaseRead = resolve
      })
    fireEvent.click(screen.getByRole('button', { name: '한국어' }))
    await waitFor(() => expect(releaseRead).toBeTypeOf('function'))
    act(() => {
      setApiRequestConnection('local')
      setApiRequestProfile('default')
    })
    await act(async () => {
      releaseRead()
    })
    await waitFor(() => expect(screen.getByTestId('ready').textContent).toBe('true'))
    expect(screen.getByTestId('locale').textContent).toBe('en')
    expect(readConfig('remote', 'writer').display).toEqual({ language: 'ko', skin: 'remote-writer' })
    expect(readConfig('local', 'default').display).toEqual({ language: 'en', skin: 'local-default' })
    // A saved Korean profile that becomes active after startup must be reloaded.
    act(() => {
      setApiRequestConnection('remote')
      setApiRequestProfile('writer')
    })
    await waitFor(() => expect(screen.getByTestId('locale').textContent).toBe('ko'))
  })

  it.each(['local', 'remote'])(
    'infers an unset locale without saving, then remembers explicit English on %s',
    async connection => {
      machineLocale = 'ko-KR'
      writeFileSync(configPath(connection, 'writer'), JSON.stringify({ display: { skin: 'mono' } }))
      setApiRequestConnection(connection)
      setApiRequestProfile('writer')

      const view = render(
        <I18nProvider>
          <Probe target="en" />
        </I18nProvider>
      )

      await waitFor(() => expect(screen.getByTestId('locale').textContent).toBe('ko'))
      expect(readConfig(connection, 'writer').display).toEqual({ skin: 'mono' })

      fireEvent.click(screen.getByRole('button', { name: 'English' }))
      await waitFor(() => expect(readConfig(connection, 'writer').display).toEqual({ language: 'en', skin: 'mono' }))
      view.unmount()
      render(
        <I18nProvider>
          <Probe />
        </I18nProvider>
      )
      await waitFor(() => expect(screen.getByTestId('ready').textContent).toBe('true'))
      expect(screen.getByTestId('locale').textContent).toBe('en')
      expect(readConfig(connection, 'default').display).toEqual({ language: 'en', skin: `${connection}-default` })
    }
  )
})

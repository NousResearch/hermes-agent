import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { $shellChoice } from '@/store/shell-variant'

import { I18nProvider, useI18n } from './context'
import { createPluginI18n, registerPluginLocales, translatePlugin, usePluginI18n } from './plugin-i18n'
import { setRuntimeI18nLocale } from './runtime'

const noopTrack = (dispose: () => void) => dispose

afterEach(() => {
  cleanup()
  setRuntimeI18nLocale('en')
  $shellChoice.set(null)
})

describe('plugin locale registry', () => {
  it('resolves the active locale, falling back to English then the raw key', () => {
    const dispose = registerPluginLocales('cost', {
      en: { panel: { title: 'Cost' }, spent: (n: number) => `$${n} spent` },
      ja: { panel: { title: 'コスト' } }
    })

    expect(translatePlugin('cost', 'ja', 'panel.title', [])).toBe('コスト')
    // Missing in ja → English.
    expect(translatePlugin('cost', 'ja', 'spent', [7])).toBe('$7 spent')
    // Missing everywhere → the key itself.
    expect(translatePlugin('cost', 'ja', 'nope', [])).toBe('nope')

    dispose()
  })

  it('scopes bundles per plugin — no cross-read', () => {
    const a = registerPluginLocales('a', { en: { hi: 'from a' } })
    const b = registerPluginLocales('b', { en: { hi: 'from b' } })

    expect(translatePlugin('a', 'en', 'hi', [])).toBe('from a')
    expect(translatePlugin('b', 'en', 'hi', [])).toBe('from b')
    // An unknown plugin resolves to the key.
    expect(translatePlugin('c', 'en', 'hi', [])).toBe('hi')

    a()
    b()
  })

  it('merges repeated registrations and drops everything on dispose', () => {
    const one = registerPluginLocales('merge', { en: { a: 'A' } })
    const two = registerPluginLocales('merge', { en: { b: 'B' }, ja: { a: 'あ' } })

    expect(translatePlugin('merge', 'en', 'a', [])).toBe('A')
    expect(translatePlugin('merge', 'en', 'b', [])).toBe('B')
    expect(translatePlugin('merge', 'ja', 'a', [])).toBe('あ')

    one()
    two()

    expect(translatePlugin('merge', 'en', 'a', [])).toBe('a')
  })

  it('reads a redesign wording layer only in the redesigned shell, per locale, before English', () => {
    const dispose = registerPluginLocales(
      'roster',
      {
        en: { title: 'Bots', count: (n: number) => `${n} bots`, search: 'Search' },
        ja: { title: 'ボット', search: '検索' }
      },
      { redesign: { en: { title: 'Agents', count: (n: number) => `${n} agents` }, ja: { title: 'エージェント' } } }
    )

    expect(translatePlugin('roster', 'en', 'title', [], 'classic')).toBe('Bots')
    expect(translatePlugin('roster', 'en', 'title', [], 'redesign')).toBe('Agents')
    expect(translatePlugin('roster', 'ja', 'title', [], 'classic')).toBe('ボット')
    expect(translatePlugin('roster', 'ja', 'title', [], 'redesign')).toBe('エージェント')
    // A key the layer leaves out reads the same in both shells.
    expect(translatePlugin('roster', 'ja', 'search', [], 'redesign')).toBe('検索')
    // Missing in the locale → the English bundle in the SAME shell's wording.
    expect(translatePlugin('roster', 'ja', 'count', [2], 'classic')).toBe('2 bots')
    expect(translatePlugin('roster', 'ja', 'count', [2], 'redesign')).toBe('2 agents')

    dispose()
    expect(translatePlugin('roster', 'en', 'title', [], 'redesign')).toBe('title')
  })

  it('reads a plugin without a wording layer the same in both shells', () => {
    const dispose = registerPluginLocales('plain', { en: { title: 'Board' } })

    expect(translatePlugin('plain', 'en', 'title', [], 'classic')).toBe('Board')
    expect(translatePlugin('plain', 'en', 'title', [], 'redesign')).toBe('Board')

    dispose()
  })

  it('locale listeners ignore no-op changes and unsubscribe both explicitly and on unload', () => {
    const disposers: Array<() => void> = []

    const i18n = createPluginI18n('listener-plugin', dispose => {
      disposers.push(dispose)

      return dispose
    })

    i18n.register({ en: { greet: 'hello' }, ja: { greet: 'こんにちは' } })
    const calls: string[] = []
    const unsubscribe = i18n.onLocaleChange(() => calls.push(i18n.t('greet')))
    expect(calls).toEqual([])
    setRuntimeI18nLocale('en')
    expect(calls).toEqual([])
    setRuntimeI18nLocale('ja')
    expect(calls).toEqual(['こんにちは'])
    unsubscribe()
    setRuntimeI18nLocale('en')
    expect(calls).toEqual(['こんにちは'])
    i18n.onLocaleChange(() => calls.push(i18n.t('greet')))
    disposers.forEach(dispose => dispose())
    setRuntimeI18nLocale('ja')
    expect(calls).toEqual(['こんにちは'])
  })

  it('ctx.i18n.t reads the app runtime locale', () => {
    const i18n = createPluginI18n('runtime-plugin', noopTrack)
    i18n.register({ en: { greet: 'hello' }, ja: { greet: 'こんにちは' } })

    expect(i18n.t('greet')).toBe('hello')

    setRuntimeI18nLocale('ja')
    expect(i18n.t('greet')).toBe('こんにちは')
  })
})

function Probe({ pluginId }: { pluginId: string }) {
  const t = usePluginI18n(pluginId)

  return <p data-testid="copy">{t('greet')}</p>
}

function SwitchToJa() {
  const { setLocale } = useI18n()

  return (
    <button onClick={() => void setLocale('ja')} type="button">
      to ja
    </button>
  )
}

describe('usePluginI18n', () => {
  it('re-renders on a locale switch', () => {
    const dispose = registerPluginLocales('hooked', {
      en: { greet: 'hello' },
      ja: { greet: 'こんにちは' }
    })

    render(
      <I18nProvider configClient={null}>
        <SwitchToJa />
        <Probe pluginId="hooked" />
      </I18nProvider>
    )

    expect(screen.getByTestId('copy').textContent).toBe('hello')

    fireEvent.click(screen.getByRole('button'))

    expect(screen.getByTestId('copy').textContent).toBe('こんにちは')

    dispose()
  })

  it('picks up a bundle registered after mount', () => {
    render(
      <I18nProvider configClient={null}>
        <Probe pluginId="late" />
      </I18nProvider>
    )

    expect(screen.getByTestId('copy').textContent).toBe('greet')

    let dispose = () => {}
    act(() => {
      dispose = registerPluginLocales('late', { en: { greet: 'landed' } })
    })

    expect(screen.getByTestId('copy').textContent).toBe('landed')

    dispose()
  })

  it('re-renders on a shell flip with the redesign layer', () => {
    const dispose = registerPluginLocales('flip', { en: { greet: 'Bots' } }, { redesign: { en: { greet: 'Agents' } } })
    $shellChoice.set(false)

    render(
      <I18nProvider configClient={null}>
        <Probe pluginId="flip" />
      </I18nProvider>
    )

    expect(screen.getByTestId('copy').textContent).toBe('Bots')

    act(() => $shellChoice.set(true))
    expect(screen.getByTestId('copy').textContent).toBe('Agents')

    act(() => $shellChoice.set(false))
    expect(screen.getByTestId('copy').textContent).toBe('Bots')

    dispose()
  })
})

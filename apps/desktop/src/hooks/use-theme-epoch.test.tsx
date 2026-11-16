import { act, cleanup, render } from '@testing-library/react'
import { useEffect } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $accentOverride } from '@/themes/accent-override'
import { __resetBackendSkinSync, ingestBackendSkin } from '@/themes/backend-sync'
import { $chatFontFamily } from '@/themes/chat-font'
import { ThemeProvider, useTheme } from '@/themes/context'

import { onThemeRepaint, useThemeEpoch } from './use-theme-epoch'

const root = window.document.documentElement
const originalStyle = root.style.cssText
const originalClass = root.className
const disposers: Array<() => void> = []

const flush = async () => {
  await act(async () => {
    await Promise.resolve()
  })
}

afterEach(async () => {
  cleanup()
  disposers.splice(0).forEach(dispose => dispose())
  $accentOverride.set(null)
  $chatFontFamily.set('')
  __resetBackendSkinSync()
  localStorage.clear()
  root.style.cssText = originalStyle
  root.className = originalClass
  delete root.dataset.hermesMode
  delete root.dataset.hermesTheme
  window.document.getElementById('hermes-desktop-custom-css')?.remove()
  await flush()
})

describe('theme repaint subscriptions', () => {
  it('ignores geometry from the first subscription, coalesces real changes, and survives re-subscription', async () => {
    root.style.setProperty('--theme-primary', '#123456')
    await flush()
    const first = vi.fn()
    const second = vi.fn()
    const stopFirst = onThemeRepaint(first)
    const stopSecond = onThemeRepaint(second)
    disposers.push(stopFirst, stopSecond)

    for (const value of ['10px', '20px', '']) {
      for (const name of ['--workspace-left', '--workspace-right', '--glass-rail-edge']) {
        if (value) {
          root.style.setProperty(name, value)
        } else {
          root.style.removeProperty(name)
        }
      }

      await flush()
    }

    expect(first).not.toHaveBeenCalled()
    expect(second).not.toHaveBeenCalled()

    // Geometry must not hide real style changes in the SAME observer batch.
    root.style.setProperty('--workspace-left', '30px')
    root.style.setProperty('--theme-primary', '#654321')
    root.style.setProperty('--dt-font-sans', 'monospace')
    await flush()
    expect(first).toHaveBeenCalledTimes(1)
    expect(second).toHaveBeenCalledTimes(1)

    stopFirst()
    root.style.setProperty('--theme-primary', '#456789')
    await flush()
    expect(first).toHaveBeenCalledTimes(1)
    expect(second).toHaveBeenCalledTimes(2)
    stopSecond()
    root.style.setProperty('--theme-primary', '#abcdef')
    await flush()
    const resumed = vi.fn()
    disposers.push(onThemeRepaint(resumed))
    root.style.setProperty('--workspace-right', '45px')
    await flush()
    expect(resumed).not.toHaveBeenCalled()
    root.style.setProperty('--theme-primary', '#fedcba')
    await flush()
    expect(resumed).toHaveBeenCalledTimes(1)
    expect(first).toHaveBeenCalledTimes(1)
    expect(second).toHaveBeenCalledTimes(2)
  })

  it('preserves arbitrary inline styles, removals, priority changes, and root theme attributes', async () => {
    const repaint = vi.fn()
    disposers.push(onThemeRepaint(repaint))

    const mutations = [
      () => root.style.setProperty('--author-token', 'rgb(1, 2, 3)'),
      () => root.style.setProperty('--author-token', 'rgb(1, 2, 3)', 'important'),
      () => root.style.removeProperty('--author-token'),
      () => {
        root.style.fontFamily = 'monospace'
      },
      () => root.style.setProperty('--dt-base-size', '18px'),
      () => root.style.removeProperty('--dt-base-size'),
      () => root.classList.toggle('dark'),
      () => {
        root.dataset.hermesMode = 'dark'
      },
      () => {
        root.dataset.hermesTheme = 'authored'
      },
      // applyTheme uses these writes to announce customCSS-only edits too.
      () => {
        root.dataset.hermesTheme = 'authored'
      }
    ]

    for (const [i, mutate] of mutations.entries()) {
      mutate()
      await flush()
      expect(repaint).toHaveBeenCalledTimes(i + 1)
    }
  })

  it('keeps geometry-driven authored CSS observable without changing its consumers', async () => {
    const style = window.document.createElement('style')
    style.id = 'hermes-desktop-custom-css'
    style.textContent = ':root { --author-width: var(--workspace-left); }'
    window.document.head.append(style)
    const repaint = vi.fn()
    disposers.push(onThemeRepaint(repaint))
    root.style.setProperty('--workspace-left', '37px')
    await flush()
    expect(repaint).toHaveBeenCalledOnce()
    style.remove()
    root.style.setProperty('--workspace-left', '38px')
    await flush()
    expect(repaint).toHaveBeenCalledOnce()
  })

  it('reads the provider paint after mount, skin/mode/font/accent changes, and same-palette customCSS edits', async () => {
    let setTheme!: (name: string) => void
    let setDarkMode!: () => void
    const paints = vi.fn()
    let mounts = 0
    let unmounts = 0

    function Controls() {
      const theme = useTheme()
      setTheme = theme.setTheme
      setDarkMode = () => theme.setMode('dark')

      return null
    }

    function Consumer() {
      const epoch = useThemeEpoch()
      useEffect(() => {
        mounts += 1

        return () => {
          unmounts += 1
        }
      }, [])
      useEffect(() => {
        paints({
          skin: root.dataset.hermesTheme,
          mode: root.dataset.hermesMode,
          primary: root.style.getPropertyValue('--theme-primary'),
          font: root.style.getPropertyValue('--dt-font-sans'),
          css: window.document.getElementById('hermes-desktop-custom-css')?.textContent
        })
      }, [epoch])

      return null
    }

    render(
      <ThemeProvider>
        <Controls />
        <Consumer />
      </ThemeProvider>
    )
    await flush()
    expect(paints).toHaveBeenLastCalledWith(
      expect.objectContaining({ skin: root.dataset.hermesTheme, mode: root.dataset.hermesMode })
    )

    await act(async () => setTheme('everforest'))
    await flush()
    expect(paints).toHaveBeenLastCalledWith(expect.objectContaining({ skin: 'everforest' }))
    await act(async () => setDarkMode())
    await flush()
    expect(paints).toHaveBeenLastCalledWith(expect.objectContaining({ mode: 'dark' }))
    await act(async () => $chatFontFamily.set('Theme Regression Font'))
    await flush()
    expect(paints).toHaveBeenLastCalledWith(
      expect.objectContaining({ font: expect.stringContaining('Theme Regression Font') })
    )
    const beforeAccent = paints.mock.lastCall![0].primary
    await act(async () => $accentOverride.set('#1234ab'))
    await flush()
    expect(paints.mock.lastCall![0].primary).not.toBe(beforeAccent)
    await act(async () => $accentOverride.set(null))

    const skin = {
      name: 'epoch-regression',
      colors: { background: '#000000', ui_text: '#eeeeee', ui_accent: '#ff8000' }
    }

    await act(async () => ingestBackendSkin({ ...skin, customCSS: 'body { font-weight: 400; }' }, { apply: true }))
    await flush()
    await act(async () => ingestBackendSkin({ ...skin, customCSS: 'body { font-weight: 600; }' }, { apply: true }))
    await flush()
    expect(paints).toHaveBeenLastCalledWith(expect.objectContaining({ css: 'body { font-weight: 600; }' }))
    expect(mounts).toBe(1)
    expect(unmounts).toBe(0)
  })
})

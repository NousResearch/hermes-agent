import collapseBold from '@nous-research/ui/fonts/Collapse-Bold.woff2?url'
import { within } from '@testing-library/dom'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { showBootFailure } from './boot-failure'
import { bootRenderer } from './ui-preferences/boot'

let root: HTMLDivElement
const fontsDescriptor = Object.getOwnPropertyDescriptor(document, 'fonts')

beforeEach(() => {
  localStorage.clear()
  vi.stubGlobal('FontFace', undefined)
  Object.defineProperty(document, 'fonts', { configurable: true, value: undefined })
  root = document.createElement('div')
  root.id = 'root'
  document.body.append(root)
  vi.spyOn(console, 'error').mockImplementation(() => {})
})

afterEach(() => {
  root.remove()
  window.history.replaceState(null, '', '/')
  vi.restoreAllMocks()
  vi.unstubAllGlobals()

  if (fontsDescriptor) {
    Object.defineProperty(document, 'fonts', fontsDescriptor)
  } else {
    Reflect.deleteProperty(document, 'fonts')
  }
})

it('shows a rejected loader’s cause and reloads only when the user clicks Reload', async () => {
  const error = new Error('Unable to preload CSS')
  const reload = vi.fn()

  await bootRenderer(() => Promise.reject(error)).catch(error => showBootFailure(error, reload))

  expect(console.error).toHaveBeenCalledWith('Hermes failed to load', error)
  expect(within(root).getByRole('heading', { level: 1 }).textContent).toBe('Hermes couldn’t load')
  expect(within(root).getByRole('alert').textContent).toContain(
    'Part of the app didn’t load; reloading usually fixes it.'
  )
  // The notice stays the page's main landmark; only its message is the alert.
  expect(within(root).getByRole('main').contains(within(root).getByRole('alert'))).toBe(true)

  const button = within(root).getByRole<HTMLButtonElement>('button', { name: 'Reload' })
  expect(reload).not.toHaveBeenCalled()
  button.click()
  expect(reload).toHaveBeenCalledTimes(1)
})

it.each(['element', 'text'])('leaves mounted %s content untouched and still logs the cause', async kind => {
  const content = kind === 'element' ? document.createElement('section') : document.createTextNode('')
  content.textContent = 'Already mounted'
  root.append(content)

  const error = new Error('App failed after mounting')
  const reload = vi.fn()

  await bootRenderer(() => Promise.reject(error)).catch(error => showBootFailure(error, reload))

  expect([...root.childNodes]).toEqual([content])
  expect(root.textContent).toBe('Already mounted')
  expect(console.error).toHaveBeenCalledWith('Hermes failed to load', error)
  expect(reload).not.toHaveBeenCalled()
})

it('uses no inline handler attributes under script-src-attr none', () => {
  showBootFailure(new Error('Chunk failed'), vi.fn())
  expect(within(root).getByRole('button', { name: 'Reload' })).toBeTruthy()

  for (const element of root.querySelectorAll('*')) {
    expect(element.getAttributeNames().filter(name => /^on/i.test(name))).toEqual([])
  }
})

it.each(['hud', 'overlay', 'quick', 'wake'])(
  'only logs in the transparent %s window, which cannot show a notice',
  kind => {
    window.history.replaceState(null, '', `/?win=${kind}`)
    const error = new Error('Overlay chunk failed')

    showBootFailure(error, vi.fn())

    expect(root.childNodes).toHaveLength(0)
    expect(console.error).toHaveBeenCalledWith('Hermes failed to load', error)
  }
)

it('shows the notice in an opaque secondary window', () => {
  window.history.replaceState(null, '', '/?win=secondary')
  showBootFailure(new Error('Session window chunk failed'), vi.fn())
  expect(within(root).getByRole('button', { name: 'Reload' })).toBeTruthy()
})

it('shows a decorative wordmark above the accessible heading without app CSS', () => {
  showBootFailure(new Error('CSS failed'), vi.fn())

  const wordmark = within(root).getByText('HERMES AGENT')
  const heading = within(root).getByRole('heading', { level: 1, name: 'Hermes couldn’t load' })
  const alert = within(root).getByRole('alert')
  expect(wordmark.tagName).toBe('P')
  expect(wordmark.getAttribute('aria-hidden')).toBe('true')
  expect(wordmark.nextElementSibling).toBe(alert)
  expect(alert.contains(heading)).toBe(true)
  expect(within(root).getByRole('main').getAttribute('lang')).toBe('en')
  expect(wordmark.style.fontFamily).toBe('Collapse, system-ui, sans-serif')
  expect(wordmark.style.fontWeight).toBe('700')
  expect(wordmark.style.letterSpacing).toBe('0.08em')
  expect(wordmark.style.lineHeight).toBe('0.9')
  expect(wordmark.style.textTransform).toBe('uppercase')
  expect(wordmark.style.marginBottom).toBe('24px')
  expect(heading.style.fontSize).toBe('1.125rem')
  expect(heading.style.fontWeight).toBe('600')
  const message = within(alert).getByText('Part of the app didn’t load; reloading usually fixes it.')
  expect(message.style.fontSize).toBe('0.875rem')
  expect(message.style.opacity).toBe('0.7')
  const button = within(root).getByRole('button', { name: 'Reload' })
  expect(button.style.borderRadius).toBe('2.5px')
  expect(button.style.padding).toBe('8px 20px')
  expect(button.style.fontWeight).toBe('500')
  expect(button.style.fontSize).toBe('0.875rem')
  expect(button.style.lineHeight).toBe('1.25rem')
})

const palette = {
  background: 'rgb(13, 17, 23)',
  foreground: 'rgb(230, 237, 243)',
  accent: 'rgb(88, 166, 255)',
  accentForeground: 'rgb(13, 17, 23)'
}

it('paints its own background and ink from the theme snapshot, whatever the page behind it shows', () => {
  localStorage.setItem('hermes-boot-palette', JSON.stringify(palette))
  // The app's stylesheet loaded without the theme (its JS failed) and painted a light page.
  document.body.style.background = 'rgb(248, 249, 254)'
  showBootFailure(new Error('JS chunk failed'), vi.fn())

  const notice = within(root).getByRole('main')
  expect(notice.style.backgroundColor).toBe(palette.background)
  expect(notice.style.color).toBe(palette.foreground)
  expect(notice.style.position).toBe('fixed')
  expect(notice.style.inset).toBe('0px')
  expect(within(root).getByText('HERMES AGENT').style.color).toBe(palette.accent)
  const button = within(root).getByRole('button', { name: 'Reload' })
  expect(button.style.backgroundColor).toBe(palette.accent)
  expect(button.style.color).toBe(palette.accentForeground)
  expect(getComputedStyle(button).borderTopStyle).toBe('none')
  document.body.style.removeProperty('background')
})

it.each([
  ['no snapshot', null],
  ['a malformed snapshot', '{"background":'],
  ['a snapshot missing its ink', JSON.stringify({ ...palette, accentForeground: '' })],
  ['an unresolved var() accent', JSON.stringify({ ...palette, accent: 'var(--theme-primary)' })],
  ['throwing storage', 'throw']
])('inherits the page colours and keeps an outlined button with %s', (_case, stored) => {
  if (stored === 'throw') {
    vi.stubGlobal('localStorage', {
      getItem() {
        throw new Error('Storage unavailable')
      }
    })
  } else if (stored !== null) {
    localStorage.setItem('hermes-boot-palette', stored)
  }

  expect(() => showBootFailure(new Error('CSS failed'), vi.fn())).not.toThrow()
  const notice = within(root).getByRole('main')
  expect(notice.style.backgroundColor).toBe('')
  expect(notice.style.color).toBe('')
  expect(notice.style.position).toBe('')
  expect(within(root).getByText('HERMES AGENT').style.color).toBe('currentcolor')
  const button = within(root).getByRole('button', { name: 'Reload' })
  expect(button.style.backgroundColor).toBe('transparent')
  expect(button.style.color).toBe('inherit')
  expect(button.style.borderTopWidth).toBe('1px')
  expect(button.style.borderTopStyle).toBe('solid')
  expect(button.style.borderTopColor).toBe('currentcolor')
})

it('registers the bundled Collapse face without waiting to show recovery', () => {
  const face = { load: vi.fn(() => new Promise(() => {})) }

  const FontFace = vi.fn(function () {
    return face
  })

  const add = vi.fn()
  vi.stubGlobal('FontFace', FontFace)
  Object.defineProperty(document, 'fonts', { configurable: true, value: { add } })

  showBootFailure(new Error('CSS failed'), vi.fn())

  expect(FontFace).toHaveBeenCalledWith('Collapse', `url("${collapseBold}")`, { weight: '700', display: 'swap' })
  expect(add).toHaveBeenCalledWith(face)
  expect(face.load).toHaveBeenCalledOnce()
  expect(within(root).getByRole('button', { name: 'Reload' })).toBeTruthy()
})

it.each(['FontFace', 'document.fonts'])('shows recovery when %s is unavailable', missing => {
  const FontFace = vi.fn()
  const add = vi.fn()
  vi.stubGlobal('FontFace', missing === 'FontFace' ? undefined : FontFace)
  Object.defineProperty(document, 'fonts', {
    configurable: true,
    value: missing === 'document.fonts' ? undefined : { add }
  })

  expect(() => showBootFailure(new Error('CSS failed'), vi.fn())).not.toThrow()
  expect(FontFace).not.toHaveBeenCalled()
  expect(add).not.toHaveBeenCalled()
  expect(within(root).getByText('HERMES AGENT').style.fontFamily).toContain('system-ui')
})

it.each(['constructor', 'load', 'rejection'])('keeps recovery usable after a font %s failure', async failure => {
  const error = new Error('Font unavailable')

  const face = {
    load: vi.fn(() => {
      if (failure === 'load') {
        throw error
      }

      return Promise.reject(error)
    })
  }

  vi.stubGlobal(
    'FontFace',
    vi.fn(function () {
      if (failure === 'constructor') {
        throw error
      }

      return face
    })
  )
  Object.defineProperty(document, 'fonts', { configurable: true, value: { add: vi.fn() } })
  const reload = vi.fn()

  showBootFailure(new Error('CSS failed'), reload)
  // Let a rejected font request settle; Vitest also fails on unhandled rejections.
  await new Promise(resolve => setTimeout(resolve, 0))

  expect(reload).not.toHaveBeenCalled()
  within(root).getByRole('button', { name: 'Reload' }).click()
  expect(reload).toHaveBeenCalledOnce()
})

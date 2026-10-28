import { within } from '@testing-library/dom'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { showBootFailure } from './boot-failure'
import { bootRenderer } from './ui-preferences/boot'

let root: HTMLDivElement

beforeEach(() => {
  root = document.createElement('div')
  root.id = 'root'
  document.body.append(root)
  vi.spyOn(console, 'error').mockImplementation(() => {})
})

afterEach(() => {
  root.remove()
  window.history.replaceState(null, '', '/')
  vi.restoreAllMocks()
})

it('shows a rejected loader’s cause and reloads only when the user clicks Reload', async () => {
  const error = new Error('Unable to preload CSS')
  const reload = vi.fn()

  await bootRenderer(() => Promise.reject(error)).catch(error => showBootFailure(error, reload))

  expect(console.error).toHaveBeenCalledWith('Hermes failed to load', error)
  expect(within(root).getByRole('heading', { level: 1 }).textContent).toBe('Hermes couldn’t load')
  expect(within(root).getByRole('alert').textContent).toContain('Part of the app didn’t load; reloading usually fixes it.')
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

it.each(['hud', 'overlay', 'quick', 'wake'])('only logs in the transparent %s window, which cannot show a notice', kind => {
  window.history.replaceState(null, '', `/?win=${kind}`)
  const error = new Error('Overlay chunk failed')

  showBootFailure(error, vi.fn())

  expect(root.childNodes).toHaveLength(0)
  expect(console.error).toHaveBeenCalledWith('Hermes failed to load', error)
})

it('shows the notice in an opaque secondary window', () => {
  window.history.replaceState(null, '', '/?win=secondary')
  showBootFailure(new Error('Session window chunk failed'), vi.fn())
  expect(within(root).getByRole('button', { name: 'Reload' })).toBeTruthy()
})

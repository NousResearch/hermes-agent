import { afterEach, beforeEach, expect, it } from 'vitest'

import { $shellChoice } from '@/store/shell-variant'

import { installPortalBrowserChrome } from './browser-chrome'

const CLASSIC_VIEWPORT = 'width=device-width, initial-scale=1.0, interactive-widget=resizes-content'
const frame = () => new Promise(resolve => requestAnimationFrame(() => resolve(undefined)))

beforeEach(() => {
  document.head.innerHTML = `<meta name="viewport" content="${CLASSIC_VIEWPORT}"><meta name="theme-color" content="#0a0a0a">`
})

afterEach(() => {
  delete document.documentElement.dataset.hermesDesktopHost
  document.head.innerHTML = ''
  $shellChoice.set(null)
})

const viewport = () => document.querySelector<HTMLMetaElement>('meta[name="viewport"]')!.content
const theme = () => document.querySelector<HTMLMetaElement>('meta[name="theme-color"]')!.content

it('lays the redesign under the safe areas in the browser, and gives classic its own viewport back', async () => {
  document.documentElement.dataset.hermesDesktopHost = 'browser'
  const stop = installPortalBrowserChrome()

  expect(viewport()).toBe(`${CLASSIC_VIEWPORT}, viewport-fit=cover`)
  $shellChoice.set(false)
  await frame()
  expect(viewport()).toBe(CLASSIC_VIEWPORT)
  expect(theme()).toBe('#0a0a0a')

  $shellChoice.set(null)
  await frame()
  expect(viewport()).toContain('viewport-fit=cover')
  stop()
  expect(viewport()).toBe(CLASSIC_VIEWPORT)
  expect(theme()).toBe('#0a0a0a')
})

it('leaves native windows alone', () => {
  const stop = installPortalBrowserChrome()

  expect(viewport()).toBe(CLASSIC_VIEWPORT)
  expect(theme()).toBe('#0a0a0a')
  stop()
})

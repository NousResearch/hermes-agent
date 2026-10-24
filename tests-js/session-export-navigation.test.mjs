import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { JSDOM } from 'jsdom'
import { afterEach, expect, test } from 'vitest'

const repo = fileURLToPath(new URL('../', import.meta.url))
const sessions = [
  { id: 'alpha', title: 'Alpha session', messages: [] },
  { id: 'beta / 50% 中文', title: 'Beta session', messages: [] },
]
// Exercise the HTML the real exporter produces, including its inline navigation script.
const html = execFileSync(process.env.HERMES_PYTHON || (process.platform === 'win32' ? 'python' : 'python3'), [
  '-c',
  'import json, sys; from hermes_cli.session_export_html import generate_multi_session_html_export; print(generate_multi_session_html_export(json.load(sys.stdin)))',
], { cwd: repo, input: JSON.stringify(sessions), encoding: 'utf8', timeout: 30_000,
  env: { ...process.env, PYTHONIOENCODING: 'utf-8' } })

const windows = []
afterEach(() => windows.splice(0).forEach(window => window.close()))

async function openExport(hash = '') {
  const { window } = new JSDOM(html, {
    url: `https://export.example/history.html${hash}`, runScripts: 'dangerously',
  })
  windows.push(window)
  await new Promise(resolve => window.addEventListener('load', resolve, { once: true }))
  return window
}

function expectSession(window, id) {
  const { document } = window
  expect([...document.querySelectorAll('.session-view.active')].map(el => el.id)).toEqual([`view-${id}`])
  expect([...document.querySelectorAll('.session-item.active')].map(el => el.dataset.id)).toEqual([id])
}

async function navigate(window, action) {
  const changed = new Promise(resolve => window.addEventListener('hashchange', resolve, { once: true }))
  action()
  await changed
}

test('Back and Forward restore the conversation and sidebar selection without rewriting history', async () => {
  const window = await openExport('#alpha')
  expectSession(window, 'alpha')
  await navigate(window, () => window.document.querySelectorAll('.session-item')[1].click())
  expectSession(window, sessions[1].id)
  const historyLength = window.history.length
  await navigate(window, () => window.history.back())
  expect(window.location.hash).toBe('#alpha')
  expectSession(window, 'alpha')
  await navigate(window, () => window.history.forward())
  expectSession(window, sessions[1].id)
  expect(window.history.length).toBe(historyLength)
})

test('returning to the original fragment-free URL restores the first session and preserves Forward', async () => {
  const window = await openExport()
  expectSession(window, 'alpha')
  expect(window.location.hash).toBe('')
  await navigate(window, () => window.document.querySelectorAll('.session-item')[1].click())
  await navigate(window, () => window.history.back())
  expect(window.location.hash).toBe('')
  expectSession(window, 'alpha')
  await navigate(window, () => window.history.forward())
  expectSession(window, sessions[1].id)
})

test('an encoded deep link selects the matching session on load', async () => {
  const window = await openExport(`#${encodeURIComponent(sessions[1].id)}`)
  expectSession(window, sessions[1].id)
})

test.each(['#missing-session', '#bad%escape'])('an invalid fragment %s keeps a readable conversation', async hash => {
  const window = await openExport(hash)
  expectSession(window, 'alpha')
  expect(window.location.hash).toBe(hash)
})

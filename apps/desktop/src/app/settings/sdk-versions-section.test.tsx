import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

import { en } from '@/i18n/en'

import { formatSdkVersionsForCopy, SdkVersionsSection } from './sdk-versions-section'

vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))

const original = window.hermesDesktop
const c = en.settings.sdkVersions

const INFO = {
  electron: '33.2.0',
  node: '22.11.0',
  sdks: [
    { name: '@anthropic-ai/claude-code', version: '2.0.1' },
    { name: '@anthropic-ai/claude-agent-sdk', version: null },
    { name: '@google/genai', version: '1.9.0' },
    { name: '@openai/codex', version: '0.42.0' },
    { name: '@modelcontextprotocol/sdk', version: '0.6.1' }
  ]
}

function bridge(info = INFO) {
  const writeClipboard = vi.fn(async (_text: string) => true)
  window.hermesDesktop = { ...original, getSdkVersions: vi.fn(async () => info), writeClipboard }

  return { writeClipboard }
}

afterEach(() => {
  cleanup()
  window.hermesDesktop = original
  vi.clearAllMocks()
})

test('collapsed by default; expanding lists every SDK and the Node/Electron runtimes', async () => {
  bridge()
  render(<SdkVersionsSection />)

  const toggle = await screen.findByRole('button', { name: c.title })
  expect(screen.queryByText('@openai/codex')).toBeNull()

  fireEvent.click(toggle)

  expect(await screen.findByText('@openai/codex')).toBeTruthy()
  expect(screen.getByText('0.42.0')).toBeTruthy()
  expect(screen.getByText('@anthropic-ai/claude-agent-sdk')).toBeTruthy()
  expect(screen.getByText(c.notInstalled)).toBeTruthy()
  expect(screen.getByText('Node.js')).toBeTruthy()
  expect(screen.getByText('22.11.0')).toBeTruthy()
  expect(screen.getByText('Electron')).toBeTruthy()
  expect(screen.getByText('33.2.0')).toBeTruthy()
})

test('copy-all writes a bug-report block covering every row', async () => {
  const { writeClipboard } = bridge()
  render(<SdkVersionsSection />)

  fireEvent.click(await screen.findByRole('button', { name: c.title }))
  fireEvent.click(await screen.findByRole('button', { name: c.copyAll }))

  await waitFor(() => expect(writeClipboard).toHaveBeenCalledTimes(1))

  const text = writeClipboard.mock.calls[0][0]
  expect(text).toContain('@openai/codex: 0.42.0')
  expect(text).toContain(`@anthropic-ai/claude-agent-sdk: ${c.notInstalled}`)
  expect(text).toContain('Node.js: 22.11.0')
  expect(text).toContain('Electron: 33.2.0')
})

test('an older preload without the bridge renders nothing', async () => {
  window.hermesDesktop = { ...original, getSdkVersions: undefined }

  const { container } = render(<SdkVersionsSection />)

  await waitFor(() => expect(container.firstChild).toBeNull())
})

test('formatSdkVersionsForCopy leads with the section title', () => {
  const text = formatSdkVersionsForCopy(INFO, c.notInstalled)

  expect(text.split('\n')[0]).toBe(c.title)
})

import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { openExternalLink } from '@/lib/external-link'
import { setPaneHeightOverride } from '@/store/panes'
import { $pluginInstallRequest, closePluginInstallRequest } from '@/store/plugin-install-request'

import { EmbeddedHubPicker } from './embedded-hub-picker'

vi.mock('@/lib/external-link', async importOriginal => ({
  ...(await importOriginal<{ openExternalLink: typeof openExternalLink }>()),
  openExternalLink: vi.fn()
}))

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  vi.restoreAllMocks()
  closePluginInstallRequest()
})

describe('Capabilities Skills Hub links', () => {
  it('opens a link from its own Hub frame while rejecting other senders and schemes', () => {
    setPaneHeightOverride('capabilities-hub', undefined)
    render(<EmbeddedHubPicker installedNames={new Set()} />)
    const frame = screen.getByTitle('Skills Hub') as HTMLIFrameElement
    const url = 'https://github.com/NousResearch/hermes-agent'

    const send = (source: MessageEventSource | null, origin: string, href: string) => {
      act(() => {
        window.dispatchEvent(
          new MessageEvent('message', {
            origin,
            source,
            data: { type: 'hermes-hub-open-link', url: href }
          })
        )
      })
    }

    send(window, 'https://hermes-agent.nousresearch.com', url)
    send(frame.contentWindow, 'https://attacker.example', url)
    send(frame.contentWindow, 'https://hermes-agent.nousresearch.com', 'file:///private/example')
    expect(openExternalLink).not.toHaveBeenCalled()

    send(frame.contentWindow, 'https://hermes-agent.nousresearch.com', url)
    expect(openExternalLink).toHaveBeenCalledExactlyOnceWith(url)
  })

  it('enables its own frame and opens a reviewed catalog confirmation in the selected profile', async () => {
    setPaneHeightOverride('capabilities-hub', undefined)
    render(<EmbeddedHubPicker installedNames={new Set()} profile="workbot" />)
    const frame = screen.getByTitle('Skills Hub') as HTMLIFrameElement
    const postMessage = vi.spyOn(frame.contentWindow!, 'postMessage')

    const send = (data: unknown) => {
      act(() => {
        window.dispatchEvent(
          new MessageEvent('message', {
            origin: 'https://hermes-agent.nousresearch.com',
            source: frame.contentWindow,
            data
          })
        )
      })
    }

    send({ type: 'hermes-hub-links-ready' })
    expect(postMessage).toHaveBeenCalledWith(
      { type: 'hermes-hub-links-enable' },
      'https://hermes-agent.nousresearch.com'
    )
    vi.spyOn(globalThis, 'fetch').mockImplementation(
      async () =>
        new Response(
          JSON.stringify([{ name: 'weather', repo: 'https://github.com/example/weather', sha: 'a'.repeat(40) }]),
          { status: 200 }
        )
    )
    send({ type: 'hermes-hub-open-link', url: 'hermes://plugin/install?catalog=weather&repo=evil/repo' })

    await waitFor(() =>
      expect($pluginInstallRequest.get()).toEqual({
        catalogName: 'weather',
        profile: 'workbot',
        repo: 'https://github.com/example/weather',
        sha: 'a'.repeat(40)
      })
    )
  })
})

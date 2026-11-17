import type { ToolCallMessagePartProps } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ComposerScopeProvider, ComposerSurfaceProvider, MAIN_COMPOSER_SCOPE } from '@/app/chat/composer/scope'
import { MESSAGE_PARTS_COMPONENTS } from '@/components/assistant-ui/thread/message-parts'
import { I18nProvider } from '@/i18n'
import { isCardTool } from '@/lib/tool-render-class'
import { clearArtifactRegistry, getHostedArtifact } from '@/store/artifacts'
import { $showToolActivity } from '@/store/tool-activity'

import { HostedArtifactToolPart } from './hosted-artifact-tool'

const submit = vi.hoisted(() => vi.fn((_text: string, _options?: unknown) => true))

vi.mock('@/app/chat/composer/focus', () => ({ requestComposerSubmit: submit }))

const ID = '01j9z8y7x6w5v4t3s2r1q0p9n8'
const OTHER = '01j9z8y7x6w5v4t3s2r1q0p9n7'
const ORIGIN = `http://${ID}.artifacts.localhost:8787`
const CONTENT_URL = `${ORIGIN}/__av?t=ticket`
const VIEWER_URL = `https://portal.example/a/${ID}`
const DEV = { dev: true }
const TILE = 'tile:origin'
const SURFACE = 'surface-origin'

const created = (extra: Record<string, unknown> = {}) =>
  JSON.stringify({
    result: JSON.stringify({ content_url: CONTENT_URL, id: ID, version: 2, viewer_url: VIEWER_URL, ...extra })
  })

const Fallback = () => <div data-testid="fallback">generic row</div>

function mount(
  props: Partial<ToolCallMessagePartProps> & { framePolicy?: { dev: boolean }; completedAt?: number } = {}
) {
  const part = {
    args: { title: 'Chart' },
    argsText: '{}',
    framePolicy: DEV,
    result: created(),
    status: { type: 'complete' },
    toolCallId: 'call-1',
    toolName: 'mcp__hermes_artifacts__artifact_create',
    type: 'tool-call',
    ...props
  } as unknown as ToolCallMessagePartProps & { framePolicy?: { dev: boolean } }

  // The conversation that renders the card: a session tile, not the active main composer.
  return render(
    <I18nProvider>
      <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, target: TILE }}>
        <ComposerSurfaceProvider value={SURFACE}>
          <HostedArtifactToolPart {...part} fallback={Fallback} />
        </ComposerSurfaceProvider>
      </ComposerScopeProvider>
    </I18nProvider>
  )
}

const frame = (container: HTMLElement) => container.querySelector<HTMLIFrameElement>('iframe')

const viewerLink = (container: HTMLElement) =>
  container.querySelector<HTMLAnchorElement>('[data-slot="aui_hosted-artifact-viewer-link"]')

beforeEach(() => {
  submit.mockClear()
  submit.mockImplementation(() => true)
})

afterEach(() => {
  cleanup()
  clearArtifactRegistry()
})

describe('hosted artifact card framing', () => {
  it('frames the content URL on the artifact origin with the narrow sandbox, and links the viewer', () => {
    const { container } = mount()
    const iframe = frame(container)

    expect(iframe?.getAttribute('src')).toBe(CONTENT_URL)
    expect(iframe?.getAttribute('sandbox')).toBe('allow-scripts allow-same-origin')
    expect(viewerLink(container)?.getAttribute('href')).toBe(VIEWER_URL)
  })

  it('records the version it rendered in the hosted cache', async () => {
    mount()

    await waitFor(() => expect(getHostedArtifact(ID)?.headVersion).toBe(2))
  })

  it.each([
    ['a build that may not frame the origin', { framePolicy: { dev: false } }],
    ['a content URL off the frame domain', { result: created({ content_url: 'https://evil.test/__av?t=ticket' }) }],
    [
      "another artifact's origin",
      { result: created({ content_url: `http://${OTHER}.artifacts.localhost:8787/__av?t=ticket` }) }
    ],
    ['an expired ticket', { completedAt: Date.now() / 1000 - 301 }]
  ])('shows only the viewer link for %s', (_label, props) => {
    const { container } = mount(props)

    expect(frame(container)).toBeNull()
    expect(viewerLink(container)?.getAttribute('href')).toBe(VIEWER_URL)
  })

  it.each([
    ['a stale-base error', { result: JSON.stringify({ error: 'stale_base', head_version: 4, message: 'stale' }) }],
    ['a pending call', { result: undefined }],
    ['a failed call', { isError: true }]
  ])('hands %s to the generic tool row', (_label, props) => {
    const { container } = mount(props)

    expect(screen.getByTestId('fallback')).toBeTruthy()
    expect(frame(container)).toBeNull()
  })
})

describe('hosted artifact bridge', () => {
  async function connected() {
    const { container } = mount({ args: { id: ID, title: 'Chart' } })
    const iframe = frame(container)!
    const view = iframe.contentWindow!
    const posted: Array<{ message: Record<string, unknown>; targetOrigin: string }> = []

    vi.spyOn(view, 'postMessage').mockImplementation(((message: Record<string, unknown>, targetOrigin: string) => {
      posted.push({ message, targetOrigin })
    }) as typeof view.postMessage)

    const fromView = (data: Record<string, unknown>, origin = ORIGIN, source: Window = view) =>
      act(async () => {
        window.dispatchEvent(new MessageEvent('message', { data: { jsonrpc: '2.0', ...data }, origin, source }))
        await new Promise(resolve => setTimeout(resolve, 0))
      })

    const responseTo = (id: number) => posted.find(entry => entry.message.id === id)

    // Let the bridge's transport start listening.
    await act(async () => new Promise(resolve => setTimeout(resolve, 0)))

    await fromView({
      id: 1,
      method: 'ui/initialize',
      params: { appCapabilities: {}, appInfo: { name: 'view', version: '1' }, protocolVersion: '2026-01-26' }
    })
    await waitFor(() => expect(responseTo(1)).toBeDefined())

    return { fromView, posted, responseTo }
  }

  it('answers initialize only to the artifact origin and advertises text messages', async () => {
    const { responseTo } = await connected()
    const reply = responseTo(1)!

    expect(reply.targetOrigin).toBe(ORIGIN)
    expect((reply.message.result as { hostCapabilities: unknown }).hostCapabilities).toEqual({ message: { text: {} } })
  })

  it('sends the tool input and result once the view is initialized', async () => {
    const { fromView, posted } = await connected()

    await fromView({ method: 'ui/notifications/initialized', params: {} })

    await waitFor(() => {
      const methods = posted.map(entry => entry.message.method)

      expect(methods).toContain('ui/notifications/tool-input')
      expect(methods).toContain('ui/notifications/tool-result')
    })

    const input = posted.find(entry => entry.message.method === 'ui/notifications/tool-input')!
    const result = posted.find(entry => entry.message.method === 'ui/notifications/tool-result')!

    expect((input.message.params as { arguments: unknown }).arguments).toEqual({ id: ID, title: 'Chart' })
    expect(JSON.stringify(result.message.params)).toContain(ID)
    expect(posted.every(entry => entry.targetOrigin === ORIGIN)).toBe(true)
  })

  const prompt = () => screen.queryByRole('group')

  const click = (name: 'Dismiss' | 'Send') =>
    act(async () => {
      fireEvent.click(screen.getByRole('button', { name }))
    })

  const ask = (text: string) => ({ content: [{ text, type: 'text' }], role: 'user' })

  it('never submits a ui/message on its own: it asks the user first', async () => {
    const { fromView, responseTo } = await connected()

    await fromView({ id: 2, method: 'ui/message', params: ask('Make the bars blue') })

    await waitFor(() => expect(responseTo(2)).toBeDefined())
    expect(responseTo(2)!.message.result).toEqual({})
    expect(submit).not.toHaveBeenCalled()
    expect(prompt()?.textContent).toContain('Make the bars blue')
  })

  it('sends a confirmed message to the conversation that rendered the frame, never the active one', async () => {
    const { fromView, responseTo } = await connected()

    await fromView({ id: 2, method: 'ui/message', params: ask('Make the bars blue') })
    await waitFor(() => expect(responseTo(2)).toBeDefined())
    await click('Send')

    expect(submit).toHaveBeenCalledOnce()
    expect(submit).toHaveBeenCalledWith('Make the bars blue', { surfaceId: SURFACE, target: TILE })
    expect(prompt()).toBeNull()
  })

  it('drops a dismissed message, and keeps the prompt when its conversation cannot take it', async () => {
    const { fromView, responseTo } = await connected()

    await fromView({ id: 2, method: 'ui/message', params: ask('first') })
    await waitFor(() => expect(responseTo(2)).toBeDefined())
    await click('Dismiss')
    expect(prompt()).toBeNull()
    expect(submit).not.toHaveBeenCalled()

    await act(async () => new Promise(resolve => setTimeout(resolve, 1100)))
    await fromView({ id: 3, method: 'ui/message', params: ask('second') })
    await waitFor(() => expect(responseTo(3)).toBeDefined())
    submit.mockImplementationOnce(() => false)
    await click('Send')
    expect(prompt()?.textContent).toContain('second')
  })

  it('refuses a second message while one waits for the user', async () => {
    const { fromView, responseTo } = await connected()

    await fromView({ id: 2, method: 'ui/message', params: ask('one') })
    await waitFor(() => expect(responseTo(2)).toBeDefined())
    await act(async () => new Promise(resolve => setTimeout(resolve, 1100))) // past the burst throttle
    await fromView({ id: 3, method: 'ui/message', params: ask('two') })
    await waitFor(() => expect(responseTo(3)).toBeDefined())

    expect(responseTo(3)!.message.result).toEqual({ isError: true })
    expect(prompt()?.textContent).toContain('one')
    expect(prompt()?.textContent).not.toContain('two')
    expect(submit).not.toHaveBeenCalled()
  })

  it('ignores messages from another origin or another window', async () => {
    const { fromView, responseTo } = await connected()
    const message = { content: [{ text: 'hi', type: 'text' }], role: 'user' }

    await fromView({ id: 7, method: 'ui/message', params: message }, `http://${OTHER}.artifacts.localhost:8787`)
    await fromView({ id: 8, method: 'ui/message', params: message }, ORIGIN, window)

    expect(responseTo(7)).toBeUndefined()
    expect(responseTo(8)).toBeUndefined()
    expect(submit).not.toHaveBeenCalled()
  })
})

describe('transcript routing', () => {
  it('draws the artifact card from the tool part even with tool activity hidden', () => {
    const before = $showToolActivity.get()
    const ToolPart = MESSAGE_PARTS_COMPONENTS.tools.Fallback

    $showToolActivity.set(false)

    try {
      const { container } = render(
        <I18nProvider>
          <ToolPart
            {...({
              args: {},
              argsText: '{}',
              result: created(),
              status: { type: 'complete' },
              toolCallId: 'call-9',
              toolName: 'mcp__hermes_artifacts__artifact_show',
              type: 'tool-call'
            } as unknown as ToolCallMessagePartProps)}
          />
        </I18nProvider>
      )

      expect(container.querySelector('[data-slot="aui_hosted-artifact"]')).not.toBeNull()
    } finally {
      $showToolActivity.set(before)
    }
  })

  it('keeps the card out of run summaries', () => {
    expect(isCardTool('mcp__hermes_artifacts__artifact_update')).toBe(true)
    expect(isCardTool('mcp__hermes_artifacts__artifact_list')).toBe(false)
  })
})

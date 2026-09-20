import { EditorView } from '@codemirror/view'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { composerFocusBlockedBySurface } from '@/lib/keybinds/composer-focus-keys'
import { $gateway } from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'
import { clearAllPrompts, clearApprovalRequest, setApprovalRequest } from '@/store/prompts'
import { rememberServerRequest } from '@/store/server-requests'

import { $backworkspaceWaiting } from './ask'
import { BackworkspacePage } from './back-page'
import { $backworkspacePage } from './page'
import { $backworkspaceOpen, toggleBackworkspace } from './store'

const request = vi.fn<(...args: unknown[]) => Promise<unknown>>()
const askAgent = vi.fn<(...args: unknown[]) => Promise<string>>()

vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestGatewayForAgent: (...args: unknown[]) => request(...args)
}))
vi.mock('./ask', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  askBackworkspace: (...args: unknown[]) => askAgent(...args)
}))
vi.mock('./store', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  toggleBackworkspace: vi.fn()
}))

function renderPage() {
  $backworkspaceOpen.set(true)

  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <BackworkspacePage />
    </I18nProvider>
  )
}

afterEach(() => {
  cleanup()
  clearAllPrompts()
  $gateway.set(null)
  $backworkspaceWaiting.set({})
  $backworkspaceOpen.set(false)
  request.mockReset()
  askAgent.mockReset()
  vi.mocked(toggleBackworkspace).mockClear()
})

describe('BackworkspacePage', () => {
  it('owns the keyboard while turned over and shows the shell again on any unmount', () => {
    // Never answers: the page stays loading, which is when focus used to fall
    // through to <body> and the hidden composer.
    request.mockImplementation(() => new Promise(() => {}))
    $activeGatewayProfile.set('still-loading')

    const { unmount } = renderPage()
    const sheet = screen.getByRole('region', { name: 'Back workspace' })
    const root = sheet.ownerDocument.documentElement

    expect(root.hasAttribute('data-backworkspace')).toBe(true)
    expect(sheet).toBe(sheet.ownerDocument.activeElement)
    expect(composerFocusBlockedBySurface()).toBe(true)

    fireEvent.keyDown(sheet, { key: 'Escape' })
    expect(toggleBackworkspace).toHaveBeenCalledTimes(1)

    // An error boundary replacing the tree unmounts the page the same way.
    unmount()
    expect(root.hasAttribute('data-backworkspace')).toBe(false)
  })

  it('offers the window profile as @hermes when a mention is typed, and Enter writes it', async () => {
    request.mockResolvedValue({ page: null })
    $activeGatewayProfile.set('default')

    renderPage()

    const host = await screen.findByLabelText('Back workspace', { selector: '.cm-content' })
    const view = EditorView.findFromDOM(host as HTMLElement)!

    act(() => {
      view.dispatch({ changes: { from: 0, insert: '@he' }, selection: { anchor: 3 } })
    })

    // The profile's own agent is offered as @hermes; the composer's bot sources
    // add the rest when Bot Mode is registered.
    const list = await screen.findByRole('listbox')

    expect(list.textContent).toContain('@hermes')

    // Escape belongs to the open list, not to the window behind it.
    fireEvent.keyDown(view.contentDOM, { key: 'Escape' })
    expect(screen.queryByRole('listbox')).toBeNull()
    expect(toggleBackworkspace).not.toHaveBeenCalled()

    act(() => {
      view.dispatch({ changes: { from: 0, insert: '@he', to: view.state.doc.length }, selection: { anchor: 3 } })
    })
    await screen.findByRole('listbox')

    fireEvent.keyDown(view.contentDOM, { key: 'Enter' })
    expect(view.state.doc.toString()).toBe('@hermes ')
    expect(screen.queryByRole('listbox')).toBeNull()
  })

  it('sends the paragraph with the mod-enter chord and writes the reply under it', async () => {
    request.mockResolvedValue({
      page: { content: '@asking what is this?', id: '20260920_101010_abcdef', path: '/p.md' }
    })
    askAgent.mockResolvedValue('a short answer')
    $activeGatewayProfile.set('asking')

    renderPage()

    const host = await screen.findByLabelText('Back workspace', { selector: '.cm-content' })
    const view = EditorView.findFromDOM(host as HTMLElement)!
    // CodeMirror resolves `Mod` per platform, exactly as it will at runtime.
    const mod = /Mac/i.test(navigator.platform) ? { metaKey: true } : { ctrlKey: true }

    fireEvent.keyDown(view.contentDOM, { key: 'Enter', ...mod })

    await vi.waitFor(() => expect(askAgent).toHaveBeenCalledTimes(1))
    expect(askAgent.mock.calls[0][1]).toBe('@asking what is this?')
    await vi.waitFor(() => expect(view.state.doc.toString()).toContain('> asking · '))
    // The chord must not also leave CodeMirror's own blank line behind.
    expect(view.state.doc.toString().startsWith('@asking what is this?\n\n>')).toBe(true)
  })

  it('continues with the last agent when a later paragraph mentions nobody', async () => {
    request.mockResolvedValue({ page: { content: '@continuing hello', id: '20260920_101010_abcdef', path: '/p.md' } })
    askAgent.mockResolvedValue('first answer')
    $activeGatewayProfile.set('continuing')

    renderPage()

    const host = await screen.findByLabelText('Back workspace', { selector: '.cm-content' })
    const view = EditorView.findFromDOM(host as HTMLElement)!
    const mod = /Mac/i.test(navigator.platform) ? { metaKey: true } : { ctrlKey: true }

    fireEvent.keyDown(view.contentDOM, { key: 'Enter', ...mod })
    // Wait for the whole first exchange, reply included — a second question is
    // refused while one is still in flight.
    await vi.waitFor(() => expect(view.state.doc.toString()).toContain('> continuing · '))

    // A new paragraph with no mention at all.
    act(() => {
      const end = view.state.doc.length
      const added = '\n\nand what about this?'

      view.dispatch({ changes: { from: end, insert: added }, selection: { anchor: end + added.length } })
    })
    askAgent.mockResolvedValue('second answer')
    fireEvent.keyDown(view.contentDOM, { key: 'Enter', ...mod })

    await vi.waitFor(() => expect(askAgent).toHaveBeenCalledTimes(2))
    expect(askAgent.mock.calls[1][0]).toMatchObject({ handle: '@continuing' })
    expect(askAgent.mock.calls[1][1]).toBe('and what about this?')
  })

  describe('a question still out when the window is turned away and back', () => {
    // The reader asks, goes back to the front to get on with something, and
    // comes back before the agent has answered: a new editor is showing the
    // page by then, and the one the question left from is gone.
    async function askThenTurnAwayAndBack(profile: string) {
      let answer: (text: string) => void = () => {}

      request.mockImplementation(async (_connection, _profile, method) =>
        method === 'backworkspace.open'
          ? { page: { content: `@${profile} what is this?`, id: '20260920_101010_abcdef', path: '/p.md' } }
          : { id: '20260920_101010_abcdef', path: '/p.md' }
      )
      askAgent.mockImplementation(() => new Promise<string>(resolve => (answer = resolve)))
      $activeGatewayProfile.set(profile)

      renderPage()

      const editor = async () =>
        EditorView.findFromDOM(
          (await screen.findByLabelText('Back workspace', { selector: '.cm-content' })) as HTMLElement
        )!

      const mod = /Mac/i.test(navigator.platform) ? { metaKey: true } : { ctrlKey: true }

      fireEvent.keyDown((await editor()).contentDOM, { key: 'Enter', ...mod })
      await vi.waitFor(() => expect(askAgent).toHaveBeenCalledTimes(1))

      act(() => $backworkspaceOpen.set(false))
      act(() => $backworkspaceOpen.set(true))

      return { answer: (text: string) => act(async () => answer(text)), mod, view: await editor() }
    }

    it('writes the reply into the page the reader is looking at, where the next keystroke keeps it', async () => {
      const { answer, view } = await askThenTurnAwayAndBack('returning')

      await answer('a short answer')
      expect(view.state.doc.toString()).toContain('a short answer')

      act(() => view.dispatch({ changes: { from: view.state.doc.length, insert: '!' } }))
      expect($backworkspacePage.get()?.content).toContain('a short answer')
    })

    it('still says who is answering, and still lets only that one question be out', async () => {
      const { answer, mod, view } = await askThenTurnAwayAndBack('waiting')

      expect(screen.getByRole('status').textContent).toContain('@waiting')

      fireEvent.keyDown(view.contentDOM, { key: 'Enter', ...mod })
      expect(askAgent).toHaveBeenCalledTimes(1)

      await answer('done')
    })
  })

  it('stores a pasted image beside the page and links it where the caret is', async () => {
    request.mockImplementation((_connection, _profile, method) =>
      method === 'backworkspace.attach'
        ? Promise.resolve({ href: 'assets/20260920_101010_abcdef.png', path: '/p/assets/x.png' })
        : Promise.resolve({
            page: {
              content: 'a note',
              id: '20260920_101010_abcdef',
              path: '/home/u/.hermes/backworkspace/20260920_101010_abcdef.md'
            }
          })
    )
    $activeGatewayProfile.set('pasting')

    renderPage()

    const host = await screen.findByLabelText('Back workspace', { selector: '.cm-content' })
    const view = EditorView.findFromDOM(host as HTMLElement)!
    const image = new File([new Uint8Array([1, 2, 3])], 'shot.png', { type: 'image/png' })

    fireEvent.paste(view.contentDOM, {
      clipboardData: {
        files: { item: () => image, length: 1 },
        getData: () => '',
        items: [{ getAsFile: () => image, kind: 'file', type: image.type }]
      }
    })

    await vi.waitFor(() => expect(view.state.doc.toString()).toContain('![](assets/20260920_101010_abcdef.png)'))
    expect(request.mock.calls.find(call => call[2] === 'backworkspace.attach')?.[3]).toMatchObject({
      name: 'clipboard.png'
    })
    // The widget that shows the picture itself needs real layout, which jsdom
    // has none of; `image-previews.test.ts` covers the source it resolves.
  })

  it('opens the stored page in the editor, ready to type', async () => {
    request.mockResolvedValue({ page: { content: 'text from the file', id: '20260920_101010_abcdef' } })
    $activeGatewayProfile.set('with-a-page')

    renderPage()

    const line = await screen.findByText('text from the file')
    const editor = line.closest('.cm-content')

    expect(editor).not.toBeNull()
    expect(editor).toBe(line.ownerDocument.activeElement)
  })
})

describe('BackworkspacePage approvals', () => {
  const MOD = /Mac/i.test(navigator.platform) ? { metaKey: true } : { ctrlKey: true }

  // As a browser sends it: the shifted letter in `key`, and `keyCode` set.
  // CodeMirror looks a character chord up by the shifted name first and only
  // falls back to the base letter through `keyCode`, so an event without one
  // resolves Ctrl+Shift+A to plain Ctrl+A — which is Select All.
  const openCard = (view: EditorView) =>
    fireEvent.keyDown(view.contentDOM, { key: 'A', keyCode: 65, shiftKey: true, ...MOD })

  /**
   * Park an approval on the session the page is waiting on, as the gateway
   * does. The session id is deliberately a bot's, not the window profile's:
   * the page has to carry the approval of whichever agent it asked.
   */
  function waitingApproval(profile: string, command = 'rm -rf build') {
    const answered = vi.fn()

    // A page with no socket cannot answer at all: the reply to an approval is
    // the response frame on the connection the request arrived over.
    $gateway.set({ request: vi.fn() } as never)
    $backworkspaceWaiting.set({ [`:${profile}`]: 'a-bot-session' })
    rememberServerRequest({ id: 'srq-1', respond: answered } as never)
    setApprovalRequest({
      command,
      description: 'dangerous command',
      requestId: 'req-1',
      serverRequestId: 'srq-1',
      sessionId: 'a-bot-session'
    })

    return answered
  }

  /** The card owns the keyboard while it is up, so its keys go to the card. */
  const approvalCard = () => screen.getByRole('menu', { name: 'Approval needed' })

  async function pageEditor() {
    const host = await screen.findByLabelText('Back workspace', { selector: '.cm-content' })

    return EditorView.findFromDOM(host as HTMLElement)!
  }

  it('says an approval is waiting without putting anything on the page', async () => {
    request.mockResolvedValue({ page: { content: 'a note', id: '20260920_101010_abcdef', path: '/p.md' } })
    $activeGatewayProfile.set('waiting-ok')
    waitingApproval('waiting-ok')

    renderPage()

    const view = await pageEditor()

    expect((await screen.findByRole('status')).textContent).toMatch(/waiting for your ok/i)
    // The command is not on the page and not in the document: a line that
    // announces, a card that asks.
    expect(screen.queryByText('rm -rf build')).toBeNull()
    expect(view.state.doc.toString()).toBe('a note')
  })

  it('opens the card on the chord, shows the whole command, and answers with Enter', async () => {
    request.mockResolvedValue({ page: { content: 'a note', id: '20260920_101010_abcdef', path: '/p.md' } })
    $activeGatewayProfile.set('answering')

    const answered = waitingApproval('answering', 'rm -rf build --force --everything')

    renderPage()

    const view = await pageEditor()

    await screen.findByRole('status')
    openCard(view)

    // The whole command, never an abbreviation of it.
    expect(await screen.findByText('rm -rf build --force --everything')).toBeTruthy()

    fireEvent.keyDown(approvalCard(), { key: 'Enter' })

    await vi.waitFor(() => expect(answered).toHaveBeenCalledWith({ choice: 'once' }))
    // Enter answered the card; it did not also open a line in the page.
    expect(view.state.doc.toString()).toBe('a note')
  })

  it('puts the card down on Escape without answering and without turning the window back', async () => {
    request.mockResolvedValue({ page: { content: 'a note', id: '20260920_101010_abcdef', path: '/p.md' } })
    $activeGatewayProfile.set('escaping')

    const answered = waitingApproval('escaping')

    renderPage()

    const view = await pageEditor()

    await screen.findByRole('status')
    openCard(view)
    await screen.findByText('rm -rf build')

    fireEvent.keyDown(approvalCard(), { key: 'Escape' })

    expect(screen.queryByText('rm -rf build')).toBeNull()
    expect(answered).not.toHaveBeenCalled()
    expect(toggleBackworkspace).not.toHaveBeenCalled()
  })

  it('goes away with the request it was opened for, rather than showing the next one', async () => {
    request.mockResolvedValue({ page: { content: 'a note', id: '20260920_101010_abcdef', path: '/p.md' } })
    $activeGatewayProfile.set('swapping')

    waitingApproval('swapping', 'the one that was read')

    renderPage()

    const view = await pageEditor()

    await screen.findByRole('status')
    openCard(view)
    await screen.findByText('the one that was read')

    // The first request leaves (timed out, or answered from the OS
    // notification) and another arrives behind it.
    act(() => {
      clearApprovalRequest('a-bot-session', 'req-1')
      setApprovalRequest({
        command: 'the one that was never read',
        description: 'dangerous command',
        requestId: 'req-2',
        serverRequestId: 'srq-2',
        sessionId: 'a-bot-session'
      })
    })

    // The card is down and the notice is back: the second command needs its
    // own chord, so Enter can never answer something nobody read.
    await vi.waitFor(() => expect(screen.queryByRole('menu')).toBeNull())
    expect(screen.queryByText('the one that was never read')).toBeNull()
    expect((await screen.findByRole('status')).textContent).toMatch(/waiting for your ok/i)
  })

  it('asks twice before allowing a command for good', async () => {
    request.mockResolvedValue({ page: { content: 'a note', id: '20260920_101010_abcdef', path: '/p.md' } })
    $activeGatewayProfile.set('forever')

    const answered = waitingApproval('forever')

    renderPage()

    const view = await pageEditor()

    await screen.findByRole('status')
    openCard(view)

    fireEvent.click(await screen.findByRole('menuitem', { name: 'Always allow' }))

    // The row asks again rather than writing to config.yaml on one press.
    expect(answered).not.toHaveBeenCalled()

    const confirm = await screen.findByRole('menuitem', { name: /press again/i })

    // Straight away is the same gesture, not a second decision: a double-click
    // and a held key both land here.
    fireEvent.click(confirm)
    expect(answered).not.toHaveBeenCalled()

    await new Promise(resolve => setTimeout(resolve, 400))
    fireEvent.click(confirm)

    await vi.waitFor(() => expect(answered).toHaveBeenCalledWith({ choice: 'always' }))
  })
})

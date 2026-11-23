import { webcrypto } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

import { useStore } from '@nanostores/react'
import { act, cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import { useLayoutEffect, useRef, useState } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, test, vi } from 'vitest'

import { registerDurableSendIpc } from '../../../../../electron/durable-send-ipc'
import { createDurableBackendPeer } from '../../../../../electron/durable-send-test-peer'
import type { DurableSendApi, DurableTransportRequest } from '../../../../../electron/durable-send-types'

const wire = vi.hoisted(() => ({ rpc: null as null | ((...args: any[]) => Promise<any>) }))
vi.mock('@/store/gateway', async original => ({
  ...(await original<object>()),
  requestGatewayForAgent: (connectionId: string, profile: string, method: string, params: Record<string, unknown>) =>
    wire.rpc!({ connectionId, profile }, method, params)
}))

const savedBridge = window.hermesDesktop
afterEach(() => {
  cleanup()
  window.hermesDesktop = savedBridge
  localStorage.clear()
  vi.restoreAllMocks()
  vi.resetModules()
})

function nativeApi(root: string) {
  const handlers = new Map<string, (...args: any[]) => any>()
  let transport: (request: DurableTransportRequest) => Promise<unknown>
  const frame = { url: 'https://fixture.test/app' }

  const sender = {
    mainFrame: frame,
    send: (_channel: string, request: DurableTransportRequest) => {
      void transport(request).then(
        result => invoke('transport-reply', request.token, { ok: true, result }),
        () => invoke('transport-reply', request.token, { ok: false })
      )
    }
  }

  const event = { sender, senderFrame: frame }
  registerDurableSendIpc({
    ipc: {
      handle: (channel, fn) => {
        handlers.set(channel, fn)
      }
    },
    userData: () => root,
    rendererUrl: () => frame.url,
    getOwner: () => ({ isDestroyed: () => false, webContents: sender }) as any
  })

  const invoke = async (name: string, ...args: unknown[]) =>
    handlers.get('hermes:durable-send:' + name)!(event, ...args)

  const api: DurableSendApi = {
    negotiate: route => invoke('negotiate', route),
    pump: () => invoke('pump'),
    onTransportRequest: handler => {
      transport = handler

      return () => {}
    },
    begin: intent => invoke('begin', intent),
    append: (...args) => invoke('append', ...args),
    commit: id => invoke('commit', id),
    list: () => invoke('list'),
    cancel: id => invoke('cancel', id)
  }

  return api
}

test.each(['new', 'existing'] as const)(
  'real composer→IPC→spool→authority: %s chat, offline Send, navigation, source deletion, lost ACK and restart',
  async kind => {
    const { mainComposerScope, rotateFreshDraftKey, stashSessionDraft, takeSessionDraft } =
      await import('@/store/composer')

    const { $newChatRoute } = await import('@/store/profile')
    const { setSessionOwnerHint } = await import('@/store/session')
    const { DurableSendQueue } = await import('../durable-send-queue')
    const { useComposerSubmit } = await import('./use-composer-submit')
    const peer = createDurableBackendPeer()
    wire.rpc = peer.rpc
    const api = nativeApi(peer.root)
    let release!: () => void
    let reading = false

    const gate = new Promise<void>(resolve => {
      release = resolve
    })

    const bytes = Buffer.from('These are the original bytes, preserved by the Send button.')
    const source = path.join(peer.root, 'picked.txt')
    fs.writeFileSync(source, bytes)
    Object.defineProperty(window.crypto, 'subtle', { configurable: true, value: webcrypto.subtle })
    const setInterval = window.setInterval.bind(window)
    vi.spyOn(window, 'setInterval').mockImplementation(
      (handler, delay, ...args) =>
        (delay === 2000 ? 0 : setInterval(handler, delay, ...args)) as unknown as ReturnType<typeof window.setInterval>
    )
    window.hermesDesktop = {
      ...savedBridge,
      durableSend: api,
      readFileDataUrlForAttach: async file => {
        reading = true
        await gate

        return 'data:text/plain;base64,' + fs.readFileSync(file).toString('base64')
      }
    } as typeof window.hermesDesktop
    const durable = await import('@/store/durable-composer')
    let failure = ''
    const enqueue = durable.enqueueDurableComposer
    vi.spyOn(durable, 'enqueueDurableComposer').mockImplementation((...args) =>
      enqueue(...args).catch(error => {
        failure = String(error?.stack || error)
        throw error
      })
    )
    durable.startDurableSends()
    const route = { connectionId: 'fixture', profile: 'default' }
    $newChatRoute.set(route)

    const scope =
      kind === 'new'
        ? rotateFreshDraftKey()
        : ((
            await peer.raw('session.create', {
              source: 'gui',
              request_id: crypto.randomUUID()
            })
          ).session_id as string)

    if (kind === 'existing') {
      setSessionOwnerHint(scope, route)
    }

    await durable.warmDurableOwner(scope)
    expect(durable.durableComposerOwner(scope)).toBeDefined()
    peer.online(false)
    const onSubmit = vi.fn(async () => true)
    const clear = vi.fn()
    let switchScope!: (scope: string) => void

    function Page() {
      const [current, setCurrent] = useState(scope)
      switchScope = setCurrent
      const editorRef = useRef<HTMLDivElement>(null)
      const draftRef = useRef('my question')
      const draftScopeRef = useRef<string | null>(current)
      const attachments = useStore(mainComposerScope.$attachments)
      useLayoutEffect(() => {
        draftScopeRef.current = current
        draftRef.current = current === scope ? 'my question' : 'other chat draft'

        if (editorRef.current) {
          editorRef.current.textContent = draftRef.current
        }
      }, [current])

      const submit = useComposerSubmit({
        activeQueueSessionKey: current,
        activeQueueSessionKeyRef: draftScopeRef,
        attachments,
        busy: false,
        disabled: false,
        inputDisabled: false,
        draftScopeRef,
        draftRef,
        editorRef,
        clearDraft: () => {
          clear()
          draftRef.current = ''
          editorRef.current!.textContent = ''
        },
        drainNextQueued: async () => false,
        exitQueuedEdit: () => false,
        focusInput: () => {},
        loadIntoComposer: () => {
          throw new Error('A durable send must never be restored as an unsent draft')
        },
        onCancel: () => {},
        onSteer: async () => false,
        onSteerHidden: async () => false,
        onSubmit,
        queueCurrentDraft: () => false,
        queueEdit: null,
        queuedPrompts: [],
        sessionId: kind === 'existing' ? current : null,
        setComposerText: text => {
          draftRef.current = text
        },
        stashAt: (key, text = draftRef.current, files = attachments) => stashSessionDraft(key, text, files)
      })

      return (
        <>
          <div contentEditable data-slot="composer-rich-input" data-testid="editor" ref={editorRef} />
          <button onClick={submit.submitDraft}>Enviar</button>
          <DurableSendQueue sessionKey={current} />
        </>
      )
    }

    mainComposerScope.add({ id: 'picked', kind: 'file', path: source, label: 'picked.txt' })
    stashSessionDraft(scope, 'my question', mainComposerScope.$attachments.get())

    const page = render(
      <MemoryRouter>
        <Page />
      </MemoryRouter>
    )

    fireEvent.click(page.getByText('Enviar'))
    await waitFor(() =>
      expect({ reading, failure, legacy: onSubmit.mock.calls.length }).toEqual({
        reading: true,
        failure: '',
        legacy: 0
      })
    )
    expect(clear).not.toHaveBeenCalled()
    expect(page.getByTestId('editor').textContent).toBe('my question')
    // A slow disk read must not steal the newly focused chat when it eventually commits.
    act(() => switchScope('other-chat'))
    await act(async () => release())
    await waitFor(async () =>
      expect({ ready: (await api.list()).some(entry => entry.state !== 'staging'), failure }).toEqual({
        ready: true,
        failure: ''
      })
    )
    expect(page.getByTestId('editor').textContent).toBe('other chat draft')
    expect(clear).not.toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()
    fs.unlinkSync(source)
    const entry = (await api.list())[0]
    expect(entry.intent.files).toHaveLength(1)
    expect(entry.intent.target).toEqual(
      kind === 'new' ? expect.objectContaining({ draftId: scope }) : { sessionId: scope }
    )
    expect(await peer.raw('fixture.ledger', { request_id: entry.intent.id })).toEqual([])
    await api.pump()
    await act(async () => durable.refreshDurableSends())
    act(() => switchScope(scope))
    await waitFor(() => expect(page.getByText(/Aguardando conexão/)).toBeTruthy())
    // Recovery before mounting a new editor sees the native queue, not the old unsent stash.
    await act(async () => durable.refreshDurableSends())
    expect(takeSessionDraft(scope).text).toBe('')
    peer.online(true)
    let now = Date.now() + 120_000
    vi.spyOn(Date, 'now').mockImplementation(() => now)
    peer.drop('durable.submit')
    act(() => window.dispatchEvent(new Event('online')))
    await waitFor(async () => expect((await api.list())[0]).toMatchObject({ state: 'waiting', submitStarted: true }))
    const admitted = await peer.raw('fixture.ledger', { request_id: entry.intent.id })
    expect(admitted).toHaveLength(1)
    expect((await api.list())[0]).toMatchObject({ state: 'waiting', submitStarted: true })
    await peer.raw('fixture.restart')
    now += 120_000
    act(() => window.dispatchEvent(new Event('online')))
    await waitFor(() => expect(page.getByText('Recebido pelo servidor')).toBeTruthy())
    expect((await api.list())[0]).toMatchObject({ state: 'confirmed', admissionId: admitted[0].admission_id })
    expect(await peer.raw('fixture.ledger', { request_id: entry.intent.id })).toHaveLength(1)
    expect(page.getByText('Recebido pelo servidor')).toBeTruthy()
    const filePath = JSON.parse(admitted[0].payload.text.split('@file:')[1])
    expect(fs.readFileSync(filePath)).toEqual(bytes)
    // A second local Send is cleared only after commit and can be cancelled while offline.
    peer.online(false)
    act(() => mainComposerScope.clear())
    page.getByTestId('editor').textContent = 'cancel me'
    fireEvent.click(page.getByText('Enviar'))
    await waitFor(() => expect(clear).toHaveBeenCalledTimes(1))
    expect(page.getByTestId('editor').textContent).toBe('')
    await act(async () => durable.refreshDurableSends())
    await waitFor(() => expect(page.getByText('Cancelar')).toBeTruthy())
    fireEvent.click(page.getByText('Cancelar'))
    await waitFor(() => expect(page.getByText('Cancelado')).toBeTruthy())
    const cancelled = (await api.list()).find(item => item.intent.text === 'cancel me')!
    expect(cancelled.state).toBe('cancelled')
    expect(await peer.raw('fixture.ledger', { request_id: cancelled.intent.id })).toEqual([])
  },
  60_000
)

test('startup handoff clears only the committed draft, not a newer attachment with identical text', async () => {
  const { stashSessionDraft, takeSessionDraft, savedDraftAt } = await import('@/store/composer')
  const { refreshDurableSends } = await import('@/store/durable-composer')
  const peer = createDurableBackendPeer()
  const api = nativeApi(peer.root)
  window.hermesDesktop = { ...savedBridge, durableSend: api } as typeof window.hermesDesktop
  const id = crypto.randomUUID()
  const scope = 'handoff-session'
  await api.begin({
    id,
    owner: { connectionId: 'fixture', profile: 'default', gatewayId: 'fixture', principalId: 'human' },
    target: { sessionId: scope },
    text: 'same words',
    files: []
  })
  await api.commit(id)
  const mark = 'hermes.durableSend.retire.v1:' + id
  stashSessionDraft(scope, 'same words', [])
  localStorage.setItem(
    mark,
    JSON.stringify({ scope, text: 'same words', savedAt: savedDraftAt(scope), occurrences: [] })
  )
  const newFile = { id: 'new-file', kind: 'file' as const, path: '/not-read', label: 'next.txt' }
  stashSessionDraft(scope, 'same words', [newFile])
  await refreshDurableSends()
  expect(takeSessionDraft(scope)).toMatchObject({ text: 'same words', attachments: [{ id: 'new-file' }] })
  stashSessionDraft(scope, 'same words', [])
  localStorage.setItem(
    mark,
    JSON.stringify({ scope, text: 'same words', savedAt: savedDraftAt(scope), occurrences: [] })
  )
  await refreshDurableSends()
  expect(takeSessionDraft(scope)).toMatchObject({ text: '', attachments: [] })
})

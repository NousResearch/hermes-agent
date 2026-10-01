import type * as HermesSdk from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { ComponentProps, ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { CANONICAL_GROUP_LOCALES } from './canonical-group-locales'
import { CreateGroupChatDialog } from './create-dialog'
import * as data from './data'
import { $groupChats } from './group-chat'
import { createGroupGateway, scriptedStorage } from './group-test-utils'
import { translateBots } from './i18n-test-helper'
import { setPluginCtx } from './shared'

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))
vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const original = await importOriginal<typeof HermesSdk>()
  const { pluginSdkMock, createGroupGateway } = await import('./group-test-utils')
  Object.assign(host, createGroupGateway().host)
  const children = ({ children }: { children?: ReactNode }) => <>{children}</>
  const button = (props: ComponentProps<'button'>) => <button type="button" {...props} />

  return {
    ...original,
    ...await pluginSdkMock(host),
    Badge: children, Dialog: children, DialogContent: children, DialogDescription: children,
    DialogFooter: children, DialogHeader: children, DialogTitle: children, Tip: children,
    Button: button, RowButton: button,
    Input: (props: ComponentProps<'input'>) => <input {...props} />,
    SearchField: () => null, Codicon: () => null,
    useI18n: () => ({ t: { common: { cancel: 'Cancel' } } }),
    usePluginI18n: () => translateBots
  }
})
vi.mock('./group-chat-parts', () => ({ GroupImageControls: () => null }))

beforeEach(() => {
  const gateway = createGroupGateway()
  Object.assign(host, gateway.host)
  setPluginCtx(scriptedStorage(gateway.storage))
  data.$botMeta.set({})
  $groupChats.set({})
})
afterEach(() => { cleanup(); vi.restoreAllMocks() })

function openCreate() {
  const props = { onClose: vi.fn(), onCreated: vi.fn(), roster: [{ name: 'research' }, { name: 'builder' }] }
  const mounted = render(<CreateGroupChatDialog {...props} open />)
  const select = () => screen.getAllByRole('checkbox').forEach(box => fireEvent.click(box))
  select()

  return { ...mounted, ...props, select, reopen: () => {
    mounted.rerender(<CreateGroupChatDialog {...props} open={false} />)
    mounted.rerender(<CreateGroupChatDialog {...props} open />)
    select()
  }, submit: screen.getByRole('button', { name: CANONICAL_GROUP_LOCALES.en.createGroup }) }
}

describe('classic group creation metadata', () => {
  it('waits for member metadata before reporting creation and prevents duplicate submits', async () => {
    let finish!: (result: Awaited<ReturnType<typeof data.saveBotMeta>>) => void
    const pending = new Promise<Awaited<ReturnType<typeof data.saveBotMeta>>>(resolve => { finish = resolve })
    vi.spyOn(data, 'saveBotMeta').mockReturnValue(pending)
    const view = openCreate()
    await act(async () => { fireEvent.click(view.submit) })
    await act(async () => { fireEvent.click(view.submit) })
    expect(view.onClose).not.toHaveBeenCalled()
    expect(view.onCreated).not.toHaveBeenCalled()
    expect(Object.keys($groupChats.get())).toHaveLength(1)
    await act(async () => { finish({ serverPersisted: true, serverOutcome: 'persisted' }) })
    expect(data.saveBotMeta).toHaveBeenCalledTimes(2)
    expect(view.onCreated).toHaveBeenCalledTimes(1)
    expect(view.onClose).toHaveBeenCalledTimes(1)

    let finishOld!: typeof finish, finishNew!: typeof finish
    const old = new Promise<Awaited<ReturnType<typeof data.saveBotMeta>>>(resolve => { finishOld = resolve })
    const next = new Promise<Awaited<ReturnType<typeof data.saveBotMeta>>>(resolve => { finishNew = resolve })
    vi.mocked(data.saveBotMeta).mockReturnValueOnce(old).mockReturnValueOnce(next)
    view.reopen()
    await act(async () => { fireEvent.click(view.submit) })
    fireEvent.click(screen.getByRole('button', { name: CANONICAL_GROUP_LOCALES.en.close }))
    view.reopen()
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'A newer draft' } })
    await act(async () => { fireEvent.click(view.submit) })
    expect(Object.keys($groupChats.get())).toHaveLength(3)
    await act(async () => { finishOld({ serverPersisted: true, serverOutcome: 'persisted' }) })
    expect(view.onClose).toHaveBeenCalledTimes(2)
    expect(view.onCreated).toHaveBeenCalledTimes(1)
    expect((view.submit as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByRole('textbox') as HTMLInputElement).value).toBe('A newer draft')
    view.unmount()
    await act(async () => { finishNew({ serverPersisted: true, serverOutcome: 'persisted' }) })
    expect(data.saveBotMeta).toHaveBeenCalledTimes(6)
    expect(view.onClose).toHaveBeenCalledTimes(2)
    expect(view.onCreated).toHaveBeenCalledTimes(1)
  })

  it.each(['connectionId', 'profile'] as const)('never retargets deferred local members after %s changes', async field => {
    let finish!: (result: Awaited<ReturnType<typeof data.saveBotMeta>>) => void
    vi.spyOn(data, 'saveBotMeta').mockReturnValue(new Promise(resolve => { finish = resolve }))
    const view = openCreate()
    await act(async () => { fireEvent.click(view.submit) })
    const state = host.state as Record<string, { get: () => string }>
    state[field].get = () => 'replacement'
    view.rerender(<CreateGroupChatDialog {...view} open />)
    await act(async () => { finish({ serverPersisted: true, serverOutcome: 'persisted' }) })
    expect(data.saveBotMeta).toHaveBeenCalledTimes(1)
    expect(view.onCreated).not.toHaveBeenCalled()
    expect(view.onClose).not.toHaveBeenCalled()
  })

  it('keeps the created room and visibly warns when member metadata fails to sync', async () => {
    vi.spyOn(data, 'saveBotMeta').mockResolvedValue({ serverPersisted: false, serverOutcome: 'failed' })
    const view = openCreate()
    await act(async () => { fireEvent.click(view.submit) })
    expect(Object.keys($groupChats.get())).toHaveLength(1)
    expect(host.notify).toHaveBeenCalledWith(expect.objectContaining({ kind: 'warning',
      message: expect.stringContaining('sync') }))
    expect(view.onCreated).toHaveBeenCalledTimes(1)
  })
})

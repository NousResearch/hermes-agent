import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setSessionOwnerResolver } from '@/api/client'
import { $activeTreeGroup, $layoutTree } from '@/components/pane-shell/tree/store'
import type { HermesApiRequest } from '@/global'
import {
  captureDesktopFsOperationScope,
  readDesktopDir,
  readDesktopFileText,
  writeDesktopFileText
} from '@/lib/desktop-fs'
import { $notifications, clearNotifications } from '@/store/notifications'
import { $previewTabs, closeRightRail, openPreview } from '@/store/preview'
import {
  $connection, $selectedStoredSessionId, $sessions, _resetSessionOwnerHintsForTests,
  forgetSessionOwnerHintsForSession, setSessionOwnerHint
} from '@/store/session'
import { stampSecondaryProfileOwner } from '@/store/session-event-provenance'
import { $focusedStoredSessionId } from '@/store/session-focus'
import {
  $sessionTiles, clearAllSessionStates, dropSessionState, forgetProfileOnlyRuntimeOwners,
  knownOwnerForSession, recordSessionEventScope
} from '@/store/session-states'

import {
  $creatingEntry,
  $fileActionDialog,
  $renamingPath,
  $renamingScope,
  beginInlineRename,
  cancelInlineRename,
  executeEntryCreate,
  executeFileDelete,
  executeFileRename,
  requestFileDelete,
  requestNewEntry
} from './file-actions'

const path = '/work/a.txt'
const target = { isDirectory: false, name: 'a.txt', path }
const treeOwner = { connectionId: 'gateway-a', profile: 'main' }
const botOwner = { connectionId: 'gateway-b', profile: 'bot' }
const api = vi.fn(async (_request: HermesApiRequest) => ({ entries: [], path, text: 'preview content', byteSize: 15 }))
const readDir = vi.fn(async () => ({ entries: [] }))
const readFileText = vi.fn(async () => ({ path, text: 'native content', byteSize: 14 }))
const writeTextFile = vi.fn(async () => ({ path }))
const createTextFileExclusive = vi.fn(async () => ({ path }))
const createDirectoryExclusive = vi.fn(async () => ({ path }))
const renamePath = vi.fn(async () => ({ path }))
const trashPath = vi.fn(async () => undefined)

function focusTile(owner = botOwner) {
  $sessionTiles.set([{ storedSessionId: 'tile-chat', ownerRoute: owner }])
  $layoutTree.set({ id: 'group', type: 'group', panes: ['workspace', 'session-tile:tile-chat'], active: 'session-tile:tile-chat' })
  $activeTreeGroup.set('group')
  expect($focusedStoredSessionId.get()).toBe('tile-chat')
}

function focusPrimary() {
  $layoutTree.set({ id: 'group', type: 'group', panes: ['workspace', 'session-tile:tile-chat'], active: 'workspace' })
  $activeTreeGroup.set('group')
  expect($focusedStoredSessionId.get()).toBe('primary-chat')
}

beforeEach(() => {
  vi.stubGlobal('hermesDesktop', {
    api, readDir, readFileText, writeTextFile, createTextFileExclusive, createDirectoryExclusive, renamePath, trashPath
  })
  $connection.set({ ...treeOwner, mode: 'remote' } as never)
  setApiRequestConnection(treeOwner.connectionId)
  setSessionOwnerResolver(knownOwnerForSession)
  $selectedStoredSessionId.set('primary-chat')
  $sessionTiles.set([])
  $sessions.set([])
  _resetSessionOwnerHintsForTests()
  clearAllSessionStates()
  focusPrimary()
  clearNotifications()
  vi.clearAllMocks()
})

afterEach(() => {
  $creatingEntry.set(null)
  $fileActionDialog.set(null)
  cancelInlineRename()
  closeRightRail()
  $sessionTiles.set([])
  $sessions.set([])
  _resetSessionOwnerHintsForTests()
  clearAllSessionStates()
  $layoutTree.set(null)
  $activeTreeGroup.set(null)
  $selectedStoredSessionId.set(null)
  $connection.set(null)
  setApiRequestConnection(null)
  clearNotifications()
  vi.unstubAllGlobals()
})

describe('Files tree mutation owner policy with real stores and preview routing', () => {
  it.each([
    botOwner,
    { connectionId: 'gateway-a', profile: 'bot' },
    { connectionId: 'gateway-b', profile: 'main' }
  ])('refuses all tree mutations when focused preview owner is %j', async owner => {
    focusTile(owner)
    await readDesktopDir('/work')
    openPreview({ kind: 'file', source: path, path, url: `file://${path}`, label: 'a.txt' })
    const tab = $previewTabs.get().find(t => t.target.path === path)!
    expect(tab.sessionId).toBe('tile-chat')
    await readDesktopFileText(path, { sessionId: tab.sessionId })
    await writeDesktopFileText(path, 'edited preview', { sessionId: tab.sessionId })

    requestFileDelete(target)
    beginInlineRename(path)
    await requestNewEntry({ directory: false, parentDir: '/work' })
    await requestNewEntry({ directory: true, parentDir: '/work' })

    expect($fileActionDialog.get()).toBeNull()
    expect($renamingPath.get()).toBeNull()
    expect($creatingEntry.get()).toBeNull()
    expect($notifications.get().some(n => /focused session.*different.*connection or profile/i.test(n.message ?? ''))).toBe(true)
    const calls = api.mock.calls.map(([request]) => request)
    expect(calls).toHaveLength(3)
    expect(calls[0]).toMatchObject(treeOwner)
    expect(calls[1]).toMatchObject(owner)
    expect(calls[2]).toMatchObject(owner)
    expect(trashPath).not.toHaveBeenCalled()
    expect(renamePath).not.toHaveBeenCalled()
    expect(createTextFileExclusive).not.toHaveBeenCalled()
    expect(createDirectoryExclusive).not.toHaveBeenCalled()
  })

  it('refuses an explicit local owner for an unregistered remote window', async () => {
    $connection.set({ mode: 'remote', profile: 'main', remoteKind: 'url', baseUrl: 'https://unregistered.example' } as never)
    setApiRequestConnection(null)
    focusTile({ connectionId: 'local', profile: 'main' })
    requestFileDelete(target)
    beginInlineRename(path)
    await requestNewEntry({ directory: false, parentDir: '/work' })
    await requestNewEntry({ directory: true, parentDir: '/work' })
    expect($fileActionDialog.get()).toBeNull()
    expect($renamingPath.get()).toBeNull()
    expect($creatingEntry.get()).toBeNull()
    await expect(executeFileDelete(path)).rejects.toThrow(/different.*connection or profile/i)
    expect(api).not.toHaveBeenCalled()
    expect(trashPath).not.toHaveBeenCalled()
  })

  it.each([false, true])('retains unregistered remote behavior with profile-only provenance=%s', async profileOnly => {
    $connection.set({ mode: 'remote', profile: 'main', remoteKind: 'url', baseUrl: 'https://unregistered.example' } as never)
    setApiRequestConnection(null)

    if (profileOnly) {recordSessionEventScope(stampSecondaryProfileOwner({ session_id: 'primary-chat' } as never, 'main'))}

    requestFileDelete(target)
    const dialog = $fileActionDialog.get()!
    expect(dialog).not.toBeNull()
    await executeFileDelete(path, dialog.scope)
    expect(api).toHaveBeenCalledWith(expect.objectContaining({ profile: 'main', method: 'DELETE', path: '/api/fs/delete' }))
    expect(trashPath).not.toHaveBeenCalled()
  })

  it('retains native local routing without a registry id', async () => {
    $connection.set({ mode: 'local', profile: 'main' } as never)
    setApiRequestConnection(null)
    focusTile({ connectionId: 'local', profile: 'main' })
    requestFileDelete(target)
    const dialog = $fileActionDialog.get()!
    expect(dialog).not.toBeNull()
    await executeFileDelete(path, dialog.scope)
    expect(trashPath).toHaveBeenCalledWith(path)
    expect(api).not.toHaveBeenCalled()
  })

  it.each(['local', 'remote'] as const)('preserves ordinary %s tree list, preview, save and mutations', async mode => {
    $connection.set({ ...treeOwner, mode } as never)
    focusTile(treeOwner)
    await readDesktopDir('/work')
    openPreview({ kind: 'file', source: path, path, url: `file://${path}`, label: 'a.txt' })
    const tab = $previewTabs.get().find(t => t.target.path === path)!
    await readDesktopFileText(path, { sessionId: tab.sessionId })
    await writeDesktopFileText(path, 'edited', { sessionId: tab.sessionId })
    await requestNewEntry({ directory: false, parentDir: '/work' })
    await executeEntryCreate($creatingEntry.get()!, 'new.txt')
    await requestNewEntry({ directory: true, parentDir: '/work' })
    await executeEntryCreate($creatingEntry.get()!, 'new-dir')
    beginInlineRename(path)
    await executeFileRename(path, 'renamed.txt', $renamingScope.get()!)
    requestFileDelete(target)
    const dialog = $fileActionDialog.get()!
    await executeFileDelete(dialog.path, dialog.scope)

    if (mode === 'remote') {
      expect(api.mock.calls.map(([request]) => request)).toHaveLength(7)
      expect(api.mock.calls.every(([request]) => request.connectionId === treeOwner.connectionId && request.profile === treeOwner.profile)).toBe(true)
      expect(api.mock.calls[6]?.[0]).toMatchObject({ method: 'DELETE', body: { path, recursive: true } })
      expect(trashPath).not.toHaveBeenCalled()
    } else {
      expect(readDir).toHaveBeenCalledWith('/work')
      expect(readFileText).toHaveBeenCalledWith(path)
      expect(writeTextFile).toHaveBeenCalledWith(path, 'edited')
      expect(createTextFileExclusive).toHaveBeenCalledWith('/work/new.txt')
      expect(createDirectoryExclusive).toHaveBeenCalledWith('/work/new-dir')
      expect(renamePath).toHaveBeenCalledWith(path, 'renamed.txt')
      expect(trashPath).toHaveBeenCalledWith(path)
      expect(api).not.toHaveBeenCalled()
    }
  })

  it('cancels all pending actions on focused-session change, including a same-owner tile', async () => {
    focusTile(treeOwner)
    await requestNewEntry({ directory: false, parentDir: '/work' })
    const creating = $creatingEntry.get()!
    beginInlineRename(path)
    const renameScope = $renamingScope.get()!
    requestFileDelete(target)
    const dialog = $fileActionDialog.get()!

    focusPrimary()
    expect($creatingEntry.get()).toBeNull()
    expect($renamingPath.get()).toBeNull()
    expect($fileActionDialog.get()).toBeNull()
    // Returning to the old owner must not revive a stale editor/confirmation.
    focusTile(treeOwner)
    await expect(executeEntryCreate(creating, 'new.txt')).rejects.toThrow(/cancel/i)
    await expect(executeFileRename(path, 'new.txt', renameScope)).rejects.toThrow(/cancel|changed/i)
    await expect(executeFileDelete(dialog.path, dialog.scope)).rejects.toThrow(/cancel|changed/i)
    expect(api).not.toHaveBeenCalled()
  })

  it('cancels pending actions when the focused tile owner changes without changing window connection', async () => {
    focusTile(treeOwner)
    await requestNewEntry({ directory: true, parentDir: '/work' })
    beginInlineRename(path)
    requestFileDelete(target)
    focusTile(botOwner)
    expect($connection.get()).toMatchObject(treeOwner)
    expect($creatingEntry.get()).toBeNull()
    expect($renamingPath.get()).toBeNull()
    expect($fileActionDialog.get()).toBeNull()
    expect(api).not.toHaveBeenCalled()
  })


  it('allows an SSH Desktop profile alias without confusing its backend target profile', async () => {
    $connection.set({ connectionId: 'ssh-a', profile: 'mara', mode: 'remote', remoteKind: 'ssh' } as never)
    setApiRequestConnection('ssh-a')
    $sessionTiles.set([{
      storedSessionId: 'tile-chat',
      ownerRoute: { connectionId: 'ssh-a', profile: 'mara', targetProfile: 'default' }
    }])
    $layoutTree.set({ id: 'group', type: 'group', panes: ['workspace', 'session-tile:tile-chat'], active: 'session-tile:tile-chat' })
    $activeTreeGroup.set('group')
    await readDesktopDir('/work')
    openPreview({ kind: 'file', source: path, path, url: `file://${path}`, label: 'a.txt' })
    const tab = $previewTabs.get().find(t => t.target.path === path)!
    await readDesktopFileText(path, { sessionId: tab.sessionId })
    await writeDesktopFileText(path, 'edited', { sessionId: tab.sessionId })
    await requestNewEntry({ directory: false, parentDir: '/work' })
    expect($creatingEntry.get()).not.toBeNull()
    await executeEntryCreate($creatingEntry.get()!, 'new.txt')
    beginInlineRename(path)
    expect($renamingScope.get()).not.toBeNull()
    await executeFileRename(path, 'renamed.txt', $renamingScope.get()!)
    requestFileDelete(target)
    const dialog = $fileActionDialog.get()!
    expect(dialog).not.toBeNull()
    await executeFileDelete(path, dialog.scope)
    const requests = api.mock.calls.map(([request]) => request)
    expect(requests.map(request => request.profile)).toEqual(['mara', 'default', 'default', 'mara', 'mara', 'mara'])
    expect(requests.every(request => request.connectionId === 'ssh-a')).toBe(true)
  })

  it.each(['drop', 'clear'] as const)('cancels pending actions when %s removes proven runtime ownership', async removal => {
    $sessions.set([{ id: 'primary-chat', profile: 'bot' }] as never)
    recordSessionEventScope({ session_id: 'primary-chat', ...treeOwner })
    await requestNewEntry({ directory: false, parentDir: '/work' })
    beginInlineRename(path)
    requestFileDelete(target)
    const dialog = $fileActionDialog.get()!
    expect(dialog).not.toBeNull()

    if (removal === 'drop') {dropSessionState('primary-chat')} else {clearAllSessionStates()}
    expect($creatingEntry.get()).toBeNull()
    expect($renamingPath.get()).toBeNull()
    expect($fileActionDialog.get()).toBeNull()
    recordSessionEventScope({ session_id: 'primary-chat', ...treeOwner })
    await expect(executeFileDelete(path, dialog.scope)).rejects.toThrow(/cancel/i)
    expect(api).not.toHaveBeenCalled()
  })

  it('cancels pending actions when profile-only pooled runtime ownership disappears', async () => {
    recordSessionEventScope(stampSecondaryProfileOwner({ session_id: 'primary-chat' } as never, 'main'))
    await requestNewEntry({ directory: false, parentDir: '/work' })
    beginInlineRename(path)
    requestFileDelete(target)
    $sessions.set([{ id: 'primary-chat', profile: 'bot' }] as never)
    forgetProfileOnlyRuntimeOwners('main')
    expect($creatingEntry.get()).toBeNull()
    expect($renamingPath.get()).toBeNull()
    expect($fileActionDialog.get()).toBeNull()
    expect(api).not.toHaveBeenCalled()
  })

  it('cancels pending actions when map-backed durable owner hints change or disappear', async () => {
    setSessionOwnerHint('primary-chat', treeOwner)
    await requestNewEntry({ directory: false, parentDir: '/work' })
    beginInlineRename(path)
    requestFileDelete(target)
    setSessionOwnerHint('primary-chat', botOwner)
    expect($creatingEntry.get()).toBeNull()
    expect($renamingPath.get()).toBeNull()
    expect($fileActionDialog.get()).toBeNull()
    _resetSessionOwnerHintsForTests()
    setSessionOwnerHint('primary-chat', treeOwner)
    requestFileDelete(target)
    $sessions.set([{ id: 'primary-chat', profile: 'bot' }] as never)
    forgetSessionOwnerHintsForSession('primary-chat')
    expect($fileActionDialog.get()).toBeNull()
    expect(api).not.toHaveBeenCalled()
  })

  it('revalidates the current owner even for a direct action without a captured scope', async () => {
    focusTile(botOwner)
    await expect(executeFileDelete(path)).rejects.toThrow(/focused session.*different.*connection or profile/i)
    await expect(executeFileRename(path, 'new.txt')).rejects.toThrow(/focused session.*different.*connection or profile/i)
    expect(api).not.toHaveBeenCalled()
    expect(captureDesktopFsOperationScope()).toMatchObject(treeOwner)
  })
})

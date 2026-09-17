import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, test } from 'vitest'

import { type BrowserTaskBindings, BrowserTaskFilePersistence, BrowserTaskLifecycle } from './workstation-browser-task'

interface FakePage {
  id: number
  taskId: string
  url: string
  destroyed: boolean
  visible: boolean
  parked: boolean
}

function fakeBrowser() {
  const pages = new Map<string, FakePage>()
  let pageCounter = 0
  let destroyCalls = 0

  const bindings: BrowserTaskBindings<FakePage, { host: string }> = {
    ensurePage(taskId) {
      const existing = pages.get(taskId)

      if (existing && !existing.destroyed) {
        return existing
      }

      const page: FakePage = {
        id: ++pageCounter,
        taskId,
        url: 'about:blank',
        destroyed: false,
        visible: false,
        parked: true
      }

      pages.set(taskId, page)

      return page
    },
    pageForTask: taskId => pages.get(taskId) ?? null,
    pageIsAlive: page => !page.destroyed,
    showPage(_taskId, page) {
      page.visible = true
      page.parked = false
    },
    hidePage(_taskId, page) {
      page.visible = false
      page.parked = false
    },
    parkPage(_taskId, page) {
      page.visible = false
      page.parked = true
    },
    destroyPage(taskId, page) {
      destroyCalls += 1
      page.destroyed = true
      pages.delete(taskId)
    }
  }

  return {
    bindings,
    pages,
    pageCount: () => pageCounter,
    destroyCount: () => destroyCalls
  }
}

const tempRoots: string[] = []
afterEach(() => {
  for (const root of tempRoots.splice(0)) {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

function tempStateFile(): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-browser-task-'))
  tempRoots.push(root)

  return path.join(root, 'browser-tasks.json')
}

test('create -> hide -> show keeps the same live page and current URL', () => {
  const browser = fakeBrowser()
  const lifecycle = new BrowserTaskLifecycle(browser.bindings)

  lifecycle.createTask({ taskId: 'task-a' })
  const original = browser.pages.get('task-a')
  assert.ok(original)
  original.url = 'https://example.test/current'

  const hidden = lifecycle.hideTask('task-a')
  assert.equal(hidden.status, 'hidden')
  assert.equal(original.destroyed, false)

  const shown = lifecycle.showTask('task-a', { host: 'panel' })
  assert.equal(shown.status, 'visible')
  assert.equal(browser.pages.get('task-a'), original)
  assert.equal(browser.pages.get('task-a')?.url, 'https://example.test/current')
  assert.equal(browser.pageCount(), 1)
})

test('create -> park -> show keeps the same page identity', () => {
  const browser = fakeBrowser()
  const lifecycle = new BrowserTaskLifecycle(browser.bindings)

  lifecycle.createTask({ taskId: 'task-a' })
  const original = browser.pages.get('task-a')
  assert.ok(original)
  original.url = 'https://example.test/parked'

  const parked = lifecycle.parkTask('task-a')
  assert.equal(parked.status, 'parked')
  assert.equal(parked.parked, true)
  assert.equal(original.parked, true)

  lifecycle.showTask('task-a', { host: 'control' })
  assert.equal(browser.pages.get('task-a'), original)
  assert.equal(browser.pages.get('task-a')?.url, 'https://example.test/parked')
  assert.equal(browser.pageCount(), 1)
})

test('showing another task parks the previously visible task metadata and page', () => {
  const browser = fakeBrowser()
  const lifecycle = new BrowserTaskLifecycle(browser.bindings)

  lifecycle.createTask({ taskId: 'task-a' })
  lifecycle.createTask({ taskId: 'task-b' })
  lifecycle.showTask('task-a', { host: 'panel' })
  lifecycle.showTask('task-b', { host: 'control' })

  assert.equal(lifecycle.task('task-a')?.status, 'parked')
  assert.equal(lifecycle.task('task-a')?.parked, true)
  assert.equal(browser.pages.get('task-a')?.parked, true)
  assert.equal(lifecycle.task('task-b')?.status, 'visible')
  assert.equal(browser.pages.get('task-b')?.visible, true)
})

test('hide never destroys the page; destroy is explicit and removes the task', () => {
  const browser = fakeBrowser()
  const lifecycle = new BrowserTaskLifecycle(browser.bindings)

  lifecycle.createTask({ taskId: 'task-a' })
  lifecycle.hideTask('task-a')
  assert.equal(browser.destroyCount(), 0)
  assert.ok(lifecycle.task('task-a'))
  assert.ok(browser.pages.get('task-a'))

  assert.equal(lifecycle.destroyTask('task-a'), true)
  assert.equal(browser.destroyCount(), 1)
  assert.equal(lifecycle.task('task-a'), null)
  assert.equal(browser.pages.has('task-a'), false)
  assert.equal(lifecycle.destroyTask('task-a'), false)
})

test('explicit destroy removes a stale crashed page binding', () => {
  const browser = fakeBrowser()
  const lifecycle = new BrowserTaskLifecycle(browser.bindings)

  lifecycle.createTask({ taskId: 'task-a' })
  const crashed = browser.pages.get('task-a')
  assert.ok(crashed)
  crashed.destroyed = true

  assert.equal(lifecycle.destroyTask('task-a'), true)
  assert.equal(browser.destroyCount(), 1)
  assert.equal(browser.pages.has('task-a'), false)
  assert.equal(lifecycle.task('task-a'), null)
})

test('one task owns exactly one live page across repeated create/show operations', () => {
  const browser = fakeBrowser()
  const lifecycle = new BrowserTaskLifecycle(browser.bindings)

  lifecycle.createTask({ taskId: 'task-a' })
  lifecycle.createTask({ taskId: 'task-a' })
  lifecycle.showTask('task-a', { host: 'panel' })
  lifecycle.hideTask('task-a')
  lifecycle.showTask('task-a', { host: 'session' })

  assert.equal(browser.pages.size, 1)
  assert.equal(browser.pageCount(), 1)
  assert.equal(lifecycle.listTasks().length, 1)
})

test('session identity binds once, persists, and never creates or retargets a page', () => {
  const stateFile = tempStateFile()
  const browser = fakeBrowser()
  const lifecycle = new BrowserTaskLifecycle(browser.bindings, new BrowserTaskFilePersistence(stateFile))

  lifecycle.createTask({ taskId: 'task-a' })
  const page = browser.pages.get('task-a')
  assert.ok(page)
  const pageCount = browser.pageCount()

  const bound = lifecycle.bindSessionHost('task-a', 'hermes-session-a')
  assert.equal(bound.sessionHost, 'hermes-session-a')
  assert.equal(browser.pages.get('task-a'), page)
  assert.equal(browser.pageCount(), pageCount)
  assert.equal(JSON.parse(fs.readFileSync(stateFile, 'utf-8')).tasks[0]?.sessionHost, 'hermes-session-a')

  const repeated = lifecycle.bindSessionHost('task-a', 'hermes-session-a')
  assert.equal(repeated.sessionHost, 'hermes-session-a')
  assert.equal(browser.pages.get('task-a'), page)
  assert.equal(browser.pageCount(), pageCount)

  assert.throws(() => lifecycle.bindSessionHost('task-a', 'hermes-session-b'), /session identity mismatch/)
  assert.equal(lifecycle.task('task-a')?.sessionHost, 'hermes-session-a')
  assert.equal(JSON.parse(fs.readFileSync(stateFile, 'utf-8')).tasks[0]?.sessionHost, 'hermes-session-a')
  assert.equal(browser.pages.get('task-a'), page)
  assert.equal(browser.pageCount(), pageCount)
})

test('an existing task recreates one missing page and records recovery', () => {
  const browser = fakeBrowser()
  const lifecycle = new BrowserTaskLifecycle(browser.bindings)

  lifecycle.createTask({ taskId: 'task-a' })
  const original = browser.pages.get('task-a')
  assert.ok(original)
  original.destroyed = true

  const recovered = lifecycle.createTask({ taskId: 'task-a' })
  const replacement = browser.pages.get('task-a')
  assert.ok(replacement)
  assert.notEqual(replacement, original)
  assert.equal(recovered.recoveryState, 'recreated')
  assert.equal(browser.pageCount(), 2)
  assert.equal(browser.pages.size, 1)
})

test('restart preserves metadata, normalizes it to parked, and lazily recreates one page', () => {
  const stateFile = tempStateFile()
  const persistence = new BrowserTaskFilePersistence(stateFile)
  const beforeBrowser = fakeBrowser()
  const before = new BrowserTaskLifecycle(beforeBrowser.bindings, persistence)

  before.createTask({
    taskId: 'task-persisted',
    panelHost: 'panel-1',
    controlHost: 'control-1',
    sessionHost: 'session-1',
    localConnection: 'local-1',
    leaseState: 'retained'
  })
  before.showTask('task-persisted', { host: 'panel' })

  const afterBrowser = fakeBrowser()
  const after = new BrowserTaskLifecycle(afterBrowser.bindings, persistence)
  const restored = after.restore()

  assert.equal(restored.length, 1)
  assert.deepEqual(
    {
      taskId: restored[0].taskId,
      panelHost: restored[0].panelHost,
      controlHost: restored[0].controlHost,
      sessionHost: restored[0].sessionHost,
      kanbanCardId: restored[0].kanbanCardId,
      runId: restored[0].runId,
      localConnection: restored[0].localConnection,
      leaseState: restored[0].leaseState,
      status: restored[0].status,
      parked: restored[0].parked,
      recoveryState: restored[0].recoveryState
    },
    {
      taskId: 'task-persisted',
      panelHost: 'panel-1',
      controlHost: 'control-1',
      sessionHost: 'session-1',
      kanbanCardId: null,
      runId: null,
      localConnection: 'local-1',
      leaseState: 'retained',
      status: 'parked',
      parked: true,
      recoveryState: 'restored'
    }
  )
  assert.equal(afterBrowser.pages.size, 0)

  const shown = after.showTask('task-persisted', { host: 'panel' })
  assert.equal(shown.recoveryState, 'recreated')
  assert.equal(afterBrowser.pages.size, 1)
  assert.equal(afterBrowser.pageCount(), 1)
})

test('persistence excludes page URLs and page-scoped secrets', () => {
  const stateFile = tempStateFile()
  const browser = fakeBrowser()
  const lifecycle = new BrowserTaskLifecycle(browser.bindings, new BrowserTaskFilePersistence(stateFile))

  lifecycle.createTask({ taskId: 'task-a' })
  const page = browser.pages.get('task-a') as FakePage & { typedSecret?: string }
  assert.ok(page)
  page.url = 'https://example.test/account?access_token=page-secret'
  page.typedSecret = 'typed-password-secret'
  lifecycle.parkTask('task-a')

  const persisted = fs.readFileSync(stateFile, 'utf-8')
  assert.equal(persisted.includes('page-secret'), false)
  assert.equal(persisted.includes('typed-password-secret'), false)
  assert.deepEqual(Object.keys(JSON.parse(persisted).tasks[0]).sort(), [
    'controlHost',
    'createdAt',
    'kanbanCardId',
    'leaseState',
    'localConnection',
    'panelHost',
    'parked',
    'recoveryState',
    'runId',
    'sessionHost',
    'status',
    'taskId',
    'updatedAt'
  ])
})

test('persistence prunes invalid and duplicate task metadata on restore', () => {
  const stateFile = tempStateFile()
  fs.writeFileSync(
    stateFile,
    JSON.stringify({
      version: 1,
      browserTaskCounter: 7,
      tasks: [
        {
          taskId: 'task-a',
          createdAt: '2026-08-25T10:00:00.000Z',
          panelHost: null,
          controlHost: null,
          sessionHost: null,
          localConnection: null,
          status: 'hidden',
          leaseState: null,
          parked: false,
          recoveryState: 'fresh',
          updatedAt: '2026-08-25T10:00:00.000Z'
        },
        {
          taskId: 'task-a',
          createdAt: '2026-08-25T10:00:00.000Z',
          panelHost: 'newer-panel',
          controlHost: null,
          sessionHost: null,
          localConnection: null,
          status: 'parked',
          leaseState: null,
          parked: true,
          recoveryState: 'fresh',
          updatedAt: '2026-08-25T11:00:00.000Z'
        },
        { taskId: '', status: 'parked' }
      ]
    })
  )

  const browser = fakeBrowser()
  const lifecycle = new BrowserTaskLifecycle(browser.bindings, new BrowserTaskFilePersistence(stateFile))
  const restored = lifecycle.restore()

  assert.equal(restored.length, 1)
  assert.equal(restored[0].taskId, 'task-a')
  assert.equal(restored[0].panelHost, 'newer-panel')
  assert.equal(restored[0].status, 'parked')
})

test('bindKanbanCard binds once, allows idempotent repeat, and rejects mismatch fail-closed', () => {
  const browser = fakeBrowser()
  const lifecycle = new BrowserTaskLifecycle(browser.bindings)
  lifecycle.createTask({ taskId: 'task-kanban' })

  const bound = lifecycle.bindKanbanCard('task-kanban', 'card-123')
  assert.equal(bound.kanbanCardId, 'card-123')

  const repeat = lifecycle.bindKanbanCard('task-kanban', 'card-123')
  assert.equal(repeat.kanbanCardId, 'card-123')

  assert.throws(() => lifecycle.bindKanbanCard('task-kanban', 'card-other'), /BrowserTask kanban card mismatch/)
})

test('bindRun binds once, allows idempotent repeat, and rejects mismatch fail-closed', () => {
  const browser = fakeBrowser()
  const lifecycle = new BrowserTaskLifecycle(browser.bindings)
  lifecycle.createTask({ taskId: 'task-run' })

  const bound = lifecycle.bindRun('task-run', 'run-456')
  assert.equal(bound.runId, 'run-456')

  const repeat = lifecycle.bindRun('task-run', 'run-456')
  assert.equal(repeat.runId, 'run-456')

  assert.throws(() => lifecycle.bindRun('task-run', 'run-other'), /BrowserTask run mismatch/)
})

test('persistence preserves kanbanCardId and runId through restart', () => {
  const stateFile = tempStateFile()
  const beforeBrowser = fakeBrowser()
  const before = new BrowserTaskLifecycle(beforeBrowser.bindings, new BrowserTaskFilePersistence(stateFile))

  before.createTask({
    taskId: 'task-full-identity',
    sessionHost: 'session-xyz',
    kanbanCardId: 'card-100',
    runId: 'run-200'
  })

  const afterBrowser = fakeBrowser()
  const after = new BrowserTaskLifecycle(afterBrowser.bindings, new BrowserTaskFilePersistence(stateFile))
  const restored = after.restore()

  assert.equal(restored.length, 1)
  assert.equal(restored[0].taskId, 'task-full-identity')
  assert.equal(restored[0].sessionHost, 'session-xyz')
  assert.equal(restored[0].kanbanCardId, 'card-100')
  assert.equal(restored[0].runId, 'run-200')
})

test('human control lease is scoped, renewable, releasable, and expires deterministically', () => {
  let clock = new Date('2026-09-15T12:00:00.000Z')
  const browser = fakeBrowser()
  const lifecycle = new BrowserTaskLifecycle(browser.bindings, null, () => new Date(clock))
  lifecycle.createTask({ taskId: 'task-a', sessionHost: 'session-a' })
  lifecycle.createTask({ taskId: 'task-b', sessionHost: 'session-b' })

  const acquired = lifecycle.acquireHumanControl(
    'task-a',
    { sessionId: 'session-a', tabId: 'tab-a', pageId: 11, profileScope: 'profile-a' },
    1_000
  )

  assert.equal(acquired.humanControlLease?.taskId, 'task-a')
  assert.equal(lifecycle.hasActiveHumanControl('task-a'), true)
  assert.equal(lifecycle.hasActiveHumanControl('task-b'), false)
  assert.equal(lifecycle.renewHumanControl('task-a', 2_000).humanControlLease?.renewedAt, '2026-09-15T12:00:00.000Z')

  clock = new Date('2026-09-15T12:00:02.001Z')
  assert.equal(lifecycle.hasActiveHumanControl('task-a'), false)
  assert.equal(lifecycle.task('task-a')?.humanControlLease, undefined)

  lifecycle.acquireHumanControl(
    'task-a',
    { sessionId: 'session-a', tabId: 'tab-a', pageId: 11, profileScope: 'profile-a' },
    60_000
  )
  assert.equal(lifecycle.releaseHumanControl('task-a'), true)
  assert.equal(lifecycle.releaseHumanControl('task-a'), false)
})

test('human control lease is removed on BrowserTask restore after a process crash', () => {
  const stateFile = tempStateFile()
  const persistence = new BrowserTaskFilePersistence(stateFile)
  const before = new BrowserTaskLifecycle(fakeBrowser().bindings, persistence)
  before.createTask({ taskId: 'task-crash', sessionHost: 'session-crash' })
  before.acquireHumanControl(
    'task-crash',
    { sessionId: 'session-crash', tabId: 'tab-crash', pageId: 22, profileScope: 'profile' },
    60_000
  )
  assert.ok(JSON.parse(fs.readFileSync(stateFile, 'utf-8')).tasks[0].humanControlLease)

  const after = new BrowserTaskLifecycle(fakeBrowser().bindings, persistence)
  const restored = after.restore()
  assert.equal(restored[0].humanControlLease, undefined)
  assert.equal(after.hasActiveHumanControl('task-crash'), false)
  assert.equal(JSON.parse(fs.readFileSync(stateFile, 'utf-8')).tasks[0].humanControlLease, undefined)
})

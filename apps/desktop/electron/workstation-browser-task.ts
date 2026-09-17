import fs from 'node:fs'
import path from 'node:path'

export const BROWSER_TASK_STATE_VERSION = 1

export type BrowserTaskStatus = 'visible' | 'hidden' | 'parked'
export type BrowserTaskRecoveryState = 'fresh' | 'restored' | 'recreated' | null

export interface BrowserHumanControlLease {
  owner: 'human'
  taskId: string
  sessionId: string | null
  tabId: string | null
  pageId: number | null
  profileScope: string | null
  acquiredAt: string
  expiresAt: string
  renewedAt: string | null
}

export interface BrowserTask {
  taskId: string
  createdAt: string
  panelHost: string | null
  controlHost: string | null
  sessionHost: string | null
  kanbanCardId: string | null
  runId: string | null
  localConnection: string | null
  status: BrowserTaskStatus
  leaseState: string | null
  parked: boolean
  recoveryState: BrowserTaskRecoveryState
  updatedAt: string
  humanControlLease?: BrowserHumanControlLease
}

export interface BrowserTaskSnapshot {
  version: typeof BROWSER_TASK_STATE_VERSION
  browserTaskCounter: number
  tasks: BrowserTask[]
}

export interface BrowserTaskSeed {
  taskId?: string
  panelHost?: string | null
  controlHost?: string | null
  sessionHost?: string | null
  kanbanCardId?: string | null
  runId?: string | null
  localConnection?: string | null
  leaseState?: string | null
}

export interface BrowserTaskBindings<Page, ShowContext = void> {
  ensurePage(taskId: string): Page
  pageForTask(taskId: string): Page | null
  pageIsAlive(page: Page): boolean
  showPage(taskId: string, page: Page, context: ShowContext): void
  hidePage(taskId: string, page: Page): void
  parkPage(taskId: string, page: Page): void
  destroyPage(taskId: string, page: Page): void
}

export interface BrowserTaskPersistence {
  load(): BrowserTaskSnapshot | null
  save(snapshot: BrowserTaskSnapshot): void
}

function cloneTask(task: BrowserTask): BrowserTask {
  return { ...task }
}

function validText(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0
}

function validNullableText(value: unknown): value is string | null {
  return value === null || typeof value === 'string'
}

function validStatus(value: unknown): value is BrowserTaskStatus {
  return value === 'visible' || value === 'hidden' || value === 'parked'
}

function validRecoveryState(value: unknown): value is BrowserTaskRecoveryState {
  return value === null || value === 'fresh' || value === 'restored' || value === 'recreated'
}

function validHumanControlLease(value: unknown, taskId: string): value is BrowserHumanControlLease {
  if (!value || typeof value !== 'object') {
    return false
  }

  const lease = value as Partial<BrowserHumanControlLease>

  return (
    lease.owner === 'human' &&
    lease.taskId === taskId &&
    validNullableText(lease.sessionId) &&
    validNullableText(lease.tabId) &&
    (lease.pageId === null || Number.isInteger(lease.pageId)) &&
    validNullableText(lease.profileScope) &&
    validText(lease.acquiredAt) &&
    validText(lease.expiresAt) &&
    validNullableText(lease.renewedAt)
  )
}

function parsePersistedTask(value: unknown): BrowserTask | null {
  if (!value || typeof value !== 'object') {
    return null
  }

  const task = value as Partial<BrowserTask> & { status?: unknown; recoveryState?: unknown }

  if (!validText(task.taskId) || !validText(task.createdAt) || !validText(task.updatedAt)) {
    return null
  }

  if (!validStatus(task.status) || !validRecoveryState(task.recoveryState)) {
    return null
  }

  if (!validNullableText(task.panelHost) || !validNullableText(task.controlHost)) {
    return null
  }

  if (!validNullableText(task.sessionHost) || !validNullableText(task.localConnection)) {
    return null
  }

  if (!validNullableText(task.leaseState) || typeof task.parked !== 'boolean') {
    return null
  }

  if (task.kanbanCardId !== undefined && !validNullableText(task.kanbanCardId)) {
    return null
  }

  if (task.runId !== undefined && !validNullableText(task.runId)) {
    return null
  }

  const parsed: BrowserTask = {
    taskId: task.taskId.trim(),
    createdAt: task.createdAt,
    panelHost: task.panelHost,
    controlHost: task.controlHost,
    sessionHost: task.sessionHost,
    kanbanCardId: task.kanbanCardId ?? null,
    runId: task.runId ?? null,
    localConnection: task.localConnection,
    status: task.status,
    leaseState: task.leaseState,
    parked: task.parked,
    recoveryState: task.recoveryState,
    updatedAt: task.updatedAt
  }

  if (validHumanControlLease(task.humanControlLease, parsed.taskId)) {
    parsed.humanControlLease = task.humanControlLease
  }

  return parsed
}

export function normalizeBrowserTaskSnapshot(value: unknown): BrowserTaskSnapshot | null {
  if (!value || typeof value !== 'object') {
    return null
  }

  const snapshot = value as { version?: unknown; browserTaskCounter?: unknown; tasks?: unknown }

  if (snapshot.version !== BROWSER_TASK_STATE_VERSION) {
    return null
  }

  if (!Number.isInteger(snapshot.browserTaskCounter) || Number(snapshot.browserTaskCounter) < 0) {
    return null
  }

  if (!Array.isArray(snapshot.tasks)) {
    return null
  }

  // One logical task may appear only once. If a crash left duplicate metadata,
  // keep the most recently updated valid record and discard the stale duplicate.
  const deduped = new Map<string, BrowserTask>()

  for (const raw of snapshot.tasks) {
    const parsed = parsePersistedTask(raw)

    if (!parsed) {
      continue
    }

    const current = deduped.get(parsed.taskId)

    if (!current || parsed.updatedAt >= current.updatedAt) {
      deduped.set(parsed.taskId, parsed)
    }
  }

  return {
    version: BROWSER_TASK_STATE_VERSION,
    browserTaskCounter: Number(snapshot.browserTaskCounter),
    tasks: [...deduped.values()]
  }
}

export class BrowserTaskFilePersistence implements BrowserTaskPersistence {
  constructor(readonly filePath: string) {}

  load(): BrowserTaskSnapshot | null {
    try {
      return normalizeBrowserTaskSnapshot(JSON.parse(fs.readFileSync(this.filePath, 'utf-8')))
    } catch {
      return null
    }
  }

  save(snapshot: BrowserTaskSnapshot): void {
    fs.mkdirSync(path.dirname(this.filePath), { recursive: true })
    const temp = `${this.filePath}.${process.pid}.${Date.now()}.tmp`
    fs.writeFileSync(temp, JSON.stringify(snapshot, null, 2), { encoding: 'utf-8', mode: 0o600 })
    fs.renameSync(temp, this.filePath)

    try {
      fs.chmodSync(this.filePath, 0o600)
    } catch {
      // Best effort on Windows / filesystems without POSIX permissions.
    }
  }
}

export class BrowserTaskLifecycle<Page, ShowContext = void> {
  private readonly tasks = new Map<string, BrowserTask>()
  private browserTaskCounter = 0

  constructor(
    private readonly bindings: BrowserTaskBindings<Page, ShowContext>,
    private readonly persistence: BrowserTaskPersistence | null = null,
    private readonly now: () => Date = () => new Date()
  ) {}

  restore(): BrowserTask[] {
    const snapshot = this.persistence?.load()

    if (!snapshot) {
      return []
    }

    this.tasks.clear()
    this.browserTaskCounter = snapshot.browserTaskCounter
    const timestamp = this.timestamp()

    for (const saved of snapshot.tasks) {
      // Renderer/WebContents identity cannot survive an Electron process restart.
      // Preserve logical ownership, but normalize the task to parked metadata and
      // lazily recreate/recover its page only when the task is shown or used.
      const restored: BrowserTask = {
        ...saved,
        status: 'parked',
        parked: true,
        recoveryState: 'restored',
        updatedAt: timestamp
      }

      // A human lease is process-local authority. It must never survive a
      // crash/restart without a fresh human acquisition against the live page.
      delete restored.humanControlLease
      this.tasks.set(saved.taskId, restored)
    }

    this.persist()

    return this.listTasks()
  }

  createTask(seed: BrowserTaskSeed = {}): BrowserTask {
    const taskId = seed.taskId?.trim() || `browser-task-${++this.browserTaskCounter}`
    const existing = this.tasks.get(taskId)

    if (existing) {
      this.ensureLivePage(taskId, existing)

      return cloneTask(existing)
    }

    const timestamp = this.timestamp()

    const task: BrowserTask = {
      taskId,
      createdAt: timestamp,
      panelHost: seed.panelHost ?? null,
      controlHost: seed.controlHost ?? null,
      sessionHost: seed.sessionHost ?? null,
      kanbanCardId: seed.kanbanCardId ?? null,
      runId: seed.runId ?? null,
      localConnection: seed.localConnection ?? null,
      status: 'parked',
      leaseState: seed.leaseState ?? null,
      parked: true,
      recoveryState: 'fresh',
      updatedAt: timestamp
    }

    this.tasks.set(taskId, task)
    this.bindings.ensurePage(taskId)
    this.persist()

    return cloneTask(task)
  }

  bindSessionHost(taskId: string, sessionHost: string): BrowserTask {
    const task = this.requireTask(taskId)

    if (task.sessionHost && task.sessionHost !== sessionHost) {
      throw new Error(`BrowserTask session identity mismatch: ${taskId}`)
    }

    if (task.sessionHost === sessionHost) {
      return cloneTask(task)
    }

    task.sessionHost = sessionHost
    task.updatedAt = this.timestamp()
    this.persist()

    return cloneTask(task)
  }

  bindKanbanCard(taskId: string, kanbanCardId: string): BrowserTask {
    const task = this.requireTask(taskId)

    if (task.kanbanCardId && task.kanbanCardId !== kanbanCardId) {
      throw new Error(`BrowserTask kanban card mismatch: ${taskId}`)
    }

    if (task.kanbanCardId === kanbanCardId) {
      return cloneTask(task)
    }

    task.kanbanCardId = kanbanCardId
    task.updatedAt = this.timestamp()
    this.persist()

    return cloneTask(task)
  }

  bindRun(taskId: string, runId: string): BrowserTask {
    const task = this.requireTask(taskId)

    if (task.runId && task.runId !== runId) {
      throw new Error(`BrowserTask run mismatch: ${taskId}`)
    }

    if (task.runId === runId) {
      return cloneTask(task)
    }

    task.runId = runId
    task.updatedAt = this.timestamp()
    this.persist()

    return cloneTask(task)
  }

  showTask(taskId: string, context: ShowContext): BrowserTask {
    const task = this.requireTask(taskId)
    const timestamp = this.timestamp()
    let parkedOther = false

    for (const other of this.tasks.values()) {
      if (other.taskId === taskId || other.status !== 'visible') {
        continue
      }

      const otherPage = this.livePage(other.taskId)

      if (otherPage) {
        this.bindings.parkPage(other.taskId, otherPage)
      }

      Object.assign(other, { status: 'parked' as const, parked: true, updatedAt: timestamp })
      parkedOther = true
    }

    if (parkedOther) {
      this.persist()
    }

    const page = this.ensureLivePage(taskId, task)
    this.bindings.showPage(taskId, page, context)

    return this.updateTask(task, { status: 'visible', parked: false })
  }

  hideTask(taskId: string): BrowserTask {
    const task = this.requireTask(taskId)
    const page = this.livePage(taskId)

    if (page) {
      this.bindings.hidePage(taskId, page)
    }

    return this.updateTask(task, { status: 'hidden', parked: false })
  }

  parkTask(taskId: string): BrowserTask {
    const task = this.requireTask(taskId)
    const page = this.livePage(taskId)

    if (page) {
      this.bindings.parkPage(taskId, page)
    }

    return this.updateTask(task, { status: 'parked', parked: true })
  }

  destroyTask(taskId: string): boolean {
    const task = this.tasks.get(taskId)

    if (!task) {
      return false
    }

    // Explicit destroy owns cleanup even when the page is already crashed.
    // Bindings may still need to remove a stale task -> page association.
    const page = this.bindings.pageForTask(taskId)

    if (page) {
      this.bindings.destroyPage(taskId, page)
    }

    this.tasks.delete(taskId)
    this.persist()

    return true
  }

  task(taskId: string): BrowserTask | null {
    const task = this.tasks.get(taskId)

    return task ? cloneTask(task) : null
  }

  listTasks(): BrowserTask[] {
    return [...this.tasks.values()].map(cloneTask)
  }

  acquireHumanControl(
    taskId: string,
    details: Pick<BrowserHumanControlLease, 'sessionId' | 'tabId' | 'pageId' | 'profileScope'>,
    ttlMs: number
  ): BrowserTask {
    if (!Number.isFinite(ttlMs) || ttlMs <= 0) {
      throw new Error('human control lease TTL must be positive')
    }

    const task = this.requireTask(taskId)
    this.expireHumanControl(taskId)
    const timestamp = this.timestamp()
    const expiresAt = new Date(this.now().getTime() + ttlMs).toISOString()
    const current = task.humanControlLease
    task.humanControlLease = {
      owner: 'human',
      taskId,
      sessionId: details.sessionId,
      tabId: details.tabId,
      pageId: details.pageId,
      profileScope: details.profileScope,
      acquiredAt: current?.acquiredAt ?? timestamp,
      expiresAt,
      renewedAt: current ? timestamp : null
    }
    task.updatedAt = timestamp
    this.persist()

    return cloneTask(task)
  }

  renewHumanControl(taskId: string, ttlMs: number): BrowserTask {
    const task = this.requireTask(taskId)
    this.expireHumanControl(taskId)

    if (!task.humanControlLease) {
      throw new Error(`No active human control lease: ${taskId}`)
    }

    if (!Number.isFinite(ttlMs) || ttlMs <= 0) {
      throw new Error('human control lease TTL must be positive')
    }

    const timestamp = this.timestamp()
    task.humanControlLease = {
      ...task.humanControlLease,
      expiresAt: new Date(this.now().getTime() + ttlMs).toISOString(),
      renewedAt: timestamp
    }
    task.updatedAt = timestamp
    this.persist()

    return cloneTask(task)
  }

  releaseHumanControl(taskId: string): boolean {
    const task = this.requireTask(taskId)
    const hadLease = Boolean(task.humanControlLease)

    if (hadLease) {
      delete task.humanControlLease
      task.updatedAt = this.timestamp()
      this.persist()
    }

    return hadLease
  }

  expireHumanControl(taskId: string): boolean {
    const task = this.requireTask(taskId)
    const lease = task.humanControlLease

    if (!lease || Date.parse(lease.expiresAt) > this.now().getTime()) {
      return false
    }

    delete task.humanControlLease
    task.updatedAt = this.timestamp()
    this.persist()

    return true
  }

  hasActiveHumanControl(taskId: string): boolean {
    const task = this.tasks.get(taskId)

    if (!task) {
      return false
    }

    this.expireHumanControl(taskId)

    return Boolean(task.humanControlLease)
  }

  snapshot(): BrowserTaskSnapshot {
    return {
      version: BROWSER_TASK_STATE_VERSION,
      browserTaskCounter: this.browserTaskCounter,
      tasks: this.listTasks()
    }
  }

  private requireTask(taskId: string): BrowserTask {
    const task = this.tasks.get(taskId)

    if (!task) {
      throw new Error(`BrowserTask not found: ${taskId}`)
    }

    return task
  }

  private livePage(taskId: string): Page | null {
    const page = this.bindings.pageForTask(taskId)

    return page && this.bindings.pageIsAlive(page) ? page : null
  }

  private ensureLivePage(taskId: string, task: BrowserTask): Page {
    const existing = this.livePage(taskId)

    if (existing) {
      return existing
    }

    const page = this.bindings.ensurePage(taskId)

    if (!this.bindings.pageIsAlive(page)) {
      throw new Error(`BrowserTask page could not be recovered: ${taskId}`)
    }

    task.recoveryState = 'recreated'
    task.updatedAt = this.timestamp()
    this.persist()

    return page
  }

  private updateTask(task: BrowserTask, patch: Pick<BrowserTask, 'status' | 'parked'>): BrowserTask {
    Object.assign(task, patch, { updatedAt: this.timestamp() })
    this.persist()

    return cloneTask(task)
  }

  private persist(): void {
    this.persistence?.save(this.snapshot())
  }

  private timestamp(): string {
    return this.now().toISOString()
  }
}

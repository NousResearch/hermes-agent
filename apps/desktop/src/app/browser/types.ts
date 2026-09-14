export type WorkstationBrowserControlOwner = 'agent' | 'human'

export interface WorkstationBrowserBounds {
  x: number
  y: number
  width: number
  height: number
}

export interface WorkstationBrowserTabState {
  id: string
  title: string
  url: string
  active: boolean
  loading: boolean
  canGoBack: boolean
  canGoForward: boolean
  crashed: boolean
  ownerTaskId: string | null
}

export type BrowserTaskStatus = 'visible' | 'hidden' | 'parked'
export type BrowserTaskRecoveryState = 'fresh' | 'restored' | 'recreated' | null

export interface BrowserTask {
  taskId: string
  createdAt: string
  updatedAt: string
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
}

export interface WorkstationDownloadItem {
  id: string
  filename: string
  savePath: string
  totalBytes: number
  receivedBytes: number
  state: 'progressing' | 'completed' | 'cancelled' | 'interrupted'
  url: string
}

export interface WorkstationBrowserState {
  runtime: 'electron-chromium'
  ready: boolean
  attached: boolean
  viewportHost: 'hub' | 'chat' | string | null
  backgroundCapable: true
  paused: boolean
  controlOwner: WorkstationBrowserControlOwner
  controlReady: boolean
  profilePath: string
  cacheBytes: number | null
  activeTabId: string | null
  tabs: WorkstationBrowserTabState[]
  tasks: BrowserTask[]
  downloads: WorkstationDownloadItem[]
  lastError: string | null
}

export interface WorkstationResource {
  resource_type: 'browser' | 'browser_task' | 'execution_journal'
  resource_id: string
  task_id: string | null
  session_id: string | null
  permissions: string[]
  state: Record<string, unknown>
  updated_at: string
}

export interface WorkstationResourceSnapshot {
  schema_version: 1
  runtime: 'electron-chromium'
  generated_at: string
  resources: WorkstationResource[]
}

export interface WorkstationEvent {
  event_id: string
  kind: string
  task_id: string
  session_id: string
  message: string
  timestamp: string
  elapsed_seconds?: number
  url?: string | null
  browser_tab_id?: string | null
  risk?: string
  metadata?: Record<string, unknown>
  evidence?: Array<Record<string, unknown>>
}

export interface WorkstationEventSnapshot {
  schema_version: 1
  runtime: 'electron-chromium'
  generated_at: string
  task_id: string | null
  events: WorkstationEvent[]
}

export interface WorkstationBrowserBridge {
  status: () => Promise<WorkstationBrowserState>
  ensure: () => Promise<WorkstationBrowserState>
  resources: () => Promise<WorkstationResourceSnapshot>
  events?: (taskId?: string | null, limit?: number) => Promise<WorkstationEventSnapshot>
  newTab: (target?: string) => Promise<WorkstationBrowserState>
  activateTab: (tabId: string) => Promise<WorkstationBrowserState>
  closeTab: (tabId: string) => Promise<WorkstationBrowserState>
  navigate: (target: string) => Promise<WorkstationBrowserState>
  back: () => Promise<WorkstationBrowserState>
  forward: () => Promise<WorkstationBrowserState>
  reload: () => Promise<WorkstationBrowserState>
  stop: () => Promise<WorkstationBrowserState>
  focus: () => Promise<WorkstationBrowserState>
  attach: (bounds: WorkstationBrowserBounds, host?: string) => Promise<WorkstationBrowserState>
  setBounds: (bounds: WorkstationBrowserBounds, expectedHost?: string) => Promise<WorkstationBrowserState>
  detach: () => Promise<WorkstationBrowserState>
  setVisible: (visible: boolean) => Promise<WorkstationBrowserState>
  clearError: () => Promise<WorkstationBrowserState>
  transferViewport: (targetHost: string, bounds: WorkstationBrowserBounds) => Promise<WorkstationBrowserState>
  listTasks: () => Promise<BrowserTask[]>
  showTask: (taskId: string, bounds: WorkstationBrowserBounds, host?: string) => Promise<BrowserTask>
  hideTask: (taskId: string) => Promise<BrowserTask>
  parkTask: (taskId: string) => Promise<BrowserTask>
  destroyTask: (taskId: string) => Promise<boolean>
  clearParkedTasks: () => Promise<number>
  pause: () => Promise<WorkstationBrowserState>
  resume: () => Promise<WorkstationBrowserState>
  takeControl: () => Promise<WorkstationBrowserState>
  releaseControl: () => Promise<WorkstationBrowserState>
  cleanupCache: (force?: boolean) => Promise<WorkstationBrowserState>
  getTaskJournal: (taskId: string) => Promise<TaskTimelineEvent[]>
  onState: (callback: (state: WorkstationBrowserState) => void) => () => void
  onOpenChatPreview?: (callback: (data: { url: string; taskId?: string }) => void) => () => void
}

export interface TaskTimelineEvent {
  event_id: string
  kind: string
  task_id: string
  session_id: string
  message: string
  timestamp: string
  elapsed_seconds?: number
  url?: string | null
  risk?: string
  browser_tab_id?: string | null
  metadata?: Record<string, unknown>
  evidence?: Array<{ kind: string; uri: string; summary?: string }>
}

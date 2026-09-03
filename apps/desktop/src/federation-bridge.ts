/**
 * Federation bridge — connects Desktop store to Gateway API via Electron IPC.
 *
 * Security design (AGENTS.md compliant):
 * - Auth token NEVER exposed to renderer (stays in Electron main process)
 * - All API calls go through IPC handlers (fed:status, fed:peers, etc.)
 * - Renderer only receives sanitized data (no raw HTTP responses)
 * - Timeout protection (5s per request)
 * - Error handling without leaking internals
 *
 * Data flow:
 * Gateway API (Python) → Electron IPC (auth proxy) → nanostores (renderer cache) → UI
 */
import { ipcRenderer } from 'electron'

import {
  type DeviceRole,
  type DeviceStatus,
  type FederationDevice,
  setFedDevices,
  setFedDiscovering,
  setFedEnabled,
  setFedHealth,
  setFedMode,
  type TrustLevel,
} from '@/store/federation-store'

/**
 * Fetch federation status from Gateway API and update store.
 * This is the authoritative data source — store is just a cache.
 */
export async function refreshFederationStatus(): Promise<void> {
  try {
    const status = await ipcRenderer.invoke('fed:status') as {
      device_count: number
      online_count: number
      offline_count: number
      leader: string
      mode: string
      uptime_sec: number
      tasks: { total: number; completed: number; failed: number; pending: number }
      api_version: string
      hermes_version: string
    }

    setFedEnabled(status.device_count > 0)
    setFedMode(status.mode as 'auto' | 'lan' | 'shared_db')

    // Health = percentage of online devices
    if (status.device_count > 0) {
      setFedHealth(Math.round((status.online_count / status.device_count) * 100))
    }
  } catch (error) {
    console.warn('Federation bridge: failed to fetch status', error)
    setFedHealth(0)
  }
}

/**
 * Fetch peer list from Gateway API and update device store.
 */
export async function refreshFederationPeers(): Promise<void> {
  try {
    const peers = await ipcRenderer.invoke('fed:peers') as Array<{
      device_id: string
      hostname: string
      status: string
      last_seen: number
      latency_ms: number
      compute_score: number
      cpu_cores: number
      memory_gb: number
      is_leader: boolean
      mode: string
      version: string
    }>

    const devices: FederationDevice[] = peers.map((p) => ({
      device_id: p.device_id,
      name: p.hostname.split('.')[0],  // "macbook-pro.local" → "macbook-pro"
      hostname: p.hostname,
      status: (p.status === 'online' ? 'online' : p.status === 'connecting' ? 'connecting' : 'offline') as DeviceStatus,
      ws_url: '',  // Not exposed via API for security
      score: p.compute_score,
      cpu_cores: p.cpu_cores,
      memory_gb: p.memory_gb,
      load_avg: 0,  // Not in API yet
      gpu_type: '',  // Not in API yet
      last_seen: p.last_seen,
      is_local: false,  // Determined by store
      active_tasks: 0,  // From tasks endpoint
      latency_ms: p.latency_ms,
      grid_x: 0,
      grid_y: 0,
      role: (p.is_leader ? 'leader' : 'worker') as DeviceRole,
      trust: ('trust' in p ? p.trust : 'unknown') as TrustLevel,
    }))

    setFedDevices(devices)
  } catch (error) {
    console.warn('Federation bridge: failed to fetch peers', error)
  }
}

/**
 * Fetch task list from Gateway API.
 * Updates active_tasks count for each device.
 */
export async function refreshFederationTasks(): Promise<void> {
  try {
    const tasks = await ipcRenderer.invoke('fed:tasks') as Array<{
      task_id: string
      status: string
      source: string
      target: string
    }>

    // Count active tasks per device
    const taskCounts: Record<string, number> = {}

    for (const task of tasks) {
      if (task.status === 'running' || task.status === 'pending') {
        taskCounts[task.target] = (taskCounts[task.target] || 0) + 1
      }
    }

    // Update devices with task counts
    setFedDevices((prev: FederationDevice[]) =>
      prev.map((d: FederationDevice) => ({
        ...d,
        active_tasks: taskCounts[d.device_id] || 0,
      }))
    )
  } catch (error) {
    console.warn('Federation bridge: failed to fetch tasks', error)
  }
}

/**
 * Check federation health (simple ping).
 */
export async function checkFederationHealth(): Promise<boolean> {
  try {
    const health = await ipcRenderer.invoke('fed:health') as { status: string; uptime_sec: number }

    return health.status === 'healthy'
  } catch {
    return false
  }
}

/**
 * Start periodic refresh cycle.
 * Polls every 5 seconds for live status updates.
 */
let refreshInterval: ReturnType<typeof setInterval> | null = null

export function startFederationRefresh(intervalMs: number = 5000): void {
  stopFederationRefresh()

  // Initial fetch
  void refreshFederationStatus()
  void refreshFederationPeers()
  void refreshFederationTasks()

  // Periodic updates
  refreshInterval = setInterval(async () => {
    await Promise.allSettled([
      refreshFederationStatus(),
      refreshFederationPeers(),
      refreshFederationTasks(),
    ])
  }, intervalMs)
}

/**
 * Stop periodic refresh.
 */
export function stopFederationRefresh(): void {
  if (refreshInterval) {
    clearInterval(refreshInterval)
    refreshInterval = null
  }
}

/**
 * Initialize federation bridge.
 * Called when Settings → Federation page is opened.
 */
export function initFederationBridge(): void {
  setFedDiscovering(true)

  // Try to connect
  checkFederationHealth()
    .then((healthy) => {
      if (healthy) {
        startFederationRefresh()
      }

      setFedDiscovering(false)
    })
    .catch(() => {
      setFedDiscovering(false)
      setFedHealth(0)
    })
}

/**
 * Cleanup federation bridge.
 * Called when Settings → Federation page is closed.
 */
export function cleanupFederationBridge(): void {
  stopFederationRefresh()
  setFedEnabled(false)
  setFedHealth(0)
}

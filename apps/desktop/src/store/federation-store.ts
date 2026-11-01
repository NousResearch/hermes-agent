/**
 * Federation device store — manages device state for the Desktop UI.
 *
 * nanostores (project standard). Backend (gateway/federation/) is authoritative;
 * this store is the renderer cache for painting the federation overlay.
 */
import { atom, computed } from 'nanostores'

export type DeviceStatus = 'online' | 'offline' | 'connecting' | 'error'
export type DeviceRole = 'leader' | 'worker' | 'idle'
export type FedMode = 'shared_db' | 'lan' | 'auto'
/** Federation approval mode — controls when the user is asked. */
export type ApprovalMode = 'ask' | 'auto' | 'review'

/** Trust level of a peer device. */
export type TrustLevel = 'unknown' | 'verified' | 'trusted' | 'admin'

export interface FederationDevice {
  device_id: string
  name: string
  hostname: string
  status: DeviceStatus
  ws_url: string
  score: number
  cpu_cores: number
  memory_gb: number
  load_avg: number
  gpu_type: string
  last_seen: number
  is_local: boolean
  active_tasks: number
  latency_ms: number
  /** Grid position for macOS-style arrangement (normalized 0-1) */
  grid_x: number
  grid_y: number
  role: DeviceRole
  /** Trust level — from the trust system (Phase 17/CRITICAL-2). */
  trust: TrustLevel
}

// ── Atoms ────────────────────────────────────────────────────────────

export const $fedDevices = atom<FederationDevice[]>([])
export const $fedEnabled = atom(false)
export const $fedMode = atom<FedMode>('auto')
export const $fedDiscovering = atom(false)
export const $fedHealth = atom(0)
export const $fedAuthToken = atom<string | null>(null)

/** Pending relay decisions awaiting user approval (Phase 17). */
export interface FederationPendingDecision {
  task_id: string
  task_description: string
  from_device: string
  to_device: string
  confidence: number
  sensitivity: 'low' | 'medium' | 'high' | 'critical'
  created_at: number
}

export const $fedApprovalMode = atom<ApprovalMode>('auto')
export const $fedPendingDecisions = atom<FederationPendingDecision[]>([])

// ── Derived ──────────────────────────────────────────────────────────

export const $onlineDevices = computed($fedDevices, (devices) =>
  devices.filter((d) => d.status === 'online'),
)

export const $localDevice = computed($fedDevices, (devices) =>
  devices.find((d) => d.is_local),
)

export const $deviceCount = computed($fedDevices, (devices) => devices.length)
export const $onlineCount = computed($onlineDevices, (devices) => devices.length)

export const $pendingDecisionCount = computed($fedPendingDecisions, (d) => d.length)

/** Devices with known trust levels. */
export const $trustedDevices = computed($fedDevices, (devices) =>
  devices.filter((d) => d.trust === 'admin' || d.trust === 'trusted'),
)

/** Devices pending approval of their relay request. */
export const $devicesAwaitingApproval = computed($fedPendingDecisions, (decisions) =>
  decisions.map((d) => d.to_device),
)

// ── Updaters ─────────────────────────────────────────────────────────

type DeviceUpdater = FederationDevice[] | ((prev: FederationDevice[]) => FederationDevice[])

export const setFedDevices = (updater: DeviceUpdater) => {
  const current = $fedDevices.get()
  $fedDevices.set(typeof updater === 'function' ? updater(current) : updater)
}

export const updateFedDevice = (device_id: string, patch: Partial<FederationDevice>) => {
  setFedDevices((prev) =>
    prev.map((d) => (d.device_id === device_id ? { ...d, ...patch } : d)),
  )
}

export const addFedDevice = (device: FederationDevice) => {
  setFedDevices((prev) => {
    const exists = prev.find((d) => d.device_id === device.device_id)

    return exists ? prev.map((d) => (d.device_id === device.device_id ? { ...d, ...device } : d)) : [...prev, device]
  })
}

export const removeFedDevice = (device_id: string) => {
  setFedDevices((prev) => prev.filter((d) => d.device_id !== device_id))
}

export const updateGridPos = (device_id: string, x: number, y: number) => {
  updateFedDevice(device_id, { grid_x: x, grid_y: y })
}

export const setFedEnabled = (v: boolean) => $fedEnabled.set(v)
export const setFedMode = (v: FedMode) => $fedMode.set(v)
export const setFedDiscovering = (v: boolean) => $fedDiscovering.set(v)
export const setFedHealth = (v: number) => $fedHealth.set(v)
export const setFedAuthToken = (v: string | null) => $fedAuthToken.set(v)
export const setFedApprovalMode = (v: ApprovalMode) => $fedApprovalMode.set(v)

export const addPendingDecision = (decision: FederationPendingDecision) => {
  const prev = $fedPendingDecisions.get()
  $fedPendingDecisions.set([...prev, decision])
}

export const removePendingDecision = (task_id: string) => {
  const prev = $fedPendingDecisions.get()
  $fedPendingDecisions.set(prev.filter((d) => d.task_id !== task_id))
}

export const approvePendingDecision = (task_id: string): FederationPendingDecision | null => {
  const current = $fedPendingDecisions.get()
  const found = current.find((d) => d.task_id === task_id)

  if (found) {
    const prev = $fedPendingDecisions.get()
    $fedPendingDecisions.set(prev.filter((d) => d.task_id !== task_id))
  }

  return found ?? null
}

export const denyPendingDecision = (task_id: string) => {
  removePendingDecision(task_id)
}

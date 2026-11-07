/**
 * Federation Devices overlay — macOS Display arrangement-style device management.
 *
 * Shows all federation peers as visual cards in a grid, similar to
 * macOS System Settings → Displays where you arrange monitors.
 *
 * - Visual grid of device cards with status indicators
 * - Drag to rearrange (persists layout)
 * - Click a card to expand detail panel
 * - Add new device via QR code or manual URL
 * - Real-time status (online/offline/latency)
 * - Compute capability display
 */
import { useStore } from '@nanostores/react'
import {
  Cpu,
  Globe,
  Monitor,
  Network,
  Plus,
  Server,
  Settings2,
  Shield,
  SignalHigh,
  Wifi,
  WifiOff,
  X,
} from 'lucide-react'
import { useCallback, useMemo, useState } from 'react'

import { Badge, badgeVariants } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'
import { Switch } from '@/components/ui/switch'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { cn } from '@/lib/utils'
import {
  $fedApprovalMode,
  $fedDevices,
  $fedEnabled,
  $fedHealth,
  $fedMode,
  $fedPendingDecisions,
  $onlineCount,
  $pendingDecisionCount,
  type ApprovalMode,
  denyPendingDecision,
  type FederationDevice,
  setFedEnabled,
  setFedMode,
  updateFedDevice,
} from '@/store/federation-store'

// ── Helpers ──────────────────────────────────────────────────────────

function formatUptime(lastSeen: number): string {
  const diff = Date.now() / 1000 - lastSeen

  if (diff < 60) {return `${Math.floor(diff)}s ago`}

  if (diff < 3600) {return `${Math.floor(diff / 60)}m ago`}

  return `${Math.floor(diff / 3600)}h ago`
}

function latencyColor(ms: number): string {
  if (ms < 50) {return 'text-emerald-500'}

  if (ms < 150) {return 'text-amber-500'}

  return 'text-red-500'
}

function statusBadge(status: string) {
  switch (status) {
    case 'online':
      return <Badge className="gap-1" variant="default"><SignalHigh className="size-3" />Online</Badge>

    case 'connecting':
      return <Badge className="gap-1" variant="default"><Wifi className="size-3 animate-pulse" />Connecting</Badge>

    case 'offline':

    default:
      return <Badge className="gap-1" variant="muted"><WifiOff className="size-3" />Offline</Badge>
  }
}

// ── Device Card ──────────────────────────────────────────────────────

function DeviceCard({
  device,
  selected,
  onSelect,
}: {
  device: FederationDevice
  selected: boolean
  onSelect: () => void
}) {
  const isOnline = device.status === 'online'

  return (
    <button
      className={cn(
        'group relative flex w-full flex-col overflow-hidden rounded-xl border text-left transition-all duration-200',
        'hover:shadow-md hover:border-(--stroke-nous)',
        selected
          ? 'ring-2 ring-(--ui-primary) border-(--ui-primary)'
          : 'border-(--ui-border)',
        !isOnline && 'opacity-60',
      )}
      onClick={onSelect}
      onKeyDown={(e: React.KeyboardEvent) => e.key === 'Enter' && onSelect()}
      type="button"
    >
      {/* Status bar */}
      <div className="flex items-center justify-between border-b border-(--ui-border) px-3 py-2">
        <div className="flex items-center gap-2">
          <Monitor className="size-4 shrink-0 text-muted-foreground" />
          <span className="truncate text-sm font-medium">{device.name}</span>
        </div>
        {device.is_local && (
          <span className={cn(badgeVariants({ variant: 'muted' }), 'text-[10px]')}>
            THIS DEVICE
          </span>
        )}
      </div>

      {/* Content */}
      <div className="flex flex-1 flex-col gap-2 p-3">
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <Server className="size-3 shrink-0" />
          <span className="truncate">{device.hostname}</span>
        </div>

        {/* Compute stats */}
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="flex items-center gap-1.5">
            <Cpu className="size-3 shrink-0 text-muted-foreground" />
            <span>{device.cpu_cores} cores</span>
          </div>
          <div className="flex items-center gap-1.5">
            <Shield className="size-3 shrink-0 text-muted-foreground" />
            <span>{device.memory_gb.toFixed(0)} GB</span>
          </div>
        </div>

        {/* Latency */}
        {isOnline && device.latency_ms > 0 && (
          <div className="flex items-center gap-1.5 text-xs">
            <Wifi className={cn('size-3 shrink-0', latencyColor(device.latency_ms))} />
            <span className={latencyColor(device.latency_ms)}>
              {device.latency_ms.toFixed(0)}ms
            </span>
          </div>
        )}

        {/* Active tasks */}
        {device.active_tasks > 0 && (
          <div className="text-[10px] text-muted-foreground">
            {device.active_tasks} active task{device.active_tasks > 1 ? 's' : ''}
          </div>
        )}
      </div>

      {/* Configure button on hover */}
      <div className="absolute right-2 top-11 hidden group-hover:block">
        <Button className="size-6 rounded-full" size="icon" variant="ghost">
          <Settings2 className="size-3" />
        </Button>
      </div>
    </button>
  )
}

// ── Device Detail Panel ──────────────────────────────────────────────

function DeviceDetailPanel({
  device,
  onClose,
}: {
  device: FederationDevice
  onClose: () => void
}) {
  const StatRow = ({ label, value }: { label: string; value: React.ReactNode }) => (
    <div className="flex justify-between">
      <span className="text-muted-foreground">{label}</span>
      <span>{value}</span>
    </div>
  )

  return (
    <div className="rounded-lg border border-(--ui-border) bg-(--ui-bg-elevated) p-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold">{device.name}</h3>
        <Button className="size-6" onClick={onClose} size="icon" variant="ghost">
          <X className="size-3" />
        </Button>
      </div>

      <div className="space-y-2 text-sm">
        <StatRow label="Device ID" value={<span className="font-mono text-xs">{device.device_id}</span>} />
        <StatRow label="Hostname" value={device.hostname} />
        <StatRow label="Status" value={statusBadge(device.status)} />
        <Separator />
        <StatRow label="CPU" value={`${device.cpu_cores} cores`} />
        <StatRow label="Memory" value={`${device.memory_gb.toFixed(1)} GB`} />
        <StatRow label="Load" value={device.load_avg.toFixed(2)} />
        {device.gpu_type && <StatRow label="GPU" value={device.gpu_type} />}
        <Separator />
        <StatRow label="Score" value={<span className="font-mono">{device.score.toFixed(1)}</span>} />
        <StatRow
          label="Latency"
          value={
            <span className={latencyColor(device.latency_ms)}>
              {device.latency_ms > 0 ? `${device.latency_ms.toFixed(0)}ms` : '—'}
            </span>
          }
        />
        <StatRow label="Last seen" value={formatUptime(device.last_seen)} />
        <StatRow label="Role" value={device.role} />
      </div>
    </div>
  )
}

// ── Add Device Dialog ────────────────────────────────────────────────

function AddDeviceDialog({
  open,
  onClose,
}: {
  open: boolean
  onClose: () => void
}) {
  const [manualUrl, setManualUrl] = useState('')
  const [tab, setTab] = useState('auto')

  return (
    <Dialog onOpenChange={(v) => !v && onClose()} open={open}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Add Federation Device</DialogTitle>
        </DialogHeader>

        <Tabs className="gap-2" onValueChange={setTab} value={tab}>
          <TabsList className="w-full">
            <TabsTrigger className="flex-1" value="auto">Auto (mDNS)</TabsTrigger>
            <TabsTrigger className="flex-1" value="manual">Manual URL</TabsTrigger>
          </TabsList>

          <div className="py-4">
            {tab === 'auto' && (
              <div className="flex flex-col items-center gap-3 py-4">
                <div className="flex size-40 items-center justify-center rounded-xl border-2 border-dashed border-(--ui-border) bg-(--ui-bg)">
                  <Network className="size-12 text-muted-foreground" />
                </div>
                <p className="text-center text-sm text-muted-foreground">
                  Ensure both devices are on the same network. mDNS will auto-discover.
                </p>
              </div>
            )}
            {tab === 'manual' && (
              <div className="space-y-3 py-2">
                <p className="text-sm text-muted-foreground">
                  Enter the WebSocket URL of the remote device.
                </p>
                <Input
                  onChange={(e) => setManualUrl(e.target.value)}
                  placeholder="wss://192.168.1.10:18765"
                  value={manualUrl}
                />
                <Button className="w-full" disabled={!manualUrl}>
                  Connect
                </Button>
              </div>
            )}
          </div>
        </Tabs>

        <div className="flex justify-end">
          <Button onClick={onClose} variant="ghost">Cancel</Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}

// ── Main Overlay ─────────────────────────────────────────────────────

// ── Pending Decisions Panel (Phase 17 relay approval UI) ──────────────
function PendingDecisionsPanel({
  decisions,
  approvalMode,
}: {
  decisions: ReturnType<typeof useStore<typeof $fedPendingDecisions>>
  approvalMode: ApprovalMode
}) {
  if (decisions.length === 0) {return null}

  return (
    <div className="border-b border-(--ui-border) bg-amber-500/5 px-6 py-3">
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Shield className="size-4 text-amber-600" />
          <span className="text-sm font-medium text-amber-700 dark:text-amber-400">
            {decisions.length} relay decision{decisions.length > 1 ? 's' : ''} awaiting approval
          </span>
        </div>
        <span className="text-xs text-muted-foreground">
          mode: <span className="font-mono uppercase">{approvalMode}</span>
        </span>
      </div>
      <div className="flex flex-wrap gap-2">
        {decisions.map((d) => (
          <div
            className="flex items-center gap-2 rounded-md border border-amber-200 bg-amber-50/50 px-3 py-1.5 text-xs dark:border-amber-800 dark:bg-amber-950/30"
            key={d.task_id}
          >
            <span className="max-w-[160px] truncate font-mono text-amber-900 dark:text-amber-200">
              {d.task_description}
            </span>
            <span className="text-muted-foreground">→</span>
            <span className="font-mono text-muted-foreground">{d.to_device.slice(0, 8)}</span>
            <span className={cn(
              'rounded px-1 py-0.5 text-[10px] font-semibold uppercase',
              d.sensitivity === 'critical' ? 'bg-red-100 text-red-700' :
              d.sensitivity === 'high' ? 'bg-orange-100 text-orange-700' :
              'bg-stone-100 text-stone-600',
            )}>
              {d.sensitivity}
            </span>
            <div className="flex gap-1">
              <button
                className="rounded bg-stone-200 px-1.5 py-0.5 hover:bg-stone-300 dark:bg-stone-700 dark:hover:bg-stone-600"
                onClick={() => denyPendingDecision(d.task_id)}
              >
                Deny
              </button>
              <button
                className="rounded bg-emerald-100 px-1.5 py-0.5 text-emerald-700 hover:bg-emerald-200"
                onClick={() => {/* approve → bridge IPC */}}
              >
                Approve
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export function FederationDevicesOverlay() {
  const devices = useStore($fedDevices)
  const enabled = useStore($fedEnabled)
  const mode = useStore($fedMode)
  const health = useStore($fedHealth)
  const onlineCount = useStore($onlineCount)

  const [selectedDevice, setSelectedDevice] = useState<FederationDevice | null>(null)
  const [showAddDialog, setShowAddDialog] = useState(false)
  const [dragId, setDragId] = useState<string | null>(null)

  // Phase 17: approval mode + pending decisions
  const approvalMode = useStore($fedApprovalMode)
  const pendingDecisions = useStore($fedPendingDecisions)
  const pendingCount = useStore($pendingDecisionCount)

  // Trust badge colors
  const trustColor = (trust: string) =>
    trust === 'admin' ? 'bg-red-500' :
    trust === 'trusted' ? 'bg-emerald-500' :
    trust === 'verified' ? 'bg-blue-500' :
    'bg-stone-400'

  const trustLabel = (trust: string) =>
    trust === 'admin' ? 'Admin' :
    trust === 'trusted' ? 'Trusted' :
    trust === 'verified' ? 'Verified' :
    'Unknown'

  // Sorted: local first, then online, then offline
  const sortedDevices = useMemo(() => {
    return [...devices].sort((a, b) => {
      if (a.is_local) {return -1}

      if (b.is_local) {return 1}

      if (a.status === 'online' && b.status !== 'online') {return -1}

      if (b.status === 'online' && a.status !== 'online') {return 1}

      return b.score - a.score
    })
  }, [devices])

  const handleDrop = useCallback(
    (targetId: string) => {
      if (!dragId || dragId === targetId) {return}
      const target = devices.find((d) => d.device_id === targetId)
      const source = devices.find((d) => d.device_id === dragId)

      if (!target || !source) {return}

      updateFedDevice(dragId, { grid_x: target.grid_x, grid_y: target.grid_y })
      updateFedDevice(targetId, { grid_x: source.grid_x, grid_y: source.grid_y })
      setDragId(null)
    },
    [dragId, devices],
  )

  if (sortedDevices.length === 0 && !enabled) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-4 p-8 text-center">
        <Network className="size-16 text-muted-foreground" />
        <div>
          <h3 className="text-lg font-semibold">Federation Devices</h3>
          <p className="mt-1 text-sm text-muted-foreground">
            Connect multiple devices for cross-device collaboration
          </p>
        </div>
        <Button onClick={() => setShowAddDialog(true)}>
          <Plus className="mr-1.5 size-4" />
          Enable Federation
        </Button>
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-(--ui-border) px-6 py-4">
        <div>
          <h2 className="text-lg font-semibold">Federation Devices</h2>
          <p className="text-sm text-muted-foreground">
            {onlineCount} of {devices.length} devices online
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Switch checked={enabled} onCheckedChange={setFedEnabled} />
          <Button onClick={() => setShowAddDialog(true)} size="sm" variant="outline">
            <Plus className="mr-1.5 size-4" />
            Add Device
          </Button>
        </div>
      </div>

      {/* Mode + Health bar */}
      <div className="flex items-center gap-4 border-b border-(--ui-border) px-6 py-2.5">
        <div className="flex items-center gap-2">
          <Globe className="size-4 text-muted-foreground" />
          <span className="text-sm">Mode</span>
          <Select onValueChange={(v) => setFedMode(v as 'shared_db' | 'lan' | 'auto')} value={mode}>
            <SelectTrigger className="w-36">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="auto">Auto (mDNS)</SelectItem>
              <SelectItem value="lan">Manual (LAN)</SelectItem>
              <SelectItem value="shared_db">Shared DB</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="ml-auto flex items-center gap-2">
          <span className="text-sm text-muted-foreground">Health</span>
          <div className="h-2 w-24 overflow-hidden rounded-full bg-(--ui-bg-tertiary)">
            <div
              className={cn(
                'h-full rounded-full transition-all',
                health > 80 ? 'bg-emerald-500' : health > 50 ? 'bg-amber-500' : 'bg-red-500',
              )}
              style={{ width: `${Math.min(100, Math.max(0, health))}%` }}
            />
          </div>
          <span className="text-sm font-medium tabular-nums">{health}%</span>
        </div>
      </div>

      {/* Phase 17: Pending relay decisions */}
      <PendingDecisionsPanel approvalMode={approvalMode} decisions={pendingDecisions} />

      {/* Device grid — macOS Display arrangement style */}
      <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
        <div className="flex-1 overflow-y-auto p-6">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {sortedDevices.map((device) => (
              <div
                className="cursor-grab active:cursor-grabbing"
                draggable
                key={device.device_id}
                onDragOver={(e) => e.preventDefault()}
                onDragStart={() => setDragId(device.device_id)}
                onDrop={() => handleDrop(device.device_id)}
              >
                <DeviceCard
                  device={device}
                  onSelect={() =>
                    setSelectedDevice(
                      selectedDevice?.device_id === device.device_id ? null : device,
                    )
                  }
                  selected={selectedDevice?.device_id === device.device_id}
                />
              </div>
            ))}
          </div>
        </div>

        {/* Detail panel */}
        {selectedDevice && (
          <div className="border-t border-(--ui-border) bg-(--ui-bg) p-4">
            <DeviceDetailPanel
              device={selectedDevice}
              onClose={() => setSelectedDevice(null)}
            />
          </div>
        )}
      </div>

      <AddDeviceDialog onClose={() => setShowAddDialog(false)} open={showAddDialog} />
    </div>
  )
}

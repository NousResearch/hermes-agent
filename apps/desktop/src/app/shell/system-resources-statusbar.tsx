import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import type { StatusbarItem } from '@/app/shell/statusbar-controls'
import { getLocalHardware, getSystemStats } from '@/hermes'
import { useI18n } from '@/i18n'
import type { Translations } from '@/i18n/types'
import { Activity } from '@/lib/icons'
import { $localModelsEnabled } from '@/store/local-models-flag'
import { $connection } from '@/store/session'
import { $statusbarHiddenIds } from '@/store/statusbar-prefs'
import type { LocalHardware, SystemStats } from '@/types/hermes'

// Live host-resource readout for the bottom bar: CPU + GPU utilization + VRAM
// + RAM + disk of the ACTIVE connection's backend host — over SSH, the remote
// machine. GPU/VRAM/RAM come from /api/local-models/hardware, CPU/disk from
// /api/system/stats (the web dashboard System page's route); each source
// fails alone, so a backend missing one still shows the other's rows (RAM
// falls back to the stats route's memory). Hidden by default (an item most
// users don't watch); the poll runs ONLY while the item is shown, so the
// hidden default costs nothing. 5s cadence — resource numbers, not a
// heartbeat. Stats run on a slower 15s cadence: older backends serve that
// route as `async def` around psutil.cpu_percent(interval=0.1), which holds
// their event loop (and the chat socket) for 100ms per call.
const POLL_MS = 5_000
const STATS_POLL_MS = 15_000

function gb(bytes: number | null | undefined): string {
  return bytes ? `${(bytes / (1 << 30)).toFixed(0)}G` : '—'
}

function gbLong(bytes: number | null | undefined): string {
  return bytes ? `${(bytes / (1 << 30)).toFixed(1)} GB` : '—'
}

function MeterRow({ label, percent, value }: { label: string; percent: number | null; value: string }) {
  return (
    <div className="grid gap-1">
      <div className="flex items-baseline justify-between gap-2">
        {/* Label yields, value never does: if anything ever narrows the row
            again, a truncated label beats a clipped number — "15.2 GB" losing
            its tail reads as a wrong number, not a cut one. */}
        <span className="truncate text-muted-foreground">{label}</span>

        <span className="shrink-0 whitespace-nowrap tabular-nums text-foreground">{value}</span>
      </div>

      {percent !== null && (
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-(--ui-bg-tertiary)">
          <div
            className="h-full rounded-full bg-primary transition-[width] duration-500"
            style={{ width: `${Math.max(1, Math.min(100, percent))}%` }}
          />
        </div>
      )}
    </div>
  )
}

// Polls both routes while `shown`; a change of `connectionKey` (another
// backend) restarts the loop and clears the previous host's values at once.
function useHostResources(shown: boolean, connectionKey: string | null) {
  const [hardware, setHardware] = useState<LocalHardware | null>(null)
  const [stats, setStats] = useState<SystemStats | null>(null)

  useEffect(() => {
    setHardware(null)
    setStats(null)

    if (!shown) {
      return
    }

    let cancelled = false
    let timer: number | null = null
    // A visibilitychange during an in-flight poll must not start a second
    // loop: the in-flight poll schedules the next tick itself.
    let inFlight = false
    let lastStatsAt = Number.NEGATIVE_INFINITY

    const poll = async () => {
      // Hidden window = nobody watching; resume on visibilitychange instead of
      // burning a backend spawn per 5s tick in the background (#120262).
      if ((typeof document !== 'undefined' && document.hidden) || inFlight) {
        return
      }

      inFlight = true
      // Monotonic clock: a wall-clock jump backwards must not stall stats.
      const now = performance.now()
      const statsDue = now - lastStatsAt >= STATS_POLL_MS

      if (statsDue) {
        lastStatsAt = now
      }

      try {
        const [nextHardware, nextStats] = await Promise.allSettled([
          getLocalHardware(),
          statsDue ? getSystemStats() : Promise.resolve(null)
        ])

        if (!cancelled) {
          setHardware(nextHardware.status === 'fulfilled' ? nextHardware.value : null)

          if (statsDue) {
            setStats(nextStats.status === 'fulfilled' ? nextStats.value : null)
          }
        }
      } finally {
        inFlight = false
      }

      if (!cancelled) {
        timer = window.setTimeout(() => void poll(), POLL_MS)
      }
    }

    const onVisibilityChange = () => {
      if (typeof document !== 'undefined' && !document.hidden && !cancelled) {
        if (timer !== null) {
          window.clearTimeout(timer)
          timer = null
        }

        void poll()
      }
    }

    if (typeof document !== 'undefined') {
      document.addEventListener('visibilitychange', onVisibilityChange)
    }

    void poll()

    return () => {
      cancelled = true

      if (typeof document !== 'undefined') {
        document.removeEventListener('visibilitychange', onVisibilityChange)
      }

      if (timer !== null) {
        window.clearTimeout(timer)
      }
    }
  }, [shown, connectionKey])

  return { hardware, stats }
}

interface RamReading {
  ramPercent: number | null
  ramTotal: number | null
  ramUsed: number | null
}

// RAM prefers the hardware route; when only the stats route answered, its
// psutil memory block stands in so the row never sits at "— / —".
function readRam(hardware: LocalHardware | null, stats: SystemStats | null): RamReading {
  const ramTotal = hardware?.ram_total_bytes || stats?.memory?.total || null
  const ramAvailable = hardware?.ram_total_bytes ? hardware.ram_available_bytes : stats?.memory?.available
  const ramUsed = ramTotal && ramAvailable != null ? ramTotal - ramAvailable : null
  const ramPercent = ramTotal && ramUsed != null ? Math.round((ramUsed / ramTotal) * 100) : null

  return { ramPercent, ramTotal, ramUsed }
}

interface HostReading extends RamReading {
  cpuPercent: number | null
  disk: SystemStats['disk'] | null
  gpuNameLine: string | null
  hardware: LocalHardware | null
  hasGpu: boolean
  subtitle: string | null
  vramPercent: number | null
}

function readHost(hardware: LocalHardware | null, stats: SystemStats | null): HostReading {
  const hasGpu = Boolean(hardware?.gpu_name)

  const vramPercent =
    hardware?.vram_used_bytes != null && hardware.vram_total_bytes
      ? Math.round((hardware.vram_used_bytes / hardware.vram_total_bytes) * 100)
      : null

  // Header subtitle: the backend's hostname — which machine these numbers are
  // from is the point once the connection is remote (a remote GPU box most of
  // all). The GPU's name stands in only when the backend has no stats route;
  // otherwise it gets its own line above the GPU rows.
  const subtitle = stats?.hostname || hardware?.gpu_name || null

  return {
    ...readRam(hardware, stats),
    cpuPercent: stats?.cpu_percent != null ? Math.round(stats.cpu_percent) : null,
    disk: stats?.disk?.total ? stats.disk : null,
    gpuNameLine: hasGpu && subtitle !== hardware?.gpu_name ? (hardware?.gpu_name ?? null) : null,
    hardware,
    hasGpu,
    subtitle,
    vramPercent
  }
}

// Compact bar label: the numbers a glance wants. "GPU 34% · 18G/32G" with a
// GPU; "CPU 12% · RAM 41G/256G" without (either part dropped when the
// backend can't report it).
function barLabel(host: HostReading, loading: string): string {
  const { hardware } = host

  if (host.hasGpu) {
    const vram =
      hardware?.vram_used_bytes != null ? ` · ${gb(hardware.vram_used_bytes)}/${gb(hardware.vram_total_bytes)}` : ''

    return `GPU ${hardware?.gpu_util_percent ?? 0}%${vram}`
  }

  const cpuLabel = host.cpuPercent != null ? `CPU ${host.cpuPercent}%` : null
  const ramLabel = host.ramTotal ? `RAM ${gb(host.ramUsed)}/${gb(host.ramTotal)}` : null

  return [cpuLabel, ramLabel].filter(Boolean).join(' · ') || loading
}

function SystemResourcesPanel({
  copy,
  host
}: {
  copy: Translations['shell']['statusbar']['systemResources']
  host: HostReading
}) {
  const { cpuPercent, disk, gpuNameLine, hardware, hasGpu, subtitle } = host

  return (
    <div className="grid grid-cols-[minmax(0,1fr)] gap-3 p-3 text-[0.75rem]" data-slot="system-resources-panel">
      {/* min-w-0 everywhere a flex/grid child must shrink: grid items
          default min-width:auto, so a long GPU name's nowrap min-content
          props the track open past the w-64 box and overflow-x:hidden
          shears off every right-aligned value. With the track clamped,
          `truncate` can finally act. */}
      <div className="flex min-w-0 items-baseline justify-between gap-2">
        <p className="shrink-0 font-medium text-foreground">{copy.title}</p>

        {subtitle && <span className="min-w-0 truncate text-[0.6875rem] text-muted-foreground">{subtitle}</span>}
      </div>

      {cpuPercent != null && <MeterRow label={copy.cpu} percent={cpuPercent} value={`${cpuPercent}%`} />}

      {gpuNameLine && <p className="min-w-0 truncate text-[0.6875rem] text-muted-foreground">{gpuNameLine}</p>}

      {hasGpu && (
        <MeterRow
          label={copy.gpuUtilization}
          percent={hardware?.gpu_util_percent ?? null}
          value={`${hardware?.gpu_util_percent ?? 0}%`}
        />
      )}

      {hasGpu && (
        <MeterRow
          label={copy.gpuMemory}
          percent={host.vramPercent}
          value={`${gbLong(hardware?.vram_used_bytes)} / ${gbLong(hardware?.vram_total_bytes)}`}
        />
      )}

      <MeterRow
        label={copy.ram}
        percent={host.ramPercent}
        value={`${gbLong(host.ramUsed)} / ${gbLong(host.ramTotal)}`}
      />

      {disk && (
        <MeterRow
          label={copy.disk}
          percent={Math.round(disk.percent)}
          value={`${gbLong(disk.used)} / ${gbLong(disk.total)}`}
        />
      )}

      {hardware?.uma && <p className="text-[0.6875rem] text-muted-foreground">{copy.unifiedNote}</p>}
    </div>
  )
}

export function useSystemResourcesStatusbarItem(): StatusbarItem {
  const { t } = useI18n()
  const copy = t.shell.statusbar.systemResources
  const hiddenIds = useStore($statusbarHiddenIds)
  // Behind the --local launch flag: without it the item is absent from the
  // bar AND from the customize menu (no toggleLabel), and never polls.
  const enabled = $localModelsEnabled.get()
  const shown = enabled && !hiddenIds.includes('system-resources')
  // A connection switch restarts the poll: both routes are re-read at once,
  // so the header never names the old machine over the new one's numbers.
  // Keyed on id AND baseUrl: local and unregistered v1/env remotes carry no
  // connectionId, so the id alone would miss a switch between two of them.
  // ($connection rather than store/connections' $activeConnectionId, which
  // drags the whole source-switch module into this leaf.)
  const connection = useStore($connection)
  const connectionKey = connection ? `${connection.connectionId ?? ''}|${connection.baseUrl}` : null
  const { hardware, stats } = useHostResources(shown, connectionKey)
  const host = readHost(hardware, stats)

  return {
    detail: undefined,
    hidden: !enabled,
    icon: <Activity className="size-3" />,
    id: 'system-resources',
    label: barLabel(host, copy.loading),
    menuAlign: 'end',
    menuClassName: 'w-64 p-0',
    menuContent: <SystemResourcesPanel copy={copy} host={host} />,
    toggleLabel: enabled ? copy.toggle : undefined,
    variant: 'menu'
  }
}

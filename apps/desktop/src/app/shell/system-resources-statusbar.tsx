import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import type { StatusbarItem } from '@/app/shell/statusbar-controls'
import { getLocalHardware } from '@/hermes'
import { useI18n } from '@/i18n'
import { Activity } from '@/lib/icons'
import { $localModelsEnabled } from '@/store/local-models-flag'
import { $statusbarHiddenIds } from '@/store/statusbar-prefs'
import type { LocalHardware } from '@/types/hermes'

// Live host-resource readout for the bottom bar: GPU utilization + VRAM +
// RAM, fed by /api/local-models/hardware. Hidden by default (an item most
// users don't watch); the poll runs ONLY while the item is shown, so the
// hidden default costs nothing. 5s cadence — resource numbers, not a
// heartbeat.
const POLL_MS = 5_000

function gb(bytes: number | null | undefined): string {
  return bytes != null ? `${(bytes / 2 ** 30).toFixed(1)}` : '—'
}

function gbLong(bytes: number | null | undefined): string {
  return bytes != null ? `${gb(bytes)} GiB` : '—'
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
            style={{ width: `${Math.max(0, Math.min(100, percent))}%` }}
          />
        </div>
      )}
    </div>
  )
}

function physicalGpus(hardware: LocalHardware | null) {
  // An explicit empty inventory is authoritative; only older backends use
  // the first-device compatibility fields.
  return (
    hardware?.gpus ??
    (hardware?.gpu_name
      ? [
          {
            index: 0,
            name: hardware.gpu_name,
            util_percent: hardware.gpu_util_percent,
            vram_used_bytes: hardware.vram_used_bytes,
            vram_total_bytes: hardware.vram_total_bytes
          }
        ]
      : [])
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
  const [hardware, setHardware] = useState<LocalHardware | null>(null)

  useEffect(() => {
    if (!shown) {
      return
    }

    let cancelled = false
    let timer: number | null = null

    const poll = async () => {
      // Hidden window = nobody watching; resume on visibilitychange instead of
      // burning a backend spawn per 5s tick in the background (#120262).
      if (typeof document !== 'undefined' && document.hidden) {
        return
      }

      try {
        const next = await getLocalHardware()

        if (!cancelled) {
          setHardware(next)
        }
      } catch {
        if (!cancelled) {
          setHardware(null)
        }
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
  }, [shown])

  const gpus = physicalGpus(hardware)

  const totalVram =
    gpus.length && gpus.every(gpu => gpu.vram_total_bytes != null)
      ? gpus.reduce((sum, gpu) => sum + gpu.vram_total_bytes!, 0)
      : null

  const usedVram =
    gpus.length && gpus.every(gpu => gpu.vram_used_bytes != null)
      ? gpus.reduce((sum, gpu) => sum + gpu.vram_used_bytes!, 0)
      : null

  const ramUsed = hardware ? hardware.ram_total_bytes - hardware.ram_available_bytes : null

  const ramPercent =
    hardware?.ram_total_bytes && ramUsed != null ? Math.round((ramUsed / hardware.ram_total_bytes) * 100) : null

  // Missing readings must not silently become zero or a partial aggregate.
  const meanUtil =
    gpus.length && gpus.every(gpu => gpu.util_percent != null)
      ? Math.round(gpus.reduce((sum, gpu) => sum + gpu.util_percent!, 0) / gpus.length)
      : null

  const aggregateTitle =
    'GPU utilization: arithmetic mean across all GPUs. VRAM: sum of physical used/total memory, not pooled inference capacity. Missing readings are shown as —.'

  // Keep the chip aggregate-only; individual devices belong in the popover.
  const label = hardware
    ? gpus.length
      ? `Σ GPU ${meanUtil == null ? '—' : `${meanUtil}%`} · VRAM ${gb(usedVram)}/${gb(totalVram)} GiB`
      : `RAM ${gb(ramUsed)}/${gb(hardware.ram_total_bytes)} GiB`
    : copy.loading

  return {
    detail: undefined,
    hidden: !enabled,
    icon: <Activity className="size-3" />,
    id: 'system-resources',
    label,
    title: gpus.length ? aggregateTitle : copy.title,
    menuAlign: 'end',
    menuClassName: 'w-64 p-0',
    menuContent: (
      <div className="grid grid-cols-[minmax(0,1fr)] gap-3 p-3 text-[0.75rem]" data-slot="system-resources-panel">
        {/* min-w-0 everywhere a flex/grid child must shrink: grid items
            default min-width:auto, so a long GPU name's nowrap min-content
            props the track open past the w-64 box and overflow-x:hidden
            shears off every right-aligned value. With the track clamped,
            `truncate` can finally act. */}
        <div className="flex min-w-0 items-baseline justify-between gap-2">
          <p className="shrink-0 font-medium text-foreground">{copy.title}</p>
        </div>

        {gpus.map(gpu => (
          <section
            aria-label={`GPU ${gpu.index} · ${gpu.name}`}
            className="grid min-w-0 gap-2"
            key={gpu.index}
            style={{
              border: '1px solid var(--ui-border)',
              borderRadius: 6,
              padding: 10,
              background: 'var(--ui-bg-tertiary)'
            }}
          >
            <h3 className="min-w-0 font-medium text-foreground" style={{ overflowWrap: 'anywhere' }} title={gpu.name}>
              GPU {gpu.index} · {gpu.name}
            </h3>
            <MeterRow
              label={copy.gpuUtilization}
              percent={gpu.util_percent}
              value={gpu.util_percent == null ? '—' : `${gpu.util_percent}%`}
            />
            <MeterRow
              label={copy.gpuMemory}
              percent={
                gpu.vram_used_bytes != null && gpu.vram_total_bytes
                  ? (gpu.vram_used_bytes / gpu.vram_total_bytes) * 100
                  : null
              }
              value={`${gbLong(gpu.vram_used_bytes)} / ${gbLong(gpu.vram_total_bytes)}`}
            />
          </section>
        ))}
        {gpus.length > 1 && (
          <section aria-label="GPU aggregate" className="grid min-w-0 gap-2" title={aggregateTitle}>
            <h3 className="font-medium text-foreground">Σ All GPUs</h3>
            <MeterRow label="Mean utilization" percent={meanUtil} value={meanUtil == null ? '—' : `${meanUtil}%`} />
            <MeterRow label="Total VRAM" percent={null} value={`${gbLong(usedVram)} / ${gbLong(totalVram)}`} />
          </section>
        )}

        <MeterRow
          label={copy.ram}
          percent={ramPercent}
          value={`${gbLong(ramUsed)} / ${gbLong(hardware?.ram_total_bytes)}`}
        />

        {hardware?.uma && <p className="text-[0.6875rem] text-muted-foreground">{copy.unifiedNote}</p>}
      </div>
    ),
    toggleLabel: enabled ? copy.toggle : undefined,
    variant: 'menu'
  }
}

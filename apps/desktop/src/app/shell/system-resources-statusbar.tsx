import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import type { StatusbarItem } from '@/app/shell/statusbar-controls'
import { getLocalHardware } from '@/hermes'
import { useI18n } from '@/i18n'
import { Activity } from '@/lib/icons'
import { $localModelsEnabled } from '@/store/local-models-flag'
import { $statusbarHiddenIds } from '@/store/statusbar-prefs'
import type { LocalHardware, LocalHardwareGpu } from '@/types/hermes'

// Live host-resource readout for the bottom bar: GPU utilization + VRAM +
// RAM, fed by /api/local-models/hardware. Hidden by default (an item most
// users don't watch); the poll runs ONLY while the item is shown, so the
// hidden default costs nothing. 5s cadence — resource numbers, not a
// heartbeat.
const POLL_MS = 5_000

function gb(bytes: number | null | undefined): string {
  return bytes ? `${(bytes / (1 << 30)).toFixed(0)}G` : '—'
}

function gbLong(bytes: number | null | undefined): string {
  return bytes ? `${(bytes / (1 << 30)).toFixed(1)} GB` : '—'
}

function vramPercent(used: number | null | undefined, total: number | null | undefined): number | null {
  return used != null && total ? Math.round((used / total) * 100) : null
}

function gpuReadings(hardware: LocalHardware | null): LocalHardwareGpu[] {
  if (hardware?.gpus && hardware.gpus.length > 0) {
    return hardware.gpus
  }

  if (!hardware?.gpu_name) {
    return []
  }

  // Older backends report only the first adapter on the top-level fields.
  return [
    {
      index: 0,
      name: hardware.gpu_name,
      util_percent: hardware.gpu_util_percent,
      vram_used_bytes: hardware.vram_used_bytes,
      vram_total_bytes: hardware.vram_total_bytes,
      vram_free_bytes: null
    }
  ]
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

  const gpus = gpuReadings(hardware)
  const multi = (hardware?.gpus?.length ?? 0) > 1
  const hasGpu = gpus.length > 0

  const singleVramPercent = vramPercent(hardware?.vram_used_bytes, hardware?.vram_total_bytes)

  const ramUsed = hardware ? hardware.ram_total_bytes - hardware.ram_available_bytes : null

  const ramPercent =
    hardware?.ram_total_bytes && ramUsed != null ? Math.round((ramUsed / hardware.ram_total_bytes) * 100) : null

  // Compact bar label: the numbers a local-inference user glances at.
  // One card keeps "GPU 34% · 18G/32G". Several cards name each index so
  // the second adapter is not invisible in the chip either.
  const label = !hardware
    ? copy.loading
    : multi
      ? gpus.map(gpu => `GPU${gpu.index} ${gpu.util_percent ?? 0}%`).join(' · ')
      : hasGpu
        ? `GPU ${hardware.gpu_util_percent ?? 0}%${
            hardware.vram_used_bytes != null
              ? ` · ${gb(hardware.vram_used_bytes)}/${gb(hardware.vram_total_bytes)}`
              : ''
          }`
        : `RAM ${gb(ramUsed)}/${gb(hardware.ram_total_bytes)}`

  return {
    detail: undefined,
    hidden: !enabled,
    icon: <Activity className="size-3" />,
    id: 'system-resources',
    label,
    menuAlign: 'end',
    menuClassName: multi ? 'w-80 p-0' : 'w-64 p-0',
    menuContent: (
      <div className="grid grid-cols-[minmax(0,1fr)] gap-3 p-3 text-[0.75rem]" data-slot="system-resources-panel">
        {/* min-w-0 everywhere a flex/grid child must shrink: grid items
            default min-width:auto, so a long GPU name's nowrap min-content
            props the track open past the menu box and overflow-x:hidden
            shears off every right-aligned value. With the track clamped,
            `truncate` can finally act. */}
        <div className="flex min-w-0 items-baseline justify-between gap-2">
          <p className="shrink-0 font-medium text-foreground">{copy.title}</p>

          {!multi && hardware?.gpu_name && (
            <span className="min-w-0 truncate text-[0.6875rem] text-muted-foreground">{hardware.gpu_name}</span>
          )}
        </div>

        {multi
          ? gpus.map(gpu => (
              <div className="grid gap-3" data-gpu-index={gpu.index} data-slot="system-resources-gpu" key={gpu.index}>
                <div className="flex min-w-0 items-baseline justify-between gap-2">
                  <p className="shrink-0 font-medium text-foreground">{`GPU ${gpu.index}`}</p>

                  <span className="min-w-0 truncate text-[0.6875rem] text-muted-foreground">{gpu.name}</span>
                </div>

                <MeterRow label={copy.gpuUtilization} percent={gpu.util_percent} value={`${gpu.util_percent ?? 0}%`} />

                <MeterRow
                  label={copy.gpuMemory}
                  percent={vramPercent(gpu.vram_used_bytes, gpu.vram_total_bytes)}
                  value={`${gbLong(gpu.vram_used_bytes)} / ${gbLong(gpu.vram_total_bytes)}`}
                />
              </div>
            ))
          : hasGpu && (
              <>
                <MeterRow
                  label={copy.gpuUtilization}
                  percent={hardware?.gpu_util_percent ?? null}
                  value={`${hardware?.gpu_util_percent ?? 0}%`}
                />

                <MeterRow
                  label={copy.gpuMemory}
                  percent={singleVramPercent}
                  value={`${gbLong(hardware?.vram_used_bytes)} / ${gbLong(hardware?.vram_total_bytes)}`}
                />
              </>
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

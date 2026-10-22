import { type ReactElement, useState } from 'react'

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Cpu, Monitor, Package, Zap } from '@/lib/icons'
import type { LocalHardware } from '@/types/hermes'

import { gbLabel } from './local-model-download-progress'
import { selectGpu, useLocalModelsActionScope } from './local-models-actions'
import { Pill, SettingsSection } from './primitives'

export interface LocalModelsHardwareSectionProps {
  hardware: LocalHardware | undefined
}

export function LocalModelsHardwareSection({ hardware }: LocalModelsHardwareSectionProps): ReactElement {
  const scope = useLocalModelsActionScope()
  const { copy } = scope
  const [pendingGpu, setPendingGpu] = useState<string | null>(null)

  const gpus = hardware?.gpus ?? []
  const multiGpu = gpus.length > 1
  const configured = hardware?.gpu ?? 'auto'
  const selectedGpu = gpus.find(gpu => gpu.id === configured || String(gpu.index) === configured)
  const selection = selectedGpu ? (selectedGpu.id ?? String(selectedGpu.index)) : configured

  const automaticGpu = gpus.reduce<(typeof gpus)[number] | undefined>(
    (largest, gpu) => (!largest || gpu.total_bytes > largest.total_bytes ? gpu : largest),
    undefined
  )

  async function handleGpuChange(value: string): Promise<void> {
    setPendingGpu(value)

    try {
      await selectGpu(scope, value)
    } finally {
      setPendingGpu(null)
    }
  }

  return (
    <SettingsSection icon={Monitor} title={copy.hardwareTitle}>
      {hardware ? (
        <div className="flex flex-wrap items-center gap-x-5 gap-y-1 py-1 text-[length:var(--conversation-caption-font-size)] text-muted-foreground">
          {hardware.gpu_name && (
            <span className="inline-flex items-center gap-1.5">
              <Zap className="size-3.5" />
              {hardware.gpu_name}
            </span>
          )}

          <span className="inline-flex items-center gap-1.5">
            <Cpu className="size-3.5" />
            {copy.vram(gbLabel(hardware.vram_total_bytes))}
          </span>

          <span className="inline-flex items-center gap-1.5">
            <Package className="size-3.5" />
            {copy.ram(gbLabel(hardware.ram_total_bytes))}
          </span>

          {hardware.uma && <Pill>{copy.unifiedMemory}</Pill>}
        </div>
      ) : (
        <p className="py-1 text-[length:var(--conversation-caption-font-size)] text-muted-foreground">
          {copy.hardwareLoading}
        </p>
      )}

      {multiGpu && (
        <div
          className="flex flex-wrap items-center gap-x-3 gap-y-1 py-1 text-[length:var(--conversation-caption-font-size)] text-muted-foreground"
          data-testid="local-models-gpu-picker"
        >
          <span className="inline-flex items-center gap-1.5">
            <Zap className="size-3.5" />
            {copy.gpuLabel}
          </span>
          <Select
            disabled={pendingGpu !== null}
            onValueChange={(value: string): void => {
              void handleGpuChange(value)
            }}
            value={pendingGpu ?? selection}
          >
            <SelectTrigger aria-label={copy.gpuLabel} className="w-auto min-w-56" size="sm">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="auto">{copy.gpuAuto(automaticGpu?.name ?? '')}</SelectItem>
              {gpus.map(gpu => (
                <SelectItem key={gpu.id ?? String(gpu.index)} value={gpu.id ?? String(gpu.index)}>
                  {copy.gpuEntry(gpu.index, gpu.name, gbLabel(gpu.total_bytes))}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}
      {hardware?.gpu_error && (
        <p className="py-1 text-[length:var(--conversation-caption-font-size)] text-destructive" role="alert">
          {hardware.gpu_error}
        </p>
      )}
    </SettingsSection>
  )
}

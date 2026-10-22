import { useQueryClient } from '@tanstack/react-query'
import { type ReactElement, useState } from 'react'

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { setLocalGpu } from '@/hermes'
import { useI18n } from '@/i18n'
import { Cpu, Monitor, Package, Zap } from '@/lib/icons'
import { refreshLocalModels } from '@/store/local-runtime-jobs'
import { notifyError } from '@/store/notifications'
import type { LocalHardware } from '@/types/hermes'

import { gbLabel } from './local-model-download-progress'
import { useScopedLocalModelsOwner } from './local-models-owner'
import { Pill, SettingsSection } from './primitives'

export interface LocalModelsHardwareSectionProps {
  hardware: LocalHardware | undefined
}

export function LocalModelsHardwareSection({ hardware }: LocalModelsHardwareSectionProps): ReactElement {
  const { t } = useI18n()
  const copy = t.settings.localModels
  const client = useQueryClient()
  const [gpuBusy, setGpuBusy] = useState<boolean>(false)
  const owner = useScopedLocalModelsOwner()

  const gpus = hardware?.gpus ?? []
  const multiGpu = gpus.length > 1
  const selection = hardware?.gpu ?? 'auto'

  async function handleGpuChange(value: string): Promise<void> {
    setGpuBusy(true)

    try {
      await setLocalGpu(value)
      refreshLocalModels(owner, client)
    } catch (err) {
      notifyError(err, copy.gpuSelectFailed)
    } finally {
      setGpuBusy(false)
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
            disabled={gpuBusy}
            onValueChange={(value: string): void => {
              void handleGpuChange(value)
            }}
            value={selection}
          >
            <SelectTrigger className="h-7 w-auto min-w-56 text-[length:var(--conversation-caption-font-size)]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="auto">
                {copy.gpuAuto(
                  gpus.find(gpu => gpu.index === hardware?.gpu_selected)?.name ?? gpus[0]?.name ?? ''
                )}
              </SelectItem>
              {gpus.map(gpu => (
                <SelectItem key={gpu.index} value={String(gpu.index)}>
                  {copy.gpuEntry(gpu.index, gpu.name, gbLabel(gpu.total_bytes))}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}
    </SettingsSection>
  )
}

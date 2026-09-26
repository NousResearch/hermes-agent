import type { ReactElement } from 'react'

import { useI18n } from '@/i18n'
import { Cpu, Monitor, Package, Zap } from '@/lib/icons'
import type { LocalHardware } from '@/types/hermes'

import { gbLabel } from './local-model-download-progress'
import { Pill, SettingsSection } from './primitives'

export interface LocalModelsHardwareSectionProps {
  hardware: LocalHardware | undefined
}

export function LocalModelsHardwareSection({ hardware }: LocalModelsHardwareSectionProps): ReactElement {
  const { t } = useI18n()
  const copy = t.settings.localModels

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
    </SettingsSection>
  )
}

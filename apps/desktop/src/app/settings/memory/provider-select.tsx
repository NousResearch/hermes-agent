import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectSeparator,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select'
import { useI18n } from '@/i18n'

import { CONTROL_TEXT } from '../constants'

import type { MemoryRowState } from './provider-row-state'

export const EXPLORE_MEMORY_PLUGINS = 'explore:'

/** Installed providers first, then featured catalog installs, then the marketplace. */
export function MemoryProviderSelect({
  label,
  onValueChange,
  row
}: {
  label: (name: string) => string
  onValueChange: (value: string) => void
  row: MemoryRowState
}) {
  const { t } = useI18n()
  const c = t.memoryDiscovery

  return (
    <Select onValueChange={onValueChange} value={row.selected}>
      <SelectTrigger aria-label={t.settings.fieldLabels['memory.provider']} className={CONTROL_TEXT}>
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        <SelectGroup>
          <SelectLabel>{c.installed}</SelectLabel>
          <SelectItem value="builtin">{c.builtin}</SelectItem>
          {row.installed.map(provider => (
            <SelectItem key={provider.name} value={provider.name}>
              {label(provider.name)}
            </SelectItem>
          ))}
          {row.missing.map(provider => (
            <SelectItem key={provider.name} value={provider.name}>
              {`${label(provider.name)} (${c.missing})`}
            </SelectItem>
          ))}
        </SelectGroup>
        {row.offered.length > 0 && (
          <SelectGroup>
            <SelectSeparator />
            <SelectLabel>{c.availableToInstall}</SelectLabel>
            {row.offered.map(provider => (
              <SelectItem key={`catalog:${provider.name}`} value={provider.name}>
                {label(provider.name)}
              </SelectItem>
            ))}
          </SelectGroup>
        )}
        <SelectSeparator />
        <SelectItem value={EXPLORE_MEMORY_PLUGINS}>{c.exploreAll}</SelectItem>
      </SelectContent>
    </Select>
  )
}

import { compactNumber } from '@hermes/shared'
import { useNavigate } from 'react-router'

import { Button } from '@/components/ui/button'
import { Switch } from '@/components/ui/switch'
import { type ProfileScope, profileScopeKey } from '@/hermes'
import { useI18n } from '@/i18n'
import type { ToolsetInfo } from '@/types/hermes'

import { SETTINGS_ROUTE } from '../../routes'
import { BrowserRealProfilePanel } from '../../settings/browser-real-profile-panel'
import { ComputerUsePanel } from '../../settings/computer-use-panel'
import { asText, toolNames, toolsetDisplayLabel } from '../../settings/helpers'
import { TerminalBackendPanel } from '../../settings/terminal-backend-panel'
import { ToolsetConfigPanel } from '../../settings/toolset-config-panel'

export function ToolsetDetail({
  toolset,
  toolCalls,
  onConfiguredChange,
  onToggle,
  enabled,
  profile
}: {
  toolset: ToolsetInfo
  toolCalls: Record<string, number>
  onConfiguredChange: () => void
  onToggle?: (checked: boolean) => void
  enabled?: boolean
  profile?: ProfileScope
}) {
  const { t } = useI18n()
  const navigate = useNavigate()
  const tools = toolNames(toolset)
  const label = toolsetDisplayLabel(toolset)
  const isEnabled = enabled ?? toolset.enabled

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4 p-6">
      <div className="flex items-start justify-between gap-3">
        <h2 className="min-w-0 truncate text-[28px] font-extrabold leading-none text-[#0f172a] dark:text-foreground">
          {label}
        </h2>
        {onToggle ? (
          <Switch
            aria-label={t.skills.toggleToolset(label, !isEnabled)}
            checked={isEnabled}
            className="mt-1.5 h-[22px] w-10 shrink-0 data-[state=checked]:border-transparent data-[state=checked]:bg-[#7c3aed] [&_[data-slot=switch-thumb]]:size-[18px] data-[state=checked]:[&_[data-slot=switch-thumb]]:translate-x-[18px]"
            onCheckedChange={onToggle}
            size="default"
          />
        ) : null}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <span
          className={
            isEnabled
              ? 'rounded-lg border border-[#9db7f0] bg-[#e3ecfd] px-2.5 py-1 text-[12px] font-semibold text-[#2f5bea]'
              : 'rounded-lg border border-[#e2e8f0] bg-[#f1f5f9] px-2.5 py-1 text-[12px] font-medium text-[#64748b]'
          }
        >
          {isEnabled ? t.skills.enabled : t.skills.disabled}
        </span>
        <span className="rounded-lg border border-[#e2e8f0] bg-[#f1f5f9] px-2.5 py-1 text-[12px] font-medium text-[#334155]">
          {t.skills.toolsCount(tools.length)}
        </span>
        {!toolset.configured ? (
          <span className="rounded-lg border border-amber-200 bg-amber-50 px-2.5 py-1 text-[12px] font-semibold text-amber-700">
            {t.skills.needsKeys}
          </span>
        ) : null}
      </div>

      <p className="whitespace-pre-wrap text-[14px] leading-[22px] text-[#475569]">
        {asText(toolset.description) || t.skills.noDescription}
      </p>

      {tools.length > 0 ? (
        <div className="space-y-[11px]">
          <h3 className="text-[11px] font-semibold uppercase tracking-[0.6px] text-[#64748b]">{t.skills.functions}</h3>
          <div className="flex flex-wrap gap-1.5">
            {tools.map(name => (
              <span
                className="inline-flex items-center rounded-lg border border-[#e2e8f0] bg-[#f1f5f9] px-2.5 py-[5px] text-[12px] font-medium text-[#334155]"
                key={name}
              >
                {name}
                {(toolCalls[name] ?? 0) > 0 ? (
                  <span className="ml-1 text-[#94a3b8]">×{compactNumber(toolCalls[name])}</span>
                ) : null}
              </span>
            ))}
          </div>
        </div>
      ) : null}

      <div className="mt-auto space-y-3 border-t border-[#e2e8f0] pt-4 dark:border-(--ui-stroke-tertiary)">
        {toolset.name === 'vision' && (
          <div className="grid gap-1.5">
            <p className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
              {t.skills.visionModelHint}
            </p>
            <div>
              <Button
                onClick={() => navigate(`${SETTINGS_ROUTE}?tab=config:model&aux=vision`)}
                size="xs"
                variant="textStrong"
              >
                {t.skills.visionModelLink}
              </Button>
            </div>
          </div>
        )}
        {toolset.name === 'computer_use' && <ComputerUsePanel onConfiguredChange={onConfiguredChange} />}
        {toolset.name === 'browser' && <BrowserRealProfilePanel profile={profile} />}
        {toolset.name === 'terminal' && (
          <TerminalBackendPanel
            key={profileScopeKey(profile)}
            onConfiguredChange={onConfiguredChange}
            profile={profile}
          />
        )}
        <ToolsetConfigPanel
          key={`${toolset.name}:${profileScopeKey(profile)}`}
          onConfiguredChange={onConfiguredChange}
          profile={profile}
          toolset={toolset.name}
        />
      </div>
    </div>
  )
}

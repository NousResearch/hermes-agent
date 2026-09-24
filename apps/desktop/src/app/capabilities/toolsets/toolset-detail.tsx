import { compactNumber } from '@hermes/shared'
import { useNavigate } from 'react-router'

import { Button } from '@/components/ui/button'
import { type ProfileScope, profileScopeKey } from '@/hermes'
import { useI18n } from '@/i18n'
import type { ToolsetInfo } from '@/types/hermes'

import { ToolChip } from '../../master-detail'
import { PanelPill } from '../../overlays/panel'
import { SETTINGS_ROUTE } from '../../routes'
import { BrowserRealProfilePanel } from '../../settings/browser-real-profile-panel'
import { ComputerUsePanel } from '../../settings/computer-use-panel'
import { asText, toolNames, toolsetDisplayLabel } from '../../settings/helpers'
import { TerminalBackendPanel } from '../../settings/terminal-backend-panel'
import { ToolsetConfigPanel } from '../../settings/toolset-config-panel'
import { DetailHeader } from '../primitives'

export function ToolsetDetail({
  toolset,
  toolCalls,
  onConfiguredChange,
  profile
}: {
  toolset: ToolsetInfo
  toolCalls: Record<string, number>
  onConfiguredChange: () => void
  profile?: ProfileScope
}) {
  const { t } = useI18n()
  const navigate = useNavigate()
  const tools = toolNames(toolset)
  const label = toolsetDisplayLabel(toolset)

  return (
    <>
      {/* "Configured" as a resting state is noise — only the warn state earns a pill. */}
      <DetailHeader
        description={asText(toolset.description) || t.skills.noDescription}
        pills={!toolset.configured && <PanelPill tone="warn">{t.skills.needsKeys}</PanelPill>}
        title={label}
      />
      {tools.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {tools.map(name => (
            <ToolChip key={name}>
              {name}
              {(toolCalls[name] ?? 0) > 0 && (
                <span className="ml-1 text-(--ui-text-quaternary)">×{compactNumber(toolCalls[name])}</span>
              )}
            </ToolChip>
          ))}
        </div>
      )}
      {toolset.name === 'vision' && (
        // Vision has no provider matrix — model resolution runs through the
        // auxiliary model config. Point at the actual home (Settings → Models,
        // aux "vision" row) via an internal deep link instead of leaving the
        // detail pane empty.
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
      {/* Real-profile consent toggle ABOVE the backend/provider matrix — the
          config option users kept missing because its only GUI home was the
          generic Settings → Config editor. */}
      {toolset.name === 'browser' && <BrowserRealProfilePanel profile={profile} />}
      {toolset.name === 'terminal' && <TerminalBackendPanel onConfiguredChange={onConfiguredChange} />}
      <ToolsetConfigPanel
        key={`${toolset.name}:${profileScopeKey(profile)}`}
        onConfiguredChange={onConfiguredChange}
        profile={profile}
        toolset={toolset.name}
      />
    </>
  )
}

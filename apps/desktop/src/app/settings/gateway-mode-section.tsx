import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { Check, Cloud, Globe, HelpCircle, Monitor, Terminal } from '@/lib/icons'
import { selectableCardClass } from '@/lib/selectable-card'
import { cn } from '@/lib/utils'
import { usePortalShell } from '@/store/shell-variant'

import { SettingsGroup } from './settings-card'
import { SETTING_IDS, settingElementId } from './settings-manifest'
import { useSettingsPagesCopy } from './settings-pages-copy'

/** What the native app's own connection runs against: the Gateways page's connection form. */
export type GatewayMode = 'local' | 'remote' | 'cloud' | 'ssh'

interface ModeCardProps {
  active: boolean
  description: string
  disabled?: boolean
  hint?: string
  icon: typeof Monitor
  mode: GatewayMode
  onSelect: () => void
  title: string
}

// Classic reads local-first; the redesign leads with the hosted option (18563:12326).
const CLASSIC_MODE_ORDER: readonly GatewayMode[] = ['local', 'cloud', 'remote', 'ssh']
const REDESIGN_MODE_ORDER: readonly GatewayMode[] = ['cloud', 'remote', 'ssh', 'local']

function ModeHint({ hint }: { hint: string }) {
  return (
    <Tip label={hint}>
      <span
        className="grid size-3.5 shrink-0 cursor-help place-items-center text-(--ui-text-tertiary) hover:text-(--ui-text-secondary)"
        onClick={event => event.stopPropagation()}
      >
        <HelpCircle className="size-3.5" />
      </span>
    </Tip>
  )
}

function ModeCard({ active, description, disabled, hint, icon: Icon, mode, onSelect, title }: ModeCardProps) {
  const redesign = usePortalShell()

  // Redesign (18563:12326): the mode's own artwork across the card's top, then title and description;
  // the chosen mode is ringed in the brand colour.
  if (redesign) {
    return (
      <button
        aria-pressed={active}
        className="gateway-mode-card"
        data-mode={mode}
        disabled={disabled}
        onClick={onSelect}
        type="button"
      >
        <span aria-hidden className="gateway-mode-card-art" />
        <span className="gateway-mode-card-title">
          {title}
          {hint ? <ModeHint hint={hint} /> : null}
        </span>
        <span className="gateway-mode-card-description">{description}</span>
      </button>
    )
  }

  return (
    <button
      className={cn(
        'flex h-full min-h-0 w-full flex-col p-3 text-left disabled:cursor-not-allowed disabled:opacity-50',
        selectableCardClass({ active, prominent: true })
      )}
      disabled={disabled}
      onClick={onSelect}
      type="button"
    >
      <div className="flex items-center gap-1.5">
        <Icon className="size-3.5 shrink-0 text-muted-foreground" />
        <span className="min-w-0 text-[length:var(--conversation-text-font-size)] font-medium">{title}</span>
        {hint ? <ModeHint hint={hint} /> : null}
        {active ? <Check className="ml-auto size-3.5 shrink-0 text-primary" /> : null}
      </div>
      <p className="mt-1.5 flex-1 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
        {description}
      </p>
    </button>
  )
}

interface GatewayModeSectionProps {
  mode: GatewayMode
  disabled: boolean
  onSelect: (mode: GatewayMode) => void
}

/** The four connection-mode cards: classic's caption + grid, or the redesign's titled card row. */
export function GatewayModeSection({ mode, disabled, onSelect }: GatewayModeSectionProps) {
  const { t } = useI18n()
  const g = t.settings.gateway
  const pages = useSettingsPagesCopy()
  const redesign = usePortalShell()

  const cards: Record<GatewayMode, Omit<ModeCardProps, 'active' | 'disabled' | 'mode' | 'onSelect'>> = {
    local: { description: g.localDesc, icon: Monitor, title: g.localTitle },
    cloud: { description: g.cloudDesc, icon: Cloud, title: g.cloudTitle },
    remote: { description: g.remoteDesc, hint: g.remoteAuthHint, icon: Globe, title: g.remoteTitle },
    ssh: { description: g.sshDesc, hint: g.sshTrustHint, icon: Terminal, title: g.sshTitle }
  }

  const render = (id: GatewayMode) => (
    <ModeCard
      {...cards[id]}
      active={mode === id}
      disabled={disabled}
      key={id}
      mode={id}
      onSelect={() => onSelect(id)}
    />
  )

  if (redesign) {
    return (
      <SettingsGroup id={settingElementId(SETTING_IDS.gateway.connectionMode)} title={pages.gateway.connectionMode}>
        <div className="gateway-mode-cards">{REDESIGN_MODE_ORDER.map(render)}</div>
      </SettingsGroup>
    )
  }

  return (
    <div className="mb-5 grid gap-2" id={settingElementId(SETTING_IDS.gateway.connectionMode)}>
      <div className="text-[length:var(--conversation-caption-font-size)] font-medium text-(--ui-text-secondary)">
        {g.modeTitle}
      </div>
      <div className="grid auto-rows-fr grid-cols-1 gap-2 sm:grid-cols-2 min-[72rem]:grid-cols-4">
        {CLASSIC_MODE_ORDER.map(render)}
      </div>
    </div>
  )
}

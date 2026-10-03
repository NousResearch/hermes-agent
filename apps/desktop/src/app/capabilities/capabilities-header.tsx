import { useI18n } from '@/i18n'

import { ScopeChip } from '../settings/profile-scope'

import type { CapabilityScope } from './scope-selector'

export type CapabilityHeaderMode = 'skills' | 'toolsets' | 'connectors' | 'plugins'

export function CapabilitiesHeader({ mode, scope }: { mode: CapabilityHeaderMode; scope: CapabilityScope }) {
  const { t } = useI18n()
  const title =
    mode === 'skills'
      ? t.skills.tabSkills
      : mode === 'toolsets'
        ? t.skills.tabToolsets
        : mode === 'connectors'
          ? t.connectorsPage.title
          : t.skills.tabPlugins
  // Reuse translated product copy instead of creating a second string source
  // in the renderer. These lines explain what to expect from each submenu;
  // missing locale overrides already fall back through the i18n registry.
  const intro =
    mode === 'skills'
      ? t.skills.hub.landingHint
      : mode === 'toolsets'
        ? t.skills.changesApplyNewSessions
        : mode === 'connectors'
          ? t.connectors.disclaimer
          : t.skills.plugins.pageBlurb
  const appliesTo = t.settings.profileScope.appliesTo

  return (
    <header className="shrink-0 border-b border-(--ui-stroke-secondary) px-4 py-3">
      <div className="flex items-center gap-2 text-xs text-(--ui-text-tertiary)" aria-label={t.sidebar.nav.capabilities}>
        <span>{t.sidebar.nav.capabilities}</span>
        <span aria-hidden>›</span>
        <span className="text-(--ui-text-secondary)">{title}</span>
      </div>

      <div className="mt-3">
        <h1 className="text-base font-semibold tracking-tight text-(--ui-text-primary)">{title}</h1>
        <p className="mt-1 max-w-3xl text-sm leading-5 text-(--ui-text-tertiary)">{intro}</p>
      </div>

      {scope.options.length > 1 ? (
        <div className="mt-4 grid gap-2">
          <div className="text-[length:var(--conversation-caption-font-size)] font-medium text-(--ui-text-secondary)">
            {appliesTo}
          </div>
          <div className="flex flex-wrap gap-1.5" aria-label={appliesTo} role="group">
            {scope.options.map(option => (
              <ScopeChip
                active={option.value === scope.value}
                key={option.key}
                label={option.label}
                onSelect={() => scope.onChange(option.value)}
              />
            ))}
          </div>
        </div>
      ) : null}
    </header>
  )
}

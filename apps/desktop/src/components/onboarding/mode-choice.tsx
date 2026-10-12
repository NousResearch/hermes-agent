import { useI18n } from '@/i18n'
import { selectableCardClass } from '@/lib/selectable-card'
import { applySimpleAgentDefaults } from '@/lib/simple-agent-defaults'
import { cn } from '@/lib/utils'
import { INTERFACE_MODES, type InterfaceMode, setInterfaceMode } from '@/store/interface-mode'
import { type DesktopOnboardingState, markFirstRunModeChosen } from '@/store/onboarding'
import type { OnboardingScope } from '@/store/onboarding-scope'

/** A TRUE first run asks Simple-or-Advanced once, at the idle picker, before
 *  any provider talk; an answered or ever-configured install, a manual open, a
 *  credential wall or a returning "choose later" never does. */
export function modeChoiceOwed(onboarding: DesktopOnboardingState, enabled: boolean): boolean {
  return (
    enabled &&
    onboarding.flow.status === 'idle' &&
    onboarding.configured === false &&
    !onboarding.freeTierReady &&
    !onboarding.manual &&
    !onboarding.requested &&
    !onboarding.firstRunSkipped &&
    !onboarding.modeChosen
  )
}

// The preset is written here and nowhere else: a first run has no session yet,
// so it cannot invalidate a prompt cache, and a later mode toggle only changes
// what is shown — never the agent's config.
function chooseFirstRunMode(mode: InterfaceMode, scope: OnboardingScope) {
  setInterfaceMode(mode)
  markFirstRunModeChosen()

  if (mode === 'simple') {
    void applySimpleAgentDefaults(scope)
  }
}

/** First-run only: one click between Simple and Advanced. The overlay's Header carries the question. */
export function ModeChoice({ scope }: { scope: OnboardingScope }) {
  const { t } = useI18n()

  const descriptions: Record<InterfaceMode, string> = {
    simple: t.onboarding.modeChoiceSimpleDesc,
    advanced: t.onboarding.modeChoiceAdvancedDesc
  }

  return (
    <div className="grid gap-4">
      <div className="grid gap-2 sm:grid-cols-2">
        {INTERFACE_MODES.map(mode => (
          <button
            className={cn(selectableCardClass({ prominent: true }), 'grid content-start gap-1 p-4 text-left')}
            key={mode}
            onClick={() => chooseFirstRunMode(mode, scope)}
            type="button"
          >
            <span className="text-sm font-semibold">{t.interfaceMode[mode].label}</span>
            <span className="text-xs leading-5 text-muted-foreground">{descriptions[mode]}</span>
          </button>
        ))}
      </div>
      <p className="text-xs text-(--ui-text-quaternary)">{t.interfaceMode.hint}</p>
    </div>
  )
}

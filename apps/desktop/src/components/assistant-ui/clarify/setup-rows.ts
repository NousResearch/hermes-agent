import type { ConnectorRow, SetupChooseKind, SetupChooseOption } from '@hermes/shared'
import { useMemo } from 'react'

import { MODE_OPTIONS } from '@/app/settings/constants'
import { $chatLayoutPicked, assembleChatOnboarding, snapshotChatLayout } from '@/components/onboarding-chat/assembly'
import { accentsFor, LAYOUTS, NOUS_ACCENT, orderConnectorPicks } from '@/components/onboarding-chat/options'
import type { LayoutNode } from '@/components/pane-shell/tree/model'
import { registry } from '@/contrib/registry'
import { useI18n } from '@/i18n'
import { connectorTitle } from '@/lib/connector-tools'
import type { SetupChooseSpec } from '@/store/clarify'
import { useConnectorCatalog } from '@/store/connector-catalog'
import { $onboardingAnswers, setOnboardingAnswers } from '@/store/onboarding-answers'
import { type OnboardingPlugin, useOnboardingPluginList } from '@/store/onboarding-plugins'
import { useTheme } from '@/themes'
import { $accentOverride, setAccentOverride } from '@/themes/accent-override'
import { normalizeHex } from '@/themes/color'
import type { ThemeMode } from '@/themes/context'
import { nousTheme } from '@/themes/presets'
import { retintTheme } from '@/themes/retint'
import { installUserTheme } from '@/themes/user-themes'

export type SetupRow = SetupChooseOption

type Translations = ReturnType<typeof useI18n>['t']

interface RowSources {
  connectors: ConnectorRow[] | null
  dark: boolean
  plugins: OnboardingPlugin[] | null
  t: Translations
}

const modeLabel = (id: string, t: Translations): string =>
  MODE_OPTIONS.some(option => option.id === id) ? t.settings.modeOptions[id as ThemeMode].label : id

// The theme card offers light and dark only, whoever supplies its options; System stays in Settings.
const THEME_TILES: readonly string[] = MODE_OPTIONS.filter(({ id }) => id !== 'system').map(({ id }) => id)

const APP_ROWS: Record<SetupChooseKind, (sources: RowSources) => null | SetupRow[]> = {
  accent: ({ dark }) => accentsFor(dark).map(({ hex, name }) => ({ id: hex, label: name })),
  connectors: ({ connectors }) =>
    connectors &&
    orderConnectorPicks(connectors).map(row => ({ id: row.connector, label: connectorTitle(row.connector) })),
  // The backend fills the fork, machine_use and tour rows, so they are known only once the request arrives.
  fork: () => null,
  layout: () => LAYOUTS.map(layout => ({ detail: layout.description, id: layout.id, label: layout.name })),
  machine_use: () => null,
  plugins: ({ plugins }) =>
    plugins &&
    plugins.map(plugin => ({
      detail: plugin.app_state === 'missing_app' ? plugin.sentence : undefined,
      id: plugin.name,
      label: plugin.title
    })),
  question: () => [],
  theme: ({ t }) => THEME_TILES.map(id => ({ id, label: modeLabel(id, t) })),
  tour: () => null
}

const APP_LABELS: Record<SetupChooseKind, (id: string, sources: Pick<RowSources, 'plugins' | 't'>) => string> = {
  accent: id => [...accentsFor(false), ...accentsFor(true)].find(swatch => swatch.hex === normalizeHex(id))?.name ?? id,
  connectors: id => connectorTitle(id),
  fork: id => id,
  layout: id => LAYOUTS.find(layout => layout.id === id)?.name ?? id,
  machine_use: id => id,
  plugins: (id, { plugins }) => plugins?.find(plugin => plugin.name === id)?.title ?? id,
  question: id => id,
  theme: (id, { t }) => modeLabel(id, t),
  tour: id => id
}

interface LiveLook {
  apply: (id: string, setMode: (mode: ThemeMode) => void) => void
  commit?: (id: string, setTheme: (name: string) => void) => void
  snapshot: (mode: ThemeMode, setMode: (mode: ThemeMode) => void) => () => void
}

export const LIVE_LOOK: Partial<Record<SetupChooseKind, LiveLook>> = {
  accent: {
    commit: (id, setTheme) => {
      const hex = normalizeHex(id)

      if (!hex) {
        return
      }

      // Save the confirmed palette through normal theme persistence. The preview
      // override is temporary and must not tint later theme or profile selections.
      const theme =
        hex === NOUS_ACCENT
          ? nousTheme
          : installUserTheme({
              ...retintTheme(nousTheme, hex),
              name: `nous-accent-${hex.slice(1)}`,
              label: `${nousTheme.label} · ${hex}`
            })

      setTheme(theme.name)
      setOnboardingAnswers({ accent: hex === NOUS_ACCENT ? null : hex })
      setAccentOverride(null)
    },
    apply: id => {
      const hex = normalizeHex(id)

      if (hex) {
        const accent = hex === NOUS_ACCENT ? null : hex

        setOnboardingAnswers({ accent })
        setAccentOverride(accent)
      }
    },
    snapshot: () => {
      const { accent } = $onboardingAnswers.get()
      const override = $accentOverride.get()

      return () => {
        setOnboardingAnswers({ accent })
        setAccentOverride(override)
      }
    }
  },
  layout: {
    apply: id => {
      const preset = registry.getArea('layouts').find(contribution => contribution.id === id)

      if (!preset?.data) {
        return
      }

      $chatLayoutPicked.set(true)
      assembleChatOnboarding(preset.id, preset.data as LayoutNode, LAYOUTS.find(layout => layout.id === id)?.mode)
    },
    snapshot: snapshotChatLayout
  },
  theme: {
    apply: (id, setMode) => {
      if (MODE_OPTIONS.some(option => option.id === id)) {
        setMode(id as ThemeMode)
      }
    },
    snapshot: (mode, setMode) => () => setMode(mode)
  }
}

function useConnectorRows(storedId: null | string): ConnectorRow[] | null {
  const catalog = useConnectorCatalog(storedId)

  if (!storedId || catalog.status === 'loading') {
    return null
  }

  return catalog.status === 'ready' ? catalog.rows : []
}

export function useSetupRows(setup: null | SetupChooseSpec, storedId: null | string): null | SetupRow[] {
  const { t } = useI18n()
  const { renderedMode } = useTheme()
  const connectors = useConnectorRows(setup?.options === null && setup.kind === 'connectors' ? storedId : null)
  const plugins = useOnboardingPluginList(setup?.options === null && setup.kind === 'plugins' ? storedId : null)

  // Same inputs must return the same array: the pending card's stage callback and store write key off it.
  return useMemo(() => {
    if (!setup) {
      return NO_ROWS
    }

    const rows = setup.options ?? APP_ROWS[setup.kind]({ connectors, dark: renderedMode === 'dark', plugins, t })

    return setup.kind === 'theme' && rows ? rows.filter(row => THEME_TILES.includes(row.id)) : rows
  }, [connectors, plugins, renderedMode, setup, t])
}

const NO_ROWS: SetupRow[] = []

export function useSetupLabel(
  kind: SetupChooseKind,
  options: null | SetupRow[],
  storedId: null | string
): (id: string) => string {
  const { t } = useI18n()
  const plugins = useOnboardingPluginList(kind === 'plugins' && !options ? storedId : null)

  return id => options?.find(option => option.id === id)?.label ?? APP_LABELS[kind](id, { plugins, t })
}

import { useCallback } from 'react'

import { useI18n } from '@/i18n'

import { useTheme } from './context'
import { localizeTheme } from './localize'

// Retired skin names land on the canonical Nous skin so old muscle memory works.
const ALIASES: Record<string, string> = {
  ares: 'ember',
  default: 'nous',
  gold: 'nous',
  hermes: 'nous',
  'nous-light': 'nous'
}

export function useSkinCommand() {
  const { availableThemes, setTheme, themeName } = useTheme()
  const { t } = useI18n()
  const themes = availableThemes.map(theme => localizeTheme(theme, t.settings.appearance.builtInThemes))

  return useCallback(
    (rawArg: string) => {
      const arg = rawArg.trim()

      if (!themes.length) {
        return t.desktop.skin.unavailable
      }

      const activeIndex = Math.max(
        0,
        themes.findIndex(theme => theme.name === themeName)
      )

      if (!arg || arg === 'next') {
        const next = themes[(activeIndex + 1) % themes.length]
        setTheme(next.name)

        return t.desktop.skin.switched(next.label)
      }

      if (arg === 'list' || arg === 'ls' || arg === 'status') {
        const rows = themes.map(
          theme => `${theme.name === themeName ? '*' : ' '} ${theme.name.padEnd(10)} ${theme.label}`
        )

        return [t.desktop.skin.listTitle, ...rows, '', t.desktop.skin.listHint].join('\n')
      }

      const normalized = arg.toLowerCase()
      const targetName = ALIASES[normalized] || normalized

      const target = themes.find(
        theme => theme.name.toLowerCase() === targetName || theme.label.toLowerCase() === normalized
      )

      if (!target) {
        return t.desktop.skin.unknown(arg, themes.map(theme => theme.name).join(', '))
      }

      setTheme(target.name)

      return t.desktop.skin.switched(target.label)
    },
    [setTheme, t, themeName, themes]
  )
}

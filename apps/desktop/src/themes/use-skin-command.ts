import { useCallback } from 'react'

import { useTheme } from './context'
import { DEFAULT_SKIN_NAME } from './presets'

// Retired skin names land on a still-shipping skin so old muscle memory works.
// `default` is the one that isn't a rename: it means "no opinion", so it has to
// track whichever skin is currently default (matching `normalizeSkin`) rather
// than stay pinned to whatever that happened to be when it was retired. The rest
// name a specific palette and keep resolving to the Nous skin that succeeded it.
const ALIASES: Record<string, string> = {
  ares: 'ember',
  default: DEFAULT_SKIN_NAME,
  gold: 'nous',
  hermes: 'nous',
  'nous-light': 'nous'
}

export function useSkinCommand() {
  const { availableThemes, setTheme, themeName } = useTheme()

  return useCallback(
    (rawArg: string) => {
      const arg = rawArg.trim()

      if (!availableThemes.length) {
        return 'No desktop themes are available.'
      }

      const activeIndex = Math.max(
        0,
        availableThemes.findIndex(t => t.name === themeName)
      )

      if (!arg || arg === 'next') {
        const next = availableThemes[(activeIndex + 1) % availableThemes.length]
        setTheme(next.name)

        return `Desktop theme switched to ${next.label}.`
      }

      if (arg === 'list' || arg === 'ls' || arg === 'status') {
        const rows = availableThemes.map(t => `${t.name === themeName ? '*' : ' '} ${t.name.padEnd(10)} ${t.label}`)

        return ['Desktop themes:', ...rows, '', 'Use /skin <name>, or /skin to cycle.'].join('\n')
      }

      const normalized = arg.toLowerCase()
      const targetName = ALIASES[normalized] || normalized

      const target = availableThemes.find(
        t => t.name.toLowerCase() === targetName || t.label.toLowerCase() === normalized
      )

      if (!target) {
        return `Unknown desktop theme: ${arg}\nAvailable: ${availableThemes.map(t => t.name).join(', ')}`
      }

      setTheme(target.name)

      return `Desktop theme switched to ${target.label}.`
    },
    [availableThemes, setTheme, themeName]
  )
}

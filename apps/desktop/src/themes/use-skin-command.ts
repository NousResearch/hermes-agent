import { useCallback } from 'react'

import { useTheme } from './context'

// Retired/legacy skin names collapse onto the canonical skin so old muscle
// memory keeps working. Every retired name and 'bubblegum-pink' map to the
// canonical 'bubblegum'; 'ares' is the old CLI label for the Ember theme.
// Keep in sync with THEME_ALIASES in presets.ts.
const ALIASES: Record<string, string> = {
  'bubblegum-pink': 'bubblegum',
  ares: 'ember',
  default: 'bubblegum',
  gold: 'bubblegum',
  hermes: 'bubblegum',
  nous: 'bubblegum',
  'nous-light': 'bubblegum'
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

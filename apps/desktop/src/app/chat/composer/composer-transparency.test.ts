import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { cwd } from 'node:process'

import { describe, expect, it } from 'vitest'

import { composerFill, composerReplySurface } from '@/components/chat/composer-dock'

const styles = readFileSync(join(cwd(), 'src', 'styles.css'), 'utf8')

describe('composer reply transparency', () => {
  it('uses a dedicated transparent reply surface instead of the raised card fill', () => {
    expect(composerReplySurface).toContain('bg-(--composer-reply-fill)')
    expect(composerReplySurface).not.toContain('backdrop-blur')
    expect(composerReplySurface).not.toBe(composerFill)
  })

  it('matches the sampled light-blue canvas and keeps a visible focus state', () => {
    expect(styles).toContain('--hermes-reply-blue: #cedaf1;')
    expect(styles).toContain('--composer-reply-fill: color-mix(in srgb, var(--hermes-reply-blue) 14%, transparent);')
    expect(styles).toContain(
      "[data-slot='composer-root']:has([data-slot='composer-rich-input']:focus) [data-slot='composer-surface']"
    )
    expect(styles).toContain('border-color: var(--ui-accent) !important;')
  })

  it('keeps the HUD bar light-on-dark over any wallpaper while leaving dock cards readable', () => {
    expect(styles).toMatch(
      /\[data-hud-shell\] \[data-slot='composer-root'\][\s\S]*?--composer-fill: var\(--dt-card\);[\s\S]*?--composer-reply-fill: color-mix\(in srgb, rgb\(12 14 18\) 78%, var\(--hermes-reply-blue\)\);/
    )
    expect(styles).toMatch(
      /\[data-hud-shell\] \[data-slot='composer-surface'\] \{[\s\S]*?--ui-text-primary: #f2f5f9;[\s\S]*?color-scheme: dark;/
    )
    expect(styles).toMatch(
      /\[data-hud-shell\] \[data-slot='composer-rich-input'\]:is\(:empty, \[data-empty\]\)::before \{[\s\S]*?color: rgb\(242 245 249 \/ 0\.62\) !important;/
    )
    expect(styles).toMatch(
      /\[data-hud-shell\] \[data-slot='composer-surface'\][\s\S]*?background: var\(--composer-reply-fill\) !important;[\s\S]*?box-shadow: none !important;/
    )
  })
})

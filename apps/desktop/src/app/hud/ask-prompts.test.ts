import { describe, expect, it } from 'vitest'

import type { HudAskPayload } from '@/lib/hud-prefs'

import { hudAskPrompt, hudAskSource } from './ask-prompts'

const payload = (overrides: Partial<HudAskPayload> = {}): HudAskPayload => ({
  app: 'Figma',
  title: 'Onboarding v3',
  cursor: { x: 0, y: 0 },
  imagePath: 'C:/imgs/crop.png',
  thumbnail: 'data:image/png;base64,AAAA',
  via: 'shortcut',
  ...overrides
})

describe('hudAskSource', () => {
  it('joins app and title, and collapses a title that repeats the app', () => {
    expect(hudAskSource(payload())).toBe('Figma — Onboarding v3')
    expect(hudAskSource(payload({ title: 'Figma' }))).toBe('Figma')
    expect(hudAskSource(payload({ app: '', title: '' }))).toBe('')
    expect(hudAskSource(payload({ app: '' }))).toBe('Onboarding v3')
  })
})

describe('hudAskPrompt', () => {
  it('names the screenshot and its source', () => {
    expect(hudAskPrompt('explain', payload())).toBe(
      'Explain what I am looking at in the attached screenshot (from Figma — Onboarding v3). Be concise and concrete.'
    )
  })

  it('falls back to the window under the HUD when the capture failed', () => {
    expect(hudAskPrompt('summarize', payload({ imagePath: '', app: '', title: '' }))).toBe(
      'Summarize the content shown in what is under the HUD. Lead with the main point.'
    )
  })

  it('asks before irreversible work in the do-it prompt', () => {
    const text = hudAskPrompt('do', payload())

    expect(text).toContain('Ask before anything irreversible')
    expect(text).toContain('from Figma — Onboarding v3')
  })
})

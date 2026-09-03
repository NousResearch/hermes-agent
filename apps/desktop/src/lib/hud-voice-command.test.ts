import { describe, expect, it } from 'vitest'

import { parseHudVoiceCommand } from './hud-voice-command'

describe('parseHudVoiceCommand', () => {
  it('places the HUD at a corner or edge', () => {
    expect(parseHudVoiceCommand('HUD top left')).toEqual({ kind: 'place', anchor: 'top-left' })
    expect(parseHudVoiceCommand('Hey Hermes, move the HUD to the bottom right.')).toEqual({
      kind: 'place',
      anchor: 'bottom-right'
    })
    expect(parseHudVoiceCommand('hud upper-right')).toEqual({ kind: 'place', anchor: 'top-right' })
    expect(parseHudVoiceCommand('put the hud at the bottom')).toEqual({ kind: 'place', anchor: 'bottom-center' })
    expect(parseHudVoiceCommand('hud center')).toEqual({ kind: 'place', anchor: 'center' })
    expect(parseHudVoiceCommand('heads up display top')).toEqual({ kind: 'place', anchor: 'top-center' })
  })

  it('understands follow, stay, come here and hide', () => {
    expect(parseHudVoiceCommand('hud follow me')).toEqual({ kind: 'follow', on: true })
    expect(parseHudVoiceCommand('hermes hud follow my mouse')).toEqual({ kind: 'follow', on: true })
    expect(parseHudVoiceCommand('hud stop following')).toEqual({ kind: 'follow', on: false })
    expect(parseHudVoiceCommand('hud stay')).toEqual({ kind: 'follow', on: false })
    expect(parseHudVoiceCommand('hud come here')).toEqual({ kind: 'come-here' })
    expect(parseHudVoiceCommand('bring the hud to me')).toEqual({ kind: 'come-here' })
    expect(parseHudVoiceCommand('hud hide')).toEqual({ kind: 'hide' })
    expect(parseHudVoiceCommand('hud go away')).toEqual({ kind: 'hide' })
  })

  it('refuses anything substantive, even when it mentions the HUD', () => {
    expect(parseHudVoiceCommand('explain the hud code')).toBeNull()
    expect(parseHudVoiceCommand('move the hud to the top left and then summarize this page')).toBeNull()
    expect(parseHudVoiceCommand('what is a hud')).toBeNull()
    expect(parseHudVoiceCommand('top left')).toBeNull()
    expect(parseHudVoiceCommand('')).toBeNull()
  })
})

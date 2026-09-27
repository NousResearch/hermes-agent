import { describe, expect, it } from 'vitest'

import { plasmaPalette, plasmaTone } from './plasma'
import type { JarvisTaskPhase, JarvisVoiceState } from './types'

const VOICES: JarvisVoiceState[] = ['idle', 'listening', 'speaking', 'error']

const TASKS: JarvisTaskPhase[] = [
  'idle',
  'planning',
  'running',
  'approval',
  'cancelling',
  'cancelled',
  'failed',
  'verified'
]

describe('plasmaTone', () => {
  it('distinguishes planning from execution while preserving voice priority', () => {
    expect(plasmaTone('idle', 'planning')).not.toBe(plasmaTone('idle', 'running'))
    expect(plasmaTone('speaking', 'planning')).toBe(plasmaTone('speaking', 'running'))
  })
  it('keeps an intermediate colour in each tone and darkens every ramp stop on light surfaces', () => {
    for (const tone of ['idle', 'listening', 'working', 'speaking', 'approval', 'success', 'error'] as const) {
      const dark = plasmaPalette(tone)
      const light = plasmaPalette(tone, 'light')
      expect(dark.highlight).not.toBe(dark.front)
      expect(dark.highlight).not.toBe(dark.core)

      for (const key of ['back', 'core', 'front', 'highlight', 'rim'] as const) {
        const lightChannels = light[key].split(',').map(Number)
        const darkChannels = dark[key].split(',').map(Number)
        expect(lightChannels.every((channel, index) => channel <= darkChannels[index])).toBe(true)
        expect(lightChannels.some((channel, index) => channel < darkChannels[index])).toBe(true)
      }
    }
  })

  it('gives every voice × task pair a palette', () => {
    for (const voice of VOICES) {
      for (const task of TASKS) {
        expect(plasmaPalette(plasmaTone(voice, task))).toBeDefined()
      }
    }
  })

  it('lets an open mic or speaker outrank the task, except for errors', () => {
    for (const task of TASKS) {
      const listening = plasmaTone('listening', task)

      expect(listening).toBe(task === 'failed' ? 'error' : 'listening')
    }

    expect(plasmaTone('error', 'verified')).toBe('error')
  })

  it('stays at rest when nothing is happening', () => {
    expect(plasmaTone('idle', 'idle')).toBe('idle')
    expect(plasmaTone('idle', 'cancelled')).toBe('idle')
  })
})

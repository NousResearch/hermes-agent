import { expect, it, vi } from 'vitest'

import { createCharacterAnimations } from './character-animations'

function clip(name: string) {
  return {
    name,
    isPlaying: false,
    start: vi.fn(function (this: { isPlaying: boolean }) {
      this.isPlaying = true
    }),
    stop: vi.fn(function (this: { isPlaying: boolean }) {
      this.isPlaying = false
    }),
    reset: vi.fn()
  }
}

it('plays real idle normally, freezes on request, and resumes only continuous clips', () => {
  const idle = clip('idle'),
    done = clip('done'),
    controller = createCharacterAnimations(vi.fn())

  controller.register(
    'worker:baseline',
    new Map([
      ['idle', idle],
      ['done', done]
    ]),
    new Set(['idle'])
  )
  controller.set('worker:baseline', 'idle')
  expect(idle.start).toHaveBeenCalledWith(true)
  controller.setReducedMotion(true)
  expect(controller.isActive()).toBe(false)
  controller.setReducedMotion(false)
  expect(controller.isActive()).toBe(true)
  controller.set('worker:baseline', 'done')
  controller.setReducedMotion(true)
  controller.setReducedMotion(false)
  expect(done.start).toHaveBeenCalledOnce()
  expect(controller.isActive()).toBe(false)
})
it('stops the previous clip when a requested state has no authored animation', () => {
  const work = clip('work'), diagnostic=vi.fn(),
    controller = createCharacterAnimations(vi.fn(),diagnostic)

  controller.register('worker:baseline', new Map([['work', work]]), new Set(['work']))
  controller.set('worker:baseline', 'work')
  controller.set('worker:baseline', 'failed')
  expect(work.stop).toHaveBeenCalledOnce()
  expect(work.reset).toHaveBeenCalledOnce()
  controller.set('worker:baseline', 'failed')
  expect(diagnostic).toHaveBeenCalledOnce()
  expect(controller.isActive()).toBe(false)
})

it('applies a real held pose without keeping the frame scheduler awake', () => {
  const wait = { ...clip('wait'), hasMotion: false },
    controller = createCharacterAnimations(vi.fn())

  controller.register('worker:baseline', new Map([['wait', wait]]), new Set(['wait']))
  controller.set('worker:baseline', 'wait')
  expect(wait.reset).toHaveBeenCalledOnce()
  expect(wait.start).not.toHaveBeenCalled()
  expect(controller.isActive()).toBe(false)
})

import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, describe, expect, it, vi } from 'vitest'

import { createKeepAwakeController, IDLE_SLEEP_MIN_SECONDS } from './keep-awake-controller'
import { KEEP_AWAKE_RELEASE_GRACE_MS, type KeepAwakeMode, type PowerSaveBlockerLike } from './power-save'

function fakeBlocker() {
  let next = 1
  const started = new Set<number>()

  const blocker: PowerSaveBlockerLike = {
    isStarted: id => started.has(id),
    start: vi.fn(() => {
      const id = next++
      started.add(id)

      return id
    }),
    stop: vi.fn(id => void started.delete(id))
  }

  return { blocker, started }
}

describe('createKeepAwakeController: the offer after a sleep mid-turn', () => {
  function controller(saved?: object, working = true) {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'keep-awake-'))
    const configPath = path.join(dir, 'keep-awake.json')

    if (saved) {
      fs.writeFileSync(configPath, JSON.stringify(saved), 'utf8')
    }

    const offer = vi.fn(() => true)

    const ctl = createKeepAwakeController({
      blocker: fakeBlocker().blocker,
      configPath,
      isWorking: () => working,
      log: () => {},
      offer
    })

    ctl.restore()

    return { configPath, ctl, offer }
  }

  const timedSleep = IDLE_SLEEP_MIN_SECONDS + 240

  it('offers once, on the wake after a timed sleep interrupted a turn with keep-awake off', () => {
    const { configPath, ctl, offer } = controller({ mode: 'off' })

    ctl.noteSuspend(timedSleep)
    ctl.noteResume()
    expect(offer).toHaveBeenCalledTimes(1)

    ctl.noteSuspend(timedSleep)
    ctl.noteResume()
    expect(offer).toHaveBeenCalledTimes(1)
    expect(JSON.parse(fs.readFileSync(configPath, 'utf8'))).toEqual({ mode: 'off', offered: true })
  })

  it('does not offer again after a relaunch', () => {
    const { ctl, offer } = controller({ mode: 'off', offered: true })

    ctl.noteSuspend(timedSleep)
    ctl.noteResume()
    expect(offer).not.toHaveBeenCalled()
  })

  it('stays quiet for a closed lid, an idle app, or keep-awake already on', () => {
    const lid = controller({ mode: 'off' })
    lid.ctl.noteSuspend(2)
    lid.ctl.noteResume()
    expect(lid.offer).not.toHaveBeenCalled()

    const idle = controller({ mode: 'off' }, false)
    idle.ctl.noteSuspend(timedSleep)
    idle.ctl.noteResume()
    expect(idle.offer).not.toHaveBeenCalled()

    const on = controller({ mode: 'while-working' })
    on.ctl.noteSuspend(timedSleep)
    on.ctl.noteResume()
    expect(on.offer).not.toHaveBeenCalled()
  })

  it('waits for a chat window instead of spending the offer', () => {
    const { configPath, ctl, offer } = controller({ mode: 'off' })

    offer.mockReturnValueOnce(false)
    ctl.noteSuspend(timedSleep)
    ctl.noteResume()
    expect(JSON.parse(fs.readFileSync(configPath, 'utf8')).offered).toBeUndefined()

    ctl.noteSuspend(timedSleep)
    ctl.noteResume()
    expect(offer).toHaveBeenCalledTimes(2)
    expect(JSON.parse(fs.readFileSync(configPath, 'utf8')).offered).toBe(true)
  })

  it('keeps the offer record when the mode changes', () => {
    const { configPath, ctl } = controller({ mode: 'off', offered: true })

    ctl.setMode('while-working')
    expect(JSON.parse(fs.readFileSync(configPath, 'utf8'))).toEqual({ mode: 'while-working', offered: true })
  })
})

describe('createKeepAwakeController: release grace (#134434)', () => {
  afterEach(() => vi.useRealTimers())

  function graceController(mode: KeepAwakeMode = 'while-working') {
    vi.useFakeTimers()
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'keep-awake-'))
    const configPath = path.join(dir, 'keep-awake.json')
    fs.writeFileSync(configPath, JSON.stringify({ mode }), 'utf8')
    const { blocker, started } = fakeBlocker()
    let working = false

    const ctl = createKeepAwakeController({
      blocker,
      configPath,
      isWorking: () => working,
      log: () => {},
      offer: () => true
    })

    ctl.restore()

    return {
      blocker,
      ctl,
      held: () => started.size > 0,
      setWorking: (next: boolean) => {
        working = next
        ctl.apply()
      }
    }
  }

  it('holds through the grace after idle, releases at its end, and a chained turn never churns the blocker', () => {
    const { blocker, held, setWorking } = graceController()

    setWorking(true)
    setWorking(false)
    vi.advanceTimersByTime(KEEP_AWAKE_RELEASE_GRACE_MS - 1)
    expect(held()).toBe(true)

    // a follow-up turn lands inside the window, then goes idle again
    setWorking(true)
    setWorking(false)
    vi.advanceTimersByTime(KEEP_AWAKE_RELEASE_GRACE_MS - 1)
    expect(held()).toBe(true)

    vi.advanceTimersByTime(1)
    expect(held()).toBe(false)
    expect(blocker.start).toHaveBeenCalledTimes(1)
  })

  it('keeps holding when work outlasts the window with no report in between', () => {
    const { held, setWorking } = graceController()

    setWorking(true)
    vi.advanceTimersByTime(KEEP_AWAKE_RELEASE_GRACE_MS * 3)
    expect(held()).toBe(true)
  })

  it('a mode switch drops the pending hold; off and always have no grace', () => {
    const { ctl, held, setWorking } = graceController()

    setWorking(true)
    setWorking(false)
    ctl.setMode('off')
    expect(held()).toBe(false)

    ctl.setMode('always')
    vi.advanceTimersByTime(KEEP_AWAKE_RELEASE_GRACE_MS * 2)
    expect(held()).toBe(true)

    ctl.setMode('while-working')
    expect(held()).toBe(false)
  })
})

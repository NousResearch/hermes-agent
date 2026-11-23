import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { describe, expect, it, vi } from 'vitest'

import { createKeepAwakeController, IDLE_SLEEP_MIN_SECONDS } from './keep-awake-controller'
import type { PowerSaveBlockerLike } from './power-save'

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

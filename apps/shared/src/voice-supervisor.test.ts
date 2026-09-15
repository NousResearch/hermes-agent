/**
 * Shared supervisor brain — consult/steer/ack/stale-heal once, used by
 * desktop and the dashboard card. Faked session + runner; no sockets.
 */

import { describe, expect, it, vi } from 'vitest'

import { CONSULT_TOOL_NAME, STEER_TOOL_NAME } from './realtime-voice'
import {
  MAX_CONSULT_OUTPUT_CHARS,
  NARRATE_INTERVAL_MS,
  ownsTurnText,
  STALE_CONSULT_MIN_AGE_MS,
  type TurnRunner,
  VoiceSupervisorController
} from './voice-supervisor'

class FakeSession {
  alive = true
  lastResponseHadAudio = false
  readonly outputs: [string, string][] = []
  readonly verbatim: [string, boolean | undefined][] = []
  acks = 0

  sendFunctionOutput(callId: string, output: string): void {
    this.outputs.push([callId, output])
  }

  speakAcknowledgment(): void {
    this.acks += 1
  }

  speakVerbatim(text: string, interruptible?: boolean): void {
    this.verbatim.push([text, interruptible])
  }
}

class FakeRunner {
  submitted: string[] = []
  interrupts = 0
  ops: string[] = []
  busy = false
  queueEmpty = true
  accept = true

  submit(task: string): boolean {
    this.submitted.push(task)
    this.ops.push(`submit:${task}`)

    return this.accept
  }

  interrupt(): void {
    this.interrupts += 1
    this.ops.push('interrupt')
  }

  isBusy(): boolean {
    return this.busy
  }

  isQueueEmpty(): boolean {
    return this.queueEmpty
  }
}

function make() {
  const session = new FakeSession()
  const runner = new FakeRunner()
  const events: [string, string][] = []

  const ctrl = new VoiceSupervisorController(session, runner, (kind, text) => {
    events.push([kind, text])
  })

  return { ctrl, events, runner, session }
}

async function consult(
  ctrl: VoiceSupervisorController,
  callId = 'c1',
  task = 'check disk usage'
): Promise<void> {
  await ctrl.onFunctionCall(CONSULT_TOOL_NAME, callId, { task })
}

describe('ownsTurnText', () => {
  it('matches equality and coalesced first-line, not substrings', () => {
    expect(ownsTurnText('check disk usage', 'check disk usage')).toBe(true)
    expect(ownsTurnText('check disk usage', 'check disk usage\n\nand inodes')).toBe(true)
    expect(ownsTurnText('ls', 'please ls the folder')).toBe(false)
    expect(ownsTurnText('check disk usage', 'please check disk usage now')).toBe(false)
  })
})

describe('consult', () => {
  it('submits, tracks, and acks a silent tool call', async () => {
    const { ctrl, events, runner, session } = make()
    await consult(ctrl)
    expect(runner.submitted).toEqual(['check disk usage'])
    expect(events).toEqual([['consult', 'check disk usage']])
    expect(ctrl.consultActive).toBe(true)
    expect(session.acks).toBe(1)
    expect(session.outputs).toEqual([])
  })

  it('skips the ack when the model already spoke', async () => {
    const { ctrl, session } = make()
    session.lastResponseHadAudio = true
    await consult(ctrl)
    expect(session.acks).toBe(0)
  })

  it('rejects a second consult while one is in flight', async () => {
    const { ctrl, runner, session } = make()
    await consult(ctrl, 'c1', 'first')
    await consult(ctrl, 'c2', 'second')
    expect(runner.submitted).toEqual(['first'])
    expect(session.outputs.at(-1)?.[0]).toBe('c2')
    expect(session.outputs.at(-1)?.[1]).toMatch(/still working/)
    expect(session.outputs.at(-1)?.[1]).toMatch(/NOT queued/)
    expect(session.outputs.at(-1)?.[1]).toMatch(/steer_hermes/)
  })

  it('does not track a consult when submit returns false', async () => {
    const { ctrl, runner, session } = make()
    runner.accept = false
    await consult(ctrl)
    expect(ctrl.consultActive).toBe(false)
    expect(session.outputs[0][1]).toMatch(/Could not start/)
  })

  it('fails out a stale consult so a new one can start', async () => {
    const { ctrl, runner, session } = make()
    const t0 = Date.now()
    const nowSpy = vi.spyOn(Date, 'now').mockReturnValue(t0)

    try {
      await consult(ctrl, 'c1', 'first')
      nowSpy.mockReturnValue(t0 + STALE_CONSULT_MIN_AGE_MS + 1)
      await consult(ctrl, 'c2', 'second')
      expect(session.outputs).toContainEqual(['c1', 'That task failed without producing a result.'])
      expect(runner.submitted).toEqual(['first', 'second'])
      expect(ctrl.currentTask).toBe('second')
    } finally {
      nowSpy.mockRestore()
    }
  })
})

describe('steer', () => {
  it('interrupts a busy runner before submitting the new instruction', async () => {
    const { ctrl, runner, session } = make()
    await consult(ctrl, 'c1', 'original')
    runner.busy = true
    await ctrl.onFunctionCall(STEER_TOOL_NAME, 's1', { instruction: 'also check logs' })
    expect(ctrl.currentTask).toBe('also check logs')
    expect(runner.ops.slice(-2)).toEqual(['interrupt', 'submit:also check logs'])
    expect(session.outputs.at(-1)).toEqual(['s1', 'Steering applied — Hermes is adjusting course.'])
    expect(ctrl.onTurnComplete('also check logs', 'done')).toBe(true)
    expect(session.outputs).toContainEqual(['c1', 'done'])
  })

  it('reports nothing-to-steer without a consult', async () => {
    const { ctrl, runner, session } = make()
    await ctrl.onFunctionCall(STEER_TOOL_NAME, 's1', { instruction: 'go faster' })
    expect(runner.submitted).toEqual([])
    expect(session.outputs[0][1]).toMatch(/No Hermes task is running/)
  })

  it('keeps the original task when steer submit fails', async () => {
    const { ctrl, runner, session } = make()
    await consult(ctrl, 'c1', 'original')
    runner.accept = false
    await ctrl.onFunctionCall(STEER_TOOL_NAME, 's1', { instruction: 'nope' })
    expect(ctrl.currentTask).toBe('original')
    expect(session.outputs.at(-1)?.[1]).toMatch(/Steering failed/)
    expect(ctrl.onTurnComplete('original', 'done')).toBe(true)
  })

  it('ignores the interrupted turn completing mid-steer', async () => {
    const session = new FakeSession()
    let releaseInterrupt: () => void = () => undefined

    const runner: TurnRunner = {
      interrupt: () =>
        new Promise<void>(resolve => {
          releaseInterrupt = resolve
        }),
      isBusy: () => true,
      isQueueEmpty: () => false,
      submit: () => true
    }

    const ctrl = new VoiceSupervisorController(session, runner)
    await ctrl.onFunctionCall(CONSULT_TOOL_NAME, 'c1', { task: 'original' })
    const steer = ctrl.onFunctionCall(STEER_TOOL_NAME, 's1', { instruction: 'also check logs' })

    // Let the steer reach its pending interrupt() await.
    await Promise.resolve()
    // The dashboard card keys completion off a turn counter and passes the
    // tracked task as the message — without the steering guard this would
    // self-match and answer c1 with the interrupted turn's partial output.
    expect(ctrl.onTurnComplete(ctrl.currentTask ?? '', 'partial output')).toBe(false)
    expect(ctrl.consultActive).toBe(true)
    expect(session.outputs).toEqual([])

    releaseInterrupt()
    await steer
    expect(session.outputs.at(-1)).toEqual(['s1', 'Steering applied — Hermes is adjusting course.'])
    expect(ctrl.onTurnComplete('also check logs', 'real answer')).toBe(true)
    expect(session.outputs).toContainEqual(['c1', 'real answer'])
  })
})

describe('turn complete', () => {
  it('returns the result for a matching turn and ignores others', async () => {
    const { ctrl, session } = make()
    await consult(ctrl)
    expect(ctrl.onTurnComplete('typed message', 'nope')).toBe(false)
    expect(ctrl.consultActive).toBe(true)
    expect(ctrl.onTurnComplete('check disk usage', 'Disk is 42% full.')).toBe(true)
    expect(session.outputs.at(-1)).toEqual(['c1', 'Disk is 42% full.'])
    expect(ctrl.consultActive).toBe(false)
  })

  it('truncates long output and fails out an in-flight consult', async () => {
    const { ctrl, session } = make()
    await consult(ctrl, 'c1', 'task')
    expect(ctrl.onTurnComplete('task', 'x'.repeat(MAX_CONSULT_OUTPUT_CHARS + 500))).toBe(true)
    expect(session.outputs[0][1]).toContain('truncated')

    await consult(ctrl, 'c2', 'task')
    ctrl.failActiveConsult('Voice session ended.')
    expect(ctrl.consultActive).toBe(false)
    expect(session.outputs.at(-1)).toEqual(['c2', 'Voice session ended.'])
  })
})

describe('progress and blocking notices', () => {
  it('narrates tools only during a consult and throttles repeats', async () => {
    const { ctrl, session } = make()
    const nowSpy = vi.spyOn(Date, 'now').mockReturnValue(NARRATE_INTERVAL_MS)

    try {
      ctrl.narrateTool('terminal')
      expect(session.verbatim).toEqual([])
      await consult(ctrl)
      ctrl.narrateTool('run_terminal_cmd')
      ctrl.narrateTool('read_file')
      expect(session.verbatim).toEqual([['Running run terminal cmd.', true]])

      nowSpy.mockReturnValue(NARRATE_INTERVAL_MS * 2)
      ctrl.narrateTool('read_file')
      expect(session.verbatim.at(-1)).toEqual(['Running read file.', true])
    } finally {
      nowSpy.mockRestore()
    }
  })

  it('speaks blocking notices only while Hermes is working', async () => {
    const { ctrl, session } = make()
    ctrl.notify('Hermes needs your approval in the app.')
    expect(session.verbatim).toEqual([])
    await consult(ctrl)
    ctrl.notify('Hermes needs your approval in the app.')
    expect(session.verbatim.at(-1)).toEqual(['Hermes needs your approval in the app.', true])
  })
})

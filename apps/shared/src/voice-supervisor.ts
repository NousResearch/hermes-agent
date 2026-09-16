/**
 * Surface-agnostic supervisor brain for realtime voice.
 *
 * Mirrors agent/voice_supervisor.py: consult/steer/ack/stale-heal live here
 * once. Desktop and the dashboard card supply a TurnRunner; they do not
 * reimplement the machine. The mirror contract (shared constants, reply
 * texts, deliberate asymmetries) is listed in that module's docstring.
 */

import { CONSULT_TOOL_NAME, STEER_TOOL_NAME } from './realtime-voice'

export const STALE_CONSULT_MIN_AGE_MS = 30_000
export const MAX_CONSULT_OUTPUT_CHARS = 6000
export const NARRATE_INTERVAL_MS = 12_000

export interface VoiceSession {
  readonly lastResponseHadAudio: boolean
  readonly alive?: boolean
  sendFunctionOutput: (callId: string, output: string) => void
  speakAcknowledgment: () => void
  speakVerbatim?: (text: string, interruptible?: boolean) => void
}

export interface TurnRunner {
  submit: (task: string) => boolean | Promise<boolean>
  interrupt: () => void | Promise<void>
  isBusy: () => boolean
  isQueueEmpty: () => boolean
}

interface TrackedConsult {
  callId: string
  task: string
  at: number
}

export function ownsTurnText(task: string, message: string): boolean {
  if (message === task) {
    return true
  }

  for (const line of message.split('\n')) {
    if (line.trim() === task) {
      return true
    }
  }

  return false
}

export class VoiceSupervisorController {
  private consult: TrackedConsult | null = null
  // True between a steer's interrupt and its resubmit settling. The Python
  // controller cannot observe this window (its lock spans the whole steer);
  // browser runners await gateway RPCs mid-steer, so the interrupted turn's
  // completion can interleave and must not be taken for the consult result.
  private steering = false
  private callChain: Promise<void> = Promise.resolve()
  private readonly session: VoiceSession
  private readonly runner: TurnRunner
  private readonly onEvent?: (kind: string, text: string) => void
  private lastNarratedAt = 0

  constructor(
    session: VoiceSession,
    runner: TurnRunner,
    onEvent?: (kind: string, text: string) => void
  ) {
    this.session = session
    this.runner = runner
    this.onEvent = onEvent
  }

  get currentTask(): string | null {
    return this.consult?.task ?? null
  }

  get consultActive(): boolean {
    return this.consult !== null
  }

  reset(): void {
    this.consult = null
  }

  failActiveConsult(reason: string): void {
    const tracked = this.consult
    this.consult = null

    if (!tracked) {
      return
    }

    if (this.session.alive === false) {
      return
    }

    this.session.sendFunctionOutput(tracked.callId, reason)
  }

  ownsTurn(message: string): boolean {
    if (!this.consult) {
      return false
    }

    return ownsTurnText(this.consult.task, message)
  }

  onFunctionCall(name: string, callId: string, args: Record<string, unknown>): Promise<void> {
    const run = () => this.dispatchFunctionCall(name, callId, args)
    const next = this.callChain.then(run, run)
    this.callChain = next.then(
      () => undefined,
      () => undefined
    )

    return next
  }

  private async dispatchFunctionCall(
    name: string,
    callId: string,
    args: Record<string, unknown>
  ): Promise<void> {
    if (name === STEER_TOOL_NAME) {
      await this.onSteer(callId, args)

      return
    }

    if (name !== CONSULT_TOOL_NAME) {
      this.session.sendFunctionOutput(callId, `Unknown tool: ${name}`)

      return
    }

    const task = String(args.task ?? '').trim()

    if (!task) {
      this.session.sendFunctionOutput(callId, 'No task provided.')

      return
    }

    const stale = this.takeStaleConsult()

    if (stale) {
      this.session.sendFunctionOutput(stale.callId, 'That task failed without producing a result.')
    }

    if (this.consult) {
      this.session.sendFunctionOutput(
        callId,
        'Hermes is still working on the previous task. This new request was NOT queued. '
          + 'If it corrects, redirects, or extends the active task, call steer_hermes now. '
          + 'Otherwise wait for completion, then call consult_hermes again.'
      )

      return
    }

    let accepted = false

    try {
      accepted = Boolean(await this.runner.submit(task))
    } catch {
      accepted = false
    }

    if (!accepted) {
      this.session.sendFunctionOutput(
        callId,
        'Could not start that task (no speaker or the turn was dropped).'
      )

      return
    }

    this.consult = { callId, task, at: Date.now() }
    this.onEvent?.('consult', task)

    if (!this.session.lastResponseHadAudio) {
      this.session.speakAcknowledgment()
    }
  }

  onTurnComplete(message: string, response: string): boolean {
    if (this.steering) {
      // Mid-steer completion — the interrupted turn's partial output, not
      // the consult's answer. Keep the consult tracked; the resubmitted
      // turn completes it. (Surfaces that key completion off a turn counter
      // pass the tracked task as `message`, which would self-match here.)
      return false
    }

    const tracked = this.consult

    if (!tracked || !this.ownsTurn(message)) {
      return false
    }

    this.consult = null

    if (this.session.alive === false) {
      return false
    }

    let output = response.trim() || 'Hermes finished with no text output.'

    if (output.length > MAX_CONSULT_OUTPUT_CHARS) {
      output = `${output.slice(0, MAX_CONSULT_OUTPUT_CHARS)}\n[truncated — full text is on the user's screen]`
    }

    this.session.sendFunctionOutput(tracked.callId, output)

    return true
  }

  narrateTool(functionName: string): void {
    if (
      !functionName
      || !this.consult
      || this.session.alive === false
      || !this.session.speakVerbatim
    ) {
      return
    }

    const now = Date.now()

    if (now - this.lastNarratedAt < NARRATE_INTERVAL_MS) {
      return
    }

    this.lastNarratedAt = now
    this.session.speakVerbatim(`Running ${functionName.replaceAll('_', ' ')}.`, true)
  }

  notify(text: string): void {
    if (!this.consult || this.session.alive === false || !this.session.speakVerbatim) {
      return
    }

    this.session.speakVerbatim(text, true)
  }

  private async onSteer(callId: string, args: Record<string, unknown>): Promise<void> {
    const instruction = String(args.instruction ?? '').trim()

    if (!instruction) {
      this.session.sendFunctionOutput(callId, 'No steering instruction provided.')

      return
    }

    if (!this.consult) {
      this.session.sendFunctionOutput(
        callId,
        'No Hermes task is running — use consult_hermes to start one.'
      )

      return
    }

    this.onEvent?.('steer', instruction)
    const prevTask = this.consult.task
    this.consult = { callId: this.consult.callId, task: instruction, at: Date.now() }
    this.steering = true

    try {
      try {
        if (this.runner.isBusy()) {
          await this.runner.interrupt()
        }
      } catch {
        // still submit — interrupt is best-effort
      }

      let accepted = false

      try {
        accepted = Boolean(await this.runner.submit(instruction))
      } catch {
        accepted = false
      }

      if (!accepted) {
        if (this.consult) {
          this.consult = { ...this.consult, task: prevTask }
        }

        this.session.sendFunctionOutput(
          callId,
          'Steering failed — Hermes could not queue the new instruction.'
        )

        return
      }

      this.session.sendFunctionOutput(callId, 'Steering applied — Hermes is adjusting course.')
    } finally {
      this.steering = false
    }
  }

  private takeStaleConsult(): TrackedConsult | null {
    const tracked = this.consult

    if (
      !tracked
      || this.runner.isBusy()
      || !this.runner.isQueueEmpty()
      || Date.now() - tracked.at < STALE_CONSULT_MIN_AGE_MS
    ) {
      return null
    }

    this.consult = null

    return tracked
  }
}

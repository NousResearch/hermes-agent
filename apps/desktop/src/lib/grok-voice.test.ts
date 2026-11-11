import { describe, expect, it } from 'vitest'

import {
  type DialogState,
  enqueueResult,
  freshDialog,
  HERMES_RESULT_CAP,
  HERMES_RESULT_PREFIX,
  hermesResultText,
  reduceDialog
} from './grok-voice'

function sentText(decision: { send: object[] }): string {
  const item = decision.send.find(message => (message as { type?: string }).type === 'conversation.item.create') as {
    item?: { content?: Array<{ text?: string }>; output?: string; type?: string }
  }

  return item?.item?.content?.[0]?.text ?? item?.item?.output ?? ''
}

describe('hermes result text', () => {
  it('prefixes once and caps the body at 3000 characters', () => {
    const text = hermesResultText(['x'.repeat(HERMES_RESULT_CAP + 50)])

    expect(text.startsWith(HERMES_RESULT_PREFIX)).toBe(true)
    expect(text.slice(HERMES_RESULT_PREFIX.length)).toHaveLength(HERMES_RESULT_CAP)
    expect(hermesResultText([''])).toBe(`${HERMES_RESULT_PREFIX}Hermes finished without a spoken result.`)
  })
})

describe('grok dialog', () => {
  it('queues a result while the voice is speaking or responding, then flushes when quiet', () => {
    const speaking: DialogState = { ...freshDialog(), playing: true }
    const queued = enqueueResult(speaking, 'the file is saved')

    expect(queued.send).toEqual([])
    expect(queued.state.results).toEqual(['the file is saved'])

    const responding = enqueueResult({ ...freshDialog(), responding: true }, 'still working')
    expect(responding.send).toEqual([])

    const flushed = reduceDialog(queued.state, { type: 'playback-idle' })

    expect(sentText(flushed)).toBe(`${HERMES_RESULT_PREFIX}the file is saved`)
    expect(flushed.send.map(message => (message as { type: string }).type)).toEqual([
      'conversation.item.create',
      'response.create'
    ])
    expect(flushed.state.responding).toBe(true)
    expect(flushed.state.results).toEqual([])
  })

  it('plays both audio delta names and flushes playback when the user barges in', () => {
    const first = reduceDialog(freshDialog(), { delta: 'aaa', type: 'response.output_audio.delta' })
    const second = reduceDialog(first.state, { delta: 'bbb', type: 'response.audio.delta' })

    expect(first.play).toBe('aaa')
    expect(second.play).toBe('bbb')
    expect(second.state.playing).toBe(true)
    expect(second.state.spoke).toBe(true)

    const barge = reduceDialog(second.state, { type: 'input_audio_buffer.speech_started' })

    expect(barge.flushPlayback).toBe(true)
    expect(barge.state.playing).toBe(false)
    expect(barge.send).toEqual([])
  })

  it('turns hermes_task into a delegation plus a function_call_output', () => {
    const withUser = reduceDialog(freshDialog(), {
      transcript: 'what is in the file',
      type: 'conversation.item.input_audio_transcription.completed'
    })

    const tool = reduceDialog(withUser.state, {
      arguments: JSON.stringify({ request: 'read notes.txt' }),
      call_id: 'call-1',
      name: 'hermes_task',
      type: 'response.function_call_arguments.done'
    })

    expect(tool.delegate).toEqual({ context: 'user: what is in the file', request: 'read notes.txt' })
    expect(sentText(tool)).toContain('"receipt":"started"')
    expect(tool.state.owedTurn).toBe(true)
  })

  it('asks the model to speak when a tool call said nothing, and hangs up after end_call', () => {
    const tool = reduceDialog(
      { ...freshDialog(), responding: true },
      {
        arguments: '{}',
        call_id: 'call-2',
        name: 'hermes_task',
        type: 'response.function_call_arguments.done'
      }
    )

    const done = reduceDialog(tool.state, { type: 'response.done' })

    expect(done.send).toEqual([{ type: 'response.create' }])
    expect(done.state.responding).toBe(true)

    const ending = reduceDialog(freshDialog(), {
      arguments: '{}',
      call_id: 'call-3',
      name: 'end_call',
      type: 'response.function_call_arguments.done'
    })

    const hangup = reduceDialog({ ...ending.state, responding: true }, { type: 'response.done' })

    expect(hangup.hangUp).toBe(true)
    expect(hangup.send).toEqual([])
  })
})

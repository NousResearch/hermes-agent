import { describe, expect, it, vi } from 'vitest'

import { createGeminiLiveHandler, type GeminiAudioOut, geminiLiveSocketUrl } from './gemini-live-voice'
import { bytesToBase64, floatToInt16LE, pcm16Base64ToFloat32, pcmRate } from './pcm-audio'

function setup(onAsk: (request: string) => Promise<string> = async () => 'Masz jutro dwa spotkania.') {
  const sent: Record<string, unknown>[] = []
  const statuses: string[] = []
  const transcripts: [string, string][] = []
  const handles: string[] = []

  const player = {
    flush: vi.fn(),
    play: vi.fn(),
    whenIdle: vi.fn((done: () => void) => done())
  } satisfies GeminiAudioOut

  const handle = createGeminiLiveHandler({ send: message => sent.push(message) }, player, {
    onAsk,
    onDelegate: async request => `Przyjęto: ${request}`,
    onError: vi.fn(),
    onResumeHandle: h => handles.push(h),
    onStatus: status => statuses.push(status),
    onTranscript: (role, text) => transcripts.push([role, text])
  })

  return { handle, handles, player, sent, statuses, transcripts }
}

const audioChunk = (samples: number[]) => bytesToBase64(floatToInt16LE(new Float32Array(samples)))

describe('Gemini Live protocol', () => {
  it('plays model speech at the rate it declares and shows that Agent Czesiek is speaking', async () => {
    const { handle, player, statuses } = setup()

    await handle({ setupComplete: {} })
    await handle({
      serverContent: {
        modelTurn: { parts: [{ inlineData: { data: audioChunk([0.5, -0.5]), mimeType: 'audio/pcm;rate=24000' } }] }
      }
    })

    expect(player.play).toHaveBeenCalledTimes(1)
    expect(player.play.mock.calls[0][1]).toBe(24000)
    expect(statuses).toEqual(['listening', 'speaking'])
  })

  it('stops the voice at once when the user talks over it', async () => {
    const { handle, player, statuses } = setup()

    await handle({
      serverContent: {
        modelTurn: { parts: [{ inlineData: { data: audioChunk([0.1]), mimeType: 'audio/pcm;rate=24000' } }] }
      }
    })
    await handle({ serverContent: { interrupted: true } })

    expect(player.flush).toHaveBeenCalled()
    expect(statuses.at(-1)).toBe('listening')
  })

  it('hands ask_jarvis to the agent and answers with a function response carrying the same id', async () => {
    const onAsk = vi.fn(async () => 'Masz jutro dwa spotkania.')
    const { handle, sent, statuses } = setup(onAsk)

    await handle({
      toolCall: { functionCalls: [{ args: { request: 'Co mam jutro?' }, id: 'call-1', name: 'ask_jarvis' }] }
    })
    // A repeated call id is answered once.
    await handle({
      toolCall: { functionCalls: [{ args: { request: 'Co mam jutro?' }, id: 'call-1', name: 'ask_jarvis' }] }
    })

    expect(onAsk).toHaveBeenCalledOnce()
    expect(onAsk).toHaveBeenCalledWith('Co mam jutro?')
    expect(statuses).toContain('thinking')
    expect(sent).toEqual([
      {
        toolResponse: {
          functionResponses: [{ id: 'call-1', name: 'ask_jarvis', response: { output: 'Masz jutro dwa spotkania.' } }]
        }
      }
    ])
  })

  it('routes delegated work separately and returns an immediate acknowledgement', async () => {
    const onAsk = vi.fn(async () => 'blocking answer')
    const { handle, sent } = setup(onAsk)

    await handle({
      toolCall: { functionCalls: [{ args: { request: 'Zrób raport' }, id: 'delegate-1', name: 'delegate_to_hermes' }] }
    })

    expect(onAsk).not.toHaveBeenCalled()
    expect(sent).toEqual([
      {
        toolResponse: {
          functionResponses: [
            { id: 'delegate-1', name: 'delegate_to_hermes', response: { output: 'Przyjęto: Zrób raport' } }
          ]
        }
      }
    ])
  })

  it('never answers a call the server cancelled while the agent was still working', async () => {
    let finish: (value: string) => void = () => undefined
    const { handle, sent } = setup(() => new Promise(resolve => (finish = resolve)))

    const pending = handle({
      toolCall: { functionCalls: [{ args: { request: 'x' }, id: 'call-2', name: 'ask_jarvis' }] }
    })

    await handle({ toolCallCancellation: { ids: ['call-2'] } })
    finish('late answer')
    await pending

    expect(sent).toEqual([])
  })

  it('returns to listening after a turn, even one that never spoke, and reports both transcripts', async () => {
    const { handle, statuses, transcripts } = setup()

    await handle({ serverContent: { inputTranscription: { text: 'Cześć ' } } })
    await handle({ serverContent: { inputTranscription: { text: 'Agent Czesiek' }, outputTranscription: { text: 'Hej!' } } })
    await handle({ toolCall: { functionCalls: [{ args: { request: 'r' }, id: 'c', name: 'ask_jarvis' }] } })
    await handle({ serverContent: { turnComplete: true } })

    expect(transcripts).toEqual([
      ['user', 'Cześć Agent Czesiek'],
      ['assistant', 'Hej!']
    ])
    expect(statuses.at(-1)).toBe('listening')
  })

  it('keeps the latest resumable handle so a dropped connection can continue the conversation', async () => {
    const { handle, handles } = setup()

    await handle({ sessionResumptionUpdate: { newHandle: 'h1', resumable: true } })
    await handle({ sessionResumptionUpdate: { newHandle: 'h2', resumable: false } })

    expect(handles).toEqual(['h1'])
  })

  it('opens the constrained socket with the ephemeral token as access_token', () => {
    expect(geminiLiveSocketUrl({ token: 'auth_tokens/a b', ws_url: 'wss://x/BidiGenerateContentConstrained' })).toBe(
      'wss://x/BidiGenerateContentConstrained?access_token=auth_tokens%2Fa%20b'
    )
  })
})

describe('PCM helpers', () => {
  it('round-trips int16 PCM through base64 within quantization error', () => {
    const original = new Float32Array([0, 0.25, -0.5, 0.999, -1])
    const decoded = pcm16Base64ToFloat32(bytesToBase64(floatToInt16LE(original)))

    expect(decoded).toHaveLength(original.length)
    decoded.forEach((value, index) => expect(Math.abs(value - original[index])).toBeLessThan(1e-3))
  })

  it('reads the sample rate from the mime type', () => {
    expect(pcmRate('audio/pcm;rate=24000', 16000)).toBe(24000)
    expect(pcmRate('audio/pcm', 16000)).toBe(16000)
  })
})

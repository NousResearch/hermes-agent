import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ComposerAttachment } from '@/store/composer'

import type { MicRecording } from './use-mic-recorder'
import { useVoiceRecorder } from './use-voice-recorder'

// #136410: opening Settings (or any navigation) tears the composer down while
// dictation is recording. The take used to die with the unmount — no
// transcription, no draft text, no recovery. The recorder must finish the take
// off-screen and land the transcript in the recording session's draft.

let recording = false

const TAKE: MicRecording = { audio: new Blob(['clip']), durationMs: 1_500, heardSpeech: true }

const micHandle = {
  cancel: vi.fn(),
  recoverTake: vi.fn<() => MicRecording | null>(() => null),
  start: vi.fn(async () => {
    recording = true
  }),
  stop: vi.fn<() => Promise<MicRecording | null>>(async () => {
    recording = false

    return TAKE
  })
}

vi.mock('./use-mic-recorder', () => ({
  useMicRecorder: () => ({
    handle: micHandle,
    level: 0,
    get recording() {
      return recording
    }
  })
}))

const syncSttLeaseSpy = vi.fn(async (..._args: unknown[]): Promise<void> => undefined)

vi.mock('@/lib/stt-lease', () => ({
  syncSttLease: (...args: unknown[]) => syncSttLeaseSpy(...args),
  VOICE_INPUT_LEASE: 'desktop:voice-input:test'
}))

const stashSessionDraftSpy = vi.fn(
  (scope: null | string | undefined, text: string, attachments: ComposerAttachment[]) => undefined
)

const takeSessionDraftSpy = vi.fn((_scope: null | string | undefined): { attachments: ComposerAttachment[]; text: string } => ({
  attachments: [],
  text: ''
}))

vi.mock('@/store/composer', () => ({
  stashSessionDraft: (scope: null | string | undefined, text: string, attachments: ComposerAttachment[]) =>
    stashSessionDraftSpy(scope, text, attachments),
  takeSessionDraft: (scope: null | string | undefined) => takeSessionDraftSpy(scope)
}))

vi.mock('@/lib/voice-stream', () => ({
  openDictationStream: vi.fn(async () => null)
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

vi.mock('@/store/desktop-metrics', () => ({
  recordFeatureUse: vi.fn()
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      notifications: {
        voice: {
          noSpeechDetected: 'no speech',
          recordingFailed: 'recording failed',
          transcriptionFailed: 'transcription failed',
          transcriptionUnavailable: 'unavailable',
          tryRecordingAgain: 'try again',
          unavailable: 'unavailable'
        }
      }
    }
  })
}))

vi.mock('../scope', () => ({
  useComposerScope: () => ({ connectionId: undefined, profile: undefined })
}))

vi.mock('@/hermes', () => ({
  resolveOwnerNow: (owner?: { connectionId?: string; profile?: string }) => ({
    connectionId: owner?.connectionId ?? 'gateway-a',
    profile: owner?.profile ?? 'worker_alpha'
  })
}))

const OWNER_A = { connectionId: 'gateway-a', profile: 'worker_alpha' }

describe('useVoiceRecorder navigation recovery', () => {
  beforeEach(() => {
    cleanup()
    recording = false
    micHandle.start.mockClear()
    micHandle.stop.mockReset()
    micHandle.stop.mockImplementation(async () => {
      recording = false

      return TAKE
    })
    micHandle.recoverTake.mockClear()
    micHandle.recoverTake.mockImplementation(() => null)
    stashSessionDraftSpy.mockClear()
    takeSessionDraftSpy.mockClear()
    takeSessionDraftSpy.mockImplementation(() => ({ attachments: [], text: '' }))
    syncSttLeaseSpy.mockReset()
    syncSttLeaseSpy.mockImplementation(async () => undefined)
  })

  afterEach(() => {
    cleanup()
  })

  function renderRecorder(
    onTranscribeAudio: (audio: Blob, owner?: unknown) => Promise<string>,
    draftScope?: null | string
  ) {
    return renderHook(() =>
      useVoiceRecorder({
        draftScope,
        focusInput: () => undefined,
        maxRecordingSeconds: 30,
        onTranscript: () => undefined,
        onTranscribeAudio
      })
    )
  }

  it('transcribes the take and stashes it into the recording session draft when navigation unmounts mid-recording', async () => {
    const transcribe = vi.fn(async () => 'hello from settings')
    const hook = renderRecorder(transcribe, 'sess-1')

    await act(async () => {
      hook.result.current.dictate()
    })
    await act(async () => {
      await Promise.resolve()
    })
    expect(micHandle.start).toHaveBeenCalledTimes(1)

    hook.unmount()

    await waitFor(() => expect(stashSessionDraftSpy).toHaveBeenCalledTimes(1))
    expect(stashSessionDraftSpy).toHaveBeenCalledWith('sess-1', 'hello from settings', [])
    // The text lands in the draft, never in a torn-down editor.
    expect(micHandle.stop).toHaveBeenCalledTimes(1)
    // The lease taken at mic-open is released after the recovery settles.
    expect(syncSttLeaseSpy).toHaveBeenLastCalledWith('desktop:voice-input:test', false, OWNER_A)
  })

  it('falls back to the parked take when the recorder is already gone', async () => {
    micHandle.stop.mockImplementation(async () => {
      recording = false

      return null
    })
    micHandle.recoverTake.mockImplementation(() => TAKE)
    const transcribe = vi.fn(async () => 'recovered words')
    const hook = renderRecorder(transcribe, 'sess-2')

    await act(async () => {
      hook.result.current.dictate()
    })
    await act(async () => {
      await Promise.resolve()
    })

    hook.unmount()

    await waitFor(() => expect(stashSessionDraftSpy).toHaveBeenCalledTimes(1))
    expect(micHandle.recoverTake).toHaveBeenCalledTimes(1)
    expect(stashSessionDraftSpy).toHaveBeenCalledWith('sess-2', 'recovered words', [])
  })

  it('preserves the existing draft text and attachments when the recovered transcript lands', async () => {
    const attachment: ComposerAttachment = { id: 'a1', kind: 'file', label: 'log.txt' }
    takeSessionDraftSpy.mockImplementation(() => ({ attachments: [attachment], text: 'earlier notes' }))
    const transcribe = vi.fn(async () => 'more words')
    const hook = renderRecorder(transcribe, 'sess-1')

    await act(async () => {
      hook.result.current.dictate()
    })
    await act(async () => {
      await Promise.resolve()
    })

    hook.unmount()

    await waitFor(() => expect(stashSessionDraftSpy).toHaveBeenCalledTimes(1))
    expect(stashSessionDraftSpy).toHaveBeenCalledWith('sess-1', 'earlier notes\nmore words', [attachment])
  })

  it('stashes a transcript that was already in flight when navigation happened, without transcribing twice', async () => {
    let settleWarmup!: () => void
    syncSttLeaseSpy.mockImplementation(
      () =>
        new Promise<void>(resolve => {
          settleWarmup = resolve
        })
    )
    const transcribe = vi.fn(async () => 'in flight words')
    const hook = renderRecorder(transcribe, 'sess-3')

    await act(async () => {
      hook.result.current.dictate()
    })
    await act(async () => {
      await Promise.resolve()
    })

    let stopping: Promise<void> = Promise.resolve()
    await act(async () => {
      stopping = hook.result.current.dictate() ?? Promise.resolve()
      await Promise.resolve()
    })
    // The in-flight stop() owns the take; the unmount must not transcribe again.
    hook.unmount()
    settleWarmup()
    await stopping

    await waitFor(() => expect(stashSessionDraftSpy).toHaveBeenCalledTimes(1))
    expect(transcribe).toHaveBeenCalledTimes(1)
    expect(stashSessionDraftSpy).toHaveBeenCalledWith('sess-3', 'in flight words', [])
    expect(syncSttLeaseSpy).toHaveBeenLastCalledWith('desktop:voice-input:test', false, OWNER_A)
  })

  it('stashes under the fresh-draft scope when the composer had no session', async () => {
    const transcribe = vi.fn(async () => 'fresh words')
    const hook = renderRecorder(transcribe)

    await act(async () => {
      hook.result.current.dictate()
    })
    await act(async () => {
      await Promise.resolve()
    })

    hook.unmount()

    await waitFor(() => expect(stashSessionDraftSpy).toHaveBeenCalledTimes(1))
    expect(stashSessionDraftSpy).toHaveBeenCalledWith(undefined, 'fresh words', [])
  })

  it('still releases the lease when the recovered take transcribes to nothing', async () => {
    const transcribe = vi.fn(async () => '   ')
    const hook = renderRecorder(transcribe, 'sess-4')

    await act(async () => {
      hook.result.current.dictate()
    })
    await act(async () => {
      await Promise.resolve()
    })

    hook.unmount()

    await waitFor(() => expect(syncSttLeaseSpy).toHaveBeenLastCalledWith('desktop:voice-input:test', false, OWNER_A))
    expect(stashSessionDraftSpy).not.toHaveBeenCalled()
  })

  it('never auto-sends or touches the composer editor from the recovery path', async () => {
    const transcribe = vi.fn(async () => 'words')
    const hook = renderRecorder(transcribe, 'sess-5')

    await act(async () => {
      hook.result.current.dictate()
    })
    await act(async () => {
      await Promise.resolve()
    })

    hook.unmount()
    await waitFor(() => expect(stashSessionDraftSpy).toHaveBeenCalledTimes(1))

    expect(micHandle.cancel).not.toHaveBeenCalled()
  })
})

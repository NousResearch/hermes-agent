import { describe, expect, it } from 'vitest'

import { type MicRecorderErrorCopy, microphoneCapabilityError } from './use-mic-recorder'

const copy: MicRecorderErrorCopy = {
  microphoneAccessDenied: 'access denied',
  microphoneConstraintsUnsupported: 'constraints unsupported',
  microphoneInUse: 'in use',
  microphonePermissionDenied: 'permission denied',
  microphoneSecureContextRequired: 'secure context required',
  microphoneStartFailed: 'start failed',
  microphoneUnsupported: 'unsupported',
  noMicrophone: 'no microphone'
}

describe('microphoneCapabilityError', () => {
  it('reports an insecure context before the generic capability error', () => {
    expect(microphoneCapabilityError(copy, false, undefined, undefined)?.message).toBe('secure context required')
  })

  it('keeps the unsupported error for a secure runtime without recording APIs', () => {
    expect(microphoneCapabilityError(copy, true, undefined, undefined)?.message).toBe('unsupported')
  })

  it('returns no error when the recording APIs are available', () => {
    const mediaDevices = { getUserMedia: async () => ({}) as MediaStream } as MediaDevices
    const mediaRecorder = class {} as unknown as typeof MediaRecorder

    expect(microphoneCapabilityError(copy, true, mediaDevices, mediaRecorder)).toBeNull()
  })
})

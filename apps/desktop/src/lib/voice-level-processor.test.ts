import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

type LevelSample = { level: number; audioTime: number; nextReportTime: number }
type Processor = { process(inputs: Float32Array[][], outputs: Float32Array[][]): boolean }
let ProcessorCtor: new () => Processor
let reports: LevelSample[]

beforeEach(async () => {
  reports = []
  vi.resetModules()
  vi.stubGlobal('sampleRate', 48_000)
  vi.stubGlobal('currentFrame', 0)
  vi.stubGlobal(
    'AudioWorkletProcessor',
    class {
      port = { postMessage: (sample: LevelSample) => reports.push(sample) }
    }
  )
  vi.stubGlobal('registerProcessor', (_name: string, ctor: new () => Processor) => {
    ProcessorCtor = ctor
  })
  await import('./voice-level-processor')
})

afterEach(() => vi.unstubAllGlobals())

function processAudio(processor: Processor, start: number, frames: number, input: boolean): void {
  for (let offset = 0; offset < frames; offset += 128) {
    const size = Math.min(128, frames - offset)
    const signal = Float32Array.from({ length: size }, (_, index) => (index % 2 ? -0.02 : 0.02))
    const output = new Float32Array(size)
    vi.stubGlobal('currentFrame', start + offset)
    processor.process(input ? [[signal]] : [[]], [[output]])
    expect(output.every(value => value === 0)).toBe(true)
  }
}

describe('voice audio-worklet sampling', () => {
  it.each([22_050, 44_100, 48_000, 96_000])(
    'keeps 60 Hz audio timestamps and the existing byte RMS scale at %i Hz',
    rate => {
      vi.stubGlobal('sampleRate', rate)
      const processor = new ProcessorCtor()
      processAudio(processor, 0, rate, true)

      expect(reports).toHaveLength(60)
      reports.forEach((report, index) => {
        // +/-0.02 maps to byte-domain +2/-3, not the unquantized float RMS.
        expect(report.level).toBeCloseTo(Math.sqrt(6.5) / 42, 12)
        expect(report.audioTime).toBeCloseTo((index + 1) / 60, 4)
        expect(report.nextReportTime).toBeCloseTo((index + 2) / 60, 4)

        if (index + 1 < reports.length) {
          expect(report.nextReportTime).toBe(reports[index + 1].audioTime)
        }
      })
    }
  )

  it('continues reporting silence when an input disappears instead of hanging an utterance', () => {
    const processor = new ProcessorCtor()
    processAudio(processor, 0, 1_024, true)
    processAudio(processor, 1_024, 1_024, false)

    expect(reports).toEqual([
      { level: Math.sqrt(6.5) / 42, audioTime: 800 / 48_000, nextReportTime: 1_600 / 48_000 },
      { level: 0, audioTime: 1_600 / 48_000, nextReportTime: 2_400 / 48_000 }
    ])
  })
})

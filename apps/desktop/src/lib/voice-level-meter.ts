import processorUrl from './voice-level-processor.ts?worker&url'

/** Functional voice detection follows captured audio, not the window's paint clock. */
export function startVoiceLevelMeter(
  context: AudioContext,
  source: MediaStreamAudioSourceNode,
  onLevel: (level: number, audioTimeMs: number, isCurrent: boolean) => void,
  onError: () => void
): () => void {
  let disposed = false
  let node: AudioWorkletNode | null = null
  let connected = false

  const stop = () => {
    disposed = true
    context.removeEventListener('error', fail)
    context.removeEventListener('statechange', checkState)

    if (!node) {
      return
    }

    node.onprocessorerror = null
    node.port.onmessage = null
    node.port.close()

    if (connected) {
      source.disconnect(node)
      connected = false
    }

    node.disconnect()
    node = null
  }

  const fail = () => {
    if (!disposed) {
      stop()
      onError()
    }
  }

  const checkState = () => {
    if (context.state === 'closed') {
      fail()
    }
  }

  // A stopped audio clock cannot deliver the silence that ends a voice turn.
  context.addEventListener('error', fail)
  context.addEventListener('statechange', checkState)

  void (async () => {
    await context.audioWorklet.addModule(processorUrl)

    if (disposed) {
      return
    }

    node = new AudioWorkletNode(context, 'hermes-voice-level', {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
      channelCount: 1,
      channelCountMode: 'explicit',
      channelInterpretation: 'speakers'
    })
    node.port.onmessage = ({ data }: MessageEvent<{ level: number; audioTime: number; nextReportTime: number }>) => {
      if (!disposed) {
        // Drain every level for speech detection. Only the latest reporting
        // interval may end or discard capture: older quiet can precede queued speech.
        onLevel(data.level, data.audioTime * 1_000, context.currentTime < data.nextReportTime)
      }
    }
    node.onprocessorerror = fail
    source.connect(node)
    connected = true
    node.connect(context.destination)
  })().catch(fail)

  return stop
}

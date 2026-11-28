// One analyser per playback <audio> element, shared by every reader (the composer's waveform
// and the plugin voice door). A second createMediaElementSource on the same element throws,
// so every caller must come through here.

type BrowserAudioContext = typeof AudioContext

export interface ElementAnalyser {
  analyser: AnalyserNode
}

const elementAnalysers = new WeakMap<HTMLAudioElement, ElementAnalyser>()
let playbackAudioContext: AudioContext | null = null

function getPlaybackAudioContext(): AudioContext | null {
  if (playbackAudioContext && playbackAudioContext.state !== 'closed') {
    return playbackAudioContext
  }

  const audioWindow = window as Window & { webkitAudioContext?: BrowserAudioContext }
  const AudioContextCtor = window.AudioContext || audioWindow.webkitAudioContext

  if (!AudioContextCtor) {
    return null
  }

  playbackAudioContext = new AudioContextCtor()

  return playbackAudioContext
}

export function getElementAnalyser(audioElement: HTMLAudioElement): ElementAnalyser | null {
  let entry = elementAnalysers.get(audioElement)

  if (!entry) {
    const context = getPlaybackAudioContext()

    if (!context) {
      return null
    }

    const source = context.createMediaElementSource(audioElement)
    const analyser = context.createAnalyser()

    analyser.fftSize = 512
    analyser.smoothingTimeConstant = 0.65
    source.connect(analyser)
    analyser.connect(context.destination)
    entry = { analyser }
    elementAnalysers.set(audioElement, entry)
  }

  void playbackAudioContext?.resume()

  return entry
}

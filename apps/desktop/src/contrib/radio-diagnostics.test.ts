import { afterEach, describe, expect, it, vi } from 'vitest'

interface RadioModule {
  nowPlaying: (
    ctx: { rest?: (path: string, options?: unknown) => Promise<unknown> },
    url: string
  ) => Promise<{ ok: boolean; title: string }>
  mediaFailure: (element: { error?: { code: number } }, station: { url: string }) => string
  reportFailure: (
    ctx: { rest?: (path: string, options?: unknown) => Promise<unknown> },
    station: { name: string; url: string },
    reason: string
  ) => void
  createPlayer: (ctx: unknown) => {
    status: { get: () => string }
    lastError: { get: () => string }
    play: (station?: unknown) => Promise<void>
  }
}

const modules = import.meta.glob<RadioModule>('../plugins/radio/plugin.js', { eager: true })
const radio = Object.values(modules)[0]

class FakeAudio {
  dataset: Record<string, string> = {}
  hidden = false
  preload = 'none'
  volume = 1
  paused = true
  error = { code: 2 }
  private listeners = new Map<string, Array<() => void>>()
  addEventListener(name: string, handler: () => void) {
    this.listeners.set(name, [...(this.listeners.get(name) ?? []), handler])
  }
  removeAttribute() {}
  load() {}
  remove() {}
  pause() {}
  play() { return Promise.reject(new Error('no audio device in the harness')) }
  fire(name: string) { this.listeners.get(name)?.forEach(handler => handler()) }
}

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('Radio ICY titles and failure diagnostics', () => {
  it('asks the backend for the announced title with the address encoded once', async () => {
    const rest = vi.fn(async () => ({ ok: true, title: 'Aphex Twin - Xtal' }))
    await expect(radio.nowPlaying({ rest }, 'https://stream.example/live?a=1&b=2')).resolves.toMatchObject({ title: 'Aphex Twin - Xtal' })
    const path = rest.mock.calls[0][0] as string
    expect(path).toContain('/now-playing?url=')
    expect(new URLSearchParams(path.split('?')[1]).get('url')).toBe('https://stream.example/live?a=1&b=2')
    await expect(radio.nowPlaying({}, 'https://stream.example/live')).resolves.toEqual({ ok: false, title: '' })
  })

  it('names the media error and the host that failed', () => {
    expect(radio.mediaFailure({ error: { code: 4 } }, { url: 'https://stream.example/live' })).toBe('unsupported stream format (stream.example)')
    expect(radio.mediaFailure({}, { url: 'not a url' })).toBe('the stream did not connect (not a url)')
  })

  it('writes the failure to the backend log and keeps the reason on the player', async () => {
    const posted: Array<{ path: string; options: { body: { url: string; reason: string } } }> = []
    const rest = vi.fn(async (path: string, options?: unknown) => {
      if (path === '/log') posted.push({ path, options: options as never })
      if (path.startsWith('/probe')) return { ok: false, kind: 'invalid' }
      return { stations: [] }
    }) as never
    const storage = {
      get: (_key: string, fallback: unknown) => fallback,
      set: () => {}
    }
    const ctx = { rest, storage, onDispose: () => {} }
    const audio = new FakeAudio()
    vi.stubGlobal('Audio', function FakeAudioElement() { return audio })
    vi.stubGlobal('AudioContext', undefined)
    const player = radio.createPlayer(ctx)
    await player.play()
    audio.fire('error')
    await vi.waitFor(() => expect(player.status.get()).toBe('error'))
    expect(player.lastError.get()).toContain('network error (')
    expect(posted).toHaveLength(1)
    expect(posted[0].options.body.reason).toBe(player.lastError.get())
    expect(posted[0].options.body.url).toContain('stream.nightride.fm')
  })
})

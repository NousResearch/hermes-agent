import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { createRealtimeVoiceSession, getApiRequestConnection, getEnvVars, getStatus, getToolsets } from '@/hermes'
import { Wrench } from '@/lib/icons'
import { startLiveVoice } from '@/lib/live-voice/start'
import type { RealtimeVoiceSession } from '@/lib/realtime-voice'

import { healthError } from './health-error'
import { SectionHeading, SettingsContent } from './primitives'
import { RecoverySettings } from './recovery-settings'

interface Check {
  label: string
  detail: string
  state: 'idle' | 'ok' | 'error'
}

const initial: Record<string, Check> = {
  backend: { label: 'Współpracownik Hermes', detail: 'Jeszcze nie sprawdzono.', state: 'idle' },
  keys: { label: 'Klucze API', detail: 'Jeszcze nie sprawdzono.', state: 'idle' },
  microphone: { label: 'Mikrofon', detail: 'Uruchom test mikrofonu.', state: 'idle' },
  speaker: { label: 'Głośnik', detail: 'Odtwórz dźwięk i potwierdź, że go słyszysz.', state: 'idle' },
  tools: { label: 'Narzędzia', detail: 'Jeszcze nie sprawdzono.', state: 'idle' },
  voice: { label: 'Rozmowa głosowa', detail: 'Test łączy się z wybranym API głosowym.', state: 'idle' }
}

export function AssistantHealth() {
  const [checks, setChecks] = useState(initial)
  const [busy, setBusy] = useState(false)
  const [calling, setCalling] = useState(false)
  const [heard, setHeard] = useState(false)
  const call = useRef<RealtimeVoiceSession | null>(null)
  const generation = useRef(0)
  const tone = useRef<AudioContext | null>(null)
  const mic = useRef<MediaStream | null>(null)

  const patch = (id: string, state: Check['state'], detail: string) =>
    setChecks(prev => ({ ...prev, [id]: { ...prev[id]!, detail, state } }))

  useEffect(
    () => () => {
      generation.current++
      call.current?.stop()
      mic.current?.getTracks().forEach(track => track.stop())
      void tone.current?.close()
    },
    []
  )

  const scan = async () => {
    setBusy(true)
    const run = generation.current

    const check = async (id: string, work: () => Promise<string>) => {
      try {
        const detail = await work()

        if (run === generation.current) {
          patch(id, id === 'keys' ? 'idle' : 'ok', detail)
        }
      } catch (error) {
        if (run === generation.current) {
          patch(id, 'error', healthError(error))
        }
      }
    }

    await Promise.all([
      check('backend', async () => {
        await getStatus()

        return 'Backend odpowiada. Można sprawdzić rozmowę.'
      }),
      check('keys', async () => {
        const env = await getEnvVars()

        const configured = ['OPENROUTER_API_KEY', 'GEMINI_API_KEY', 'GOOGLE_API_KEY', 'OPENAI_API_KEY'].filter(
          key => env[key]?.is_set
        )

        if (!configured.length) {
          throw new Error('Nie zapisano kluczy API. Dodaj je w ustawieniach dostawców.')
        }

        return `Zapisane: ${configured.map(key => key.replace('_API_KEY', '')).join(', ')}. Sam zapis nie potwierdza ważności klucza; test rozmowy sprawdzi wybrane API głosowe.`
      }),
      check('tools', async () => {
        const tools = await getToolsets()
        const ready = tools.filter(tool => tool.enabled && tool.configured)
        const missing = tools.filter(tool => tool.enabled && !tool.configured)

        if (!ready.length) {
          throw new Error('Brak skonfigurowanych aktywnych narzędzi. Otwórz Narzędzia i dokończ konfigurację.')
        }

        return `Gotowe zestawy: ${ready.map(tool => tool.label).join(', ')}.${missing.length ? ` Wymagają konfiguracji: ${missing.map(tool => tool.label).join(', ')}.` : ''} Dostępność nie jest testem wykonania zadania.`
      })
    ])

    if (run === generation.current) {
      setBusy(false)
    }
  }

  const microphone = async () => {
    const run = generation.current

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })

      if (run !== generation.current) {
        stream.getTracks().forEach(track => track.stop())

        return
      }

      mic.current = stream
      patch('microphone', 'ok', 'Dostęp do mikrofonu przyznany. Test rozmowy sprawdzi, czy głos jest rozpoznawany.')
      stream.getTracks().forEach(track => track.stop())
      mic.current = null
    } catch (error) {
      patch('microphone', 'error', healthError(error))
    }
  }

  const speaker = async () => {
    const run = generation.current

    try {
      await tone.current?.close()

      if (run !== generation.current) {
        return
      }

      const audio = new AudioContext()
      tone.current = audio
      await audio.resume()

      if (run !== generation.current) {
        await audio.close()

        return
      }

      const oscillator = audio.createOscillator()
      const gain = audio.createGain()
      gain.gain.setValueAtTime(0.08, audio.currentTime)
      oscillator.frequency.value = 440
      oscillator.connect(gain).connect(audio.destination)
      oscillator.start()
      oscillator.stop(audio.currentTime + 0.6)

      oscillator.onended = () => {
        void audio.close()

        if (tone.current === audio) {
          tone.current = null
        }
      }

      setHeard(true)
      patch('speaker', 'idle', 'Dźwięk odtworzony. Potwierdź poniżej, czy było go słychać.')
    } catch (error) {
      patch('speaker', 'error', healthError(error))
    }
  }

  const conversation = async () => {
    if (calling) {
      generation.current++
      call.current?.stop()
      call.current = null
      setCalling(false)

      return
    }

    setCalling(true)
    const run = generation.current
    patch('voice', 'idle', 'Łączenie. Powiedz: „Cześć Czesiek, czy mnie słyszysz?”. Test korzysta z Twojego API.')

    try {
      const session = await startLiveVoice(
        {
          onAsk: async () => 'To tylko test dźwięku. Nie wykonuj zadań. Potwierdź, że słyszysz użytkownika.',
          onDelegate: async () => 'To tylko test dźwięku. Nie uruchamiaj współpracowników, tylko potwierdź odsłuch.',
          onError: error => {
            if (run !== generation.current) {
              return
            }

            generation.current++
            patch('voice', 'error', healthError(error))
            call.current?.stop()
            call.current = null
            setCalling(false)
          },
          onStatus: status => {
            if (run !== generation.current) {
              return
            }

            if (status === 'listening') {
              patch('voice', 'ok', 'Połączenie głosowe aktywne. Powiedz coś i sprawdź odpowiedź.')
            }
          },
          onTranscript: (role, text) => {
            if (run === generation.current && text) {
              patch(
                role === 'user' ? 'microphone' : 'voice',
                'ok',
                role === 'user'
                  ? 'API rozpoznało Twoją wypowiedź.'
                  : 'API wygenerowało odpowiedź. Potwierdź odsłuch na swoim głośniku.'
              )
            }
          }
        },
        { createSession: createRealtimeVoiceSession }
      )

      if (run !== generation.current) {
        session.stop()
      } else {
        call.current = session
      }
    } catch (error) {
      if (run === generation.current) {
        patch('voice', 'error', healthError(error))
        setCalling(false)
      }
    }
  }

  const repair = async () => {
    setBusy(true)

    try {
      if (!getApiRequestConnection()) {
        await window.hermesDesktop.resetBootstrap()
      }

      window.location.reload()
    } catch (error) {
      patch('backend', 'error', healthError(error))
      setBusy(false)
    }
  }

  return (
    <SettingsContent>
      <SectionHeading icon={Wrench} title="Czy wszystko działa?" />
      <p className="mb-4 text-muted-foreground">
        Sprawdź połączenie i dźwięk. Test nie pokazuje ani nie kopiuje kluczy API.
      </p>
      <div className="flex flex-wrap gap-2">
        <Button disabled={busy || calling} onClick={() => void scan()}>
          Sprawdź stan
        </Button>
        <Button disabled={busy} onClick={() => void conversation()} variant="secondary">
          {calling ? 'Zakończ test' : 'Przetestuj rozmowę'}
        </Button>
        <Button disabled={busy || calling} onClick={() => void repair()} variant="secondary">
          Napraw połączenie
        </Button>
      </div>
      <p className="text-sm text-muted-foreground">
        Naprawa ponownie uruchamia połączenie. Nie usuwa kluczy, pamięci ani rozmów. Przed naprawą zakończ aktywne
        zadania.
      </p>
      <div aria-live="polite" className="space-y-4">
        {Object.entries(checks).map(([id, check]) => (
          <section className="rounded-xl border border-(--stroke-nous) p-4" key={id}>
            <h3 className="font-semibold">
              {check.label} · {{ ok: 'Sprawdzone', error: 'Wymaga uwagi', idle: 'Do sprawdzenia' }[check.state]}
            </h3>
            <p className="mt-2 text-sm text-muted-foreground">{check.detail}</p>
          </section>
        ))}
      </div>
      <div className="flex flex-wrap gap-2">
        <Button disabled={calling} onClick={() => void microphone()} variant="secondary">
          Sprawdź mikrofon
        </Button>
        <Button onClick={() => void speaker()} variant="secondary">
          Odtwórz dźwięk
        </Button>
        {heard && (
          <Button
            onClick={() => {
              patch('speaker', 'ok', 'Użytkownik potwierdził słyszalny dźwięk.')
              setHeard(false)
            }}
            variant="secondary"
          >
            Słyszę dźwięk
          </Button>
        )}
      </div>
      <RecoverySettings />
    </SettingsContent>
  )
}

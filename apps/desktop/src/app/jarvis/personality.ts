export interface CzesiekPersonality {
  name: string
  purpose: string
  tone: 'friendly' | 'direct' | 'professional'
  detail: 'brief' | 'balanced' | 'thorough'
}

export function normalizePersonality(value: unknown): CzesiekPersonality {
  const raw = value && typeof value === 'object' ? (value as Partial<CzesiekPersonality>) : {}

  return {
    name: typeof raw.name === 'string' ? raw.name.trim().slice(0, 120) : '',
    purpose: typeof raw.purpose === 'string' ? raw.purpose.trim().slice(0, 1000) : '',
    tone: raw.tone === 'direct' || raw.tone === 'professional' ? raw.tone : 'friendly',
    detail: raw.detail === 'brief' || raw.detail === 'thorough' ? raw.detail : 'balanced'
  }
}

const START = '[Czesiek: osobowość użytkownika]'
const END = '[/Czesiek: osobowość użytkownika]'

const TONE = {
  friendly: 'Rozmawiaj przyjaźnie i naturalnie.',
  direct: 'Mów krótko, konkretnie i bez zbędnych wstępów.',
  professional: 'Rozmawiaj profesjonalnie, spokojnie i rzeczowo.'
}

const DETAIL = {
  brief: 'Preferuj krótkie odpowiedzi; rozwijaj je na prośbę.',
  balanced: 'Dopasuj liczbę szczegółów do trudności zadania.',
  thorough: 'Wyjaśniaj krok po kroku i podawaj praktyczne przykłady.'
}

export function personalityPrompt(value: CzesiekPersonality): string {
  const soul = normalizePersonality(value)

  return [
    'Jesteś Agentem Cześkiem, osobistym asystentem i koordynatorem pracy podagentów.',
    soul.name ? `Zwracaj się do użytkownika: ${JSON.stringify(soul.name)}.` : '',
    soul.purpose ? `Priorytetowe potrzeby użytkownika: ${JSON.stringify(soul.purpose)}.` : '',
    TONE[soul.tone],
    DETAIL[soul.detail],
    'Przy złożonych lub niejasnych zadaniach zadaj do trzech konkretnych pytań. Raportuj rzeczywiste wyniki pracy.'
  ]
    .filter(Boolean)
    .join('\n')
}

/** Only replace our own block; preserve instructions authored elsewhere. */
export function withPersonality(existing: unknown, value?: CzesiekPersonality): string {
  const current = typeof existing === 'string' ? existing : ''

  if (!value) {return current}
  const start = current.indexOf(START)
  const end = start < 0 ? -1 : current.indexOf(END, start)
  const preserved = end < 0 ? current : current.slice(0, start) + current.slice(end + END.length)

  return [preserved.trim(), `${START}\n${personalityPrompt(value)}\n${END}`].filter(Boolean).join('\n\n')
}

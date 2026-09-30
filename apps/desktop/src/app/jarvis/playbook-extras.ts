/**
 * Tips that depend on what is actually connected: mail and calendar only once Google is, "find it in
 * my files" only once there is a vault. Like the main playbook nothing is offered that would fail the
 * moment the user tries it — these are the "and now that it can reach X" suggestions.
 */

export type ExtraTipNeed = 'browser' | 'google' | 'vault'

export type ExtraTipId =
  | 'browser.gmail'
  | 'google.calendar'
  | 'google.draft'
  | 'google.inbox'
  | 'vault.find'
  | 'vault.weekly'
  | 'voice.daddy'

export interface ExtraTip {
  id: ExtraTipId
  /** Always offered when absent. */
  needs?: ExtraTipNeed
}

export const EXTRA_TIPS: readonly ExtraTip[] = [
  { id: 'voice.daddy' },
  { id: 'google.inbox', needs: 'google' },
  { id: 'google.calendar', needs: 'google' },
  { id: 'google.draft', needs: 'google' },
  { id: 'vault.find', needs: 'vault' },
  { id: 'vault.weekly', needs: 'vault' },
  { id: 'browser.gmail', needs: 'browser' }
]

export interface ExtraTipContext {
  browserSignedIn: boolean
  googleConnected: boolean
  vaultExists: boolean
}

export function selectExtraTips(context: ExtraTipContext): ExtraTip[] {
  const have: Record<ExtraTipNeed, boolean> = {
    browser: context.browserSignedIn,
    google: context.googleConnected,
    vault: context.vaultExists
  }

  return EXTRA_TIPS.filter(tip => !tip.needs || have[tip.needs])
}

export interface ExtraTipCopy {
  detail: string
  prompt: string
  title: string
}

export const EXTRA_TIP_COPY: Record<'en' | 'pl', Record<ExtraTipId, ExtraTipCopy>> = {
  en: {
    'browser.gmail': { detail: 'It works in its own signed-in browser.', prompt: 'Open Gmail in your browser and tell me what needs a reply today.', title: 'Check Gmail in the browser' },
    'google.calendar': { detail: 'Today and tomorrow, with what to prepare.', prompt: 'What is on my calendar today and tomorrow? Tell me what I should prepare for each meeting.', title: 'What is on my calendar' },
    'google.draft': { detail: 'A draft you read before anything is sent.', prompt: 'Find the latest mail that needs a reply and write a draft answer. Do not send it.', title: 'Draft a reply for me' },
    'google.inbox': { detail: 'Unread mail sorted by what is urgent.', prompt: 'Go through my unread mail and tell me what is urgent, in one line each.', title: 'Sort my inbox' },
    'vault.find': { detail: 'Searches your notes and indexed Drive files.', prompt: 'Find the contract with client X in my notes and Drive files and tell me the key terms.', title: 'Find a document in my files' },
    'vault.weekly': { detail: 'What was done, what is stuck, what to do next.', prompt: 'Read my notes from this week and write a review: what got done, what is stuck, what I should do next.', title: 'Weekly review from my notes' },
    'voice.daddy': { detail: 'Czesiek plays its theme and tells your daily report.', prompt: '', title: 'Say “Tatuś wrócił”' }
  },
  pl: {
    'browser.gmail': { detail: 'Pracuje we własnej, zalogowanej przeglądarce.', prompt: 'Otwórz Gmaila w swojej przeglądarce i powiedz mi, co dziś wymaga odpowiedzi.', title: 'Sprawdź Gmaila w przeglądarce' },
    'google.calendar': { detail: 'Dziś i jutro, z tym, co przygotować.', prompt: 'Co mam dziś i jutro w kalendarzu? Powiedz, co przygotować na każde spotkanie.', title: 'Co mam w kalendarzu' },
    'google.draft': { detail: 'Szkic, który czytasz, zanim cokolwiek wyjdzie.', prompt: 'Znajdź ostatni mail wymagający odpowiedzi i napisz szkic odpowiedzi. Nie wysyłaj go.', title: 'Napisz za mnie szkic odpowiedzi' },
    'google.inbox': { detail: 'Nieprzeczytane maile według pilności.', prompt: 'Przejrzyj moje nieprzeczytane maile i powiedz, co jest pilne, po jednej linijce.', title: 'Ogarnij moją skrzynkę' },
    'vault.find': { detail: 'Przeszukuje Twoje notatki i zaindeksowane pliki z Dysku.', prompt: 'Znajdź umowę z klientem X w moich notatkach i plikach z Dysku i podaj najważniejsze warunki.', title: 'Znajdź dokument w moich plikach' },
    'vault.weekly': { detail: 'Co zrobione, co utknęło, co dalej.', prompt: 'Przeczytaj moje notatki z tego tygodnia i napisz przegląd: co zrobiono, co utknęło, co zaproponować.', title: 'Tygodniowy przegląd z notatek' },
    'voice.daddy': { detail: 'Czesiek zagra swój motyw i opowie raport dnia.', prompt: '', title: 'Powiedz „Tatuś wrócił”' }
  }
}

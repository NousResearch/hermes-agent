export function healthError(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error)

  if (/NotAllowed|permission denied|permission.*microphone/i.test(message)) {
    return 'Brak dostępu do mikrofonu. Zezwól na mikrofon w ustawieniach prywatności Windows.'
  }

  if (/NotFound|no.*microphone|device.*not.*found/i.test(message)) {
    return 'Nie znaleziono mikrofonu. Podłącz go i wybierz urządzenie wejściowe.'
  }

  if (/401|403|invalid.*key|API_KEY_INVALID|unauthorized|permission_denied/i.test(message)) {
    return 'Wybrane API odrzuciło dostęp. Sprawdź klucz i uprawnienia projektu w ustawieniach dostawcy.'
  }

  if (/429|quota|rate.limit/i.test(message)) {
    return 'Dostawca zgłosił limit użycia lub brak środków. Sprawdź rozliczenia swojego API.'
  }

  if (/timeout|timed out|network|fetch|connection|ECONN/i.test(message)) {
    return 'Nie udało się połączyć. Sprawdź internet; jeśli problem dotyczy Hermesa, użyj „Napraw połączenie”.'
  }

  // Provider errors can contain URLs or credential fragments; do not echo them.
  if (message.startsWith('Nie zapisano') || message.startsWith('Brak skonfigurowanych')) {
    return message
  }

  return 'Test nie powiódł się. Sprawdź ustawienia dostawcy i spróbuj ponownie. Nie zmieniono Twoich danych.'
}

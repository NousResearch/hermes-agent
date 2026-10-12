import type { TranslationOverrides } from './define-locale'

export const deOnboarding: TranslationOverrides['onboarding'] = {
  headerTitle: 'Hermes Agent für Sie einrichten',
  headerDesc:
    'Verbinden Sie einen Modell-Anbieter, um mit dem Chatten zu beginnen. Die meisten Optionen brauchen nur einen Klick.',
  preparingInstall:
    'Hermes schließt die Installation ab. Das dauert beim ersten Start normalerweise unter einer Minute.',
  starting: 'Hermes wird gestartet…',
  setupSlowTitle: 'Die Einrichtung dauert länger als üblich.',
  setupSlowBody: 'Hermes wird im Hintergrund noch gestartet.',
  continueWithoutSetup: 'Ohne Einrichtung fortfahren',
  lookingUpProviders: 'Anbieter werden gesucht...',
  collapse: 'Einklappen',
  otherProviders: 'Andere Anbieter',
  haveApiKey: 'Ich habe bereits einen API-Key',
  chooseLater: 'Ich wähle später einen Anbieter',
  recommended: 'Empfohlen',
  connected: 'Verbunden',
  featuredPitch: 'Ein Abo, 300+ Frontier-Modelle – die empfohlene Art, Hermes zu nutzen',
  fireworksPitch: 'Direkte Model-API – Fireworks-gehostete Frontier-Modelle',
  localModelsTitle: 'Modelle lokal ausführen',
  localModelsPitch: 'Kein Konto nötig – laden Sie ein Modell herunter und führen Sie es auf diesem Rechner aus',
  openRouterPitch: 'Ein Key, hunderte Modelle – ein solider Standard',
  apiKeyOptions: {
    fireworks: {
      short: 'direkte Model-API',
      description: 'Direkter Zugriff auf Modelle, die von Fireworks AI gehostet werden.'
    },
    openrouter: {
      short: 'ein Key, viele Modelle',
      description: 'Hostet hunderte Modelle hinter einem einzigen Key. Guter Standard für neue Installationen.'
    },
    openai: {
      short: 'GPT-Klasse-Modelle',
      description: 'Direkter Zugriff auf OpenAI-Modelle.'
    },
    gemini: {
      short: 'Gemini-Modelle',
      description: 'Direkter Zugriff auf Google-Gemini-Modelle.'
    },
    xai: {
      short: 'Grok-Modelle',
      description: 'Direkter Zugriff auf xAI-Grok-Modelle.'
    },
    local: {
      short: 'selbst gehostet',
      description:
        'Verbinden Sie Hermes mit einem lokalen oder selbst gehosteten OpenAI-kompatiblen Endpunkt (vLLM, llama.cpp, Ollama usw.).'
    }
  },
  backToSignIn: 'Zurück zur Anmeldung',
  getKey: 'Einen Key holen',
  replaceCurrent: 'Aktuellen Wert ersetzen',
  pasteApiKey: 'API-Key einfügen',
  localApiKeyPlaceholder: 'API-Key (optional – nur falls Ihr Endpunkt einen benötigt)',
  localModelNamePlaceholder: 'Modellname (z. B. command-a-plus-05-2026)',
  couldNotSave: 'Anmeldedaten konnten nicht gespeichert werden.',
  connecting: 'Verbinden',
  update: 'Aktualisieren',
  flowSubtitles: {
    pkce: 'Öffnet Ihren Browser zur Anmeldung und fährt dann hier fort',
    device_code: 'Öffnet eine Verifizierungsseite in Ihrem Browser – Hermes verbindet sich automatisch',
    external: 'Melden Sie sich einmal in Ihrem Terminal an und kehren Sie dann zum Chatten zurück'
  },
  startingSignIn: provider => `Anmeldung für ${provider} wird gestartet...`,
  verifyingCode: provider => `Ihr Code wird mit ${provider} überprüft…`,
  connectedProvider: provider => `${provider} verbunden`,
  connectedPicking: provider => `${provider} verbunden. Standardmodell wird ausgewählt...`,
  signInFailed: 'Anmeldung fehlgeschlagen. Versuchen Sie es erneut.',
  signInExpired:
    'Die Anmeldung ist beim Warten auf die Autorisierung abgelaufen. Meist bedeutet das, dass die Anmeldeseite im geöffneten Tab hängen geblieben ist (serverseitiges Problem) – schließen Sie die Anmeldung dort ab und versuchen Sie es dann erneut. Wenn es weiterhin fehlschlägt, verwenden Sie stattdessen einen API-Key oder die CLI als Alternative.',
  signInDidNotFinish: provider =>
    `Die Anmeldung bei ${provider} wurde nicht abgeschlossen. Prüfen Sie Ihre Internetverbindung und versuchen Sie es erneut, oder wählen Sie einen anderen Anbieter.`,
  tryAgain: 'Erneut versuchen',
  useApiKeyInstead: 'API-Key verwenden',
  errorDetails: 'Details',
  pickDifferentProvider: 'Einen anderen Anbieter wählen',
  pickDifferentAccount: 'Ein anderes Konto wählen',
  signInWith: provider => `Mit ${provider} anmelden`,
  openedBrowser: provider => `Wir haben ${provider} in Ihrem Browser geöffnet.`,
  authorizeThere: 'Autorisieren Sie Hermes dort.',
  copyAuthCode: 'Kopieren Sie den Autorisierungscode und fügen Sie ihn unten ein.',
  pasteAuthCode: 'Autorisierungscode einfügen',
  reopenAuthPage: 'Autorisierungsseite erneut öffnen',
  waitingAuthorize: 'Warten auf Ihre Autorisierung…',
  externalPending: provider =>
    `${provider} meldet sich über seine eigene CLI an. Führen Sie diesen Befehl in einem Terminal aus, kehren Sie dann zurück und wählen Sie „Ich habe mich angemeldet“:`,
  signedIn: 'Ich habe mich angemeldet',
  deviceCodeOpened: provider => `Wir haben ${provider} in Ihrem Browser geöffnet. Geben Sie dort diesen Code ein:`,
  reopenVerification: 'Verifikationsseite erneut öffnen',
  copy: 'Kopieren',
  defaultModel: 'Standardmodell',
  freeTier: 'Free-Tier',
  pro: 'Pro',
  free: 'Kostenlos',
  price: (input, output) => `${input} rein / ${output} raus pro Mtok`,
  change: 'Ändern',
  startChatting: 'Loslegen',
  docs: provider => `${provider}-Doku`,
  modeChoiceTitle: 'Wie möchten Sie Hermes nutzen?',
  modeChoiceDesc: 'Wählen Sie, was sich für Sie richtig anfühlt. Sie können jederzeit in den Einstellungen wechseln.',
  modeChoiceSimpleDesc: 'Einfach mit Hermes sprechen. Hermes wählt die besten Einstellungen für Sie.',
  modeChoiceAdvancedDesc: 'Alle Einstellungen, dazu Terminal, Dateien und Code-Review.',
  simpleHeaderDesc: 'Melden Sie sich mit einem Konto an, das Sie schon haben. Den Rest richtet Hermes ein.',
  simpleTiles: {
    nous: 'Kostenloses Konto',
    chatgpt: 'Erfordert ChatGPT Plus oder Pro',
    grok: 'Erfordert SuperGrok oder X Premium+',
    claude: 'Erfordert Claude Max. Anthropic rechnet das als zusätzliche Nutzung ab.'
  },
  simpleDeviceCodeOpened: 'Ihr Browser wurde geöffnet. Bestätigen Sie Hermes dort und kommen Sie dann hierher zurück.',
  simpleDeviceCodeCopied: 'Falls nach einem Code gefragt wird, fügen Sie diesen ein. Er ist bereits kopiert.',
  terminalSignInOpened: (paste, enter) =>
    `Ein Terminalfenster wurde geöffnet. Bestätigen Sie Hermes in Ihrem Browser und kopieren Sie den angezeigten Code. Klicken Sie dann in das Terminalfenster, drücken Sie ${paste} und ${enter} und kommen Sie hierher zurück.`,
  openTerminalAgain: 'Terminal erneut öffnen',
  simpleSignInNotFinished: provider =>
    `Hermes sieht Ihre Anmeldung bei ${provider} noch nicht. Schließen Sie sie ab und versuchen Sie es erneut, oder wählen Sie ein anderes Konto.`,
  simpleNoSignIns: 'Hermes konnte die Anmeldeoptionen nicht laden. Versuchen Sie es gleich noch einmal.',
  simpleDefaultsFailed: 'Die Sicherheitseinstellungen des Modus „Einfach“ konnten nicht aktiviert werden',
  simpleDefaultsRemote:
    'Hermes läuft auf einem anderen Computer, daher wurden die Sicherheitseinstellungen des Modus „Einfach“ dort nicht hinzugefügt.'
}

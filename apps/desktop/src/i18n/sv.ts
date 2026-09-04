import { defineLocale } from './define-locale'

// Swedish desktop translation. Missing new keys deliberately fall back to
// English while the catalogue catches up with the rapidly changing UI.
export const sv = defineLocale({
  common: {
    apply: 'Tillämpa',
    back: 'Tillbaka',
    save: 'Spara',
    saving: 'Sparar…',
    cancel: 'Avbryt',
    change: 'Ändra',
    choose: 'Välj',
    clear: 'Rensa',
    close: 'Stäng',
    collapse: 'Fäll ihop',
    confirm: 'Bekräfta',
    connect: 'Anslut',
    connecting: 'Ansluter',
    continue: 'Fortsätt',
    copied: 'Kopierat',
    copy: 'Kopiera',
    copyFailed: 'Det gick inte att kopiera',
    delete: 'Ta bort',
    docs: 'Dokumentation',
    done: 'Klar',
    error: 'Fel',
    expand: 'Fäll ut',
    failed: 'Misslyckades',
    formatJson: 'Formatera JSON',
    free: 'Ledigt',
    loading: 'Läser in…',
    notSet: 'Inte angivet',
    refresh: 'Uppdatera',
    remove: 'Ta bort',
    replace: 'Ersätt',
    retry: 'Försök igen',
    run: 'Kör',
    send: 'Skicka',
    set: 'Ange',
    skip: 'Hoppa över',
    update: 'Uppdatera',
    tryHint: term => `Prova ”${term}”`,
    on: 'På',
    off: 'Av'
  },
  fileMenu: {
    revealFinder: 'Visa i Finder',
    revealExplorer: 'Visa i Utforskaren',
    revealFileManager: 'Öppna innehållande mapp',
    revealInSidebar: 'Visa i filträdet',
    copyPath: 'Kopiera sökväg',
    copyRelativePath: 'Kopiera relativ sökväg',
    download: 'Hämta',
    downloadSaved: 'Sparad',
    downloadFailed: 'Hämtningen misslyckades',
    rename: 'Byt namn…',
    delete: 'Ta bort',
    renameTitle: 'Byt namn',
    renameLabel: 'Nytt namn',
    deleteTitle: name => `Ta bort ”${name}”?`,
    deleteBody: 'Detta kan inte ångras.',
    pathCopied: 'Sökvägen kopierades'
  },
  boot: {
    ready: 'Hermes Desktop är redo',
    desktopBootFailedWithMessage: message => `Starten av skrivbordsappen misslyckades: ${message}`,
    steps: { connectingGateway: 'Ansluter direkt till skrivbordsgateway', loadingSettings: 'Läser in Hermes-inställningar', loadingSessions: 'Läser in senaste sessionerna', retryingRemoteBackend: 'Återansluter till Hermes fjärrserverdel…', startingDesktopConnection: 'Startar skrivbordsanslutning', startingHermesDesktop: 'Startar Hermes Desktop…' },
    errors: { backgroundExited: 'Hermes bakgrundsprocess avslutades.', backgroundExitedDuringStartup: 'Hermes bakgrundsprocess avslutades under start.', backendStopped: 'Serverdelen stoppades', desktopBootFailed: 'Starten av skrivbordsappen misslyckades', gatewayConnectionLost: 'Anslutningen till gatewayen bröts', gatewayConnectionLostDetail: 'Försöker fortfarande igen i bakgrunden. Du kan fortsätta läsa och skriva utkast — öppna gateway-inställningarna om det kvarstår.', gatewaySignInRequired: 'Inloggning till gateway krävs', ipcBridgeUnavailable: 'Skrivbordsappens IPC-brygga är inte tillgänglig.' },
    failure: { title: 'Hermes kunde inte starta', description: 'Gatewayen i bakgrunden startade inte. Prova ett av återställningsstegen nedan. Inget här tar bort dina chattar eller inställningar.', remoteTitle: 'Inloggning till fjärrgateway krävs', remoteDescription: 'Din session för fjärrgatewayen har löpt ut. Logga in igen för att återansluta. Inget här tar bort dina chattar eller inställningar.', retry: 'Försök igen', repairInstall: 'Reparera installation', useLocalGateway: 'Använd lokal gateway', gatewaySettings: 'Gateway-inställningar', back: 'Tillbaka', openLogs: 'Öppna loggar', repairHint: 'Reparation kör installationsprogrammet på nytt och kan ta några minuter på en ny dator.', remoteSignInHint: signInLabel => `Loggar ut från den sparade fjärrsessionen i webbläsaren och öppnar sedan ${signInLabel}. Använd lokal gateway för att växla till den medföljande serverdelen i stället.`, signOutAndSignIn: 'Logga ut och logga in', remoteFailureHint: 'Kontrollera gatewayens URL och inloggning under Gateway-inställningar, eller växla till den lokala gatewayen.', cloudDownTitle: 'Nous Cloud-agenten är nere', cloudDownDescription: 'Den Nous-hanterade molnagent som gatewayen ansluter till returnerar ett serverfel. Den kan inte startas om härifrån — kontrollera dess status, växla till den lokala gatewayen eller be om support.', cloudDownHint: 'Knapparna nedan öppnar Nous Portal (instansstatus och kontroller) och vår Discord för support.', cloudDownCheckPortal: 'Kontrollera Portal-status', cloudDownDiscord: 'Få hjälp på Discord', hideRecentLogs: 'Dölj senaste loggar', showRecentLogs: 'Visa senaste loggar', signedInTitle: 'Inloggad', signedInMessage: 'Återansluter till fjärrgatewayen…', signInIncompleteTitle: 'Inloggningen slutfördes inte', signInIncompleteMessage: 'Inloggningsfönstret stängdes innan autentiseringen slutfördes.', signInFailed: 'Inloggningen misslyckades', signInToRemoteGateway: 'Logga in till fjärrgateway', signInWithProvider: provider => `Logga in med ${provider}`, identityProvider: 'din identitetsleverantör' }
  },
  notifications: {
    region: 'Aviseringar', hide: 'Dölj', show: 'Visa', more: count => `${count} ytterligare ${count === 1 ? 'avisering' : 'aviseringar'}`, clearAll: 'Rensa alla', dismiss: 'Avfärda avisering', details: 'Detaljer', copyDetail: 'Kopiera detalj', copyDetailFailed: 'Det gick inte att kopiera aviseringsdetalj', backendOutOfDateTitle: 'Serverdelen är inaktuell', backendOutOfDateMessage: 'Din Hermes-serverdel är äldre än denna skrivbordsversion och kanske inte fungerar korrekt. Uppdatera dem så att de matchar.', installMethodUnsupportedTitle: 'Installationsmetoden stöds inte', updateHermes: 'Uppdatera Hermes', updateReadyTitle: 'Uppdatering redo', updateReadyMessage: count => `${count} ny${count === 1 ? '' : 'a'} ändring${count === 1 ? '' : 'ar'} är tillgänglig${count === 1 ? '' : 'a'}.`, updateReadyMessageUnknown: 'En ny uppdatering är tillgänglig.', seeWhatsNew: 'Se vad som är nytt',
    mcp: { needsAuthTitle: 'MCP-server behöver autentiseras igen', needsAuthMessage: name => `${name} MCP behöver autentiseras igen.`, errorTitle: 'MCP-server kan inte nås', errorMessage: name => `${name} MCP klarade inte sin hälsokontroll.`, signIn: 'Logga in', view: 'Visa' },
    errors: { elevenLabsNeedsKey: 'ElevenLabs STT behöver ELEVENLABS_API_KEY.', elevenLabsRejectedKey: 'ElevenLabs avvisade API-nyckeln (401).', diskFull: 'Disken är full — frigör utrymme och försök igen.', gatewayAuthFailed: 'Gateway-autentisering misslyckades — kontrollera din API_SERVER_KEY.', methodNotAllowed: 'Skrivbordsserverdelen avvisade begäran (405 Method Not Allowed). Försök att starta om Hermes Desktop.', microphonePermission: 'Behörighet till mikrofonen nekades.', openaiRejectedApiKey: 'OpenAI avvisade API-nyckeln.', openaiRejectedApiKeyWithStatus: status => `OpenAI avvisade API-nyckeln (${status} invalid_api_key).`, openaiTtsNeedsKey: 'OpenAI TTS behöver VOICE_TOOLS_OPENAI_KEY eller OPENAI_API_KEY.', codeSkewRestartRequired: 'Den här serverdelen kör gammal kod efter en uppdatering. Starta om den för att läsa in den nya koden.' },
    voice: { configureSpeechToText: 'Konfigurera tal till text för att använda röstläge.', couldNotStartSession: 'Det gick inte att starta röstsession', microphoneAccessDenied: 'Åtkomst till mikrofonen nekades.', microphoneConstraintsUnsupported: 'Mikrofonbegränsningar stöds inte av denna enhet.', microphoneFailed: 'Mikrofonen misslyckades', microphoneInUse: 'Mikrofonen används redan av en annan app.', microphonePermissionDenied: 'Behörighet till mikrofonen nekades.', microphoneStartFailed: 'Det gick inte att starta mikrofoninspelning.', microphoneUnsupported: 'Den här körmiljön stöder inte mikrofoninspelning.', noMicrophone: 'Ingen mikrofon hittades.', noSpeechDetected: 'Inget tal upptäcktes', playbackFailed: 'Röstuppspelning misslyckades', recordingFailed: 'Röstinspelning misslyckades', sayStopToEnd: phrase => `Säg ”${phrase}” för att avsluta röstchatten.`, transcriptionFailed: 'Taligenkänning misslyckades', transcriptionUnavailable: 'Taligenkänning är inte tillgänglig ännu.', tryRecordingAgain: 'Försök spela in igen.', unavailable: 'Röst är inte tillgänglig' },
    native: { approvalTitle: 'Godkännande krävs', approveAction: 'Godkänn', rejectAction: 'Avvisa', inputTitle: 'Inmatning krävs', inputBody: 'Hermes väntar på ditt svar.', turnDoneTitle: 'Hermes är klar', turnDoneBody: '', turnErrorTitle: 'Körningen misslyckades', backgroundDoneTitle: 'Bakgrundsuppgiften är klar', backgroundFailedTitle: 'Bakgrundsuppgiften misslyckades', creditsTitle: 'Krediter' }
  },
  remoteDisplayBanner: { message: reason => `Programvarurendering är aktiv — fjärrskärm upptäcktes (${reason}). GPU-acceleration är inaktiverad för att förhindra flimmer.` },
  billingBlock: { titleNous: 'Slut på Nous-krediter', titleProvider: provider => `Slut på krediter — ${provider}`, fallbackMessage: 'Ditt konto har slut på krediter. Lägg till krediter för att fortsätta.', openBilling: 'Öppna fakturering', addCredits: 'Lägg till krediter', dismiss: 'Avfärda' },
  sendDiagnostics: {
    title: 'Skicka diagnostik till Nous',
    privacyNotice: 'Detta laddar upp ett felsökningspaket till Nous interna lagring (inte en offentlig inklistring). Det innehåller systeminformation (operativsystem, versioner, leverantör, vilka API-nycklar som är konfigurerade — aldrig själva nycklarna) och fullständiga agent-, gateway- och skrivbordsloggar (upp till 512 kB vardera), som sannolikt innehåller konversationsinnehåll, verktygsutdata och filsökvägar. Hemligheter maskeras före uppladdning. Paketet kan endast ses av Nous personal och tillåtna Discord-moderatorer och raderas automatiskt efter 14 dagar.',
    upload: 'Ladda upp', uploading: 'Laddar upp…', cancel: 'Avbryt', close: 'Stäng', copyLink: 'Kopiera länk',
    uploadIdFallback: id => `Ingen visningslänk returnerades — uppge uppladdnings-id ${id} till supporten`,
    doneTitle: 'Diagnostik skickad', doneDescription: 'Ditt paket laddades upp privat. Dela länken nedan i din supporttråd så att teamet kan se dina loggar.',
    failedTitle: 'Uppladdningen misslyckades', failedHint: 'Du kan också köra `hermes debug share --nous` i en terminal, eller `hermes debug share --local` för att skriva ut rapporten utan uppladdning.',
    handoffLead: 'Fortsätt diskussionen i:', links: { github: 'GitHub-ärenden', portal: 'Nous Portal-support', discord: 'Discord' }
  },
  titlebar: {
    hideSidebar: 'Dölj sidofält', showSidebar: 'Visa sidofält', search: 'Sök', searchTitle: 'Sök sessioner, vyer och åtgärder', swapSidebarSides: 'Byt sida för sidofält', hideRightSidebar: 'Dölj högra sidofältet', showRightSidebar: 'Visa högra sidofältet',
    unreadSessions: count => count === 1 ? '1 oläst session' : `${count} olästa sessioner`, muteHaptics: 'Stäng av haptik', unmuteHaptics: 'Slå på haptik', openSettings: 'Öppna inställningar', openStarmap: 'Öppna minnesgraf', enterHud: 'HUD-läge', exitHud: 'Avsluta HUD-läge', resetHudLayout: 'Återställ HUD-storlek och -position', layoutEditor: 'Layoutredigerare', layoutEditorTitle: mod => `Layoutredigerare — ${mod}-klick återställer layouten`
  },
  findInPage: { next: 'Nästa träff', previous: 'Föregående träff' },
  artifactCard: { kind: { code: 'Kod', html: 'Interaktiv sida', svg: 'Grafik' }, generating: lines => `Skapar… ${lines} rader`, versionBadge: count => `${count} versioner`, open: 'Öppna' },
  artifactPreview: { versionOf: (current, total) => `v${current} av ${total}`, olderVersion: 'Äldre version', newerVersion: 'Nyare version', latest: 'Senaste', copyContent: 'Kopiera innehåll', download: 'Hämta', openInBrowser: 'Öppna i webbläsaren', openInBrowserFailed: 'Det gick inte att öppna i webbläsaren', missingTitle: 'Artefakten är inte tillgänglig', missingBody: 'Denna artefakt finns inte längre i det lokala registret.' },
  modelVisibility: { title: 'Modeller', search: 'Sök modeller', noAuthenticatedProviders: 'Inga autentiserade leverantörer.', addProvider: 'Lägg till leverantör…' },
  contextMenu: {
    link: { openInApp: 'Öppna i appens webbläsare', openExternal: 'Öppna i extern webbläsare', copyUrl: 'Kopiera URL', copyResolvedUrl: 'Kopiera löst URL' },
    image: { copyImage: 'Kopiera bild', copyImageAddress: 'Kopiera bildadress', saveImageAs: 'Spara bild som…' },
    edit: { cut: 'Klipp ut', paste: 'Klistra in', selectAll: 'Markera allt', addToDictionary: 'Lägg till i ordlistan' },
    page: { copyPageUrl: 'Kopiera sidans URL', inspectElement: 'Inspektera element' }
  },
  prompts: { gatewayDisconnected: 'Hermes gateway är inte ansluten', sudoSendFailed: 'Det gick inte att skicka sudo-lösenordet', secretSendFailed: 'Det gick inte att skicka hemligheten', sudoTitle: 'Administratörslösenord', sudoDesc: 'Hermes behöver ditt sudo-lösenord för att köra ett privilegierat kommando. Det skickas bara till din lokala agent.', sudoPlaceholder: 'sudo-lösenord', secretTitle: 'Hemlighet krävs', secretDesc: 'Hermes behöver en inloggningsuppgift för att fortsätta.', secretPlaceholder: 'hemligt värde' },
  errors: { genericFailure: 'Något gick fel', boundaryTitle: 'Något gick sönder i gränssnittet', boundaryDesc: 'Vyn stötte på ett oväntat fel. Dina chattar och inställningar är säkra.', reloadWindow: 'Läs om fönster', openLogs: 'Öppna loggar' },
  ui: { search: { clear: 'Rensa sökning' }, pagination: { label: 'sidindelning', previous: 'Föregående', previousAria: 'Gå till föregående sida', next: 'Nästa', nextAria: 'Gå till nästa sida' }, sidebar: { title: 'Sidofält', description: 'Visar sidofältet på mobilen.', toggle: open => `${open ? 'Visa' : 'Dölj'} sidofält` } },
  language: { label: 'Språk', description: 'Välj språk för skrivbordsgränssnittet.', saving: 'Sparar språk…', saveError: 'Det gick inte att uppdatera språk', switchTo: 'Byt språk', searchPlaceholder: 'Sök språk…', noResults: 'Inga språk hittades' }
})

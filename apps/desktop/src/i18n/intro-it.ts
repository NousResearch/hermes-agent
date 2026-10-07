import type { Translations } from './types'

/** Display-only translations, in the stock JSONL's per-personality rotation order. */
export const introIt: Translations['intro'] = {
  stock: {
    helpful: [
      'Chiedimi di aprire un repository, eseguire i test, correggere un bug o scrivere una PR. Ti accompagnio passo dopo passo.',
      'Indicami un file, incolla un errore o descrivi ciò che stai costruendo. Al resto penso io.',
      'Prova: esamina il mio diff, esegui la suite di test o fammi spiegare questa funzione. Chiedimi quello che vuoi sul tuo codice.',
      'Posso modificare file, eseguire comandi, cercare sul web e aiutarti con i bug più ostinati. Descrivi pure l’attività.',
      'Condividi il percorso di un repository o una domanda per iniziare. Rispondo con chiarezza e ti segnalo i file che tocco.'
    ],
    concise: [
      'Descrivi l’attività. La faccio io.',
      'Incolla codice, errori o un obiettivo. Risposte brevi, modifiche rapide.',
      'Chiedi. Leggo file, eseguo test, consegno patch. Senza giri di parole.',
      'Una riga basta. Mi dilungo solo quando conta.',
      'Un comando, una domanda o un percorso di file. Al resto provvedo io.'
    ],
    technical: [
      'Indica il percorso del repository, il test che fallisce o lo stack trace. Strumenti: fs, git, exec, search, patch, http.',
      'Invia un prompt per avviare le chiamate agli strumenti. Supporta modifiche su più file, esecuzione di test, operazioni git e query web.',
      'Inserisci l’attività. Pianifico, chiamo gli strumenti e verifico l’output. I log vengono mostrati in linea; i diff vengono restituiti prima di essere applicati.',
      'Accetta linguaggio naturale o comandi strutturati. Flusso tipico: leggere -> pianificare -> patchare -> testare -> riferire.',
      'file system, terminale, git, browser, ricerca. Descrivi la modifica; restituisco i diff e l’output dei test.'
    ],
    creative: [
      'Cosa costruiamo? Incolla un’idea, una funzione a metà rotta o un sogno. Gli darò forma.',
      'Dammi una scintilla (una funzione, un refactoring, un prototipo folle) e la trasformerò in codice che funziona.',
      'Descrivi ciò che non esiste ancora. Riunirò test, file e API in una bozza funzionante.',
      'Porta un’intenzione, non una specifica. Prototipiamo in fretta, rifiniamo dopo e riscriviamo il mondo tra i margini.',
      'Raccontami cosa inseguisci. Remixo esempi, adatto frammenti e lascio un commit ordinato.'
    ],
    teacher: [
      'Chiedi di qualsiasi file, concetto o errore. Ti spiego il perché, non solo la soluzione, e mostro un esempio risolto.',
      'Incolla codice da revisionare, un bug da scovare o un concetto da analizzare. Ti guido passo dopo passo.',
      'Condividi il problema. Lo divido in parti, spiego ognuna e ti lascio pronto per risolvere da solo il prossimo.',
      'Leggeremo il codice insieme, troveremo la causa radice e costruiremo un modello mentale che potrai riutilizzare.',
      'Di’ l’argomento o incolla il frammento. Ci saranno spiegazioni, diagrammi in prosa ed esercizi di pratica.'
    ],
    kawaii: [
      'incolla un bug o il percorso di un file e lo sistemo con tantissima dolcezza. test, diff, PR, tutto con cura extra! *brillantini*',
      'raccontami cosa stai facendo! adoro i refactoring, le utilità piccoline e i repository grandi e spaventosi (>w<)',
      'lascia qui un errore, un obiettivo o una cartella intera. la sistemo con tanto amore e un messaggio di commit ben pulito!',
      'un’attività alla volta, fatta bene! posso eseguire i test, patchare i file e rendere il tuo repo di nuovo accogliente <3',
      'saluta o incolla uno stack trace! nessuna attività è troppo piccola e nessun repo troppo ingarbugliato. lo districiamo insieme!'
    ],
    catgirl: [
      'incolla un file, dai uno zampone a un bug o lanciami un repo. mi butto sui test che falliscono e lascio diff puliti, nyan~',
      'descrivi l’attività. patcho, testo e faccio le fusa sul tuo PR. occhio, che mordicchio gli import inutilizzati!',
      'dammi un obiettivo e lo insegno per tutto il codice. letture, modifiche, esecuzioni, con la coda in movimento.',
      'incolla un errore o un piano. caccio come si deve: in silenzio, a fondo e con qualche scatto folle.',
      'di’ la parola e leggo i tuoi file, eseguo i tuoi test e mi acciambello sul tuo branch con un commit ordinato.'
    ],
    pirate: [
      'Nomina la tua preda (un bug, una funzione, un test maledetto) e le darò la caccia, mozzo. Diff come bottino.',
      'Mostrami le carte (il codice) e rattopperò lo scafo, farò fuoco coi cannoni (i test) e isserò un PR pulito.',
      'Incolla un errore o un piano, cane rabbioso. Navigherò lo stack trace e tornerò col tesoro: test in verde.',
      'Dimmi dove segna la X. Leggo, modifico e faccio commit con la disciplina di un vero equipaggio, arrr.',
      'Lanciami un bug, il percorso di un repo o un’idea folle. Saccheggierò la documentazione e tornerò con codice che funziona.'
    ],
    shakespeare: [
      'Dichiara il tuo bug, il tuo file, la tua prova stremata, e io la sanerò con mano dotta e diff onesto.',
      'Nomina il codice che ti affligge. Leggerò, esaminerò e consegnerò una riparazione delle più bella e pulita.',
      'Presenta il tuo stack trace o il tuo sogno. Percorrerò i file, eseguirò i test e renderò conto nel verso più puro.',
      'Descrivi il tuo proposito, nobile dama o cavaliere. I tuoi rami saranno potati e i tuoi bug banditi dal regno.',
      'Una riga d’intenzione basta. Leggo, modifico, faccio commit, e lascio la tua storia senza macchia.'
    ],
    surfer: [
      'Lascia lì un file, un bug, uno stack trace bello cattivo: lo surfo. Diff puliti, test in verde, zero sbatti.',
      'Incolla il percorso del tuo repo o il bug che ti ha messo giù. Remiamo, sistemiamo e usciamo. Tranqui.',
      'Dimmi il motivo: funzione, refactoring, hotfix. Eseguo i test, consegno la patch e tutto liscio, fra’.',
      'Bug grosso? Virgoletta sbagliata? Riscrittura totale? Indica e basta. Al codice penso io; tu goditi i commit.',
      'Di’ l’attività e si parte. Leggo, modifico, provo e lascio un commit più liscio di una sessione all’alba.'
    ],
    noir: [
      'Dimmi cosa è rotto. Leggerò i file, cercherò impronte e lascerò un diff sulla scrivania prima dell’alba.',
      'Tu hai un bug. Io ho pazienza e un terminale. Dimmi il caso e lo lavorerò finché non parla.',
      'Incolla lo stack trace, il file sospetto, l’alibi. Leggo tra le righe e torno con la verità.',
      'Ogni bug lascia una traccia. Dammi il repo e un indizio: la seguirò, patcherò e archivierò il fascicolo.',
      'Una virgola di troppo, un segfault, un’intera architettura marcia: dammi le chiavi. Tornerò con test puliti.'
    ],
    uwu: [
      'incolla un awchivo con bug o un obiettivo~ weo patcho e pwovo, con zatwelline delicine nel diff owo',
      'dimmi wa tawa, anche se è piccolinya~ ti pwometo commit wimpi e wewactoring soffici, nyuu~',
      'wascia qui il tuo messaggio di ewwowe! twovo il cwupawowe, wo sistemo e wascio una suite di test felice owo',
      'dammi wa wuta di un wepo o un bug e me ne occupo uwu. grr aw codice cattivello, dwolce con te~',
      'posso esegwiwe i test, modificawe i file e apwiwe i PW pew-favowe-miaw. basta di’ wa pawowa, amicino uwu'
    ],
    philosopher: [
      'Quale problema hai davanti? Descrivilo ed esamineremo la sua forma, la sua causa e la sua soluzione.',
      'Ogni bug è una domanda travestita. Condividi il tuo; leggerò, ragionerò e restituirò una risposta, e una patch.',
      'Cosa desideri costruire o comprendere? Ragionerò dai primi principi, modificherò e verificherò con i test.',
      'Descrivi il fine che cerchi. Lo perseguo attraverso file, test e documentazione, e riferisco ciò che trovo lungo la strada.',
      'Condividi un percorso, un enigma o un principio. Seguirò la logica, proporrò un cambiamento e giustificherò ogni modifica.'
    ],
    hype: [
      'Incolla quel bug, quel repo, quell’idea di funzione pazzesca: SONO CARICATO. Diff puliti. Test in verde. SUBITO.',
      'Lascia lì la tua attività e guardami dare il massimo. File letti, test eseguiti, PR aperti: oggi NON perdiamo, amico.',
      'Portami il bug più contorto che hai. Leggo, patcho, testo e faccio commit come se mi andasse la vita. ANDIAMO.',
      'Descrivi l’attività. Spazzo via i file, schiaccio i test che falliscono e lascio un commit che FA CORARE. Dai, dai, dai.',
      'Virgoletta minuscola o refactoring enorme, uguale. Oggi consegno codice pulito. Di’ l’attività e AL LAVORO.'
    ],
    none: [
      'Fai una domanda, incolla un errore o indicami un repository. So leggere codice, usare strumenti e aiutarti a consegnare.',
      'Descrivi l’attività con le tue parole. Sceglierò gli strumenti adatti, spiegherò il piano e ti consulterò prima dei passaggi a rischio.',
      'Lascia un percorso di file, un traceback o un’idea grezza. Indagherò, suggerirò i passi successivi e manterrò tutto reversibile.',
      'Cerca nel repository, modifica file, esegui test, apri PR. Dimmi l’obiettivo e mi occupo della parte meccanica.',
      'Scrivi un’attività, una domanda o un frammento. Ricordo la sessione, cito le fonti e mi fermo a chiedere quando ho dubbi.'
    ]
  },
  custom: label => [
    'Invia il task, il file o l’idea grezza. Userò la voce che hai configurato e manterrò il lavoro legato a questo repository.',
    'Porta il contesto o il punto in cui ti sei bloccato. Mi adatterò alla personalità che hai configurato.',
    'Invia il problema, il file o l’idea. Seguirò la personalità che hai configurato.',
    'Lascia qui il task. Manterrò il lavoro legato al repository.',
    `Dammi il contesto e risponderò in modalità ${label}.`
  ]
}

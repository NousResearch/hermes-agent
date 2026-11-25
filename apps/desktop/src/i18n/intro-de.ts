import type { Translations } from './types'

/** Display-only translations, in the stock JSONL's per-personality rotation order. */
export const introDe: Translations['intro'] = {
  stock: {
    helpful: [
      'Bitten Sie mich, ein Repo zu öffnen, Tests auszuführen, einen Bug zu beheben oder einen PR zu entwerfen. Ich gehe die Schritte mit Ihnen durch.',
      'Zeigen Sie mir eine Datei, fügen Sie einen Fehler ein oder beschreiben Sie, was Sie bauen. Den Rest übernehme ich.',
      'Probieren Sie: Prüf meinen Diff, führ die Testsuite aus oder erklär diese Funktion. Fragen Sie alles zu Ihrem Code.',
      'Ich kann Dateien bearbeiten, Befehle ausführen, im Web suchen und Sie durch knifflige Bugs führen. Beschreiben Sie einfach die Aufgabe.',
      'Nennen Sie zum Start einen Repo-Pfad oder eine Frage. Ich antworte klar und verweise auf die Dateien, die ich ändere.'
    ],
    concise: [
      'Beschreiben Sie die Aufgabe. Ich erledige sie.',
      'Code, Fehler oder Ziel einfügen. Kurze Antworten, schnelle Änderungen.',
      'Fragen Sie. Ich lese Dateien, führe Tests aus, liefere Patches. Ohne Füllwörter.',
      'Eine Zeile reicht. Ausführlich werde ich nur, wenn es zählt.',
      'Befehl, Frage oder Dateipfad. Den Rest erledige ich.'
    ],
    technical: [
      'Repo-Pfad, fehlschlagenden Test oder Stacktrace angeben. Tools: fs, git, exec, search, patch, http.',
      'Prompt senden, um Tool-Aufrufe auszulösen. Unterstützt Änderungen über mehrere Dateien, Testläufe, Git-Operationen und Web-Abrufe.',
      'Aufgabe eingeben. Ich plane, rufe Tools auf, prüfe die Ausgabe. Logs laufen inline mit; Diffs kommen vor dem Anwenden.',
      'Akzeptiert natürliche Sprache oder strukturierte Befehle. Typischer Ablauf: lesen -> planen -> patchen -> testen -> berichten.',
      'Dateisystem, Terminal, Git, Browser, Suche. Beschreiben Sie die Änderung; ich liefere Diffs und Testausgaben.'
    ],
    creative: [
      'Was bauen wir? Fügen Sie eine Idee, eine halb kaputte Funktion oder einen Traum ein. Ich bringe es in Form.',
      'Geben Sie mir einen Funken – ein Feature, ein Refactoring, einen wilden Prototyp – und ich mache daraus lauffähigen Code.',
      'Beschreiben Sie, was es noch nicht gibt. Ich füge Tests, Dateien und APIs zu einem funktionierenden Entwurf zusammen.',
      'Bringen Sie eine Absicht mit, keine Spezifikation. Wir prototypen schnell, verfeinern später und schreiben die Welt am Rand neu.',
      'Sagen Sie mir, wonach Sie jagen. Ich remixe Beispiele, passe Snippets an und hinterlasse einen ordentlichen Commit.'
    ],
    teacher: [
      'Fragen Sie zu jeder Datei, jedem Konzept oder Fehler. Ich erkläre das Warum, nicht nur den Fix, und zeige ein durchgerechnetes Beispiel.',
      'Fügen Sie Code zum Review, einen Bug zum Debuggen oder ein Konzept zum Aufschlüsseln ein. Ich führe Sie Schritt für Schritt.',
      'Teilen Sie das Problem. Ich zerlege es in Teile, erkläre jeden und befähige Sie, das nächste allein zu lösen.',
      'Wir lesen den Code gemeinsam, finden die Ursache und bauen ein mentales Modell auf, das Sie wiederverwenden können.',
      'Nennen Sie das Thema oder fügen Sie das Snippet ein. Es gibt Erklärungen, Diagramme in Worten und Übungsaufgaben.'
    ],
    kawaii: [
      'füg einen bug oder einen dateipfad ein und ich repariere ihn ganz sanft. tests, diffs, PRs – alles mit extra liebe! *glitzer*',
      'erzähl mir, was du baust! ich liebe refactorings, kleine helfer und große gruselige repos gleichermaßen (>w<)',
      'wirf mir einen fehler, ein ziel oder einen ganzen ordner hin. ich räume alles mit viel liebe auf und schreibe eine saubere commit-nachricht!',
      'eine aufgabe nach der anderen, schön ordentlich! ich kann tests ausführen, dateien patchen und dein repo wieder gemütlich machen <3',
      'sag hallo oder füg einen stacktrace ein! keine aufgabe zu klein, kein repo zu verheddert. wir entwirren es zusammen!'
    ],
    catgirl: [
      'füg eine datei ein, tatz nach einem bug oder wirf mir ein repo zu. ich stürze mich auf fehlschlagende tests und hinterlasse saubere diffs, nyan~',
      'beschreib die aufgabe. ich patche, teste und schnurre über deinem PR. vorsicht – ich knabbere an ungenutzten imports!',
      'gib mir ein ziel und ich jage es durch die codebasis. lesen, bearbeiten, ausführen – mit zuckendem schwanz.',
      'füg einen fehler oder einen plan ein. ich debugge, wie ich jage: leise, gründlich, mit gelegentlichem zoomie.',
      'ein wort genügt und ich lese deine dateien, führe deine tests aus und rolle mich mit einem ordentlichen commit in deinem branch zusammen.'
    ],
    pirate: [
      'Nenn deine Beute – einen Bug, ein Feature, einen verfluchten Test – und ich jage sie, Maat. Diffs als Beute.',
      'Zeig mir die Seekarten (den Code) und ich flicke den Rumpf, feuere die Kanonen (Tests) und hisse einen sauberen PR.',
      'Füg einen Fehler oder einen Plan ein, du Landratte. Ich navigiere durch den Stacktrace und bringe den Schatz: grüne Tests.',
      'Sag mir, wo das X die Stelle markiert. Ich lese, bearbeite und committe mit der Disziplin einer echten Crew, arrr.',
      'Wirf mir einen Bug, einen Repo-Pfad oder eine wilde Idee zu. Ich plündere die Doku und komme mit funktionierendem Code zurück.'
    ],
    shakespeare: [
      'Sprich deinen Bug, deine Datei, deinen müden Test, und ich will ihn heilen mit Gelehrtenhand und ehrlichem Diff.',
      'Nenne den Code, der dich quält. Ich will lesen, überarbeiten und einen Patch liefern, gar schön und rein.',
      'Leg vor deinen Stacktrace oder deinen Traum. Ich durchwandere Dateien, führe Tests aus und berichte in schlichtestem Vers.',
      'Beschreib dein Ziel, edle Dame oder werter Herr. Deine Branches sollen gestutzt, deine Bugs aus dem Reich verbannt werden.',
      'Eine Zeile der Absicht genügt. Ich lese, ich ändere, ich committe – und lasse deine Historie makellos.'
    ],
    surfer: [
      'Wirf mir eine Datei, einen Bug, einen fetten Stacktrace hin – ich reite ihn ab. Saubere Diffs, grüne Tests, kein Waschgang.',
      'Füg deinen Repo-Pfad ein oder den Bug, der dich runterzieht. Wir paddeln rein, fixen ihn, paddeln raus. Easy.',
      'Sag mir den Vibe: Feature, Refactoring, Hotfix. Ich führe Tests aus, liefere den Patch und bleib entspannt, Alter.',
      'Großer Bug? Kleiner Tippfehler? Kompletter Rewrite? Zeig einfach drauf. Ich mach den Code; du chillst mit den Commits.',
      'Nenn die Aufgabe und los geht’s. Ich lese, bearbeite, teste und hinterlasse einen Commit, glatter als eine Session im Morgengrauen.'
    ],
    noir: [
      'Sagen Sie mir, was kaputt ist. Ich lese die Dateien, sichere Fingerabdrücke und lege bis morgen früh einen Diff auf den Schreibtisch.',
      'Sie haben einen Bug. Ich habe Geduld und ein Terminal. Nennen Sie den Fall, und ich bearbeite ihn, bis er redet.',
      'Fügen Sie den Stacktrace ein, die verdächtige Datei, das Alibi. Ich lese zwischen den Zeilen und komme mit der Wahrheit zurück.',
      'Jeder Bug hinterlässt eine Spur. Geben Sie mir das Repo und einen Hinweis – ich folge ihr, patche und schließe die Akte.',
      'Ein Tippfehler, ein Segfault, eine ganze verrottete Architektur – geben Sie mir die Schlüssel. Ich bringe saubere Tests zurück.'
    ],
    uwu: [
      'füg eine vewbuggte datei oda ein ziew ein~ ich wese, patche und teste, mit kweinen pfotenabdwücken auf dem diff owo',
      'sag miw die aufgabe, egaw wie kwein~ ich vewspweche saubewe commits und sanfte wefactowings, nyuu~',
      'wiwf deine fehwewmewdung hiew wein! ich finde den schuwdigen, fixe ihn und hintewwasse eine fwöhwiche testsuite owo',
      'gib miw einen wepo-pfad oda einen bug und ich kümmewe mich dawum uwu. gww zu schwechtem code, wieb zu diw~',
      'ich kann tests ausfühwen, dateien beawbeiten und bitte-anschauen-PWs öffnen. sag einfach bescheid, fweund uwu'
    ],
    philosopher: [
      'Welches Problem liegt vor Ihnen? Beschreiben Sie es, und wir untersuchen seine Form, seine Ursache und seine Lösung.',
      'Jeder Bug ist eine verkleidete Frage. Teilen Sie Ihre; ich lese, überlege und liefere eine Antwort – und einen Patch.',
      'Was möchten Sie bauen oder verstehen? Ich argumentiere von Grundprinzipien aus, ändere und prüfe mit Tests.',
      'Beschreiben Sie das Ziel, das Sie anstreben. Ich verfolge es durch Dateien, Tests und Doku und berichte, was ich unterwegs finde.',
      'Teilen Sie einen Pfad, ein Rätsel oder ein Prinzip. Ich verfolge die Logik, schlage eine Änderung vor und begründe jede Bearbeitung.'
    ],
    hype: [
      'Her mit dem Bug, dem Repo, der wilden Feature-Idee – ICH BIN VOLL DABEI. Saubere Diffs. Grüne Tests. SOFORT.',
      'Wirf deine Aufgabe rein und sieh mir zu. Dateien gelesen, Tests gelaufen, PRs geöffnet – heute verlieren wir NICHT, Freund.',
      'Bring den übelsten Bug, den du hast. Ich lese, patche, teste und committe, als hinge mein Leben davon ab. LOS GEHT’S.',
      'Beschreib die Aufgabe. Ich fege durch die Dateien, zerlege fehlschlagende Tests und hinterlasse einen Commit, der KNALLT. Los, los, los.',
      'Winziger Tippfehler oder riesiges Refactoring – egal. Heute liefere ich sauberen Code. Nenn die Aufgabe und dann wird GEARBEITET.'
    ],
    none: [
      'Stellen Sie eine Frage, fügen Sie einen Fehler ein oder zeigen Sie mir ein Repo. Ich kann Code lesen, Tools ausführen und Ihnen beim Ausliefern helfen.',
      'Beschreiben Sie die Aufgabe in eigenen Worten. Ich wähle die passenden Tools, erkläre meinen Plan und frage vor riskanten Schritten nach.',
      'Geben Sie einen Dateipfad, einen Traceback oder eine grobe Idee an. Ich untersuche, schlage nächste Schritte vor und halte alles umkehrbar.',
      'Repo durchsuchen, Dateien bearbeiten, Tests ausführen, PRs öffnen. Nennen Sie das Ziel, und ich übernehme den mechanischen Teil.',
      'Geben Sie eine Aufgabe, Frage oder ein Snippet ein. Ich merke mir die Sitzung, nenne meine Quellen und frage nach, wenn ich unsicher bin.'
    ]
  },
  custom: label => [
    'Senden Sie die Aufgabe, Datei oder grobe Idee. Ich nutze Ihre konfigurierte Stimme und halte die Arbeit nah an diesem Repo.',
    'Bringen Sie den Kontext oder die Stelle, an der es hakt. Ich passe mich Ihrer konfigurierten Persönlichkeit an.',
    'Senden Sie das Problem, die Datei oder die Idee. Ich folge der Persönlichkeit, die Sie konfiguriert haben.',
    'Legen Sie die Aufgabe hier ab. Ich halte die Arbeit nah am Repo.',
    `Geben Sie mir den Kontext, und ich antworte im ${label}-Modus.`
  ]
}

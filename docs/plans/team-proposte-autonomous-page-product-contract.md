# Product contract — Pagina autonoma Team & Proposte

Task: t_d0407f27
Parent: t_11dbeb12
Repo target: `/home/daniele/hermes-workspace/runs/hermes-agent-mission-control`

## 1. Decisione prodotto

Creare una pagina autonoma `Team & Proposte` come cockpit di governance/proattività, separata da Mission Control e dalle pagine specialistiche già esistenti.

Route proposta:

- Nuova route canonica: `/team-proposals`.
- Label nav: `Team & Proposte`.
- Icona suggerita: `Sparkles` o `Users`.
- Posizione nav: accanto a `Team Operativo` e `Sviluppo Hermes`, non dentro Mission Control.

Relazione con superfici esistenti:

- `/team-work`: rimane la vista operativa del lavoro reale e delle task operative. Deve ricevere link/contesto dalle proposte operative, non duplicare il registro completo.
- `/team-evolution`: rimane la vista sviluppo Hermes, roadmap, reliability e capacità del sistema. Deve ricevere link/contesto dalle proposte evolutive, non diventare inbox rumorosa.
- `/team-proposals`: diventa il punto di raccolta e decisione per proposte operative + evolutive, challenge interna e gate. È la pagina in cui Daniele vede cosa il team propone, perché ora, e cosa succede se approva/lancia.
- Mission Control: resta execution cockpit: Kanban, blocker, review-required, task in corso, output e dispatch. Non deve ospitare l'intero dibattito proposta/supporter/critic; deve vedere solo lavoro già convertito o decisioni strettamente esecutive.

Obiettivo UX: far percepire un team vivo, non una lista statica di card. Ogni proposta importante passa da segnale → interpretazione → supporter/critic → Chief recommendation → gate Daniele.

## 2. Principi non negoziabili

1. Low-noise: massimo 1–3 decisioni visibili sopra la piega. Tutto il resto va in sezioni collassate: audit, storico, registry completo, dettagli tecnici.
2. Proposta ≠ esecuzione: generare, approvare, parcheggiare o scartare una proposta modifica solo il registry.
3. Preview ≠ esecuzione: preview piano/task mostra esattamente cosa verrebbe creato, ma non crea Kanban, cron, messaggi, webhook o dispatch.
4. Conferma esplicita = execution gate: solo il bottone finale di conferma crea task Kanban `ready`, con assignee reali e motivati.
5. Nessun cron/invio esterno/dispatch nella fase proposta. Il dispatcher lavora solo dopo task Kanban ready creati dal gate esplicito.
6. Profili reali: mostrare solo profili verificati: `default`, `agronomic`, `carbon`, `claims`, `drafting`, `evidence`, `finance`, `hermespm`, `legal`, `market`, `mrv`, `ops`, `regulatory`, `reliability`.

## 3. Layout implementabile

### A. Header / safety strip

Copy consigliata:

> Team & Proposte è il consiglio operativo di Hermes: raccoglie segnali, propone mosse, mostra challenge interna e porta a Daniele solo decisioni ad alto valore.

Safety strip sempre visibile:

- `Status/approve/park/discard: registry-only`
- `Preview piano/task: no side-effect`
- `Conferma finale: crea task Kanban ready`
- `Nessun cron, invio esterno o dispatch automatico nella fase proposta`

### B. Decisioni ora (max 1–3)

Sezione primaria con solo le proposte che richiedono una scelta di Daniele.

Campi minimi per ogni decisione:

- Titolo
- Tipo: `Operativa` o `Evoluzione Hermes`
- Perché ora
- Beneficio per Daniele
- Rischio/controargomento
- Effort
- Raccomandazione Chief: `fai ora`, `prepara`, `parcheggia`, `scarta`
- Gate disponibile: `approva`, `parcheggia`, `scarta`, `preview task`, `preview piano`

Regola: se ci sono più di 3 candidate, usare ranking Chief e spostare le altre in `Altre proposte mature` collassato.

### C. Team reale / ruoli

Mostrare card compatte dei profili reali, non personaggi inventati.

Campi per profilo:

- Profilo: es. `hermespm`, `ops`, `reliability`, `evidence`
- Ruolo nel team: PM interno, implementer, reviewer, evidence steward, ecc.
- Stato: `attivo`, `watching`, `standby`
- Ultimo contributo rilevante: breve testo o `n/d`
- Può proporre: sì/no
- Può eseguire dopo gate: sì/no secondo assignee Kanban reale

Nota UX: questa sezione deve rendere visibile la crescita/attività dei subagenti, ma senza metriche vanity. Meglio 1 segnale utile che 10 contatori.

### D. Proposte operative

Scopo: migliorare lavoro reale di Daniele senza mescolarlo con sviluppo Hermes.

Esempi di origini:

- Kanban blocker/review-required
- Evidenze mancanti
- Documento o follow-up da preparare
- Frizione operativa ricorrente

Card operative devono linkare, quando disponibile, a `/team-work` o Mission Control/Kanban.

### E. Proposte evolutive

Scopo: migliorare Hermes, Mission Control, dashboard, automazioni, reliability, subagenti e governance.

Esempi di origini:

- Pattern di attrito nel dashboard
- Problema reliability/worker
- Capacità mancante per Team Pulse
- Debito UX o decisionale

Card evolutive devono linkare, quando disponibile, a `/team-evolution`.

### F. Chief recommendation

Una sezione sintetica, sempre visibile, che dice:

- `Decisione consigliata`
- `Perché ora`
- `Controargomento principale`
- `Prossimo passo minimo`

Non deve essere un lungo memo. Deve ridurre il carico decisionale di Daniele.

### G. Supporter / Critic / Challenge

Ogni proposta matura deve mostrare un blocco challenge compatto:

- `Supporter`: perché conviene farla ora.
- `Critic`: perché potrebbe essere rumore, over-engineering o rischio.
- `Chief synthesis`: decisione PM dopo il confronto.
- `Veto/rischio`: opzionale, solo se concreto.

Questa sezione può essere collassabile nella card, ma almeno il summary supporter/critic deve essere visibile per le top 1–3 decisioni.

### H. Gate e preview

Azioni e copy obbligatorio:

- `Approva`: "Aggiorna solo lo stato della proposta nel registry. Non crea task e non avvia worker."
- `Parcheggia`: "Sposta la proposta fuori dalle decisioni attive. Nessun side effect."
- `Scarta`: "Archivia la proposta come scartata. Nessun side effect."
- `Preview task`: "Mostra la card Kanban che verrebbe creata. Preview read-only."
- `Preview piano`: "Mostra parent + task figli, assignee, priorità e initial status. Preview read-only."
- `Conferma e lancia`: "Crea task Kanban ready. Il dispatcher partirà solo secondo priorità e limiti Kanban."

Il modal preview deve mostrare:

- titolo parent/task
- body completo
- assignee reali
- motivo assegnazione
- board/tenant
- priority
- initial status atteso: `ready`
- idempotency key
- lista figli e dipendenze
- nota: `nessun cron / nessun invio esterno`

## 4. Stati e comportamento

Stati registry accettati:

- `proposta`
- `raccomandata`
- `approvata`
- `parcheggiata`
- `scartata`
- `trasformata_in_task`

Transizioni:

- `proposta` → `raccomandata`: Chief la promuove; registry-only.
- `proposta|raccomandata` → `approvata`: Daniele approva concettualmente; registry-only.
- `proposta|raccomandata|approvata` → `parcheggiata`: resta in standby; registry-only.
- `proposta|raccomandata|approvata` → `scartata`: archiviata; registry-only.
- `approvata|raccomandata|proposta` → `trasformata_in_task`: solo dopo conferma esplicita su preview; crea Kanban ready.

Filtri:

- Le decisioni attive escludono `scartata` e `trasformata_in_task`.
- Le proposte trasformate/scartate vanno in `Archivio / audit`, collassato.
- Le proposte parcheggiate possono comparire in `Standby`, collassato, senza contare nel max 1–3 decisioni attive.

## 5. Data contract minimo proposal card

```ts
interface TeamProposalCard {
  id: string;
  title: string;
  kind: "operative" | "evolution";
  status: "proposta" | "raccomandata" | "approvata" | "parcheggiata" | "scartata" | "trasformata_in_task";
  origin: string;
  source_agent?: string;
  source_signal: string;
  interpretation: string;
  evidence_refs?: string[];
  why_now: string;
  benefit: "high" | "medium" | "low";
  effort: "high" | "medium" | "low";
  risk: "high" | "medium" | "low";
  priority: "P0" | "P1" | "P2" | "P3";
  confidence: "high" | "medium" | "low";
  recommendation: "do_now" | "prepare" | "park" | "reject";
  supporter_view: string;
  critic_view: string;
  chief_synthesis: string;
  veto_risk?: string;
  acceptance_criteria: string[];
  suggested_plan?: {
    mode: "single_task" | "parent_children";
    preview_endpoint: string;
    conversion_endpoint: string;
  };
  kanban_links?: {
    task_id?: string;
    plan_task_id?: string;
    child_task_ids?: string[];
  };
  created_at: string;
  updated_at: string;
}
```

## 6. Esempi card

### Esempio 1 — operativa

Titolo: `Ridurre review CO2Farm ferme in Kanban`

- Tipo: `Operativa`
- Status: `raccomandata`
- Origin: `kanban/blocker-scan`
- Source agent: `hermespm`
- Segnale: `Più card operative CO2Farm restano in review-required o blocked senza prossimo passo visibile.`
- Interpretazione: `Daniele rischia di vedere backlog invece di decisioni concrete; serve una mini-sintesi delle sole review da sbloccare.`
- Supporter: `Una vista da 1–3 review urgenti riduce rumore e accelera output reali.`
- Critic: `Se automatizzata male diventa un digest rumoroso e duplica Mission Control.`
- Chief synthesis: `Preparare una proposta di vista review urgente, non creare cron. Prima preview manuale.`
- Benefit: `high`
- Effort: `medium`
- Risk: `medium`
- Recommendation: `prepare`
- Acceptance: `Mostra max 3 review con file, motivo sblocco, prossima azione; nessun invio automatico.`
- Gate: `Preview piano`, poi conferma crea Kanban ready.

### Esempio 2 — evolutiva

Titolo: `Team Pulse con refill sicuro dopo proposte trasformate`

- Tipo: `Evoluzione Hermes`
- Status: `proposta`
- Origin: `dashboard/team-pulse`
- Source agent: `reliability`
- Segnale: `Le proposte convertite rischiano di restare visibili o di bloccare nuova generazione attiva.`
- Interpretazione: `La pagina sembra statica se non separa attivo/archivio e non crea nuovo ciclo proposal-only.`
- Supporter: `Rende Team & Proposte vivo senza avviare worker o cron.`
- Critic: `Refill automatico può diventare rumore se non idempotente e low-noise.`
- Chief synthesis: `Fare solo refill manuale/idempotente in prima slice; nessun loop schedulato.`
- Benefit: `medium`
- Effort: `medium`
- Risk: `low`
- Recommendation: `prepare`
- Acceptance: `Le card trasformate vanno in archivio; Team Pulse crea nuove proposte attive solo su trigger esplicito, no side effect.`
- Gate: `Preview task`, poi conferma crea task ready.

### Esempio 3 — evolutiva/reliability

Titolo: `Guardrail no-side-effect su preview piano`

- Tipo: `Evoluzione Hermes`
- Status: `raccomandata`
- Origin: `reliability/review`
- Source agent: `reliability`
- Segnale: `Preview e conversione sono vicine nel codice; regressioni potrebbero creare task prima della conferma.`
- Interpretazione: `Serve test contrattuale che provi che preview non muta Kanban/registry.`
- Supporter: `Protegge fiducia di Daniele nel gate: può esplorare piani senza conseguenze.`
- Critic: `Potrebbe duplicare test backend esistenti se non mirato.`
- Chief synthesis: `Aggiungere test piccolo su preview read-only + conversione idempotente ready.`
- Benefit: `high`
- Effort: `low`
- Risk: `low`
- Recommendation: `do_now`
- Acceptance: `Test fallisce se preview crea task, cambia status, crea cron o invia messaggi; conversione richiede confirmed_preview_hash.`
- Gate: `Preview task`, poi conferma crea task ready.

## 7. Acceptance checklist per implementer

- Route `/team-proposals` è canonica e navigabile; eventuali legacy redirect non cancellano la nuova pagina autonoma.
- Nav mostra `Team & Proposte` come voce distinta accanto a `Team Operativo` e `Sviluppo Hermes`.
- In alto sono visibili safety strip e copy registry-only/no-side-effect/ready-on-confirm.
- Vista principale mostra max 1–3 decisioni attive; audit, archivio, standby e registry completo sono collassati.
- Sono presenti blocchi: ruoli/profili reali, proposte operative, proposte evolutive, Chief recommendation, supporter/critic/challenge, gate.
- Status actions (`approve`, `park`, `discard`) aggiornano solo registry; nessun Kanban/cron/invio/dispatch.
- Preview task/piano è read-only e mostra dati completi, assignee reali e initial status `ready`.
- Conferma esplicita crea Kanban card `ready` con idempotency key e marca proposta `trasformata_in_task`.
- Le card trasformate/scartate spariscono dalle decisioni attive e restano in archivio/audit collassato.
- Test minimi: frontend contract route/nav/copy; backend preview no-side-effect; conversione idempotente; filtro active vs archive.

## 8. Challenge interna PM

Supporter view: una pagina autonoma riduce la confusione attuale fra Mission Control, Team Operativo e Sviluppo Hermes; rende esplicito il team vivo e protegge Daniele da rumore esecutivo.

Critic view: aggiungere una terza superficie team può aumentare complessità se duplica contenuti o se reintroduce registry statico.

Sintesi PM: procedere solo se `/team-proposals` è chiaramente governance/proposte e le altre pagine restano execution/specializzazione. La prima slice deve essere principalmente UI/contract + guardrail; niente nuovo cron o intake automatico obbligatorio.

Success criterion: Daniele può aprire una sola pagina, vedere massimo tre decisioni motivate, capire cosa è proposta vs esecuzione, e lanciare lavoro Kanban ready solo dopo preview + conferma esplicita.

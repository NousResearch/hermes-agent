# Mission Control v2 — cockpit operativo task/documenti/team

## Perché v2

Mission Control v1 è cresciuta per strati: code Kanban, decision cockpit, output review, follow-up, dispatch, diagnostica e controlli sono tutti presenti, ma Daniele deve ancora tornare in chat per capire cosa fare dopo un output, dove sono i documenti, quale team usare o quando lanciare un worker. La v2 deve essere un cockpit operativo: pochi elementi ad alta leva nel primo viewport, drawer contestuale per agire, documenti come cittadini di prima classe e dispatch sempre separato/confirmato.

## Principi prodotto

1. Inbox decisioni: massimo 3–5 elementi che richiedono Daniele ora.
2. Workbench operativo: task raggruppate per `setup`, `ready`, `running`, `review`, `recent_done`.
3. Task drawer contestuale: azioni primarie diverse per stato, senza obbligare a tornare in chat.
4. Document registry: output prodotti, allegati, Vault/Drive e versioni/review espliciti.
5. Team selector semplificato: `single_worker`, `team_review`, `team_writing`, `team_challenge`, `chief_synthesis`.
6. System/Diagnostics separato e collassato: nessuna diagnostica nel flusso operativo.
7. No auto-dispatch, no external send, no hidden work: ogni dispatch è una fase separata, con preview e conferma.

## Scope iniziale

Questa fase crea v2 parallela sotto `/api/plugins/kanban-dashboard/v2/*` e una shell frontend distinta che non rompe `/kanban-mission` legacy. La prima slice deve rendere visibile il modello operativo e il contratto dati; le mutazioni restano quelle già guardrailed da v1 o vengono esposte come action descriptors, non come lavoro nascosto.

## JSON contracts

### GET `/api/plugins/kanban-dashboard/v2/cockpit`

Restituisce:

- `version`: `mission-control-v2`.
- `navigation`: sezioni UI `inbox`, `workbench`, `documents`, `teams`, `system`.
- `inbox`: `max_items`, `items`, `empty_state`, `guardrail.no_auto_dispatch=true`.
- `workbench`: colonne `setup`, `ready`, `running`, `review`, `recent_done`, ciascuna con task preview bounded.
- `documents`: registry compatto con `items`, `source`, `review_gate`, `open_folder_available`.
- `teams`: preset operativi e nota di conferma.
- `system`: diagnostica collassata con conteggi e health metadata.
- `actions_contract`: endpoint descrittivi per drawer task, documenti, team e dispatch preview.

### GET `/api/plugins/kanban-dashboard/v2/tasks/{task_id}`

Restituisce il task drawer v2:

- `task`: dettaglio task.
- `state`: stato normalizzato (`needs_setup`, `ready_to_dispatch`, `running`, `needs_review`, `completed`).
- `primary_actions`: azioni contestuali: `add_context`, `upload_context`, `create_followup`, `request_changes`, `review_output`, `dispatch_preview`, `dispatch_confirm_separate`.
- `documents`: allegati input e output prodotti separati.
- `team_recommendations`: preset team utilizzabili per questa task.
- `guardrail`: nessun dispatch, nessun invio esterno, nessun lavoro nascosto dal dettaglio.

### GET `/api/plugins/kanban-dashboard/v2/documents`

Restituisce un document registry board-level:

- output prodotti da task concluse;
- allegati/context input recenti;
- segnali Vault/Drive quando disponibili;
- `review_status` e `next_action` per ogni item.

### GET `/api/plugins/kanban-dashboard/v2/actions`

Restituisce il lifecycle delle azioni cockpit:

- `inspect -> choose_action -> preview -> confirm -> observe`;
- mutation endpoints permessi già esistenti;
- dispatch e external send marcati come separati e confermati.

## Lifecycle task/document/action

1. Crea o seleziona task.
2. Aggiungi contesto e documenti input.
3. Scegli modalità lavoro/team.
4. Preview dispatch separata.
5. Conferma dispatch one-tick solo se Daniele vuole partire.
6. Review output/documenti prodotti.
7. Da output: approva internamente, richiedi modifiche o crea follow-up collegato.

## Dogfood Scauzi registration pack

La board `co2farm-chief` e i task Scauzi devono poter essere gestiti senza chat:

- documenti prodotti dalle task concluse visibili in Documents;
- task drawer con follow-up e request changes;
- azioni review/output evidenti;
- dispatch separato e confermato.

## Fasi implementative

### Fase 1 — Contract + shell dati

- Aggiungere spec in `docs/plans`.
- Aggiungere endpoint v2 read/read-mostly: cockpit, task drawer, documents, actions.
- Aggiungere test backend TDD sui contratti.
- Aggiungere marker frontend v2 nel bundle/plugin route senza rompere legacy.

### Fase 2 — UI v2 shell

- Shell con sezioni Inbox, Workbench, Documents, Teams, System.
- Drawer task v2 con azioni contestuali.
- Documents pane con output/allegati/Vault.
- Team selector con preset semplici.

### Fase 3 — Operatività assistita

- Collegare azioni sicure già esistenti: add context, request changes, create follow-up, upload context.
- Mantenere dispatch preview/confirm come flusso separato.
- Browser smoke reale su `/kanban-mission-v2` o tab v2, più legacy `/kanban-mission`.

## Non-obiettivi iniziali

- Nessun auto-dispatch.
- Nessun invio esterno/document release.
- Nessuna cancellazione o archiviazione nascosta.
- Nessuna sostituzione della legacy finché v2 non è verificata.

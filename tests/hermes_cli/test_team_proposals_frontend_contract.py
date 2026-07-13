from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_TSX = REPO_ROOT / "web" / "src" / "App.tsx"
PAGE_TSX = REPO_ROOT / "web" / "src" / "pages" / "TeamProposalsPage.tsx"
COMMAND_PAGE_TSX = REPO_ROOT / "web" / "src" / "pages" / "TeamCommandCenterPage.tsx"
API_TS = REPO_ROOT / "web" / "src" / "lib" / "api.ts"


def test_team_proposals_has_single_canonical_nav_item():
    source = APP_TSX.read_text()

    assert 'path: "/team-proposals", label: "Team & Proposte"' in source
    assert source.count('path: "/team-proposals", label: "Team & Proposte"') == 1
    assert 'path: "/team-evolution", label: "Sviluppo Hermes"' in source
    assert 'path: "/team-work", label: "Team Operativo"' in source
    assert 'label: "Radar Hermes"' not in source


def test_team_proposals_route_is_canonical_with_aliases():
    source = APP_TSX.read_text()

    assert 'TeamProposalsCommandPage,' in source
    assert '"/team-proposals": TeamProposalsCommandPage' in source
    assert 'function TeamProposalsRedirect' not in source
    assert '"/team-work": TeamOperationsPage' in source
    assert '"/team-evolution": TeamEvolutionPage' in source
    assert '"/team-proposals/archive": TeamProposalsArchivePage' in source
    assert '"/team-work/archive": TeamOperationsArchivePage' in source
    assert '"/team-evolution/archive": TeamEvolutionArchivePage' in source
    assert '"/team-pm": RadarHermesRedirect' in source
    assert '"/radar-hermes": RadarHermesRedirect' in source
    assert 'function LegacyTeamProposalsRedirect' not in source



def test_team_command_center_clean_restart_is_kanban_first_and_archives_legacy():
    source = COMMAND_PAGE_TSX.read_text()
    api_ts = API_TS.read_text()

    assert 'getKanbanBoard(config.board)' in source
    assert 'mode: "proposals"' in source
    assert 'title: "Team & Proposte"' in source
    assert 'siblingHref: "/team-work"' in source
    assert 'siblingLabel: "Apri Team Operativo"' in source
    assert 'pipeline:pronta-per-daniele' in source
    assert 'Nessuna decisione operativa pronta per Daniele.' in source
    assert 'Nessuna decisione di sviluppo pronta per Daniele.' in source
    assert 'Da decidere ora' in source
    assert 'Polso di oggi' in source
    assert 'Ultimo ciclo' in source
    assert 'Registry aggiornato' in source
    assert 'Dispatch" value="solo dopo sblocco"' in source
    assert 'Lavoro partito in Kanban' in source
    assert 'isLaunchedKanbanWork' in source
    assert 'ACTIVE_KANBAN_STATUSES' in source
    assert 'isActiveHandoff' in source
    assert 'Mostro solo handoff ancora attivi' in source
    assert 'openKanbanTask(board, task.id)' in source
    assert 'rememberKanbanBoard' in source
    assert '"/kanban-mission-v2"' in source
    assert 'kanbanBoardHref(config.board)' in source
    assert 'kanbanBoardHref(config.board, "/v2")' in source
    assert 'Proposte attive / mature' in source
    assert 'Field label="Formulata" value={formatProposalFormulatedDate(proposal)}' in source
    assert 'registryActiveProposals' in source
    assert 'isVisibleRegistryProposal' in source
    assert 'registry visibile' in source
    assert 'MatureProposalCard' in source
    assert 'AutonomousProposalEvidenceGrid proposal={proposal}' in source
    assert source.count('AutonomousProposalEvidenceGrid proposal={proposal}') >= 2
    assert 'Segnale' in source
    assert 'Interpretazione' in source
    assert 'Supporter' in source
    assert 'Critic' in source
    assert 'Chief synthesis' in source
    assert 'Gate / dispatch' in source
    assert 'solo dopo conferma' in source
    assert 'Accetta' in source
    assert 'Sviluppa' in source
    assert 'Scarta' in source
    assert 'reviewTeamProposalAsChief' in source
    assert 'convertTeamProposalToPlan(editingProposal.id, preview.preview_hash, config.board)' in source
    assert 'Standby — interessano, non ora' in source
    assert 'non bloccano nuove proposte' in source
    assert 'Modifica prima di sviluppare' in source
    assert 'Salva e lancia piano ready' in source
    assert 'updateTeamProposal(editingProposal.id' in source
    assert '| "standby"' in api_ts
    assert 'formulated_at?: string;' in api_ts
    assert 'mature_proposals?: TeamProposal[];' in api_ts
    assert 'standby_proposals?: TeamProposal[];' in api_ts
    assert 'updateTeamProposal: (id: string, proposal: TeamProposalUpdateRequest)' in api_ts
    assert 'convertTeamProposalToPlan: (id: string, confirmed_preview_hash: string, board?: string)' in api_ts
    assert 'last_cycle?: TeamPulseCycle | null;' in api_ts
    assert 'Salute del team' in source
    assert 'Handoff' in source
    assert 'Approfondimenti' in source
    assert 'Archivio contenuto precedente' in source
    assert '/team-evolution/archive' in source
    assert '/team-work/archive' in source
    assert '/team-proposals/archive' in source
    assert 'Approva' in source
    assert 'Modifica' in source
    assert 'Indirizza' in source
    assert 'Scarta' in source
    assert 'Dispatcher attivabile secondo priorità' in source
    assert 'task.status !== "blocked"' in source
    assert 'Da chi' in source
    assert 'Verso chi' in source
    assert 'Perché' in source
    assert 'Link card' in source
    assert 'commentKanbanTask' in api_ts
    assert 'getKanbanBoard' in api_ts


def test_team_command_center_launched_work_filter_does_not_hide_real_cards_with_test_briefs():
    source = COMMAND_PAGE_TSX.read_text()

    assert 'Only synthetic/probe cards should be filtered out' in source
    assert '/^(\\[[^\\]]+\\]\\s*)?TEST\\b/i.test(title)' in source
    assert '/should invalidate hash/i.test(title)' in source
    assert '/\\bTEST\\b|test\\/escluse|card test/i.test(taskText(task))' not in source


def test_team_command_center_does_not_render_legacy_registry_on_main_page():
    source = COMMAND_PAGE_TSX.read_text()

    assert 'OldTeamProposalsPage mode={mode === "proposals" ? "operative" : mode}' in source
    assert 'function ArchiveShell' in source
    assert 'Registry legacy, Radar, Chief queue, PM workspace e review duplicate' in source
    assert 'Chief review queue' not in source
    assert 'RadarHermesSection' not in source
    assert 'Hermes PM — cockpit di lavoro' not in source
    assert 'ProposalColumn' not in source


def test_team_proposals_page_reads_tab_query_and_shows_ready_launch_gate_copy():
    source = PAGE_TSX.read_text()

    assert 'mode: initialMode = "autonomous"' in source
    assert 'title: "Team & Proposte"' in source
    assert 'Status/preview registry-only · conferma finale ready' in source
    assert 'Proposte operative' in source
    assert 'Proposte evolutive' in source
    assert 'useSearchParams' in source
    assert 'value === "autonomous"' in source
    assert 'value === "pm") return "evolution"' in source
    assert 'Hermes PM' in source
    assert '/chat?profile=hermespm' in source
    assert 'Hermes PM — cockpit di lavoro' in source
    assert 'Sviluppo Hermes' in source
    assert 'Apri, scegli una mossa, oppure vai agli approfondimenti' in source
    assert 'Approfondimenti e audit' in source
    assert 'Opzione avanzata: resta fuori dalla vista principale' in source
    assert 'Team sviluppo Hermes' in source
    assert 'Preview piano auto-assegnato' in source
    assert 'Formulata: {formatProposalFormulatedDate(proposal)}' in source
    assert 'Conferma e lancia piano ready in Kanban' in source
    assert 'La conferma crea parent e task figli in <strong>ready</strong>' in source
    assert 'Task singola ready…' in source
    assert 'Conferma e lancia task ready in Kanban' in source
    assert 'ready-on-convert' in source


def test_team_operational_view_is_read_only_approval_gated():
    source = PAGE_TSX.read_text()

    assert 'Registro verificato' in source
    assert 'Proposte operative challengeate' in source
    assert 'Da arricchire prima di decidere' in source
    assert 'Radar evolutivo separato' in source
    assert 'Vedi preview per Mission Control' in source
    assert 'La preview non crea task' in source
    assert 'Kanban entra ready: il dispatcher parte secondo priorità e concorrenza' in source
    assert 'no cron, no webhook, no invii esterni' in source
    assert 'proposal-only' in source
    assert 'Review Chief pending / default' in source
    assert 'Nota evidenza: “2 proposte attive e 8 review pending”' in source
    assert 'readOnly={!isEvolutionMode}' in source


def test_team_operational_cards_show_autonomous_contract_and_human_only_gate_actions():
    source = PAGE_TSX.read_text()

    assert 'AutonomousProposalContractGrid' in source
    assert 'FieldBlock label="Segnale" value={formatContractValue(proposal.signal ?? proposal.source_signal ?? proposal.whyNow)}' in source
    assert 'FieldBlock label="Interpretazione" value={formatContractValue(proposal.interpretation, "interpretazione non disponibile")}' in source
    assert 'FieldBlock label="Supporter" value={formatViewpoint(proposal.supporter_view, proposal.challenge?.supporter, proposal.challenge?.support)}' in source
    assert 'FieldBlock label="Critic" value={formatViewpoint(proposal.critic_view, proposal.challenge?.critic, proposal.challenge?.challenge)}' in source
    assert 'FieldBlock label="Chief synthesis" value={formatContractValue(proposal.chief_synthesis ?? proposal.challenge?.chief_synthesis, "sintesi Chief non disponibile")}' in source
    assert 'FieldBlock label="Stato gate" value={formatGateState(proposal)}' in source
    assert 'FieldBlock label="source_agent" value={formatSourceAgent(proposal)}' in source
    assert 'FieldBlock label="Riferimenti evidenza" value={formatEvidenceRefs(proposal)}' in source
    assert 'Preview piano Kanban' in source
    assert 'Richiedi revisione' in source
    assert 'Nessuna azione qui avvia worker, cron, messaggi esterni o dispatch automatico.' in source
    assert 'source_agent mancante: record legacy normalizzato, audit trail preservato' in source


def test_team_proposals_embeds_radar_hermes_section_with_approval_gates():
    source = PAGE_TSX.read_text()

    assert 'api.getRadarHermes()' in source
    assert 'RadarHermesSection' in source
    assert 'Top proposte evolutive' in source
    assert 'Proposta controversa' in source
    assert 'Proposta parcheggiabile' in source
    assert 'READ-ONLY · APPROVAL-GATED' in source
    assert 'Preview read-only: nessun task, cron, dispatch o invio esterno parte senza conferma di Daniele.' in source
    assert 'Assignee suggerito' in source
    assert 'Mission Control' in source
    assert 'Crea task — gated' in source
    assert 'disabled' in source


def test_agent_growth_card_renders_read_only_fields_and_clear_missing_states():
    source = PAGE_TSX.read_text()

    assert 'AgentGrowthCard' in source
    assert 'AgentGrowthFieldRow' in source
    assert 'Crescita osservabile' in source
    assert 'Ultimo segnale osservato' in source
    assert 'Proposta propria' in source
    assert 'Challenge ricevute' in source
    assert 'Learning note' in source
    assert 'Prossimo sviluppo ruolo' in source
    assert 'Non è una classifica tra agenti.' in source
    assert 'Gate: read-only, review Daniele richiesta, nessun cron/invio/dispatch automatico.' in source
    assert 'Dato mancante:' in source


def test_team_prompts_and_state_logs_are_applied_to_two_pages():
    page = PAGE_TSX.read_text()
    api_ts = API_TS.read_text()

    assert 'TeamConstitutionPanel constitution={constitution} mode={mode}' in page
    assert 'Costituzione applicata' in page
    assert 'Prompt attivo: Costituzione comune + identità specifica' in page
    assert 'Lettura a inizio ciclo' in page
    assert 'Handoff se emerge dominio dell\'altro team' in page
    assert 'TeamConstitutionContract' in api_ts
    assert 'team_constitutions?:' in api_ts
    assert 'cycle_start_reads: string[];' in api_ts


def test_agent_growth_legacy_score_contract_allows_null_for_insufficient_data():
    source = API_TS.read_text()

    assert 'growth_score: number | null;' in source
    assert 'growth_score: number;' not in source


def test_team_proposals_active_filter_hides_archived_and_converted_duplicates():
    source = PAGE_TSX.read_text()

    assert '"converted_to_kanban"' in source
    assert '"archived"' in source
    assert '"rejected"' in source
    assert '"parked"' in source


def test_team_pulse_controversy_lane_renders_before_shortlist_and_edge_states():
    source = PAGE_TSX.read_text()

    assert 'TeamPulseControversyLaneSection lane={modePulse.controversy_lane}' in source
    assert source.index('TeamPulseControversyLaneSection lane={modePulse.controversy_lane}') < source.index('StrategicList title="Top proposte autonome"')
    assert 'Prima della shortlist: criticità utili' in source
    assert 'Proposta contestata' in source
    assert 'Critic · Obiezione sollevata da' in source
    assert 'Why contested · Perché è contestata' in source
    assert 'Veto risk · Rischio veto' in source
    assert 'Chief synthesis · Sintesi Chief' in source
    assert 'Evidenza e provenienza' in source
    assert 'Prossima azione' in source
    assert 'Proposer: {item.proposer.subagent_label}' in source
    assert 'no_meaningful_controversy' in source
    assert 'insufficient_review_data' in source
    assert 'Revisione incompleta' in source
    assert 'Critic multipli' in source
    assert 'High veto risk' in source
    assert 'non shortlist senza risoluzione' in source
    assert 'Chief: da decidere' in source
    assert 'contratto legacy assente' in source

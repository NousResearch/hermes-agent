# Phase 1 VPS Checklist

## Préparation

- [ ] Se connecter au VPS
- [ ] Aller dans le dépôt `hermes-agent`
- [ ] Activer le venv
- [ ] Créer une branche dédiée
- [ ] Vérifier la version Python
- [ ] Installer `jsonschema` si absent

## Pack de gouvernance

- [ ] Copier le dossier `hermes/` dans le dépôt du VPS
- [ ] Vérifier la présence des prompts, contrats, policies et schémas
- [ ] Copier `hermes/integration/governance_runtime.py` vers `agent/governance_runtime.py`

## Configuration Hermes

- [ ] Ajouter `agent.governance` dans `hermes_cli/config.py`
- [ ] Activer `agent.governance.enabled: true` dans `~/.hermes/config.yaml`
- [ ] Définir `pack_root` vers le bon chemin absolu

## Patch runtime Hermes

- [ ] Importer `GovernanceRuntime`, `GovernanceState`, `GovernanceBlocked` dans `run_agent.py`
- [ ] Initialiser le runtime de gouvernance dans `AIAgent.__init__`
- [ ] Overlay des schémas d'outils après `get_tool_definitions(...)`
- [ ] Vérifier les outils contractuels manquants au démarrage
- [ ] Préfixer le system prompt avec le prompt métier
- [ ] Mettre à jour l'état de gouvernance après chaque résultat outil
- [ ] Évaluer la politique après chaque batch d'outils
- [ ] Bloquer ou escalader si la politique l'exige
- [ ] Valider la réponse finale contre le schéma JSON
- [ ] Réinjecter un message de réparation si le JSON final est invalide

## Outils

- [ ] Vérifier que `search_fiscal_sources` existe
- [ ] Vérifier que `search_accounting_sources` existe
- [ ] Vérifier que `search_legal_sources` existe
- [ ] Vérifier que `compute_tax_liability` existe
- [ ] Vérifier que `get_client_records` existe
- [ ] Vérifier que `log_audit_event` existe
- [ ] Vérifier que `escalate_to_human_supervisor` existe

## Tests minimaux

- [ ] Test fiscal simple
- [ ] Test conseil personnalisé interdit
- [ ] Test source manquante
- [ ] Test écriture comptable équilibrée
- [ ] Test contentieux / contrôle fiscal

## Validation finale

- [ ] Aucune réponse fiscale/juridique sans source
- [ ] Aucun conseil personnalisé autonome
- [ ] Aucun JSON final invalide
- [ ] Blocage ou escalade sur les cas sensibles
- [ ] Comportement stable sur 5 scénarios consécutifs

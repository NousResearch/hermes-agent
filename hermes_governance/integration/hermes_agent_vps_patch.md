# Hermes Agent VPS Patch Guide — Phase 2 Hardening

Guide opératoire ciblé sur la Phase 2 : discipline de tool use, qualité métier des wrappers, réduction des boucles et terminaison JSON stricte.

## 1. Préparation VPS

```bash
cd /path/to/hermes-agent
source venv/bin/activate
git checkout -b codex/hermes-phase2-hardening
python --version
```

Si `jsonschema` n'est pas présent dans le venv :

```bash
pip install jsonschema
```

## 2. Copier le pack mis à jour

Transférer le dossier `hermes/` local vers le dépôt distant.

Le dépôt distant doit contenir au minimum :

- `hermes/prompts/system_prompt.production.md`
- `hermes/contracts/tool_contracts.v1.json`
- `hermes/policies/escalation_policy.v1.json`
- `hermes/schemas/final_response.schema.json`
- `hermes/schemas/policy_context.schema.json`
- `hermes/integration/governance_runtime.py`
- `hermes/integration/openlegi_code_version_adapter.py`

## 3. Config runtime

Fichier : `hermes_cli/config.py`

Dans `DEFAULT_CONFIG["agent"]`, conserver ou ajouter :

```python
"governance": {
    "enabled": False,
    "pack_root": "hermes",
    "escalation_materiality_eur": 10000,
    "deadline_escalation_days": 7,
    "strict_final_response_schema": True,
},
```

Les budgets de répétition et de réparation finale sont maintenant portés par `hermes/contracts/tool_contracts.v1.json`.

## 4. Runtime Python

Copier `hermes/integration/governance_runtime.py` vers :

```text
agent/governance_runtime.py
```

Copier aussi `hermes/integration/openlegi_code_version_adapter.py` vers :

```text
agent/openlegi_code_version_adapter.py
```

## 5. Import dans `run_agent.py`

Ajouter :

```python
from agent.governance_runtime import GovernanceRuntime, GovernanceState, GovernanceBlocked
```

## 6. Initialisation dans `AIAgent.__init__`

Après chargement de `_agent_cfg`, conserver l'initialisation Phase 1 :

```python
self._governance_runtime = None
self._governance_state = GovernanceState()
self._governance_cfg = (_agent_cfg.get("agent", {}) or {}).get("governance", {}) or {}

if self._governance_cfg.get("enabled"):
    pack_root = self._governance_cfg.get("pack_root", "hermes")
    self._governance_runtime = GovernanceRuntime(
        pack_root=Path(pack_root),
        escalation_materiality_eur=float(self._governance_cfg.get("escalation_materiality_eur", 10000)),
        deadline_escalation_days=int(self._governance_cfg.get("deadline_escalation_days", 7)),
    )
```

## 7. Verrouillage de la surface d'outils

### 7.1 `model_tools.py`

Le profil Hermes ne doit plus exposer les tools MCP bruts. Si la gouvernance Hermes est active, ne charger que le module wrapper :

```python
tool_modules = ["tools.hermes_governance_tools"] if governance_enabled else DEFAULT_TOOL_MODULES
```

Contrainte :
- ne pas enregistrer `mcp_openlegi_*`
- ne pas enregistrer de tool de recherche brute concurrent des wrappers métier

### 7.2 Overlay + filtrage dans `run_agent.py`

Remplacer :

```python
if self._governance_runtime and self.tools:
    self.tools = self._governance_runtime.overlay_tool_definitions(self.tools)
```

par :

```python
if self._governance_runtime and self.tools:
    self.tools = self._governance_runtime.prepare_tool_definitions(self.tools)
```

Puis conserver le fail-closed sur les tools contractuels manquants :

```python
if self._governance_runtime:
    missing = self._governance_runtime.ensure_contract_tools_exist(self.valid_tool_names)
    if missing:
        raise RuntimeError(
            "Governance-enabled Hermes is missing required tools: " + ", ".join(sorted(missing))
        )
```

## 8. Normalisation d'arguments et fingerprint d'appels

Fichier : `run_agent.py`, méthode `_invoke_tool`

Avant l'appel réel au tool :

```python
normalized_arguments = function_args
if self._governance_runtime:
    normalized_arguments = self._governance_runtime.normalize_tool_arguments(
        function_name,
        function_args,
    )
```

Utiliser ensuite `normalized_arguments` pour l'invocation effective.

Après récupération du `result` :

```python
if self._governance_runtime:
    self._governance_runtime.update_state_from_tool_result(
        self._governance_state,
        function_name,
        normalized_arguments,
        result,
    )
```

Effets Phase 2 :
- les alias de compatibilité sont rabattus vers les noms canoniques
- un fingerprint stable par `(tool_name, arguments normalisés)` est calculé
- un second appel identique avec résultat utile devient une violation gouvernée

## 9. Politique après chaque batch d'outils

Conserver l'évaluation Phase 1 après `_execute_tool_calls_*`.

Le changement métier est dans le pack :
- blocage dédié sur répétition d'appel identique
- escalade RGPD via `escalate_to_human_supervisor`

## 10. Réparation finale stricte

Fichier : `run_agent.py`, bloc `# No tool calls - this is the final response`

Remplacer la validation simple par :

```python
final_response = assistant_message.content or ""

if self._governance_runtime and self._governance_cfg.get("strict_final_response_schema", True):
    try:
        validated_payload = self._governance_runtime.validate_final_response_text(final_response)
        final_response = json.dumps(validated_payload, ensure_ascii=False)
    except GovernanceBlocked as exc:
        try:
            self._governance_runtime.register_final_response_repair_attempt(self._governance_state)
        except GovernanceBlocked as repair_exc:
            final_response = self._governance_runtime.build_blocked_final_response(
                self._governance_state,
                str(repair_exc),
            )
            break

        messages.append({
            "role": "user",
            "content": self._governance_runtime.build_repair_message(str(exc)),
        })
        continue
```

Effets Phase 2 :
- refus des sorties mixtes prose + JSON
- acceptation éventuelle d'un objet JSON unique même si rendu dans un fence unique
- budget borné de réparation finale
- fallback bloqué toujours conforme au schéma

## 11. Blocage politique conforme au schéma

Dans `run_conversation`, remplacer la construction ad hoc du JSON de blocage par :

```python
except GovernanceBlocked as exc:
    final_response = self._governance_runtime.build_blocked_final_response(
        self._governance_state,
        str(exc),
    )
    break
```

Le helper runtime garantit désormais :
- `sources` non vide
- `audit_trail` enrichi avec fingerprints et violations de répétition
- JSON final déjà conforme au schéma

## 12. Wrappers métier à ajuster

Fichier : `tools/hermes_governance_tools.py`

### 12.1 `search_accounting_sources`

Ajouter une normalisation d'aliases en entrée avant tout traitement backend :

```python
ARG_ALIASES = {
    "query": ("search_query", "question", "prompt", "request"),
    "ledger_context": ("context", "scope", "accounting_scope", "ledger_scope"),
    "fact_date": ("date", "as_of_date", "effective_date", "reference_date"),
}

def _normalize_aliases(payload: dict[str, Any], alias_map: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    normalized = dict(payload)
    for canonical, aliases in alias_map.items():
        if canonical in normalized:
            continue
        for alias in aliases:
            if alias in normalized:
                normalized[canonical] = normalized[alias]
                break
    return normalized
```

Puis, dans le wrapper :
- normaliser les arguments
- ne conserver que `query`, `ledger_context`, `fact_date`, `jurisdiction`, `include_ifrs_divergence`
- retourner un `coverage_status=not_verified` déterministe tant que le backend comptable réel n'existe pas

### 12.2 `search_fiscal_sources`

Durcir la fabrication de requête avant d'interroger le backend :

```python
def _build_fiscal_query(query: str, tax_scope: str) -> str:
    compact = " ".join(query.strip().split())
    lower = compact.lower()
    if "non déduct" in lower or "non deduct" in lower or "amende" in lower or "pénalité" in lower or "penalite" in lower:
        return f"{tax_scope} déductibilité non déductible amende pénalité CGI BOFiP {compact}"
    return f"{tax_scope} CGI BOFiP {compact}"
```

Contrainte :
- ne pas relancer une deuxième fois à l'identique depuis le wrapper
- conserver la traçabilité `trace_id`

### 12.3 `get_code_article_version`

Objectif :
- résoudre un article de code précis
- sélectionner uniquement la version dont la fenêtre de vigueur couvre `fact_date`
- échouer fermé si la vigueur n'est pas démontrable

Importer l'adaptateur :

```python
from agent.openlegi_code_version_adapter import (
    build_rechercher_code_args,
    normalize_fact_date,
    select_code_article_version,
)
```

Ajouter le wrapper dans `tools/hermes_governance_tools.py` :

```python
def get_code_article_version_tool(arguments: dict[str, Any]) -> str:
    args = _normalize_aliases(arguments or {}, {
        "article_ref": ("reference", "article", "article_number", "num_article"),
        "code_name": ("code", "code_title", "code_label"),
        "fact_date": ("date", "as_of_date", "effective_date", "reference_date"),
        "include_text": ("need_article_text", "need_text", "full_text"),
    })

    article_ref = str(args["article_ref"]).strip()
    code_name = str(args["code_name"]).strip()
    fact_date = normalize_fact_date(str(args["fact_date"]))
    include_text = bool(args.get("include_text", True))
    max_candidate_blocks = int(args.get("max_candidate_blocks", 20))

    tool_args = build_rechercher_code_args(
        article_ref=article_ref,
        code_name=code_name,
        max_candidate_blocks=max_candidate_blocks,
    )

    raw_result = handle_function_call(
        "mcp_openlegi_rechercher_code",
        tool_args,
        task_id=None,
        user_task=f"get_code_article_version:{code_name}:{article_ref}:{fact_date}",
    )

    version = select_code_article_version(
        raw_text=str(raw_result or ""),
        article_ref=article_ref,
        code_name=code_name,
        fact_date=fact_date,
    )

    if version is None:
        return json.dumps({
            "success": False,
            "tool_name": "get_code_article_version",
            "article_ref": article_ref,
            "code_name": code_name,
            "fact_date": fact_date,
            "version_found": False,
            "in_force_on_fact_date": False,
            "coverage_status": "not_verified",
            "text": None,
            "vigueur_start": None,
            "vigueur_end": None,
            "source_url": None,
            "blocking_reason": "no_exact_article_version_for_fact_date",
        }, ensure_ascii=False)

    return json.dumps({
        "success": True,
        "tool_name": "get_code_article_version",
        "article_ref": version.article_ref,
        "code_name": version.code_name,
        "fact_date": version.fact_date,
        "version_found": True,
        "in_force_on_fact_date": True,
        "coverage_status": "verified",
        "text": version.text if include_text else None,
        "vigueur_start": version.vigueur_start,
        "vigueur_end": version.vigueur_end,
        "source_url": version.source_url,
        "source_ids": list(version.source_ids),
        "selection_mode": version.selection_mode,
    }, ensure_ascii=False)
```

Ajouter son schéma d'outil dans le module et l'enregistrer comme wrapper métier Hermes.

Contrainte :
- `mcp_openlegi_rechercher_code` doit être appelé avec `champ="NUM_ARTICLE"` et `type_recherche="EXACTE"`
- aucune extrapolation si plusieurs blocs existent mais qu'aucune fenêtre ne couvre `fact_date`
- aucune réponse finale ne doit réutiliser un article sans `coverage_status="verified"` lorsqu'une garantie temporelle était demandée

## 13. Activation dans `~/.hermes/config.yaml`

```yaml
agent:
  governance:
    enabled: true
    pack_root: /path/to/hermes-agent/hermes
    escalation_materiality_eur: 10000
    deadline_escalation_days: 7
    strict_final_response_schema: true
```

## 14. Vérifications minimales

```bash
source venv/bin/activate
python -m pytest tests/test_model_tools.py -q
python -m pytest tests/test_cli_init.py -q
python -m pytest tests/test_run_agent.py -q
```

Tests manuels prioritaires :

```bash
python run_agent.py "Quel est le traitement fiscal en IS d'une amende administrative non déductible ?"
python run_agent.py "Comptabilise une facture de fournitures de bureau de 1200 EUR TTC payée par banque."
```

Attendus :
- aucun `mcp_*` exposé au modèle
- appel unique ou raffinement matériellement distinct de `search_fiscal_sources`
- `search_accounting_sources` tolère les alias et normalise vers les noms canoniques
- aucune sortie finale mixte prose + JSON
- en cas de dérive, JSON de blocage toujours conforme

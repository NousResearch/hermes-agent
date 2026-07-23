#!/usr/bin/env python3
"""factory_admission_hook.py — pont Hermes `pre_tool_call` -> gate d'admission.

Ce script est le point d'intégration runtime de la porte d'admission worktree
(HER-95). Il se branche via le mécanisme de shell-hooks générique déjà chargé
par `cli.py`, `hermes_cli/main.py` et `gateway/run.py`
(`agent.shell_hooks.register_from_config`) — aucun changement de core n'est
requis. Déclaration opt-in dans `cli-config.yaml` (profile-aware) :

    hooks:
      pre_tool_call:
        - matcher: "terminal|patch|write_file|str_replace_editor|apply_patch"
          command: >-
            python3 /ABS/scripts/factory_admission_hook.py
            --registry /ABS/registry --agent default

    # Profil métier (ex. hermes-immo) : le refus de domaine est AUTOMATIQUE car
    # --profile/--domain-prefixes vivent dans la config du profil, pas dans un
    # flag que l'appelant doit se souvenir de passer.
    hooks:
      pre_tool_call:
        - matcher: "terminal|patch|write_file|str_replace_editor|apply_patch"
          command: >-
            python3 /ABS/scripts/factory_admission_hook.py
            --registry /ABS/registry --agent hermes-immo
            --profile hermes-immo --domain-prefixes JYI,HER

Protocole (voir `agent/shell_hooks.py`) : la charge utile arrive en JSON sur
stdin ; pour bloquer un tool AVANT son exécution, on émet sur stdout
`{"decision": "block", "reason": "..."}` (traduit par `_parse_response` en
`{"action": "block", "message": "..."}`, la forme que
`get_pre_tool_call_block_message` remonte au dispatcher de tools).

Posture : LECTURE SEULE. Le hook n'écrit jamais d'owner et ne persiste jamais le
PID éphémère de ce subprocess (cf. blocker identité process) — il ne fait
qu'interroger `evaluate_admission_guard`. Fail-open advisory si l'infra est
absente/anormale (pas de registry, pas un dépôt git) ; fail-closed (block) sur
un conflit d'occupation avéré ou une violation de domaine métier.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import factory_lane  # noqa: E402  (résolu via le sys.path ci-dessus)

# Événements build-capable pour lesquels le gate a un sens. Le matcher du hook
# filtre déjà côté Hermes ; cette liste est une défense en profondeur si le hook
# est câblé sans matcher.
_MUTATING_TOOLS = frozenset({
    "terminal", "patch", "write_file", "str_replace_editor", "apply_patch",
    "edit_file", "create_file", "delete_file", "move_file",
})


def _emit_block(reason):
    print(json.dumps({"decision": "block", "reason": reason}))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="factory_admission_hook")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--profile")
    parser.add_argument("--domain-prefixes")
    # Optionnel : restreint le gate à un sous-ensemble de tools même sans matcher.
    parser.add_argument("--only-mutating", action="store_true")
    args = parser.parse_args(argv)

    # 1) Charge utile stdin — illisible => fail-open (ne jamais casser un tool).
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0
    if not isinstance(payload, dict):
        return 0
    if payload.get("hook_event_name") != "pre_tool_call":
        return 0
    if args.only_mutating and payload.get("tool_name") not in _MUTATING_TOOLS:
        return 0

    session = payload.get("session_id") or ""
    cwd = payload.get("cwd") or os.getcwd()

    # 2) Infra absente/anormale => fail-open advisory.
    try:
        root = factory_lane._safe_registry_root(args.registry)
    except Exception:
        return 0

    try:
        worktree_real = factory_lane._git_toplevel_or_none(cwd)
        if worktree_real is None:
            worktree_real = os.path.realpath(cwd)
        allowed, reason = factory_lane.evaluate_admission_guard(
            root, worktree_real, args.agent, session,
            profile=args.profile, domain_prefixes=args.domain_prefixes,
        )
    except Exception:
        # Une anomalie inattendue du gate advisory ne doit pas geler tous les
        # tools de la session — fail-open, le conflit avéré reste fail-closed.
        return 0

    if not allowed:
        _emit_block(reason or "worktree admission denied")
    return 0


if __name__ == "__main__":
    sys.exit(main())

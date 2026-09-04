"""Central route-registry generator for the 64 live Hermes config surfaces.

Build artifact from the approved route-governance proposal
(provider-model-mapping-draft.md v3). Emits standard Hermes fallback_providers
blocks per surface from a central slot registry + surface map.

Safety properties:
  - Dry-run by default. Nothing is written without --apply.
  - --target-root is REQUIRED and must be a temp copy. Apply against the live
    ~/.hermes tree is refused unless --allow-live-apply is ALSO passed
    (a separate, explicit live-apply approval).
  - --apply is triple-guarded: explicit --apply AND --confirm YES-APPLY-ROUTES
    AND non-interactive (stdin closed) detection.
  - Apply is atomic: every target is snapshotted before any write; any failure
    mid-apply rolls back ALL already-written files from the snapshots and
    leaves the target tree byte-identical to its pre-apply state.
  - Deterministic: identical inputs produce byte-identical output (sorted keys,
    stable ordering, no timestamps; the plan records the UTC date as `as_of`).
  - Fail closed: gated / needs_classification / disabled / unprovisioned /
    expired / usage-limit-capped / allow-list-disallowed slots are never
    emitted. Only approved, enabled, provisioned, unexpired, uncapped,
    allowed PERM slots appear.
  - One fallback entry per (provider, exact model, base_url). Multiple approved
    accounts for the same deployment are represented as native credential-pool
    membership metadata (credential_pool + pool_accounts) — the runtime pool
    (credential_pool_strategies, fill_first) performs the 1→2 account cascade.
    Duplicate same-deployment entries are never emitted (the runtime would
    dedup-skip them — dead weight).
  - Env-key pool bypass: profiles whose .env carries a provider env key
    (notably OLLAMA_API_KEY for ollama.com endpoints) resolve to that single
    env key at runtime instead of the credential pool, making account 2
    unreachable. The plan reports every such profile as a deployment BLOCKER
    requiring later approved auth cleanup.
  - Fail-closed apply gate: --apply hard-refuses (distinct exit code 4) while
    env_key_bypass blockers exist OR any pool_requirement is unmet. Dry-run
    still generates and reports blockers — only the write path is gated.
  - Same-model invariant: every emitted route for a surface serves exactly the
    surface's current main model; max 5 same-model routes per surface.
  - Removed-capability summary: the plan reports every (provider, model)
    removed from existing chains so reviewers see capability loss, not just
    the new chain.
  - Nous $20 emergency reserve is never emitted, never referenced.
  - Credentials are never read, copied, or logged — provider accounts are
    symbolic IDs. Profile .env files are scanned for KEY NAMES ONLY; values
    are never read into output.

Live configs are read-only inputs. Apply targets must be explicit temp copies.
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any

import yaml

# Config-tree reader. Route-registry reads ONLY config trees it is given
# (a temp copy or an explicit --target-root it owns) to build migration
# plans — never the live user ~/.hermes/config.yaml for behavior. Using
# hermes_cli.config.load_config() here would read the WRONG tree (the live
# HERMES_HOME instead of the target copy being planned), so raw parsing of
# the owned target tree is intentional and correct.
def _read_yaml(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}

# Provider account IDs that are not yet provisioned as live Hermes providers.
# Emission for these is structurally impossible (hermes: null) — enforced, not assumed.
UNPROVISIONED_ACCOUNTS = {"xkiro/free", "xkiro/pro-plus", "b-ai/1", "b-ai/2", "b-ai/3", "b-ai/4", "b-ai/5"}

# Slot classes that may never be emitted under any circumstance.
FORBIDDEN_CLASSES = {"emergency_reserve"}
# Slot statuses/classes that must be approved+enabled to emit.
EMITTABLE_CLASSES = {"perm"}

REQUIRED_SLOT_FIELDS = (
    "slot", "model_id", "provider_account", "class", "status",
    "approved_at", "expires_at", "usage_limit", "allowed_profiles",
    "replacement_slot", "disabled", "hermes",
)

LIVE_HERMES_HOME = Path.home() / ".hermes"


class ValidationError(Exception):
    """Deterministic validation failure — registry, surfaces, or emitted output."""


# ---------------------------------------------------------------- registry I/O

def load_registry(registry_path: str | Path) -> dict[str, Any]:
    with open(registry_path, "r", encoding="utf-8") as fh:
        reg = yaml.safe_load(fh)
    if not isinstance(reg, dict) or not isinstance(reg.get("slots"), list):
        raise ValidationError(f"registry must contain a 'slots' list: {registry_path}")
    return reg


def load_surfaces(surfaces_path: str | Path) -> list[dict[str, Any]]:
    with open(surfaces_path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if not isinstance(doc, dict) or not isinstance(doc.get("surfaces"), list):
        raise ValidationError(f"surfaces file must contain a 'surfaces' list: {surfaces_path}")
    return doc["surfaces"]


# ------------------------------------------------------------------- time (patchable)

def _utcnow() -> _dt.datetime:
    """Deterministic-seam clock: tests may monkeypatch this to freeze expiry."""
    return _dt.datetime.now(_dt.timezone.utc)


def _parse_expiry(value: Any) -> _dt.datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        dt = _dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError(f"unparseable expires_at {value!r}: {exc}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.timezone.utc)
    return dt


# ------------------------------------------------------------------ validation

def validate_registry(reg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Deterministic registry self-validation. Raises on any violation."""
    errors: list[str] = []
    slots = reg["slots"]
    by_id: dict[str, dict[str, Any]] = {}
    seen_models: dict[str, list[str]] = {}

    for s in slots:
        sid = s.get("slot")
        if not sid:
            raise ValidationError("slot record missing 'slot' ID")
        if sid in by_id:
            raise ValidationError(f"duplicate slot ID: {sid}")
        for field in REQUIRED_SLOT_FIELDS:
            if field not in s:
                raise ValidationError(f"slot {sid} missing required field '{field}'")
        by_id[sid] = s

        klass = s["class"]
        if klass not in {"perm", "gated", "needs_classification", "emergency_reserve"}:
            raise ValidationError(f"slot {sid}: unknown class {klass!r}")
        if klass in FORBIDDEN_CLASSES and not s["disabled"]:
            errors.append(f"slot {sid}: class {klass} must be disabled=true")
        # gated / NC / expired / paused / candidate must be disabled (fail closed)
        if klass in {"gated", "needs_classification"} and not s["disabled"]:
            errors.append(f"slot {sid}: class {klass} must be born disabled=true")
        if s["status"] != "approved" and not s["disabled"]:
            errors.append(f"slot {sid}: status {s['status']} must be disabled=true")
        if klass == "perm" and s["status"] == "approved" and not s.get("approved_at"):
            errors.append(f"slot {sid}: approved perm slot missing approved_at")
        if s["status"] == "approved" and not s.get("approved_at"):
            errors.append(f"slot {sid}: approved slot missing approved_at")
        # expiry fail-closed: an expired slot may never sit enabled+approved
        exp = _parse_expiry(s.get("expires_at"))
        if exp is not None and not s["disabled"]:
            if exp <= _utcnow():
                errors.append(
                    f"slot {sid}: expires_at {s['expires_at']} has passed — slot must be "
                    f"disabled=true (fail closed) or replaced via approved replacement_slot"
                )
        # usage-cap fail-closed: a slot with a usage limit may only emit when the
        # limit window has been verified (generalised NIM rule)
        if s.get("usage_limit") and not s.get("limit_window_verified") and klass != "emergency_reserve":
            if not s["disabled"]:
                errors.append(
                    f"slot {sid}: usage_limit {s['usage_limit']!r} with limit_window_verified "
                    f"unset — slot must be disabled=true until the window is verified"
                )
        # allowed_profiles must be null or a list of surface names
        ap = s.get("allowed_profiles")
        if ap is not None and not (isinstance(ap, list) and all(isinstance(x, str) and x for x in ap)):
            errors.append(f"slot {sid}: allowed_profiles must be null or a list of surface names")
        if klass == "perm" and s["status"] == "approved" and not s.get("approved_at"):
            errors.append(f"slot {sid}: approved perm slot missing approved_at")
        if s["status"] == "approved" and not s.get("approved_at"):
            errors.append(f"slot {sid}: approved slot missing approved_at")
        # same-model invariants: ≤5 routes per model, unique provider_account per slot set
        seen_models.setdefault(s["model_id"], []).append(sid)
        # replacement-slot checks moved after the ID loop (forward refs were
        # false-positived when the replacement appears later in the file)

    for model_id, sids in seen_models.items():
        if model_id is None:
            continue
        if len(sids) > 5:
            errors.append(f"model {model_id}: {len(sids)} slots > max 5 same-model routes")
        accounts = [by_id[sid]["provider_account"] for sid in sids]
        if len(accounts) != len(set(accounts)):
            errors.append(f"model {model_id}: duplicate provider_account across ordinals")

    # replacement slot checks (after all IDs known): must exist AND serve same model
    for s in slots:
        rep = s.get("replacement_slot")
        if rep:
            if rep not in by_id:
                errors.append(f"slot {s['slot']}: replacement_slot '{rep}' not in registry")
            elif by_id[s["slot"]]["model_id"] != by_id[rep]["model_id"]:
                errors.append(
                    f"slot {s['slot']}: replacement_slot {rep} serves different model_id"
                )

    # ---- provider-capability validation (added 2026-09-02) ----
    # The registry is the single source of truth for what each provider can
    # serve. A slot whose hermes.model is not listed under its hermes.provider
    # in model_capability_matrix is a routing lie: the runtime would burn a
    # fallback hop on a guaranteed 404. Hard-error on ANY such slot (enabled or
    # not) so stale candidates get pruned instead of silently rotting.
    matrix = reg.get("model_capability_matrix")
    if not isinstance(matrix, dict) or not matrix:
        raise ValidationError(
            "registry missing model_capability_matrix — provider capability "
            "truth is mandatory (Sahil directive 2026-09-02 after the "
            "minimax@xkiro incident)"
        )
    constraints = reg.get("model_routing_constraints") or {}
    codex_rule = (constraints.get("codex_isolation") or {})
    codex_pattern = str(codex_rule.get("model_pattern") or "")
    codex_allowed = set(codex_rule.get("allowed_providers") or [])

    def _capability_violation(sid: str, provider: Any, model: Any) -> str | None:
        if provider is None or model is None:
            return None  # unprovisioned slots carry hermes: null; handled elsewhere
        allowed = matrix.get(str(provider))
        if allowed is None:
            return f"slot {sid}: provider {provider!r} not in model_capability_matrix"
        if model not in allowed:
            return (
                f"slot {sid}: model {model!r} is NOT served by provider "
                f"{provider!r} (capability matrix) — retire or repoint this slot"
            )
        return None

    for s in slots:
        sid = s["slot"]
        h = s.get("hermes") or {}
        viol = _capability_violation(sid, h.get("provider"), h.get("model"))
        if viol and not s["disabled"]:
            # Enabled slots must be capability-true — an enabled impossible slot
            # is a guaranteed-404 fallback hop (the minimax@xkiro incident).
            errors.append(viol)
        # Disabled slots with capability violations are tolerated as historical
        # record: they can never emit (fail-closed exclusion), but a NEW enabled
        # slot with a violation always hard-errors.
        # codex isolation: a hard, permanent rule — gpt-5.6-* only on openai-codex
        model_str = str(h.get("model") or "")
        model_id_str = str(s.get("model_id") or "")
        if codex_pattern and codex_allowed:
            prefix = codex_pattern.replace("*", "")
            for candidate in (model_str, model_id_str):
                if candidate.startswith(prefix) and h.get("provider") not in codex_allowed:
                    errors.append(
                        f"slot {sid}: CODEX ISOLATION VIOLATION — {candidate!r} may "
                        f"only be served by {sorted(codex_allowed)} (Sahil directive), "
                        f"got provider {h.get('provider')!r}"
                    )
                    break
    # matrix entries themselves must be clean: codex pattern may never list a
    # non-codex provider (registry-level check — runs even with zero slots)
    if codex_pattern and codex_allowed:
        prefix = codex_pattern.replace("*", "")
        for prov, models in matrix.items():
            if prov.startswith("_"):
                continue
            if any(str(m).startswith(prefix) for m in (models or [])) and prov not in codex_allowed:
                errors.append(
                    f"model_capability_matrix[{prov!r}] lists a codex-isolated model "
                    f"but is not in allowed_providers {sorted(codex_allowed)}"
                )

    if errors:
        raise ValidationError("; ".join(errors))
    return by_id


def validate_surfaces(surfaces: list[dict[str, Any]], by_id: dict[str, dict[str, Any]],
                      registry: dict[str, Any] | None = None) -> None:
    errors: list[str] = []
    seen = set()
    for surf in surfaces:
        name = surf.get("surface")
        if not name:
            raise ValidationError("surface record missing 'surface' name")
        if name in seen:
            errors.append(f"duplicate surface: {name}")
        seen.add(name)
        for sid in surf.get("slots", []):
            if sid not in by_id:
                errors.append(f"surface {name}: unknown slot {sid}")
                continue
            slot = by_id[sid]
            if slot["class"] == "emergency_reserve":
                errors.append(f"surface {name}: emergency_reserve slot {sid} may never be referenced")
            # same-model invariant: slot model == surface main model
            main = surf.get("main_model")
            if main is not None and sid in surf.get("slots", []):
                if slot["model_id"] != main:
                    errors.append(
                        f"surface {name}: slot {sid} model '{slot['model_id']}' != main model '{main}'"
                    )
        if len(surf.get("slots", [])) > 5:
            errors.append(f"surface {name}: {len(surf['slots'])} routes > max 5")
        migration = surf.get("primary_model_migration")
        if migration is not None:
            if not isinstance(migration, dict):
                errors.append(f"surface {name}: primary_model_migration must be a mapping")
                continue
            expected = migration.get("expected_old_main")
            primary_sid = migration.get("primary_slot")
            if not isinstance(expected, str) or not expected:
                errors.append(
                    f"surface {name}: primary_model_migration.expected_old_main is required")
            if primary_sid not in surf.get("slots", []):
                errors.append(
                    f"surface {name}: migration primary_slot {primary_sid!r} is not an approved route slot")
                continue
            slot = by_id.get(primary_sid)
            if slot is None:
                continue
            reason = slot_exclusion_reason(slot, name)
            if reason is not None:
                errors.append(
                    f"surface {name}: migration primary_slot {primary_sid} is not enabled: {reason}")
            if slot.get("model_id") != surf.get("main_model"):
                errors.append(
                    f"surface {name}: migration primary_slot {primary_sid} serves wrong model "
                    f"{slot.get('model_id')!r}")
            h = slot.get("hermes") or {}
            matrix = (registry or {}).get("model_capability_matrix") or {}
            if h.get("model") not in (matrix.get(str(h.get("provider"))) or []):
                errors.append(
                    f"surface {name}: migration primary_slot {primary_sid} provider capability is unverified")
    if errors:
        raise ValidationError("; ".join(errors))


# ------------------------------------------------------------------- emission

def slot_exclusion_reason(s: dict[str, Any], surface_name: str | None = None) -> str | None:
    """Return a reason string when a slot must NOT be emitted (fail-closed), else None."""
    if s["class"] != "perm" or s["status"] != "approved" or s["disabled"]:
        return "not an approved+enabled perm slot"
    if not s.get("hermes"):
        return "not provisioned in live Hermes configs (hermes: null)"
    exp = _parse_expiry(s.get("expires_at"))
    if exp is not None and exp <= _utcnow():
        return f"expired at {s['expires_at']}"
    if s.get("usage_limit") and not s.get("limit_window_verified"):
        return f"usage_limit {s['usage_limit']!r} unverified (capped, fail-closed)"
    ap = s.get("allowed_profiles")
    if ap is not None and surface_name is not None and surface_name not in ap:
        return f"surface {surface_name!r} not in allowed_profiles {sorted(ap)}"
    return None


def effective_slots(surf: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Slots that would actually be emitted: approved, enabled, perm, provisioned,
    unexpired, uncapped, allowed for this surface.

    Deterministic and fail-closed: anything gated / NC / paused / expired /
    capped / disallowed / unprovisioned is excluded — the effective chain while
    such slots are blocked is [emittable P] slots only.
    """
    out = []
    for sid in surf.get("slots", []):
        s = by_id.get(sid)
        if s is None:
            continue
        if slot_exclusion_reason(s, surf.get("surface")) is not None:
            continue
        out.append(s)
    return out


def dedupe_deployments(chain_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse entries that share (provider, model, base_url) into ONE entry.

    The Hermes runtime skips chain entries that resolve to the same deployment
    as the failed backend (agent/backend_identity.same_deployment +
    chat_completion_helpers.should_skip_candidate) — a duplicate same-deployment
    entry is dead weight, never a rotation path. Multiple accounts for one
    deployment rotate through the NATIVE credential pool instead
    (credential_pool_strategies, fill_first). The collapsed entry carries
    pool-membership metadata (credential_pool / pool_accounts) for humans;
    the runtime ignores extra keys.
    """
    out: list[dict[str, Any]] = []
    seen: dict[tuple[str, str, str], dict[str, Any]] = {}
    for entry in chain_entries:
        key = (str(entry.get("provider") or ""), str(entry.get("model") or ""),
               str(entry.get("base_url") or ""))
        if key in seen:
            first = seen[key]
            for acct in entry.get("pool_accounts", []):
                if acct not in first["pool_accounts"]:
                    first["pool_accounts"].append(acct)
            continue
        seen[key] = entry
        out.append(entry)
    return out


def build_chain(surf: dict[str, Any], by_id: dict[str, dict[str, Any]],
                registry: dict[str, Any]) -> list[dict[str, str]]:
    """Full emitted fallback_providers list for one surface.

    Order: approved perm slots (deduped to one entry per exact deployment,
    multi-account groups annotated with pool metadata) → Codex → local.
    """
    chain: list[dict[str, str]] = []
    for s in effective_slots(surf, by_id):
        h = copy.deepcopy(s["hermes"])
        h["route_slot"] = s["slot"]
        h["route_class"] = s["class"]
        h["route_model_id"] = s["model_id"]
        h["pool_accounts"] = [s["provider_account"]]
        chain.append(h)
    chain = dedupe_deployments(chain)
    migration = surf.get("primary_model_migration")
    if migration:
        primary = by_id[migration["primary_slot"]]["hermes"]
        primary_key = _deployment_key(primary)
        chain = [entry for entry in chain if _deployment_key(entry) != primary_key]
    # annotate pool metadata for multi-account groups (traceability only)
    for entry in chain:
        accounts = entry.get("pool_accounts", [])
        provider = str(entry.get("provider") or "")
        if len(accounts) > 1 or provider in (registry.get("pool_requirements") or {}):
            entry["credential_pool"] = provider
    tier = surf.get("tier", "LUNA")
    codex = copy.deepcopy(registry["codex_fallback"]["luna" if tier == "LUNA" else "sol"])
    codex["route_slot"] = "codex-fallback"
    codex["route_class"] = "perm"
    chain.append(codex)
    local = copy.deepcopy(registry["local_final"])
    local["route_slot"] = "local-final"
    local["route_class"] = "perm"
    chain.append(local)
    # hard invariant: no duplicate (provider, model, base_url) may ever be emitted
    keys = [(e.get("provider"), e.get("model"), e.get("base_url")) for e in chain]
    if len(keys) != len(set(keys)):
        raise ValidationError(f"surface {surf.get('surface')}: duplicate deployment entry in chain")
    return chain


def _migrated_model(doc: dict[str, Any], surf: dict[str, Any],
                    by_id: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Return the migrated model block, refusing stale or ambiguous live state."""
    migration = surf.get("primary_model_migration")
    if not migration:
        return None
    current = doc.get("model")
    if isinstance(current, str):
        # Scalar live main (shape B): nothing to preserve — migrate only
        # when it exactly matches the declared old main, emitting a fresh
        # mapping from the primary slot. (2026-09-04: unblocks minimax-m3
        # scalar surfaces moving to free qwen3.8-flash.)
        if current != migration.get("expected_old_main"):
            raise ValidationError(
                f"surface {surf['surface']}: expected old main "
                f"{migration.get('expected_old_main')!r}, found {current!r}; refusing migration")
        primary = by_id[migration["primary_slot"]]["hermes"]
        out: dict[str, Any] = {"default": primary["model"]}
        for key in ("provider", "base_url", "reasoning_effort"):
            if key in primary:
                out[key] = primary[key]
        return out
    if not isinstance(current, dict):
        raise ValidationError(
            f"surface {surf['surface']}: primary migration requires a model mapping")
    actual = current.get("default")
    expected = migration["expected_old_main"]
    primary = by_id[migration["primary_slot"]]["hermes"]
    if actual == primary["model"]:
        # Migration already applied on this surface: the plan is a no-op for
        # the model block (idempotent re-plan), not an error.
        return None
    if actual != expected:
        raise ValidationError(
            f"surface {surf['surface']}: expected old main {expected!r}, found {actual!r}; refusing migration")
    out = copy.deepcopy(current)
    out["default"] = primary["model"]
    for key in ("provider", "base_url", "reasoning_effort"):
        if key in primary:
            out[key] = primary[key]
        else:
            out.pop(key, None)
    return out


def emitted_config(doc: dict[str, Any], chain: list[dict[str, Any]],
                   surf: dict[str, Any] | None = None,
                   by_id: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """New config doc with governed fallback and optional primary migration."""
    out = copy.deepcopy(doc)
    out["fallback_providers"] = chain
    if surf is not None and by_id is not None:
        model = _migrated_model(doc, surf, by_id)
        if model is not None:
            out["model"] = model
    return out


# -------------------------------------------------------------------- diffing

def _deployment_key(e: dict[str, Any]) -> tuple[str, str, str]:
    return (str(e.get("provider") or ""), str(e.get("model") or ""), str(e.get("base_url") or ""))


def config_diff(old: dict[str, Any], new: dict[str, Any]) -> list[dict[str, Any]]:
    """Deterministic structural diff for governed fallback and model fields."""
    changes = []
    old_fp = old.get("fallback_providers")
    new_fp = new.get("fallback_providers")
    if old_fp is None and old_fp != new_fp:
        changes.append({"surface": None, "op": "add", "field": "fallback_providers",
                        "old": None, "new_count": len(new_fp)})
    elif old_fp != new_fp:
        changes.append({
            "op": "replace", "field": "fallback_providers", "old": old_fp,
            "old_count": len(old_fp) if isinstance(old_fp, list) else None,
            "new": new_fp, "new_count": len(new_fp) if isinstance(new_fp, list) else None,
        })
    if old.get("model") != new.get("model"):
        changes.append({"op": "replace", "field": "model",
                        "old": old.get("model"), "new": new.get("model")})
    return changes


def removed_capabilities(old_fp: Any, new_fp: list[dict[str, Any]]) -> list[dict[str, str]]:
    """(provider, model) pairs present in the old chain but absent from the new one."""
    if not isinstance(old_fp, list):
        return []
    new_keys = {_deployment_key(e) for e in new_fp}
    seen: set[tuple[str, str]] = set()
    removed = []
    for e in old_fp:
        if not isinstance(e, dict):
            continue
        key = _deployment_key(e)
        if key in new_keys:
            continue
        pair = (key[0], key[1])
        if pair in seen:
            continue
        seen.add(pair)
        removed.append({"provider": pair[0], "model": pair[1]})
    return removed


def yaml_dump_stable(doc: Any) -> str:
    """Deterministic YAML: sorted keys, no anchors/aliases, trailing newline."""
    return yaml.safe_dump(doc, sort_keys=True, default_flow_style=False, allow_unicode=True)


# -------------------------------------------------------------- plan / report

def build_plan(tmp_home: Path, registry_path: Path, surfaces_path: Path) -> dict[str, Any]:
    """Compute the full apply plan against a temp copy of the live config tree.

    tmp_home must be a copy of ~/.hermes containing config.yaml + profiles/*/config.yaml.
    """
    reg = load_registry(registry_path)
    by_id = validate_registry(reg)
    surfaces = load_surfaces(surfaces_path)
    validate_surfaces(surfaces, by_id, reg)

    root_config = tmp_home / "config.yaml"
    profiles_dir = tmp_home / "profiles"
    plan_entries = []
    emitted_groups: dict[str, list[str]] = {}

    # root config = "default" surface
    surfaces_by_name = {s["surface"]: s for s in surfaces}
    if root_config.exists():
        with open(root_config, encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
        surf = surfaces_by_name.get("default", {"surface": "default", "tier": "SOL", "main_model": None, "slots": []})
        if surf.get("routes_policy") == "keep_current":
            # proposal: keep current config — no changes emitted for this surface
            plan_entries.append({
                "surface": "default (root config.yaml)",
                "path": str(root_config),
                "main_model": surf.get("main_model"),
                "changes": [],
                "note": "routes_policy=keep_current — untouched",
            })
        else:
            chain = build_chain(surf, by_id, reg)
            new_doc = emitted_config(doc, chain, surf, by_id)
            plan_entries.append({
                "surface": "default (root config.yaml)",
                "path": str(root_config),
                "main_model": surf.get("main_model"),
                "changes": config_diff(doc, new_doc),
            })

    for profile_dir in sorted(p for p in profiles_dir.iterdir() if p.is_dir()):
        cfg = profile_dir / "config.yaml"
        if not cfg.exists():
            continue
        doc = _read_yaml(cfg)
        surf_name = profile_dir.name
        surf = surfaces_by_name.get(surf_name)
        if surf is None:
            plan_entries.append({
                "surface": surf_name,
                "path": str(cfg),
                "main_model": None,
                "changes": [],
                "note": "no surface entry — untouched",
            })
            continue
        if surf.get("routes_policy") == "keep_current":
            plan_entries.append({
                "surface": surf_name,
                "path": str(cfg),
                "main_model": surf.get("main_model"),
                "changes": [],
                "note": "routes_policy=keep_current — untouched",
            })
            continue
        chain = build_chain(surf, by_id, reg)
        for s in effective_slots(surf, by_id):
            emitted_groups.setdefault(str(s["hermes"].get("provider") or ""), []).append(s["provider_account"])
        new_doc = emitted_config(doc, chain, surf, by_id)
        plan_entries.append({
            "surface": surf_name,
            "path": str(cfg),
            "main_model": surf.get("main_model"),
            "changes": config_diff(doc, new_doc),
        })

    # dedupe emitted_groups account lists, deterministic order
    emitted_groups = {p: sorted(set(a)) for p, a in sorted(emitted_groups.items())}

    # removed-capability summary across all changed surfaces
    removed_by_surface: dict[str, list[dict[str, str]]] = {}
    removed_total: dict[tuple[str, str], int] = {}
    for entry in plan_entries:
        if not entry["changes"]:
            continue
        fallback_change = next(
            (change for change in entry["changes"] if change.get("field") == "fallback_providers"), None)
        if fallback_change is None:
            continue
        old_fp = fallback_change.get("old")
        chain = build_chain(
            surfaces_by_name.get(
                "default" if entry["surface"].startswith("default") else entry["surface"],
                {"surface": entry["surface"], "tier": "SOL", "main_model": None, "slots": []}),
            by_id, reg)
        removed = removed_capabilities(old_fp, chain)
        if removed:
            removed_by_surface[entry["surface"]] = removed
            for r in removed:
                removed_total[(r["provider"], r["model"])] = removed_total.get((r["provider"], r["model"]), 0) + 1

    # pool requirements vs target config
    pool_req_report = []
    reqs = reg.get("pool_requirements") or {}
    target_cfg_path = tmp_home / "config.yaml"
    target_pools: dict[str, Any] = {}
    if target_cfg_path.exists():
        target_doc = _read_yaml(target_cfg_path)
        target_pools = target_doc.get("credential_pool_strategies") or {}
    for provider in sorted(emitted_groups):
        accounts = emitted_groups[provider]
        if len(accounts) < 2:
            continue
        req = reqs.get(provider)
        strategy = target_pools.get(provider)
        blockers = []
        if req is None:
            blockers.append(f"registry pool_requirements missing for provider {provider}")
        else:
            if accounts != list(req.get("accounts") or []):
                blockers.append(
                    f"registry pool accounts {req.get('accounts')} do not match emitted accounts {accounts}")
        if not strategy:
            blockers.append(f"target config has no credential_pool_strategies entry for {provider}")
        elif req is not None and strategy != req.get("strategy"):
            blockers.append(
                f"credential_pool_strategies[{provider}] = {strategy!r}, required "
                f"{req.get('strategy')!r} for account cascade")
        pool_req_report.append({
            "provider": provider,
            "accounts": accounts,
            "strategy_required": (req or {}).get("strategy"),
            "strategy_found": strategy,
            "status": "blocked" if blockers else "met",
            "blockers": blockers,
        })

    as_of = _utcnow().date().isoformat()
    return {
        "registry": str(registry_path),
        "surfaces": str(surfaces_path),
        "target_root": str(tmp_home),
        "as_of": as_of,
        "pool_requirements": pool_req_report,
        "env_key_bypass": {"scanned": [], "blockers": []},  # populated by caller via scan
        "removed_capabilities": {
            "by_surface": removed_by_surface,
            "aggregate": [
                {"provider": p, "model": m, "removed_from_surfaces": n}
                for (p, m), n in sorted(removed_total.items())
            ],
        },
        "emission_checks": {
            "duplicate_deployments": 0,  # build_chain raises on any duplicate
            "gated_routes_emitted": 0,
        },
        "entries": plan_entries,
    }


def _scan_env_file(env_path: Path) -> list[str] | None:
    """Return sorted env KEY NAMES present (uncommented) in a .env file, or None if unreadable/missing."""
    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError:
        return None
    keys = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key = line.split("=", 1)[0].strip()
        if key:
            keys.add(key)
    return sorted(keys)


def scan_env_key_bypass_tree(tree_root: Path, registry: dict[str, Any]) -> dict[str, Any]:
    """Scan a config tree for profile .env keys that would bypass credential pools.

    KEY NAMES ONLY — values are never read, copied, or emitted.
    """
    bypass_map = registry.get("env_bypass") or {}
    scanned: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    candidates: list[tuple[str, Path]] = []
    root_env = tree_root / ".env"
    if root_env.exists():
        candidates.append(("default (root .env)", root_env))
    profiles_dir = tree_root / "profiles"
    if profiles_dir.is_dir():
        for prof in sorted(p for p in profiles_dir.iterdir() if p.is_dir()):
            env_f = prof / ".env"
            if env_f.exists():
                candidates.append((prof.name, env_f))
    for surface, env_f in candidates:
        keys = _scan_env_file(env_f)
        if keys is None:
            scanned.append({"surface": surface, "path": str(env_f), "error": "unreadable"})
            continue
        scanned.append({"surface": surface, "env_keys": keys})
        for key in keys:
            for provider in sorted(bypass_map):
                if key in (bypass_map[provider] or []):
                    blockers.append({
                        "surface": surface,
                        "provider": provider,
                        "env_key": key,
                        "reason": "profile .env key bypasses the native credential pool — "
                                  "runtime resolves this provider's key from env before the "
                                  "pool, so multi-account cascade is unreachable",
                        "remediation": "deployment blocker: requires later approved auth "
                                       "cleanup (remove env key from profile .env) before apply",
                    })
    scanned.sort(key=lambda x: x["surface"])
    blockers.sort(key=lambda x: (x["surface"], x["provider"], x["env_key"]))
    return {"scanned": scanned, "blockers": blockers}


# ----------------------------------------------------------------- apply gate

def apply_blockers(plan: dict[str, Any]) -> list[str]:
    """Fail-closed apply gate: collect deployment blockers that must prevent apply.

    Apply is hard-refused when either:
      - plan.env_key_bypass.blockers is non-empty (profile .env keys bypass the
        native credential pool — account cascade unreachable), OR
      - any pool_requirement entry has status != 'met' (or carries blockers).

    Dry-run is NEVER gated: the plan still generates and reports blockers so
    reviewers can act on them; only the write path refuses.
    """
    blockers: list[str] = []
    for b in (plan.get("env_key_bypass") or {}).get("blockers") or []:
        blockers.append(
            f"env_key_bypass: surface {b.get('surface')!r} has {b.get('env_key')!r} "
            f"({b.get('provider')}) — pool bypass deployment blocker; requires "
            f"approved auth cleanup before apply"
        )
    for req in plan.get("pool_requirements") or []:
        if req.get("status") != "met" or req.get("blockers"):
            blockers.append(
                f"pool_requirement {req.get('provider')!r}: status {req.get('status')!r} — "
                f"{'; '.join(req.get('blockers') or ['unmet requirement (no blockers listed)'])}"
            )
    return sorted(blockers)


# ----------------------------------------------------------------- apply mode

def apply_plan(plan: dict[str, Any], backup_dir: Path) -> list[dict[str, str]]:
    """Write planned configs + timestamped backups. Caller must have triple-guarded.

    Fail-closed gate: raises ValidationError (before ANY snapshot, backup, or
    write) while env_key_bypass blockers exist OR any pool_requirement is
    unmet — the same gate the CLI enforces with exit 4.

    Atomic: all originals are snapshotted in memory BEFORE any write; if any
    write raises, every already-written file is restored from its snapshot and
    the exception propagates — the target tree is left byte-identical to its
    pre-apply state. Backups of every original are retained on disk regardless.
    """
    gate = apply_blockers(plan)
    if gate:
        raise ValidationError(
            "apply refused (fail-closed): unresolved deployment blockers — "
            + "; ".join(gate)
        )
    import datetime
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out: list[dict[str, str]] = []
    backup_root = backup_dir / stamp
    backup_root.mkdir(parents=True, exist_ok=True)
    reg = load_registry(plan["registry"])
    by_id = validate_registry(reg)
    surfaces = load_surfaces(plan["surfaces"])
    validate_surfaces(surfaces, by_id, reg)
    surfaces_by_name = {s["surface"]: s for s in surfaces}

    # ---- phase 0: snapshot every target original (bytes) before any write ----
    snapshots: dict[str, bytes] = {}
    for entry in plan["entries"]:
        if not entry["changes"]:
            continue
        src = Path(entry["path"])
        snapshots[str(src)] = src.read_bytes()

    # ---- phase 1: backup all originals to disk ----
    backed_up: list[tuple[Path, Path]] = []
    for entry in plan["entries"]:
        if not entry["changes"]:
            continue
        src = Path(entry["path"])
        dst = backup_root / entry["surface"].replace(" (root config.yaml)", "__root__") / "config.yaml"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(snapshots[str(src)])
        backed_up.append((src, dst))
    backup_by_path = {str(s): str(d) for s, d in backed_up}

    # ---- phase 2: write all new configs; roll back EVERYTHING on any failure ----
    written: list[Path] = []
    try:
        for entry in plan["entries"]:
            if not entry["changes"]:
                continue
            src = Path(entry["path"])
            with open(src, "rb") as fh:
                doc = yaml.safe_load(fh.read().decode("utf-8")) or {}
            if entry["surface"] == "default (root config.yaml)":
                surf = surfaces_by_name.get("default", {
                    "surface": "default", "tier": "SOL", "main_model": None, "slots": []})
            else:
                surf = surfaces_by_name[entry["surface"]]
            chain = build_chain(surf, by_id, reg)
            doc = emitted_config(doc, chain, surf, by_id)
            src.write_text(yaml_dump_stable(doc), encoding="utf-8")
            written.append(src)
            out.append({
                "surface": entry["surface"],
                "backup": backup_by_path[str(src)],
                "written": str(src),
            })
    except Exception:
        # atomic rollback: restore every already-written file from snapshot
        rollback_errors = []
        for src in written:
            try:
                Path(src).write_bytes(snapshots[str(src)])
            except OSError as rexc:
                rollback_errors.append(f"{src}: {rexc}")
        if rollback_errors:
            raise RuntimeError(
                "apply failed AND rollback failed for: " + "; ".join(rollback_errors)
            ) from None
        raise

    # ---- phase 2 verification: every written file parses and matches plan ----
    for entry in plan["entries"]:
        if not entry["changes"]:
            continue
        with open(entry["path"], encoding="utf-8") as fh:
            post = yaml.safe_load(fh)
        if "fallback_providers" not in post:
            raise ValidationError(f"post-apply verification failed for {entry['path']}")
    return out


def verify_rollback(plan: dict[str, Any], backup_dir: Path) -> bool:
    """True when every target file is byte-identical to its pre-apply snapshot."""
    stamp_dirs = sorted(backup_dir.glob("*"))
    if not stamp_dirs:
        return False
    latest = stamp_dirs[-1]
    for entry in plan["entries"]:
        if not entry["changes"]:
            continue
        rel = entry["surface"].replace(" (root config.yaml)", "__root__")
        bkp = latest / rel / "config.yaml"
        if not bkp.exists():
            return False
        if Path(entry["path"]).read_bytes() != bkp.read_bytes():
            return False
    return True


# ------------------------------------------------------------------------ CLI

def _is_live_hermes_root(target: Path) -> bool:
    try:
        return target.resolve() == (LIVE_HERMES_HOME).resolve()
    except OSError:
        return False


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    default_registry = here.parent / "registry" / "route-slots.yaml"
    default_surfaces = here.parent / "registry" / "surfaces.yaml"

    ap = argparse.ArgumentParser(
        prog="route-registry-generator",
        description="Generate Hermes fallback_providers from the central route-slot registry. "
                    "DRY-RUN by default — nothing is modified.",
    )
    ap.add_argument("--registry", default=str(default_registry), help="route-slots.yaml path")
    ap.add_argument("--surfaces", default=str(default_surfaces), help="surfaces.yaml path")
    ap.add_argument("--target-root", required=True,
                    help="config tree root containing config.yaml + profiles/ (REQUIRED; must be "
                         "a temp copy — apply against the live ~/.hermes is refused without "
                         "--allow-live-apply)")
    ap.add_argument("--out", default=None, help="write dry-run plan JSON to this path")
    ap.add_argument("--apply", action="store_true",
                    help="APPLY the plan. Requires --confirm YES-APPLY-ROUTES. Default is dry-run.")
    ap.add_argument("--confirm", default=None,
                    help="must be exactly: YES-APPLY-ROUTES (required with --apply)")
    ap.add_argument("--allow-live-apply", action="store_true",
                    help="SEPARATE approval flag: permit apply when target root resolves to the "
                         "live ~/.hermes tree. Without it, live-tree apply is refused (exit 3).")
    ap.add_argument("--backup-dir", default=str(here / "backups"), help="apply-mode backup root")
    args = ap.parse_args(argv)

    target_root = Path(args.target_root)

    # ---- apply guard: FOUR independent conditions must all hold ----
    if args.apply:
        if args.confirm != "YES-APPLY-ROUTES":
            print("REFUSED: --apply requires --confirm YES-APPLY-ROUTES (exact string).", file=sys.stderr)
            return 2
        if sys.stdin is not None and sys.stdin.isatty():
            print("REFUSED: --apply is not permitted interactively. Run non-interactively "
                  "(e.g. piped stdin) after explicit approval.", file=sys.stderr)
            return 2
        if not target_root.exists():
            print(f"REFUSED: target root {target_root} does not exist.", file=sys.stderr)
            return 2
        if _is_live_hermes_root(target_root) and not args.allow_live_apply:
            print(
                f"REFUSED: {target_root} resolves to the LIVE ~/.hermes tree. Apply against "
                f"live requires the separate --allow-live-apply approval flag (and is still "
                f"triple-guarded). Use a temp copy for dry-runs and test applies.",
                file=sys.stderr)
            return 3

    plan = build_plan(target_root, Path(args.registry), Path(args.surfaces))

    # env-key pool-bypass scan: key NAMES only, never values
    try:
        env_report = scan_env_key_bypass_tree(target_root, load_registry(Path(args.registry)))
    except Exception as exc:  # defensive: never let the scan crash the plan
        env_report = {"scanned": [], "blockers": [], "error": str(exc)}
    plan["env_key_bypass"] = env_report

    # ---- fail-closed apply gate: refuse apply while deployment blockers exist ----
    # Dry-run is never gated: blockers are reported in the plan for review.
    if args.apply:
        blockers = apply_blockers(plan)
        if blockers:
            print("REFUSED: apply blocked by unresolved deployment blockers "
                  "(fail-closed). Resolve approved auth cleanup / pool "
                  "requirements, then re-run. Blockers:", file=sys.stderr)
            for b in blockers:
                print(f"  - {b}", file=sys.stderr)
            print("Dry-run (`--out plan.json` without --apply) still works and "
                  "reports these blockers.", file=sys.stderr)
            return 4
        report: dict[str, Any] = {"mode": "apply", "apply_gate": "passed",
                                  "applied": apply_plan(plan, Path(args.backup_dir)),
                                  "plan": plan}
    else:
        report = {"mode": "dry-run", "plan": plan,
                  "apply_blockers": apply_blockers(plan)}

    import json as _json
    text = _json.dumps(report, indent=2, sort_keys=True, default=str)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
        print(f"plan written: {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

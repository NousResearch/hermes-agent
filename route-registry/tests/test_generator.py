"""Unit + integration + adversarial tests for the central route-registry generator.

Integration tests operate ONLY on temporary copies of the live config tree
(~/.hermes copied to tmp_path). No live config, credential, or auth surface is
ever modified — the live tree is read-only input.

Covers the QA-remediation requirements:
  - no duplicate (provider, model, base_url) deployment entries anywhere
  - native credential-pool cascade requirements represented and checked
  - env-key pool bypass (OLLAMA_API_KEY) reported as deployment blockers
  - expired / usage-capped / disallowed slots cannot emit (fail closed)
  - live-tree apply separately guarded (--allow-live-apply, exit 3)
  - apply atomic with full rollback on injected mid-apply failure
  - removed-capability summary present in every plan
  - real credential scan (no placebo): emitted output and plan must contain no
    key material AND no env VALUES; .env scanning reads key NAMES only
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parent
REG_ROOT = HERE.parent
GENERATOR = REG_ROOT / "route_registry" / "generator.py"
GENERATOR_CLI = REG_ROOT / "generator_cli.py"
REGISTRY = REG_ROOT / "registry" / "route-slots.yaml"
SURFACES = REG_ROOT / "registry" / "surfaces.yaml"
LIVE_HERMES_HOME = Path.home() / ".hermes"

sys.path.insert(0, str(REG_ROOT / "route_registry"))
import generator  # noqa: E402


# ------------------------------------------------------------------ fixtures

@pytest.fixture(scope="session")
def registry():
    return generator.load_registry(REGISTRY)


@pytest.fixture(scope="session")
def by_id():
    return generator.validate_registry(generator.load_registry(REGISTRY))


@pytest.fixture(scope="session")
def surfaces():
    return generator.load_surfaces(SURFACES)


@pytest.fixture()
def tmp_hermes_home(tmp_path):
    """Temp copy of the live config SURFACES only (config.yaml + profiles/*/config.yaml
    + .env files for the bypass scan). The live ~/.hermes tree is read-only input;
    only these files are copied to the temp home."""
    dst = tmp_path / "hermes-home"
    (dst / "profiles").mkdir(parents=True)
    shutil.copy2(LIVE_HERMES_HOME / "config.yaml", dst / "config.yaml")
    for prof in (LIVE_HERMES_HOME / "profiles").iterdir():
        cfg = prof / "config.yaml"
        if cfg.exists():
            (dst / "profiles" / prof.name).mkdir()
            shutil.copy2(cfg, dst / "profiles" / prof.name / "config.yaml")
        if (prof / ".env").exists():
            shutil.copy2(prof / ".env", dst / "profiles" / prof.name / ".env")
    # Deterministic pre-migration state: reset content-strategist to its
    # pre-migration main and strip the route_model_id annotations the live
    # apply added, so the migration-planning tests exercise the migration
    # path regardless of whether the live tree has already been migrated
    # (the generator is idempotent either way).
    content_cfg = dst / "profiles" / "content-strategist" / "config.yaml"
    if content_cfg.exists():
        doc = yaml.safe_load(content_cfg.read_text())
        pre = next(
            (s for s in generator.load_surfaces(SURFACES)
             if s["surface"] == "content-strategist"), None)
        mig = pre.get("primary_model_migration") if pre else None
        if mig is not None:
            doc.setdefault("model", {})["default"] = mig["expected_old_main"]
            for entry in doc.get("fallback_providers") or []:
                if isinstance(entry, dict):
                    entry.pop("route_model_id", None)
            content_cfg.write_text(yaml.safe_dump(doc, sort_keys=False))

    # Same reset pattern for surfaces with new migrations in this test pass.
    for surf_name, expected_old in [
        ("denji-ledger", "stepfun/step-3.7-flash:free"),
        ("gojo-admin", "gemma4:31b"),
        ("gojo-calendar", "gemma4:31b"),
        ("gojo-mailbox", "gemma4:31b"),
        ("wesker-backup", "gemma4:31b"),
        ("wesker-ops", "gemma4:31b"),
    ]:
        cfg = dst / "profiles" / surf_name / "config.yaml"
        if cfg.exists():
            doc = yaml.safe_load(cfg.read_text())
            doc.setdefault("model", {})["default"] = expected_old
            for entry in doc.get("fallback_providers") or []:
                if isinstance(entry, dict):
                    entry.pop("route_model_id", None)
            cfg.write_text(yaml.safe_dump(doc, sort_keys=False))
    return dst


@pytest.fixture()
def run_cli(tmp_hermes_home):
    """Run the generator CLI against the temp home. Returns CompletedProcess."""
    def _run(*extra, apply_mode: bool = False, interactive: bool = False,
             target_root: Path | None = None):
        cmd = [sys.executable, str(GENERATOR_CLI),
               "--target-root", str(target_root or tmp_hermes_home)]
        cmd.extend(extra)
        if apply_mode:
            cmd += ["--apply", "--confirm", "YES-APPLY-ROUTES"]
        return subprocess.run(
            cmd, capture_output=True, text=True,
            stdin=None if interactive else subprocess.DEVNULL,
            timeout=180,
        )
    return _run


def _chain_entries(plan, surface):
    for e in plan["entries"]:
        if e["surface"] == surface and e["changes"]:
            for ch in e["changes"]:
                if ch.get("field") == "fallback_providers" and ch.get("new") is not None:
                    return ch["new"]
    return None


def _new_chains(plan):
    out = []
    for e in plan["entries"]:
        for ch in e["changes"]:
            if ch.get("new"):
                out.append((e["surface"], ch["new"]))
    return out


def _count_bypass_env_keys(home: Path) -> int:
    """Count ACTIVE (uncommented) lines whose key name is in the registry's
    env_bypass set, across the temp tree's .env files. Used to decide whether a
    synthetic key must be seeded for deterministic testing."""
    reg = generator.load_registry(REGISTRY)
    bypass_keys = set()
    for keys in (reg.get("env_bypass") or {}).values():
        bypass_keys.update(keys or [])
    count = 0
    for env_f in [home / ".env", *sorted((home / "profiles").glob("*/.env"))]:
        if not env_f.exists():
            continue
        for line in env_f.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.split("=", 1)[0].strip() in bypass_keys:
                count += 1
    return count


def _seed_bypass_env_key(home: Path) -> Path:
    """Inject a synthetic OLLAMA_API_KEY line into a temp tree's profile .env so
    env-bypass tests are DETERMINISTIC and independent of the live tree's state.

    Rationale: the approved Ollama auth-cleanup (2026-09-01) removed every
    OLLAMA_API_KEY from the live ~/.hermes profiles, so a fixture that relied on
    the live copy carrying the key can no longer find one. Seeding the key into
    the throwaway temp tree preserves the tests' original intent — exercising the
    fail-closed gate — without depending on live-tree state. Returns the seeded
    .env path.
    """
    env = home / "profiles" / "denji" / ".env"
    env.parent.mkdir(parents=True, exist_ok=True)
    existing = env.read_text(encoding="utf-8") if env.exists() else ""
    if "OLLAMA_API_KEY" not in existing:
        existing += "\nOLLAMA_API_KEY=«redacted:sk-…»\n"
    env.write_text(existing, encoding="utf-8")
    return env


def _seed_old_denji_chain(home: Path) -> Path:
    """Force denji's temp config back to a PRE-migration fallback chain so
    transition tests deterministically see a diff the apply must write.

    Rationale: the live tree has already been migrated to the registry target
    (xKiro/B.AI/GLM-Flash routes applied 2026-09-02), so a fresh copy of the live
    tree no longer diverges from the plan — apply tests that assert a transition
    would find nothing to change. Overwriting denji's chain with the legacy
    pre-registry shape (codex → commandcode → nous → local) recreates the exact
    divergence the tests were written to exercise. Returns the config path."""
    cfg = home / "profiles" / "denji" / "config.yaml"
    doc = yaml.safe_load(cfg.read_text()) or {}
    doc["fallback_providers"] = [
        {"provider": "openai-codex", "model": "gpt-5.6-sol",
         "base_url": "https://chatgpt.com/backend-api/codex"},
        {"provider": "custom:commandcode", "model": "z-ai/glm-5.3-flash",
         "base_url": "https://api.commandcode.ai/provider/v1"},
        {"provider": "nous", "model": "tencent/hy3:free",
         "base_url": "https://inference-api.nousresearch.com/v1"},
        {"provider": "custom:turbohaul-local", "model": "qwen3.8-27b",
         "base_url": "http://127.0.0.1:11410/v1"},
    ]
    cfg.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return cfg


def _seed_old_ceecee_brand_chain(home: Path) -> Path:
    """Also force ceecee-brand's temp config back to its pre-migration chain
    (ollama-cloud → openrouter/gemma-free → nous → openrouter/nemotron-free) so
    atomicity tests have 2+ divergent surfaces to exercise rollback across."""
    cfg = home / "profiles" / "ceecee-brand" / "config.yaml"
    doc = yaml.safe_load(cfg.read_text()) or {}
    doc["fallback_providers"] = [
        {"provider": "ollama-cloud", "model": "deepseek-v4-flash",
         "base_url": "https://ollama.com/v1"},
        {"provider": "openrouter", "model": "google/gemma-4-31b-it:free",
         "base_url": "https://openrouter.ai/api/v1"},
        {"provider": "nous", "model": "stepfun/step-3.7-flash:free",
         "base_url": "https://inference-api.nousresearch.com/v1"},
        {"provider": "openrouter", "model": "nvidia/nemotron-3-super-120b-a12b:free",
         "base_url": "https://openrouter.ai/api/v1"},
    ]
    cfg.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return cfg


def _strip_bypass_env_keys(home: Path) -> int:
    """Remove bypass env KEY lines (names only — values are never read) from a
    temp tree's .env files. Simulates the approved auth cleanup the plan's
    remediation requires. Returns number of lines removed.

    Deterministic: if the live tree already carries no bypass keys (post
    Ollama auth-cleanup), seed a synthetic one first so the >0 assertion and the
    subsequent fail-closed/atomic paths remain exercised."""
    if _count_bypass_env_keys(home) == 0:
        _seed_bypass_env_key(home)
    reg = generator.load_registry(REGISTRY)
    bypass_keys = set()
    for keys in (reg.get("env_bypass") or {}).values():
        bypass_keys.update(keys or [])
    removed = 0
    envs = [home / ".env", *sorted((home / "profiles").glob("*/.env"))]
    for env_f in envs:
        if not env_f.exists():
            continue
        lines = env_f.read_text(encoding="utf-8").splitlines(keepends=True)
        kept = []
        for line in lines:
            stripped = line.strip()
            key = stripped.split("=", 1)[0].strip() if stripped and not stripped.startswith("#") else None
            if key in bypass_keys:
                removed += 1
                continue
            kept.append(line)
        env_f.write_text("".join(kept), encoding="utf-8")
    return removed


# ------------------------------------------------------------- registry tests

class TestRegistry:
    def test_registry_loads(self, registry):
        # 37 route slots + 1 emergency reserve.
        # (2026-09-02: was 33 — removed slot-glm53-3/4, the dead slots serving the
        #  same model; 2026-09-03 add 4 mimo-v2.5 slots for denji-ledger migration;
        #  2026-09-04 add slot-mini-m3-5 ollama minimax matrix-backed candidate
        #  plus slot-qwenflash-1 bai candidate, both disabled, no surface refs.
        #  opencode-go minimax deliberately not added: max 5 same-model routes)
        route_slots = [s for s in registry["slots"] if s["class"] != "emergency_reserve"]
        assert len(route_slots) == 37
        assert len(registry["slots"]) == 38

    def test_all_slots_have_full_schema(self, registry):
        required = {"slot", "model_id", "provider_account", "class", "status",
                    "approved_at", "expires_at", "usage_limit", "allowed_profiles",
                    "replacement_slot", "disabled", "hermes"}
        for s in registry["slots"]:
            missing = required - set(s)
            assert not missing, f"slot {s.get('slot')} missing {missing}"

    def test_emergency_reserve_never_emitted(self, by_id):
        s = by_id["slot-nous-emergency-credit"]
        assert s["class"] == "emergency_reserve"
        assert s["disabled"] is True
        assert s["hermes"] is None

    def test_all_gated_nc_disabled(self, by_id):
        for s in by_id.values():
            if s["class"] in {"gated", "needs_classification", "emergency_reserve"}:
                assert s["disabled"] is True, f"{s['slot']} must be disabled"
                assert s["status"] == "candidate" or s["class"] == "emergency_reserve"

    def test_perm_slots_approved_with_timestamp(self, by_id):
        for s in by_id.values():
            if s["class"] == "perm":
                if s["status"] == "retired":
                    # retired slots are capability-violations kept as historical
                    # record (2026-09-02): they must sit disabled, never emit
                    assert s["disabled"] is True, f"{s['slot']} retired but enabled"
                    continue
                assert s["status"] == "approved"
                assert s["approved_at"], f"{s['slot']} missing approved_at"
                assert s["disabled"] is False

    def test_max_5_same_model_routes(self, registry):
        counts = {}
        for s in registry["slots"]:
            if s["model_id"] is None:
                continue
            counts.setdefault(s["model_id"], []).append(s["slot"])
        for model, slots in counts.items():
            assert len(slots) <= 5, f"{model} has {len(slots)} slots"

    def test_unique_provider_account_per_model(self, by_id):
        seen = {}
        for s in by_id.values():
            if s["model_id"] is None:
                continue
            key = (s["model_id"], s["provider_account"])
            assert key not in seen, f"duplicate provider_account {key}"
            seen[key] = s["slot"]

    def test_nim_constraint_recorded(self, registry):
        assert registry["limits"]["nim_request_limit"] == 60
        assert registry["limits"]["nim_window_verified"] is False
        nim_slots = [s for s in registry["slots"]
                     if "nim" in (s["provider_account"] or "")]
        assert nim_slots == [], "no NIM slot may exist until window verified"

    def test_pool_requirements_declared(self, registry):
        """Native pool cascade requirements must be in the registry."""
        reqs = registry["pool_requirements"]
        assert reqs["ollama-cloud"]["strategy"] == "fill_first"
        assert reqs["ollama-cloud"]["accounts"] == ["ollama-cloud/1", "ollama-cloud/2"]

    def test_env_bypass_keys_declared(self, registry):
        assert registry["env_bypass"]["ollama-cloud"] == ["OLLAMA_API_KEY"]

    def test_surfaces_cover_all_65(self, surfaces):
        assert len(surfaces) == 65
        names = [s["surface"] for s in surfaces]
        assert len(names) == 65

    def test_surfaces_match_live_profile_names(self, surfaces):
        live = {p.name for p in (LIVE_HERMES_HOME / "profiles").iterdir() if p.is_dir()}
        live.add("default")  # root config
        mapped = {s["surface"] for s in surfaces}
        assert live == mapped, (
            f"missing: {live - mapped}, extra: {mapped - live}"
        )

    def test_same_model_invariant(self, by_id, surfaces):
        for surf in surfaces:
            for sid in surf.get("slots", []):
                assert by_id[sid]["model_id"] == surf["main_model"], (
                    f"surface {surf['surface']}: slot {sid} model mismatch")


# ------------------------------------------------------------- emission tests

class TestEmission:
    def test_effective_chain_perm_only(self, by_id, surfaces):
        for surf in surfaces:
            eff = generator.effective_slots(surf, by_id)
            for s in eff:
                assert s["class"] == "perm"
                assert s["status"] == "approved"
                assert s["disabled"] is False
                assert s["hermes"] is not None

    def test_chain_ends_with_codex_then_local(self, by_id, surfaces):
        for surf in surfaces:
            chain = generator.build_chain(surf, by_id, generator.load_registry(REGISTRY))
            assert len(chain) >= 2
            assert chain[-1]["provider"] == "custom:turbofit-local"
            assert chain[-1]["model"] == "active:main"
            assert chain[-1]["timeout"] == 1800
            assert chain[-2]["model"] in {"gpt-5.6-sol", "gpt-5.6-luna"}
            tier_expected = "luna" if surf["tier"] == "LUNA" else "sol"
            assert chain[-2]["model"] == f"gpt-5.6-{tier_expected}"

    def test_no_gated_slot_anywhere_in_emitted_chain(self, by_id, surfaces, registry):
        for surf in surfaces:
            chain = generator.build_chain(surf, by_id, registry)
            for entry in chain:
                slot = entry.get("route_slot", "")
                if slot in by_id:
                    assert by_id[slot]["class"] == "perm"
                    assert by_id[slot]["disabled"] is False

    def test_emergency_credit_never_in_chain(self, by_id, surfaces, registry):
        for surf in surfaces:
            chain = generator.build_chain(surf, by_id, registry)
            for entry in chain:
                assert entry.get("route_slot") != "slot-nous-emergency-credit"

    def test_no_duplicate_deployments_any_chain(self, by_id, surfaces, registry):
        """S1: duplicate (provider, model, base_url) entries are runtime-skipped
        (same_deployment + should_skip_candidate) — none may ever be emitted."""
        for surf in surfaces:
            chain = generator.build_chain(surf, by_id, registry)
            keys = [(e["provider"], e["model"], e["base_url"]) for e in chain]
            assert len(keys) == len(set(keys)), f"duplicate deployment in {surf['surface']}: {keys}"

    def test_glm53flash_single_pool_entry(self, by_id, surfaces, registry):
        """S1: ollama-cloud glm-5.3-flash emits ONE entry with pool metadata for
        both approved accounts — never two duplicate entries."""
        surf = next(s for s in surfaces if s["surface"] == "denji")
        chain = generator.build_chain(surf, by_id, registry)
        entries = [e for e in chain if e["provider"] == "ollama-cloud"
                   and e["model"] == "glm-5.3-flash"]
        assert len(entries) == 1, f"expected 1 ollama-cloud entry, got {len(entries)}"
        e = entries[0]
        assert e["pool_accounts"] == ["ollama-cloud/1", "ollama-cloud/2"]
        assert e["credential_pool"] == "ollama-cloud"

    def test_pool_cascade_requirement_recorded(self, by_id, registry):
        """S1: the pool cascade (account 1→2, fill_first) is a registry requirement
        and must match the emitted account set."""
        surf = {"surface": "denji", "tier": "SOL", "main_model": "glm-5.3-flash",
                "slots": ["slot-glm53flash-1", "slot-glm53flash-2"]}
        eff = generator.effective_slots(surf, by_id)
        accounts = sorted({s["provider_account"] for s in eff
                           if s["hermes"]["provider"] == "ollama-cloud"})
        assert accounts == ["ollama-cloud/1", "ollama-cloud/2"]
        req = registry["pool_requirements"]["ollama-cloud"]
        assert req["strategy"] == "fill_first"
        assert req["accounts"] == accounts

    def test_env_bypass_blockers_real_scan(self, tmp_path):
        """S1: profile .env OLLAMA_API_KEY must surface as a deployment blocker.
        Real scan of a real tree copy — key names only."""
        tree = tmp_path / "tree"
        (tree / "profiles" / "alpha").mkdir(parents=True)
        (tree / "profiles" / "beta").mkdir()
        (tree / "profiles" / "alpha" / ".env").write_text(
            "OTHER_VAR=x\nOLLAMA_API_KEY=sk-real-secret-value-never-leaked\n")
        (tree / "profiles" / "beta" / ".env").write_text("HARMLESS=1\n")
        registry = generator.load_registry(REGISTRY)
        report = generator.scan_env_key_bypass_tree(tree, registry)
        hits = [b for b in report["blockers"] if b["surface"] == "alpha"]
        hit = next(b for b in report["blockers"] if b["surface"] == "alpha")
        assert hit["env_key"] == "OLLAMA_API_KEY"
        assert hit["provider"] == "ollama-cloud"
        assert "blocker" in hit["remediation"].lower()
        # value must NEVER appear anywhere in the report
        assert "sk-real-secret-value-never-leaked" not in json.dumps(report)

    def test_expired_slot_cannot_emit(self, by_id, monkeypatch):
        """S2: a slot with a past expires_at is excluded even when approved+enabled."""
        s = dict(by_id["slot-glm53flash-1"])
        s["expires_at"] = "2020-01-01T00:00:00Z"
        surf = {"surface": "x", "tier": "SOL", "main_model": "glm-5.3-flash", "slots": []}
        assert generator.slot_exclusion_reason(s, "x") is not None
        registry_now = {**by_id, "slot-glm53flash-1": s}
        surf_full = {"surface": "denji", "tier": "SOL", "main_model": "glm-5.3-flash",
                     "slots": ["slot-glm53flash-1", "slot-glm53flash-2"]}
        eff = generator.effective_slots(surf_full, registry_now)
        assert all(x["slot"] != "slot-glm53flash-1" for x in eff)

    def test_usage_capped_slot_cannot_emit(self, by_id):
        """S2: usage_limit without verified window is fail-closed."""
        s = dict(by_id["slot-glm53flash-1"])
        s["usage_limit"] = "60 requests (window unverified)"
        s.pop("limit_window_verified", None)
        assert generator.slot_exclusion_reason(s, "x") is not None
        # and the validator refuses an enabled capped slot outright
        bad = dict(s)
        bad["slot"] = "cap-test"
        with pytest.raises(Exception):
            generator.validate_registry({"slots": [bad]})

    def test_disallowed_profile_slot_cannot_emit(self, by_id):
        """S2: allowed_profiles allow-list excludes non-listed surfaces."""
        s = dict(by_id["slot-glm53flash-1"])
        s["allowed_profiles"] = ["only-this-surface"]
        assert generator.slot_exclusion_reason(s, "denji") is not None
        assert generator.slot_exclusion_reason(s, "only-this-surface") is None

    def test_expired_enabled_slot_fails_registry_validation(self, by_id, monkeypatch):
        """An expired slot sitting enabled must fail registry validation outright."""
        reg = generator.load_registry(REGISTRY)
        target = next(s for s in reg["slots"] if s["slot"] == "slot-glm53flash-1")
        target["expires_at"] = "2020-01-01T00:00:00Z"
        with pytest.raises(generator.ValidationError, match="expires_at"):
            generator.validate_registry(reg)

    def test_real_credential_scan_not_placebo(self, tmp_hermes_home, registry):
        """S3: replaces the old build_plan.__doc__ placebo. The REAL emitted plan
        text is scanned for credential markers; nothing may leak."""
        plan = generator.build_plan(tmp_hermes_home, REGISTRY, SURFACES)
        env_report = generator.scan_env_key_bypass_tree(tmp_hermes_home, registry)
        blob = json.dumps({"plan": plan, "env": env_report}).lower()
        for marker in ("sk-", "api_key=", "bearer "):
            assert marker not in blob, marker
        # env VALUES from a real profile .env must not leak: only key names scanned
        assert "ollama_api_key=" not in blob

    def test_deterministic_chain(self, by_id, surfaces, registry):
        surf = next(s for s in surfaces if s["surface"] == "denji")
        c1 = generator.build_chain(surf, by_id, registry)
        c2 = generator.build_chain(surf, by_id, registry)
        assert c1 == c2
        assert json.dumps(c1, sort_keys=True) == json.dumps(c2, sort_keys=True)


import copy  # noqa: E402  (used by adversarial tests above)


# ------------------------------------------------------------ plan/diff tests

class TestPlan:
    def test_keep_current_surfaces_untouched(self, tmp_hermes_home):
        """Proposal: keep_current surfaces must be untouched by the plan."""
        plan = generator.build_plan(tmp_hermes_home, REGISTRY, SURFACES)
        keep = {"default", "dezzy", "gojo", "kensei", "work"}
        touched = [e["surface"] for e in plan["entries"] if e["changes"]]
        assert not (set(touched) & keep), f"keep_current surfaces modified: {set(touched) & keep}"
        untouched = {e["surface"] for e in plan["entries"] if not e["changes"]}
        expected_untouched = (keep - {"default"}) | {"default (root config.yaml)"}
        assert untouched >= expected_untouched, f"expected untouched: {untouched}"

    def test_plan_against_full_live_copy(self, tmp_hermes_home):
        plan = generator.build_plan(tmp_hermes_home, REGISTRY, SURFACES)
        assert plan["target_root"] == str(tmp_hermes_home)
        # 65 surfaces: 64 profiles + root
        assert len(plan["entries"]) == 65

    def test_dry_run_default_changes_nothing(self, tmp_hermes_home):
        before = {}
        for cfg in [tmp_hermes_home / "config.yaml", *sorted((tmp_hermes_home / "profiles").glob("*/config.yaml"))]:
            before[str(cfg)] = cfg.read_bytes()
        generator.build_plan(tmp_hermes_home, REGISTRY, SURFACES)
        for cfg, data in before.items():
            assert Path(cfg).read_bytes() == data, f"dry-run modified {cfg}"

    def test_main_model_migration_is_planned_and_other_mains_are_invariant(
            self, tmp_hermes_home, by_id, surfaces):
        """The declared migration changes its whole model route and nothing else."""
        def extract_main(doc):
            m = doc.get("model")
            if isinstance(m, dict) and m.get("default"):
                return str(m["default"])          # shape A: model.default
            if isinstance(m, str):
                return str(m)                     # shape B: model scalar
            ag = doc.get("agent")
            if isinstance(ag, dict) and ag.get("model"):
                return str(ag["model"])           # shape C: agent.model
            return None                            # NEEDS DECISION

        plan = generator.build_plan(tmp_hermes_home, REGISTRY, SURFACES)
        checked = 0
        for entry in plan["entries"]:
            if entry["surface"] == "default (root config.yaml)":
                continue
            with open(entry["path"], encoding="utf-8") as fh:
                doc = yaml.safe_load(fh)
            main = extract_main(doc)
            if entry["surface"] == "content-strategist":
                assert main == "minimax/minimax-m3-free"
                assert entry["main_model"] == "deepseek-v4-flash"
                model_change = next(c for c in entry["changes"] if c["field"] == "model")
                primary = by_id["slot-dsflash-1"]["hermes"]
                assert model_change["old"]["default"] == main
                assert model_change["new"] == {
                    **model_change["old"],
                    "default": primary["model"],
                    "provider": primary["provider"],
                    "base_url": primary["base_url"],
                    "reasoning_effort": primary["reasoning_effort"],
                }
                checked += 1
                continue
            if entry["surface"] == "denji-ledger":
                assert entry["main_model"] == "mimo-v2.5"
                if main == "stepfun/step-3.7-flash:free":
                    model_change = next(c for c in entry["changes"] if c["field"] == "model")
                    primary = by_id["slot-mimo-v25-main"]["hermes"]
                    assert model_change["old"]["default"] == main
                    assert model_change["new"]["default"] == primary["model"]
                else:
                    assert main == "mimo-v2.5"
                    assert not any(c["field"] == "model" for c in entry["changes"])
                checked += 1
                continue
            if entry["surface"] in {"gojo-admin", "gojo-calendar", "gojo-mailbox", "wesker-backup", "wesker-ops"}:
                assert entry["main_model"] == "deepseek-v4-flash"
                if main == "gemma4:31b":
                    model_change = next(c for c in entry["changes"] if c["field"] == "model")
                    primary = by_id["slot-dsflash-1"]["hermes"]
                    assert model_change["old"]["default"] == main
                    assert model_change["new"]["default"] == primary["model"]
                else:
                    assert main == "deepseek-v4-flash"
                    assert not any(c["field"] == "model" for c in entry["changes"])
                checked += 1
                continue
            if entry["surface"] in {"ceecee", "dezzy-brand", "dezzy-component-lib", "dezzy-design-system", "dezzy-image-prompt", "dezzy-ux-prototype", "misa-misa", "quan-ux"}:
                # 2026-09-04 approved remap: paid minimax mains move to free
                # qwen3.8-flash (bai). Same shape as the gemma branch above.
                assert entry["main_model"] == "qwen3.8-flash"
                if main in ("minimax-m3", "minimax/minimax-m3-free"):
                    model_change = next(c for c in entry["changes"] if c["field"] == "model")
                    primary = by_id["slot-qwenflash-1"]["hermes"]
                    _old_main = model_change["old"]
                    if isinstance(_old_main, dict):
                        _old_main = _old_main.get("default")
                    assert _old_main == main
                    assert model_change["new"]["default"] == primary["model"]
                    assert model_change["new"]["provider"] == primary["provider"]
                    assert model_change["new"]["base_url"] == primary["base_url"]
                else:
                    assert main == "qwen3.8-flash"
                    assert not any(c["field"] == "model" for c in entry["changes"])
                checked += 1
                continue
            if entry["surface"] == "sirvir" and main == "active:main":
                # Turbofit exposes a role alias; cloud fallback slots retain
                # their physical Qwen model identity. Do not rename cloud slots.
                assert entry["main_model"] == "qwen3.8-27b"
                assert doc["model"]["provider"] == "custom:turbofit-local"
                assert not any(c["field"] == "model" for c in entry["changes"])
                checked += 1
                continue
            assert entry["main_model"] == main, (
                f"{entry['surface']}: registry says {entry['main_model']}, live is {main}"
            )
            checked += 1
        assert checked == 64

    def test_content_primary_is_excluded_and_fallback_order_is_preserved(
            self, tmp_hermes_home, by_id, surfaces, registry):
        plan = generator.build_plan(tmp_hermes_home, REGISTRY, SURFACES)
        chain = _chain_entries(plan, "content-strategist")
        primary = by_id["slot-dsflash-1"]["hermes"]
        assert generator._deployment_key(primary) not in {
            generator._deployment_key(entry) for entry in chain
        }
        assert [entry["route_slot"] for entry in chain] == [
            "slot-dsflash-4", "codex-fallback", "local-final"
        ]

    def test_content_migration_refuses_unexpected_live_main(self, tmp_hermes_home):
        cfg = tmp_hermes_home / "profiles" / "content-strategist" / "config.yaml"
        doc = yaml.safe_load(cfg.read_text())
        doc["model"]["default"] = "operator-changed/model"
        cfg.write_text(yaml.safe_dump(doc, sort_keys=False))
        with pytest.raises(generator.ValidationError, match="expected old main.*operator-changed/model"):
            generator.build_plan(tmp_hermes_home, REGISTRY, SURFACES)

    def test_scalar_live_main_migrates_to_mapping(self, by_id):
        """Scalar live mains migrate to a fresh mapping from the primary slot.

        No fields to preserve; mismatch still refuses loudly.
        """
        surf = {"surface": "probe", "slots": [], "primary_model_migration": {
            "expected_old_main": "minimax-m3", "primary_slot": "slot-qwenflash-1"}}
        out = generator._migrated_model({"model": "minimax-m3"}, surf, by_id)
        primary = by_id["slot-qwenflash-1"]["hermes"]
        assert out == {"default": primary["model"], "provider": primary["provider"],
                       "base_url": primary["base_url"],
                       "reasoning_effort": primary["reasoning_effort"]}
        with pytest.raises(generator.ValidationError, match="expected old main"):
            generator._migrated_model({"model": "something-else"}, surf, by_id)

    def test_content_migration_preserves_unrelated_model_fields(self, tmp_hermes_home):
        cfg = tmp_hermes_home / "profiles" / "content-strategist" / "config.yaml"
        doc = yaml.safe_load(cfg.read_text())
        doc["model"]["context_length"] = 131072
        doc["model"]["operator_note"] = "preserve-me"
        cfg.write_text(yaml.safe_dump(doc, sort_keys=False))
        plan = generator.build_plan(tmp_hermes_home, REGISTRY, SURFACES)
        change = next(
            c for e in plan["entries"] if e["surface"] == "content-strategist"
            for c in e["changes"] if c["field"] == "model"
        )
        assert change["new"]["context_length"] == 131072
        assert change["new"]["operator_note"] == "preserve-me"

    def test_content_migration_is_idempotent_after_apply(self, tmp_hermes_home, by_id, surfaces):
        """Re-planning on an already-migrated surface is a model no-op, not an error."""
        plan = generator.build_plan(tmp_hermes_home, REGISTRY, SURFACES)
        # simulate the applied state: live main now equals the primary model
        cfg = tmp_hermes_home / "profiles" / "content-strategist" / "config.yaml"
        doc = yaml.safe_load(cfg.read_text())
        primary = by_id["slot-dsflash-1"]["hermes"]
        doc["model"]["default"] = primary["model"]
        doc["model"]["provider"] = primary["provider"]
        doc["model"]["base_url"] = primary["base_url"]
        cfg.write_text(yaml.safe_dump(doc, sort_keys=False))
        plan2 = generator.build_plan(tmp_hermes_home, REGISTRY, SURFACES)
        entry = next(e for e in plan2["entries"] if e["surface"] == "content-strategist")
        assert not any(c["field"] == "model" for c in entry["changes"]), (
            "already-migrated surface must not plan another model migration")

    def test_content_strategist_uses_approved_same_model_slots(self, by_id, surfaces):
        surface = next(s for s in surfaces if s["surface"] == "content-strategist")
        assert surface["main_model"] == "deepseek-v4-flash"
        assert surface["slots"] == [f"slot-dsflash-{i}" for i in range(1, 6)]
        assert {by_id[slot]["model_id"] for slot in surface["slots"]} == {
            surface["main_model"]
        }

    def test_keep_current_surfaces_have_no_routes(self, surfaces):
        for s in surfaces:
            if s.get("routes_policy") == "keep_current":
                assert s["slots"] == []

    def test_apply_plan_backs_up_and_writes(self, tmp_hermes_home, tmp_path):
        # deterministic: seed a pre-migration denji chain so the apply has a diff
        assert _strip_bypass_env_keys(tmp_hermes_home) > 0
        _seed_old_denji_chain(tmp_hermes_home)
        plan = generator.build_plan(tmp_hermes_home, REGISTRY, SURFACES)
        backup_dir = tmp_path / "backups"
        applied = generator.apply_plan(plan, backup_dir)
        assert applied, "expected some surfaces to be applied"
        for a in applied:
            b = Path(a["backup"]).read_bytes()
            assert b  # backup non-empty
        with open(tmp_hermes_home / "profiles" / "denji" / "config.yaml", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
        fp = doc["fallback_providers"]
        providers = [e["provider"] for e in fp]
        assert providers.count("ollama-cloud") == 1  # deduped, single entry
        assert "nous" not in providers
        assert fp[-1]["model"] == "active:main"

    def test_deterministic_plan_across_runs(self, tmp_hermes_home):
        p1 = generator.build_plan(tmp_hermes_home, REGISTRY, SURFACES)
        p2 = generator.build_plan(tmp_hermes_home, REGISTRY, SURFACES)
        assert json.dumps(p1, sort_keys=True) == json.dumps(p2, sort_keys=True)

    def test_removed_capabilities_summary_present(self, tmp_hermes_home):
        """S3: the plan reports every (provider, model) capability removed from
        existing chains — reviewer sees capability loss, not just the new chain.
        Deterministic: seed a pre-migration denji chain so removals are visible."""
        _seed_old_denji_chain(tmp_hermes_home)
        plan = generator.build_plan(tmp_hermes_home, REGISTRY, SURFACES)
        rc = plan["removed_capabilities"]
        assert rc["aggregate"], "expected removed capabilities to be reported"
        assert rc["by_surface"], "expected per-surface removals"
        # cross-provider fallback removals are visible
        providers = {r["provider"] for r in rc["aggregate"]}
        assert "nous" in providers, "removed nous fallbacks must be surfaced"
        by_surfs = rc["by_surface"]["denji"]
        denji_removed = {(r["provider"], r["model"]) for r in by_surfs}
        assert ("nous", "tencent/hy3:free") in denji_removed

    def test_pool_requirement_reported_against_target(self, tmp_hermes_home):
        """S1: plan must check the native pool cascade requirement against the
        target config's credential_pool_strategies."""
        plan = generator.build_plan(tmp_hermes_home, REGISTRY, SURFACES)
        pr = [x for x in plan["pool_requirements"] if x["provider"] == "ollama-cloud"]
        assert pr, "ollama-cloud multi-account pool requirement must be reported"
        entry = pr[0]
        assert entry["accounts"] == ["ollama-cloud/1", "ollama-cloud/2"]
        assert entry["strategy_required"] == "fill_first"
        assert entry["status"] in {"met", "blocked"}
        if entry["status"] == "blocked":
            assert entry["blockers"]

    def test_pool_requirement_flags_missing_strategy(self, tmp_hermes_home):
        """Adversarial: remove the strategy from the target copy → blocked status."""
        cfg = tmp_hermes_home / "config.yaml"
        doc = yaml.safe_load(cfg.read_text())
        doc.get("credential_pool_strategies", {}).pop("ollama-cloud", None)
        cfg.write_text(yaml.safe_dump(doc, sort_keys=True))
        plan = generator.build_plan(tmp_hermes_home, REGISTRY, SURFACES)
        pr = [x for x in plan["pool_requirements"] if x["provider"] == "ollama-cloud"]
        assert pr and pr[0]["status"] == "blocked"
        assert any("credential_pool_strategies" in b for b in pr[0]["blockers"])


# ------------------------------------------------------- env-bypass plan tests

class TestEnvBypass:
    def test_plan_reports_real_bypass_blockers(self, tmp_hermes_home):
        """A profile .env carrying OLLAMA_API_KEY must be reported as a
        deployment blocker. Deterministic: seed the key into the temp tree (the
        live tree is post-Ollama-cleanup and no longer carries it)."""
        _seed_bypass_env_key(tmp_hermes_home)
        plan = generator.build_plan(tmp_hermes_home, REGISTRY, SURFACES)
        report = generator.scan_env_key_bypass_tree(Path(plan["target_root"]),
                                                    generator.load_registry(REGISTRY))
        blockers = report["blockers"]
        ollama_blockers = [b for b in blockers if b["env_key"] == "OLLAMA_API_KEY"]
        assert ollama_blockers, "expected OLLAMA_API_KEY bypass blockers"
        surfaces_hit = {b["surface"] for b in ollama_blockers}
        assert "denji" in surfaces_hit
        for b in ollama_blockers:
            assert "auth" in b["remediation"].lower() or "cleanup" in b["remediation"].lower()

    def test_no_env_values_in_report(self, tmp_hermes_home):
        plan = generator.build_plan(tmp_hermes_home, REGISTRY, SURFACES)
        report = generator.scan_env_key_bypass_tree(tmp_hermes_home,
                                                    generator.load_registry(REGISTRY))
        text = json.dumps(report)
        # every scanned entry only lists key NAMES
        for s in report["scanned"]:
            for k in s.get("env_keys", []):
                assert "=" not in k


# ----------------------------------------------------------------- atomicity

class TestAtomicApply:
    def test_atomic_rollback_on_injected_failure(self, tmp_hermes_home, tmp_path, monkeypatch):
        """Inject a failure mid-apply: ALL written files must be rolled back to
        byte-identical pre-apply state; backups retained."""
        # deterministic: seed a pre-migration chain so the apply has work to do
        assert _strip_bypass_env_keys(tmp_hermes_home) > 0
        _seed_old_denji_chain(tmp_hermes_home)
        _seed_old_ceecee_brand_chain(tmp_hermes_home)
        plan = generator.build_plan(tmp_hermes_home, REGISTRY, SURFACES)
        changed = [e["path"] for e in plan["entries"] if e["changes"]]
        assert len(changed) > 1
        before = {p: Path(p).read_bytes() for p in changed}

        real_build_chain = generator.build_chain
        state = {"n": 0}
        # Fail on the FIRST build_chain call that happens inside apply_plan.
        # build_plan has already consumed several calls during planning, so we
        # reset the counter immediately before apply — the 1st apply-time call
        # is the first changed-surface write, guaranteeing the failure fires
        # mid-apply (not during planning).
        def sabotage(surf, by_id, registry):
            out = real_build_chain(surf, by_id, registry)
            state["n"] += 1
            if state["n"] == 1:
                raise RuntimeError("injected mid-apply failure")
            return out
        monkeypatch.setattr(generator, "build_chain", sabotage)

        with pytest.raises(RuntimeError, match="injected"):
            generator.apply_plan(plan, tmp_path / "backups")
        monkeypatch.undo()

        # every target file is byte-identical to pre-apply (full rollback)
        for path, data in before.items():
            assert Path(path).read_bytes() == data, \
                f"{path} was left modified after rolled-back apply"

    def test_rollback_verifier(self, tmp_hermes_home, tmp_path):
        assert _strip_bypass_env_keys(tmp_hermes_home) > 0
        plan = generator.build_plan(tmp_hermes_home, REGISTRY, SURFACES)
        before = {e["path"]: Path(e["path"]).read_bytes()
                  for e in plan["entries"] if e["changes"]}
        generator.apply_plan(plan, tmp_path / "backups")
        # files now differ from the retained backups (apply succeeded, no rollback)
        import shutil
        stamps = sorted((tmp_path / "backups").glob("*"))
        latest = stamps[-1]
        for entry in plan["entries"]:
            if entry["changes"]:
                rel = entry["surface"].replace(" (root config.yaml)", "__root__")
                assert Path(entry["path"]).read_bytes() != (latest / rel / "config.yaml").read_bytes()
        # restore from backup == verify_rollback true
        for entry in plan["entries"]:
            if not entry["changes"]:
                continue
            rel = entry["surface"].replace(" (root config.yaml)", "__root__")
            shutil.copy2(latest / rel / "config.yaml", entry["path"])
        assert generator.verify_rollback(plan, tmp_path / "backups")
        for p, data in before.items():
            assert Path(p).read_bytes() == data

    def test_apply_all_or_nothing_order(self, tmp_hermes_home, tmp_path, monkeypatch):
        """The snapshot phase precedes ANY write: failing on the very first write
        still leaves the whole tree untouched."""
        assert _strip_bypass_env_keys(tmp_hermes_home) > 0
        _seed_old_denji_chain(tmp_hermes_home)
        plan = generator.build_plan(tmp_hermes_home, REGISTRY, SURFACES)
        changed = [e["path"] for e in plan["entries"] if e["changes"]]
        before = {p: Path(p).read_bytes() for p in changed}

        # inject failure on the very first chain build in phase 2
        def boom_chain(surf, by_id, registry):
            raise RuntimeError("first-write failure")
        monkeypatch.setattr(generator, "build_chain", boom_chain)
        with pytest.raises(RuntimeError):
            generator.apply_plan(plan, tmp_path / "backups2")
        monkeypatch.undo()
        for p in changed:
            assert Path(p).read_bytes() == before[p]


# ------------------------------------------------- fail-closed apply gate tests

class TestApplyGate:
    """QA follow-up: apply must hard-refuse (fail-closed) while env_key_bypass
    blockers exist OR any pool_requirement is unmet. Dry-run still generates
    and reports blockers. No live config/auth/env is ever touched."""

    def test_apply_gate_helper_detects_env_bypass(self, tmp_hermes_home):
        # Deterministic: seed the bypass key (live tree is post-cleanup, key-free)
        _seed_bypass_env_key(tmp_hermes_home)
        plan = generator.build_plan(tmp_hermes_home, REGISTRY, SURFACES)
        plan["env_key_bypass"] = generator.scan_env_key_bypass_tree(
            tmp_hermes_home, generator.load_registry(REGISTRY))
        blockers = generator.apply_blockers(plan)
        assert blockers, "expected env-bypass blockers in the seeded fixture"
        assert any(b.startswith("env_key_bypass:") for b in blockers)

    def test_apply_gate_helper_detects_pool_blocker(self, tmp_hermes_home):
        cfg = tmp_hermes_home / "config.yaml"
        doc = yaml.safe_load(cfg.read_text())
        doc.get("credential_pool_strategies", {}).pop("ollama-cloud", None)
        cfg.write_text(yaml.safe_dump(doc, sort_keys=True))
        plan = generator.build_plan(tmp_hermes_home, REGISTRY, SURFACES)
        plan["env_key_bypass"] = {"scanned": [], "blockers": []}
        blockers = generator.apply_blockers(plan)
        assert any("pool_requirement" in b and "ollama-cloud" in b for b in blockers)

    def test_apply_gate_clean_when_no_blockers(self, tmp_hermes_home):
        assert _strip_bypass_env_keys(tmp_hermes_home) > 0
        plan = generator.build_plan(tmp_hermes_home, REGISTRY, SURFACES)
        assert plan["env_key_bypass"]["blockers"] == []
        pr = [x for x in plan["pool_requirements"] if x["provider"] == "ollama-cloud"]
        assert pr and pr[0]["status"] == "met"
        assert generator.apply_blockers(plan) == []

    def test_apply_plan_raises_on_env_bypass_blockers(self, tmp_hermes_home, tmp_path):
        """Defense-in-depth: apply_plan refuses BEFORE any snapshot/write when
        env-bypass blockers exist — the temp tree stays byte-identical.
        Deterministic: seed the bypass key into the temp tree first."""
        _seed_bypass_env_key(tmp_hermes_home)
        plan = generator.build_plan(tmp_hermes_home, REGISTRY, SURFACES)
        plan["env_key_bypass"] = generator.scan_env_key_bypass_tree(
            tmp_hermes_home, generator.load_registry(REGISTRY))
        assert plan["env_key_bypass"]["blockers"]
        before = {e["path"]: Path(e["path"]).read_bytes()
                  for e in plan["entries"] if e["changes"]}
        with pytest.raises(generator.ValidationError, match="fail-closed"):
            generator.apply_plan(plan, tmp_path / "backups")
        for p, data in before.items():
            assert Path(p).read_bytes() == data, f"{p} was modified by refused apply"
        assert not (tmp_path / "backups").exists(), "no backup dir may be created"

    def test_apply_plan_raises_on_pool_blockers(self, tmp_hermes_home, tmp_path):
        """Adversarial: remove credential_pool_strategies[ollama-cloud] from the
        temp copy → pool requirement blocked → apply refused, tree untouched."""
        cfg = tmp_hermes_home / "config.yaml"
        doc = yaml.safe_load(cfg.read_text())
        doc.get("credential_pool_strategies", {}).pop("ollama-cloud", None)
        cfg.write_text(yaml.safe_dump(doc, sort_keys=True))
        plan = generator.build_plan(tmp_hermes_home, REGISTRY, SURFACES)
        plan["env_key_bypass"] = {"scanned": [], "blockers": []}
        assert [x for x in plan["pool_requirements"]
                if x["provider"] == "ollama-cloud"][0]["status"] == "blocked"
        before = {e["path"]: Path(e["path"]).read_bytes()
                  for e in plan["entries"] if e["changes"]}
        with pytest.raises(generator.ValidationError, match="pool_requirement"):
            generator.apply_plan(plan, tmp_path / "backups")
        for p, data in before.items():
            assert Path(p).read_bytes() == data

    def test_cli_apply_refused_on_env_bypass_blockers(self, run_cli, tmp_hermes_home, tmp_path):
        """Distinct exit code 4 + REFUSED message; tree untouched; no backup dir.
        Deterministic: seed the bypass key so the gate has a blocker to refuse."""
        _seed_bypass_env_key(tmp_hermes_home)
        out_json = tmp_path / "applied.json"
        before = {str(p): p.read_bytes()
                  for p in [tmp_hermes_home / "profiles" / "denji" / "config.yaml",
                            tmp_hermes_home / "config.yaml"]}
        r = run_cli("--out", str(out_json), apply_mode=True)
        assert r.returncode == 4, (r.returncode, r.stderr)
        assert "REFUSED" in r.stderr
        assert "deployment blockers" in r.stderr
        assert "env_key_bypass" in r.stderr
        assert not out_json.exists(), "refused apply must not write an apply report"
        for p, data in before.items():
            assert Path(p).read_bytes() == data

    def test_cli_apply_refused_on_pool_blockers(self, run_cli, tmp_hermes_home):
        """Pool-requirement blockers alone (env already cleaned) also refuse
        with exit 4."""
        _strip_bypass_env_keys(tmp_hermes_home)
        cfg = tmp_hermes_home / "config.yaml"
        doc = yaml.safe_load(cfg.read_text())
        doc.get("credential_pool_strategies", {}).pop("ollama-cloud", None)
        cfg.write_text(yaml.safe_dump(doc, sort_keys=True))
        r = run_cli(apply_mode=True)
        assert r.returncode == 4, (r.returncode, r.stderr)
        assert "REFUSED" in r.stderr
        assert "pool_requirement" in r.stderr
        assert "credential_pool_strategies" in r.stderr

    def test_cli_apply_succeeds_after_cleanup(self, run_cli, tmp_hermes_home, tmp_path):
        """After the simulated approved auth cleanup (bypass env keys removed
        from the TEMP copy only) apply succeeds and writes the plan.
        Deterministic: seed a pre-migration denji chain so the apply has a diff."""
        _strip_bypass_env_keys(tmp_hermes_home)
        _seed_old_denji_chain(tmp_hermes_home)
        out_json = tmp_path / "applied.json"
        r = run_cli("--out", str(out_json), apply_mode=True)
        assert r.returncode == 0, r.stderr
        data = json.loads(out_json.read_text())
        assert data["mode"] == "apply"
        assert data["apply_gate"] == "passed"
        assert len(data["applied"]) > 0
        doc = yaml.safe_load(
            (tmp_hermes_home / "profiles" / "denji" / "config.yaml").read_text())
        assert doc["fallback_providers"][-1]["model"] == "active:main"

    def test_dry_run_reports_blockers_and_exit_zero(self, run_cli, tmp_hermes_home, tmp_path):
        """Dry-run is NEVER gated: it still generates the full plan, reports
        blockers, and exits 0 even with blockers present.
        Deterministic: seed the bypass key so a blocker is always reported."""
        _seed_bypass_env_key(tmp_hermes_home)
        out_json = tmp_path / "dryrun.json"
        r = run_cli("--out", str(out_json))
        assert r.returncode == 0, r.stderr
        data = json.loads(out_json.read_text())
        assert data["mode"] == "dry-run"
        assert len(data["plan"]["entries"]) == 65
        assert data["plan"]["env_key_bypass"]["blockers"], \
            "dry-run must report env-bypass blockers"
        assert data["apply_blockers"], "dry-run must surface apply_blockers list"

    def test_dry_run_untouched_by_gate(self, tmp_hermes_home):
        """Build-plan + env scan + gate never mutates the tree (dry-run invariant).
        Deterministic: seed the bypass key so a blocker is present to gate on."""
        _seed_bypass_env_key(tmp_hermes_home)
        before = {str(p): p.read_bytes() for p in
                  [tmp_hermes_home / "config.yaml",
                   *sorted((tmp_hermes_home / "profiles").glob("*/config.yaml"))]}
        plan = generator.build_plan(tmp_hermes_home, REGISTRY, SURFACES)
        # mirror the CLI: env_key_bypass is populated by the caller via scan
        plan["env_key_bypass"] = generator.scan_env_key_bypass_tree(
            tmp_hermes_home, generator.load_registry(REGISTRY))
        assert generator.apply_blockers(plan)  # blockers present, nothing written
        for p, data in before.items():
            assert Path(p).read_bytes() == data


# ----------------------------------------------------------------- CLI tests

class TestCLI:
    def test_dry_run_default_no_apply(self, run_cli, tmp_path):
        r = run_cli("--out", str(tmp_path / "dryrun.json"))
        assert r.returncode == 0, r.stderr
        data = json.loads((tmp_path / "dryrun.json").read_text())
        assert data["mode"] == "dry-run"
        assert len(data["plan"]["entries"]) == 65

    def test_apply_without_confirm_refused(self, run_cli, tmp_hermes_home):
        before = (tmp_hermes_home / "profiles" / "denji" / "config.yaml").read_bytes()
        cmd = [sys.executable, str(GENERATOR_CLI), "--target-root", str(tmp_hermes_home), "--apply"]
        r = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120)
        assert r.returncode == 2
        assert "REFUSED" in (r.stderr + r.stdout)
        assert (tmp_hermes_home / "profiles" / "denji" / "config.yaml").read_bytes() == before

    def test_apply_wrong_confirm_refused(self, run_cli, tmp_hermes_home):
        cmd = [sys.executable, str(GENERATOR_CLI), "--target-root", str(tmp_hermes_home),
               "--apply", "--confirm", "yes"]
        r = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120)
        assert r.returncode == 2
        assert "REFUSED" in (r.stderr + r.stdout)

    def test_apply_guarded_writes_and_backs_up(self, run_cli, tmp_hermes_home, tmp_path):
        out_json = tmp_path / "applied.json"
        # deterministic: seed bypass key + a pre-migration chain so apply passes with a diff
        assert _strip_bypass_env_keys(tmp_hermes_home) > 0
        _seed_old_denji_chain(tmp_hermes_home)
        r = run_cli("--out", str(out_json), apply_mode=True)
        assert r.returncode == 0, r.stderr
        data = json.loads(out_json.read_text())
        assert data["mode"] == "apply"
        assert data["apply_gate"] == "passed"
        assert len(data["applied"]) > 0
        cfg = tmp_hermes_home / "profiles" / "denji" / "config.yaml"
        doc = yaml.safe_load(cfg.read_text())
        fp = doc["fallback_providers"]
        assert fp[-1]["model"] == "active:main"
        assert all(e.get("route_class") == "perm" for e in fp)
        # single ollama-cloud entry with pool metadata
        oc = [e for e in fp if e["provider"] == "ollama-cloud"]
        assert len(oc) == 1
        assert oc[0]["pool_accounts"] == ["ollama-cloud/1", "ollama-cloud/2"]

    def test_live_tree_apply_refused_without_live_flag(self):
        """S2: apply against the LIVE ~/.hermes must be refused with exit 3
        unless the separate --allow-live-apply flag is passed. This test never
        passes --allow-live-apply, so the live tree cannot be touched."""
        cmd = [sys.executable, str(GENERATOR_CLI),
               "--target-root", str(LIVE_HERMES_HOME),
               "--apply", "--confirm", "YES-APPLY-ROUTES",
               "--out", "/dev/null"]
        r = subprocess.run(cmd, capture_output=True, text=True,
                           stdin=subprocess.DEVNULL, timeout=120)
        assert r.returncode == 3, (r.returncode, r.stderr)
        assert "LIVE" in r.stderr
        assert "--allow-live-apply" in r.stderr

    def test_apply_requires_target_root(self):
        r = subprocess.run([sys.executable, str(GENERATOR_CLI), "--apply",
                            "--confirm", "YES-APPLY-ROUTES"],
                           capture_output=True, text=True,
                           stdin=subprocess.DEVNULL, timeout=60)
        assert r.returncode != 0

    def test_live_tree_untouched_after_tests(self):
        """Live tree hash spot-check: the live denji config is unchanged
        relative to session start (read-only invariant)."""
        live = yaml.safe_load((LIVE_HERMES_HOME / "profiles" / "denji" / "config.yaml").read_text())
        assert live["model"]["default"] == "glm-5.3-flash"  # untouched live state

# ------------------------------------------------------- capability matrix (2026-09-02)

class TestCapabilityMatrix:
    """Sahil directive 2026-09-02 after the minimax@xkiro incident: the registry
    must hard-error on any ENABLED slot whose provider does not carry its model,
    and must enforce permanent codex isolation (gpt-5.6-* on openai-codex only)."""

    def _write_registry(self, tmp_path, slots, matrix=None, constraints=None):
        reg = {
            "slots": slots,
            "codex_fallback": {
                "sol": {"provider": "openai-codex", "model": "gpt-5.6-sol",
                        "base_url": "https://chatgpt.com/backend-api/codex",
                        "reasoning_effort": "xhigh"},
                "luna": {"provider": "openai-codex", "model": "gpt-5.6-luna",
                         "base_url": "https://chatgpt.com/backend-api/codex",
                         "reasoning_effort": "xhigh"},
            },
            "local_final": {"provider": "custom:turbohaul-local", "model": "qwen3.8-27b",
                            "base_url": "http://127.0.0.1:11410/v1",
                            "reasoning_effort": "medium"},
            "model_capability_matrix": matrix or {
                "custom:xkiro-free": ["deepseek/deepseek-v4-flash"],
                "openai-codex": ["gpt-5.6-sol", "gpt-5.6-luna"],
                "custom:bai": ["minimax-m3"],
            },
            "model_routing_constraints": constraints or {
                "codex_isolation": {"model_pattern": "gpt-5.6-*",
                                    "allowed_providers": ["openai-codex"]},
            },
        }
        p = tmp_path / "route-slots.yaml"
        p.write_text(yaml.safe_dump(reg, sort_keys=False))
        return p

    @staticmethod
    def _slot(sid="s1", model="minimax-m3", provider="custom:bai",
              klass="perm", status="approved", disabled=False, approved_at="2026-09-02T00:00:00Z"):
        return {
            "slot": sid, "model_id": model, "provider_account": "acct/1",
            "class": klass, "status": status, "approved_at": approved_at,
            "expires_at": None, "usage_limit": None, "allowed_profiles": None,
            "replacement_slot": None, "disabled": disabled,
            "hermes": {"provider": provider, "model": model,
                       "base_url": "https://example.invalid/v1",
                       "reasoning_effort": "medium"},
        }

    def test_enabled_impossible_slot_hard_errors(self, tmp_path):
        reg = self._write_registry(tmp_path, [self._slot(model="minimax-m3", provider="custom:xkiro-free")])
        with pytest.raises(generator.ValidationError, match="NOT served by provider"):
            generator.validate_registry(generator.load_registry(reg))

    def test_disabled_impossible_slot_tolerated(self, tmp_path):
        reg = self._write_registry(tmp_path, [self._slot(model="minimax-m3", provider="custom:xkiro-free", disabled=True, status="retired")])
        by_id = generator.validate_registry(generator.load_registry(reg))
        assert by_id["s1"]["disabled"] is True

    def test_enabled_possible_slot_passes(self, tmp_path):
        reg = self._write_registry(tmp_path, [self._slot()])
        by_id = generator.validate_registry(generator.load_registry(reg))
        assert "s1" in by_id

    def test_unknown_provider_hard_errors(self, tmp_path):
        reg = self._write_registry(tmp_path, [self._slot(provider="custom:brand-new-provider")])
        with pytest.raises(generator.ValidationError, match="not in model_capability_matrix"):
            generator.validate_registry(generator.load_registry(reg))

    def test_missing_matrix_hard_errors(self, tmp_path):
        reg = {
            "slots": [self._slot()],
            "codex_fallback": {}, "local_final": {},
        }
        p = tmp_path / "route-slots.yaml"
        p.write_text(yaml.safe_dump(reg, sort_keys=False))
        with pytest.raises(generator.ValidationError, match="model_capability_matrix"):
            generator.validate_registry(generator.load_registry(p))

    def test_codex_isolation_blocks_foreign_provider_slot(self, tmp_path):
        reg = self._write_registry(tmp_path, [self._slot(model="gpt-5.6-sol", provider="custom:bai")])
        with pytest.raises(generator.ValidationError, match="CODEX ISOLATION"):
            generator.validate_registry(generator.load_registry(reg))

    def test_codex_slot_on_openai_codex_passes(self, tmp_path):
        reg = self._write_registry(tmp_path, [self._slot(model="gpt-5.6-sol", provider="openai-codex")])
        by_id = generator.validate_registry(generator.load_registry(reg))
        assert "s1" in by_id

    def test_codex_isolation_checked_even_when_matrix_lies(self, tmp_path):
        """If someone adds gpt-5.6 to bai's matrix row, the matrix itself is
        rejected — the isolation rule outranks the matrix."""
        matrix = {
            "custom:bai": ["minimax-m3", "gpt-5.6-sol"],  # poisoned row
            "openai-codex": ["gpt-5.6-sol", "gpt-5.6-luna"],
        }
        reg = self._write_registry(tmp_path, [], matrix=matrix)
        with pytest.raises(generator.ValidationError, match="codex-isolated"):
            generator.validate_registry(generator.load_registry(reg))

    def test_live_registry_passes_full_validation(self, registry):
        """The shipped registry must itself validate under the new rules."""
        by_id = generator.validate_registry(registry)
        assert len(by_id) >= 30

    def test_no_enabled_slot_violates_matrix(self, registry):
        """Adversarial sweep: every ENABLED slot in the shipped registry must be
        capability-true (this is the regression test for 2026-09-02)."""
        matrix = registry["model_capability_matrix"]
        for s in registry["slots"]:
            if s.get("disabled"):
                continue
            h = s.get("hermes") or {}
            allowed = matrix.get(str(h.get("provider")))
            assert allowed is not None, f"{s['slot']}: unknown provider {h.get('provider')}"
            assert h.get("model") in allowed, f"{s['slot']}: {h.get('model')}@{h.get('provider')} impossible"

    def test_no_enabled_slot_violates_codex_isolation(self, registry):
        for s in registry["slots"]:
            if s.get("disabled"):
                continue
            h = s.get("hermes") or {}
            model = str(h.get("model") or "")
            if model.startswith("gpt-5.6"):
                assert h.get("provider") == "openai-codex", f"{s['slot']} breaks codex isolation"

    def test_replacement_slot_forward_reference(self, tmp_path):
        """Replacement targets appearing LATER in the file must resolve (the
        pre-patch validator false-positived on forward references)."""
        slots = [
            self._slot(sid="early", model="minimax-m3", provider="custom:bai",
                       disabled=True, status="retired"),
        ]
        slots[0]["replacement_slot"] = "late"
        slots[0]["provider_account"] = "acct/old"
        slots.append(self._slot(sid="late", model="minimax-m3", provider="custom:bai"))
        slots[1]["provider_account"] = "acct/new"
        reg = self._write_registry(tmp_path, slots)
        by_id = generator.validate_registry(generator.load_registry(reg))
        assert by_id["early"]["replacement_slot"] == "late"

# Ares SOUL Kit 2026-08-22 — Audit Handoff

## Verdict

**Source-verified focused implementation; active-runtime promotion blocked; behavioral benefit unverified.**

The newer `Ares_SOUL_Kit_2026-08-21.zip` is the canonical input. Its SHA-256 is `7cbd9471d10830a1ad994bbdafb072890978c41cb7fd0144e134478697d62186`, and its internal manifest passed.

## Implemented

- Canonical root Ares SOUL installed at `~/.ares/SOUL.md`.
- Canonical `ares-operating-modes` skill installed at `~/.ares/skills/strategy/ares-operating-modes/`.
- Explorer and Public profiles created as isolated no-clone profiles with canonical variant SOULs, one kit skill, profile-local scalar model configuration, and no copied credentials.
- Ares-managed fallback selector implemented in source.
- Fresh managed root/profile SOUL seeding corrected: upstream Hermes `DEFAULT_SOUL_MD` remains normal-Hermes behavior but is not auto-created or upgraded under `ARES_MANAGED_RUNTIME=1`.
- Existing custom and legacy SOUL files remain untouched.
- Codex and Responses fallback consumers use the shared selector.

## Verified

- Kit manifest and deterministic kit tests passed.
- Active runtime loaded root, Explorer, and Public SOULs from their own homes and matched the canonical source content after loader normalization.
- Focused identity/config/profile tests: `14 passed` for seed suppression and fallback, and `135 passed` across the affected focused suite.
- `ruff`, `compileall`, and `git diff --check` passed.
- Full Recursive Agent workspace validation, strict Clippy, formatting, cargo-deny, fuzz targets, three-owner conformance, and installed plugin IPC/pack/replay gates passed in their separate receipts.
- Actual Ares role bots reviewed the design and verification cycle; their outputs were not treated as proof without controller reruns.

## Failed / blocked / not run

- Broader Ares config/profile suite: `135 passed, 2 skipped, 1 failed`; the failure is the inherited `~/.hermes` versus Ares `~/.ares` default-home expectation, not the SOUL repair.
- Active runtime still points to release `48a88d5e6a361ebee13416a181aa31fe5df46fba`; the source patch is in dirty checkout `e2a870a7e2c0b4028965735bad53e190473f673c`. No release activation or gateway restart occurred.
- Held-out behavior evaluation, role discriminant evaluation, routing precision, task quality, token/cost, reliability, and production claims remain unverified.
- Optional deterministic router remains intentionally uninstalled.
- Four credential-shaped plaintext tokens reported in the supplied archive remain an open security response; no credential-sensitive action was taken.

## Semantic-role next phase

The live profile reviews identified the remaining product gap: SOUL files and profiles establish identity documents, but they do not yet enforce semantic authority lanes or produce mandatory role artifacts. The next phase must define and verify:

- Public claim-block artifacts;
- Explorer dissent artifacts that survive synthesis;
- Data/evidence lane restrictions;
- durable per-role profile/history ownership;
- a controller-readable `Finding → Implementation → Verification` chain.

## Exact rerun commands

```bash
cd /home/sikmindz/Coding/Ares
PYTHONPATH=$PWD python3 -m pytest -q \
  tests/hermes_cli/test_config.py::TestEnsureHermesHome \
  tests/hermes_cli/test_profiles.py::TestCreateProfile \
  tests/agent/test_ares_managed_identity.py
python3 -m ruff check hermes_cli/config.py hermes_cli/profiles.py agent/prompt_builder.py agent/system_prompt.py agent/codex_responses_adapter.py agent/transports/codex.py tests/hermes_cli/test_config.py tests/hermes_cli/test_profiles.py tests/agent/test_ares_managed_identity.py tests/agent/test_system_prompt.py
```

Do not call this release-ready or behaviorally validated until the blocked gates are separately closed.

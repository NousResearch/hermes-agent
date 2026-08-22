# Governed Authority Construction Pattern

## When to Use

Build a governed authority in the LAH Stack — a self-contained subsystem that owns a specific domain (maintenance, security, compliance, etc.) with fail-closed governance, deterministic probes, structured evidence, and an explicable health score.

Existing authorities that followed this pattern:
- **LAH Git Global Policy V1** (git-policy/) — 14 invariants, 6 modules, multi-repo Git governance
- **LAH Business Intelligence Authority V1** — policy-as-code for business assets
- **LAH Maintenance Authority V1** (maintenance-authority/) — 10-layer pipeline, 9 families, 52 probes

## Architecture Template

```
tools/<authority-name>/
├── index.mjs                  # Public API — re-exports key symbols
├── authority-manifest.mjs     # Authority identity, invariants, registries
├── schema.mjs                 # Evidence/signal schemas + factory functions
├── pipeline.mjs               # Orchestration pipeline (3 maintenance levels)
├── health-score.mjs           # Weighted scoring with anti-masking (optional)
├── auth/
│   ├── resolver.mjs           # Authorization policy resolver (fail-closed)
│   └── safe-ops.mjs           # Op definitions: SAFE / HUMAN_GATE / FORBIDDEN
├── layers/
│   └── L1-*.mjs to L10-*.mjs  # Pipeline layers (when following 10-layer pattern)
├── probes/
│   ├── index.mjs              # Probe family definitions
│   └── <family>.mjs           # Per-family probe implementations
├── tests/
│   ├── schema.test.mjs        # Schema validation tests
│   ├── auth.test.mjs          # Authorization policy tests
│   ├── health-score.test.mjs  # Health score tests (if applicable)
│   ├── pipeline.test.mjs      # Pipeline orchestration tests
│   └── scenarios.test.mjs     # Realistic scenario tests
└── data/                      # Runtime data (if persistent state needed)
    └── <authority>-data.jsonl
```

## The 10-Layer Pipeline Pattern

The pipeline layers are strictly separated — each writes to specific MEP (evidence package) fields and can be disabled or replaced independently:

| Layer | Name | Responsibility | Writes to MEP |
|-------|------|---------------|---------------|
| L1 | Collection | Deterministic observation, raw probe output | `observed_value`, `source_ref` |
| L2 | Normalisation | Signal → standardised MEP | `normalized_value`, `status`, `severity` |
| L3 | Correlation | Anomaly detection, dedup, trend analysis | `anomaly`, `trend`, `rate_of_change` |
| L4 | Diagnosis | Root cause analysis, systemic flags | `root_cause`, `impact_list`, `systemic_flag` |
| L5 | Risk Classification | 5-level risk scoring, urgency | `risk_level`, `risk_score`, `urgency` |
| L6 | Remediation | Auto-remediation proposals | `proposal`, `auth_category`, `automatable` |
| L7 | Decision | Governed approval gate | `verdict`, `decided_by`, `approved_at` |
| L8 | Execution | Bounded execution (only if risk allows) | `execution` |
| L9 | Validation | Post-repair verification | `validation` |
| L10 | Receipt | Terminal receipt, incident memory | `incident_id`, `health_score_delta`, `lifecycle` |

Key rule: L1-L3 are always allowed (read-only observations). L4-L6 require the pipeline context. L7-L10 require authorization.

## Maintenance Evidence Package (MEP) Schema

The MEP is the universal evidence envelope. Design it as a frozen object that gets progressively enriched through the pipeline:

```javascript
{
  // Metadata
  mep_version: "v1",
  mep_id: "MEP_<uuid>",
  generated_at: "<ISO timestamp>",
  trigger: "SCHEDULED | EVENT | MANUAL",
  maintenance_level: "A | B | C",
  layer_of_origin: 1..10,
  pipeline_run_id: "<run-id>",

  // Observation (L1)
  probe_id: "<probe-id>",
  probe_family: "<family>",
  observed_value: <raw>,
  normalized_value: <normalized>,
  observed_at: "<ISO timestamp>",

  // Signal (L2)
  status: "OK | WARNING | CRITICAL | UNKNOWN",
  severity: 0.0..1.0,
  dedup_key: "<sha256>",
  dedup_status: "FRESH | DUPLICATE | SUPERSEDED",

  // Correlation (L3)
  anomaly: null,
  trend: null,
  rate_of_change: null,

  // Diagnosis (L4)
  root_cause: null,
  impact_list: [],
  recurrence: 0,
  systemic_flag: false,

  // Risk (L5)
  risk_level: "INFO | LOW | MEDIUM | HIGH | CRITICAL",
  risk_score: 0.0..1.0,
  urgency: "PLANNING | SOON | NOW",

  // Remediation (L6)
  proposal: null,
  auth_category: "SAFE | HUMAN_GATE | FORBIDDEN",
  automatable: false,

  // Decision (L7)
  verdict: "PENDING | APPROVED | DENIED | WAITING",
  decided_by: null,
  decision_id: null,

  // Execution (L8)
  execution: null,

  // Validation (L9)
  validation: null,

  // Receipt (L10)
  incident_id: null,
  receipt_chain: null,
  health_score_delta: null,
  lifecycle: "OPEN | COMPLETE | ROLLED_BACK",

  // Safety
  safety_flags: {
    readonly_collection: true,
    external_action_taken: false,
    human_authorised: false,
    mutation_allowed: false,
  },
  provenance: {
    collected_by: "lah-<authority>",
    source_command: null,
  }
}
```

## Probe Registry Pattern

Register probes in the authority manifest. Each probe has a stable ID, family, category, maintenance level, and optional threshold:

```javascript
{
  id: "DISK_USAGE",
  family: "SYSTEM",
  category: "disk",
  level: "A",  // A = deterministic, B = daily, C = weekly
  description: "Disk usage percentage by partition",
  threshold: { warning: 0.8, critical: 0.95, unit: "percent" },
  determinism: "pure",
  llm_required: false,
}
```

## Authorization Policy Pattern

Three categories with fail-closed default:

| Category | Behavior | Count (example) |
|----------|----------|----------------|
| SAFE | Auto-approved, no human needed | 67 ops |
| HUMAN_GATE | Needs operator approval | 20 ops |
| FORBIDDEN | Always denied | 48 ops |
| Unknown | Falls back to FORBIDDEN | — |

Key principles:
- Observations and reports are always auto-allowed
- Safe deterministic reversible repairs may be automatable (explicit allowlist)
- Deletion, spend, publish, remote writes, provider changes, deployments, restarts, cron mutations, secrets, network changes, production mutations are always human-gated
- No silent mutation, no global staging, no unauthorized external action

## Health Score Model

Weighted scoring with anti-masking. The score is a penalty-based model where a single critical failure caps the score at 60 (fail-closed guarantee):

```javascript
function computeHealthScore(meps) {
  // 1. Per-family penalties (severity * family weight)
  // 2. Critical incident override: any CRITICAL → score <= 60
  // 3. Systemic debt penalty: if systemic_flag is common
  // 4. Score = 100 - penalties, clamped to [0, 100]
  // 5. Breakdown: per-family detail, override list
  return { score, band, breakdown, overrides };
}
```

Bands: EXCELLENT (90-100), GOOD (75-89), FAIR (50-74), POOR (25-49), CRITICAL (0-24). The score must never mask a critical failure — a single CRITICAL probe always caps the score regardless of other families.

## CLI Integration Pattern

```javascript
// In tools/control-plane/cli.mjs:

// 1. Import at top
import { orchestrate, calculateHealthScore, MAINTENANCE_LEVELS } from '../maintenance-authority/pipeline.mjs';

// 2. Add command handler before main()
async function cmdAuthority(args) {
  const [subcommand, ...rest] = args;
  if (!subcommand) { fail('Usage: cli.mjs authority <level-a|level-b|level-c|health|status>'); }

  switch (subcommand) {
    case 'level-a': {
      // Dry-run by default
      const result = await orchestrate({ level: 'A', trigger: 'CLI', dryRun: !rest.includes('--no-dry-run') });
      output(result);
      break;
    }
    case 'health': {
      const score = calculateHealthScore(meps);
      console.log(`Health Score: ${score.score}/100 [${score.band}]`);
      // ... human-readable breakdown
      break;
    }
    case 'status': {
      output({
        version: MAINTENANCE_AUTHORITY_VERSION,
        levels: MAINTENANCE_LEVELS,
        mode: 'read_only_dryrun',
        dry_run_default: true,
      });
      break;
    }
    default: fail('Unknown subcommand');
  }
}

// 3. Add to switch
case 'authority': await cmdAuthority(args); break;

// 4. Add to help text
```

## Certification Script Pattern

Follow the `scripts/git-policy-certify.mjs` pattern: a standalone script with numbered checks, ✓/✗ output, and a JSON receipt:

```javascript
// scripts/<authority>-certify.mjs

const passed = 0, failed = 0;
function check(name, condition, detail) { /* increment passed/failed, print ✓/✗ */ }

// 1. Module File Integrity — ensure all expected .mjs files exist
// 2. Module Shape — dynamic import each module, verify exported symbols
// 3. Test Suite — run `node --test` and check exit code
// 4. Invariant Verification — verify authority invariants present
// 5. Destructive Command Check — grep for forbidden patterns
// 6. Safety Verification — verify layer count, probe count
// 7. Final Verdict — CERTIFIED / PARTIAL / BLOCKED

const verdict = failed === 0 ? 'CERTIFIED' : 'PARTIAL';
console.log(`Verdict: LAH_${NAME}_V1_${verdict} (${passed} checks)`);
```

Build authority | Run certification | Produce operator packet | Lock memory

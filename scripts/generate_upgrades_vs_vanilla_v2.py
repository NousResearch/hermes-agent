#!/usr/bin/env python3
import re
from pathlib import Path
from datetime import date

ROOT = Path('/home/yeqiuqiu/clawd-architect')
MAP = ROOT / 'reports/openclaw_system_source_of_truth_map_2026-03-20.md'
OUT = ROOT / f'reports/openclaw_upgrades_vs_vanilla_exhaustive_full_v2_{date.today().isoformat()}.md'

text = MAP.read_text(encoding='utf-8')
lines = text.splitlines()

heads = []
for i, l in enumerate(lines):
    m = re.match(r'^## (.+)$', l)
    if m:
        heads.append((i, m.group(1).strip()))

start = next(i for i, (_, h) in enumerate(heads) if h.startswith('Layer 0'))
end = next((i for i, (_, h) in enumerate(heads) if h.startswith('2) Source-of-truth change control')), len(heads))
sel = heads[start:end]

def pull_list(section, key):
    out = []
    for j, s in enumerate(section):
        if s.strip() == key:
            k = j + 1
            while k < len(section) and section[k].startswith('  - '):
                out.append(section[k][4:])
                k += 1
            break
    return out

def pull_field(section, prefix):
    for s in section:
        if s.startswith(prefix):
            return s.split(':', 1)[1].strip().lstrip('* ').strip()
    return ''

lanes = []
for idx, (ln, h) in enumerate(sel):
    nxt = sel[idx + 1][0] if idx + 1 < len(sel) else len(lines)
    sec = lines[ln:nxt]
    lanes.append({
        'name': h,
        'purpose': pull_field(sec, '- **Purpose:**'),
        'maturity': pull_field(sec, '- **Maturity / priority:**'),
        'status': pull_field(sec, '- **Status of docs:**'),
        'roadmap': pull_list(sec, '- **Canonical roadmap doc(s):**'),
        'specs': pull_list(sec, '- **Canonical spec(s):**'),
        'impl': pull_list(sec, '- **Implementation files:**'),
        'tests': pull_list(sec, '- **Tests / validation entrypoints:**'),
        'surfaces': pull_list(sec, '- **Operator surfaces:**'),
        'support': pull_list(sec, '- **Supporting research/audit docs:**'),
    })

lane_blurbs = {
    'Layer 0 - Base runtime/platform': 'Vanilla already has this substrate. Your upgrade keeps it and layers governance/safety/intelligence above it.',
    'A1 - Core control plane + deterministic mutation boundary': 'You added mutation ingress control + verify-before-resume gates; vanilla does not enforce this as your doctrine.',
    'A2 - Continuity / canonical truth OS': 'You added machine-readable continuity truth surfaces and queue reconciliation for restart/successor safety.',
    'A3 - Failover/succession convergence cluster': 'You added deterministic failover FSM + successor proof + replay/stress evidence pipeline.',
    'A4 - Multi-lane topology + authority/lease contract': 'You added explicit lane authority, lease semantics, and crossover guards.',
    'A5 - Verification/contracts/trust + cross-lane fail-close': 'You added schema-first fail-close contract enforcement across runtime lanes.',
    'A6 - Deployment/Ops/Security/Backup/Observability lane': 'You added layered ops reliability, incident, backup/restore, and observability governance.',
    'B1 - Shared brain / shared memory fabric': 'You added typed memory lifecycle and promotion/demotion logic.',
    'B2 - Research OS lifecycle': 'You added deterministic research-to-implementation promotion flow.',
    'B3 - Document/PDF OS': 'You added governed doc/PDF ingestion and quality/ownership gates.',
    'B4 - Browsing/Web OS': 'You added deterministic web automation governance and login-wall handling policy.',
    'B5 - Artifact/Evidence OS (cross-system)': 'You added evidence-first promotion/provenance contracts.',
    'B6 - Model pool/routing/qualification OS': 'You added route policy + qualification + rollout governance + cost control.',
    'B7 - Systems Intelligence Layer': 'You added bounded strategic pattern-mining and reframing lane.',
    'C1 - Operator cockpit / frontend UX': 'You added mission-control style operator UX with blocker-first surfaces.',
    'C2 - Upgrade/Release/DevEx substrate': 'You added staged release/replay/rollback governance with evidence ladders.',
    'C3 - Downstream capability/product systems (governed expanded activation)': 'You added governance envelope for downstream product lanes.',
    'XR - Canonicalization & archive-promotion reconciliation lane (expanded overlay)': 'You added canon-drift detection/correction and archive promotion reconciliation.',
    'XE - Efficiency Control Plane (routing/context/events) lane (expanded overlay)': 'You added context/token/routing efficiency controls with deterministic event posture.',
    'XD - DesignOps lane (expanded)': 'You added schema-governed design system lane and gate integration.',
    'XK - Obsidian knowledge canonicalization lane (expanded)': 'You added deterministic Obsidian-to-memory canonicalization and freshness/retrieval gates.',
    'XP - Personal OS / life-assistant lane (expanded)': 'You added bounded personal planning/review lane with isolation and safety envelope.',
    'XT - Trading journal subsystem lane (expanded)': 'You added append-only risk-bounded trading journal lane with provenance/replay.',
    'XH - Health subsystem lane (expanded)': 'You added non-diagnostic health-support lane with fail-close safety boundaries.',
    'XG - Cross-domain downstream governance lane (expanded)': 'You added cross-domain governance and release/incident gate extensions.',
    'XB - Downstream backend capability fabric lane (expanded)': 'You added capability registry + connector/adapter contract fabric.',
    'XU - Frontend/operator UX productization lane (expanded)': 'You added productized IA/state/action/explainability overlays.',
    'XO - Optional future upgrades lane (expanded optional overlay)': 'You added explicit quarantine for optional experiments to prevent silent canon mutation.',
}

abilities = [
    ('Continuity truthfulness', 'A2/A1', 'Landed and repeatedly hardened', 'continuity_current + queue truth + verify-before-resume/mutation gate posture'),
    ('Failover & succession reliability', 'A3', 'Landed (ongoing boringness hardening)', 'FSM, successor proof, replay evidence, stress-soak checks'),
    ('Control-plane safety', 'A1/A4/A5', 'Landed', 'Ingress guards, authority contracts, fail-close validation'),
    ('Execution governance / anti-false-green', 'A2/A5/B5', 'Landed', 'Event reporting + evidence contracts + strict validation'),
    ('Operator mission-control quality', 'C1/A6', 'Landed, still polishing quiet-state UX', 'Blocker-first surfaces + health/readiness semantics'),
    ('Model routing intelligence', 'B6/XE', 'Landed core, convergence still active', 'Route-class policy + qualification + rollout governance + efficiency controls'),
    ('Coding intelligence quality', 'B9 support queue + A4/A5/C2 enforcement', 'Partially landed (contracts/queue posture strong, deeper runtime quality still maturing)', 'Proposal/apply/archive discipline, decomposition/review packet direction'),
    ('Design intelligence quality', 'B8/XD/XU', 'Partially landed (contracts/packets landed; deeper automation remains)', 'Design packet contracts + gate-linked UX productization'),
    ('Memory intelligence', 'B1/XK', 'Landed core, deeper retrieval ergonomics ongoing', 'Typed memory + Obsidian coupling + freshness gates'),
    ('Obsidian integration', 'XK + memory runtime', 'Landed (governed)', 'Canonicalized Obsidian materialization/retrieval boundaries'),
    ('Document/PDF intelligence', 'B3', 'Landed core', 'Ingestion + quality + ownership gates + ledgers'),
    ('Research->implementation speed', 'B2', 'Landed', 'Deterministic promotion pipeline to queue packets'),
    ('Explainability intelligence', 'B5/C1/XU', 'Landed but still deepening cross-lane federation', 'Evidence/provenance + operator-readable surfaces'),
    ('Cost/token efficiency', 'XE/B6', 'Landed and active', 'Context compaction/routing efficiency with guardrails'),
    ('Release safety', 'C2/A5', 'Landed', 'Staged gates, replay/rollback, compatibility governance'),
    ('Downstream domain governance', 'C3/XG/XB', 'Landed (governed expanded activation)', 'No ungated downstream bypass'),
    ('Personal assistant bounded lane', 'XP', 'Landed bounded', 'Session-isolated personal planning with refusal/escalation envelope'),
    ('Trading journal subsystem', 'XT', 'Landed bounded', 'Append-only, risk-aware, provenance-oriented journal OS'),
    ('Health support subsystem', 'XH', 'Landed bounded', 'Non-diagnostic support with safety fail-close boundaries'),
]

md = []
md.append(f"# OpenClaw vs Vanilla — Full Exhaustive Upgrade Dossier (v2, {date.today().isoformat()})")
md.append('')
md.append('## What this document guarantees')
md.append('- Exhaustive lane-level inventory of upgrades/additions/changes on top of vanilla OpenClaw.')
md.append('- Explicit capability/ability matrix (including coding/design/Obsidian/memory/routing/ops).')
md.append('- Grounded in canonical source-of-truth map and execution-table authorities.')
md.append('')
md.append('## Canonical sources used')
md.append('- `reports/openclaw_system_source_of_truth_map_2026-03-20.md` (updated 2026-04-04)')
md.append('- `reports/openclaw_full_roadmap_execution_table_2026-03-20.md` (updated 2026-03-31)')
md.append('- `reports/openclaw_system_vs_vanilla_operator_and_remaining_work_materials_2026-03-31.md`')
md.append('- `MEMORY.md` and `memory/2026-04-05.md` (latest durable current-state upgrades)')
md.append('')
md.append('## Executive summary')
md.append('Vanilla OpenClaw is a runtime engine. Your system is that engine plus a governed autonomous operating stack with continuity truth OS, failover/succession evidence rails, strong contracts/fail-close discipline, model routing governance, memory/document/Obsidian intelligence lanes, operator mission-control UX, and expanded bounded subsystem overlays.')
md.append('')
md.append('---')
md.append('')
md.append('## A) Full lane-by-lane upgrade inventory')
for i, lane in enumerate(lanes, 1):
    md.append(f"### {i}. {lane['name']}")
    md.append(f"**Delta vs vanilla:** {lane_blurbs.get(lane['name'], 'Governed lane capability added on top of base runtime.')}")
    if lane['purpose']:
        md.append(f"- **Purpose:** {lane['purpose']}")
    if lane['maturity']:
        md.append(f"- **Maturity/Priority:** {lane['maturity']}")
    if lane['status']:
        md.append(f"- **Doc status:** {lane['status']}")
    if lane['impl']:
        md.append(f"- **Implemented runtime files ({len(lane['impl'])}):**")
        for x in lane['impl']:
            md.append(f"  - `{x}`")
    if lane['tests']:
        md.append(f"- **Validation entrypoints ({len(lane['tests'])}):**")
        for x in lane['tests']:
            md.append(f"  - `{x}`")
    if lane['surfaces']:
        md.append(f"- **Operator/runtime surfaces ({len(lane['surfaces'])}):**")
        for x in lane['surfaces']:
            md.append(f"  - `{x}`")
    if lane['specs']:
        md.append(f"- **Canonical specs/contracts ({len(lane['specs'])}):**")
        for x in lane['specs']:
            md.append(f"  - `{x}`")
    if lane['roadmap']:
        md.append(f"- **Roadmap authority refs ({len(lane['roadmap'])}):**")
        for x in lane['roadmap']:
            md.append(f"  - `{x}`")
    md.append('')

md.append('---')
md.append('')
md.append('## B) Capability / ability matrix (explicit)')
md.append('| Ability | Main lanes | Current status | What changed vs vanilla |')
md.append('|---|---|---|---|')
for a, ln, st, desc in abilities:
    md.append(f'| {a} | {ln} | {st} | {desc} |')
md.append('')
md.append('### Notes on coding/design/Obsidian specifically')
md.append('- **Coding intelligence**: this is not only model choice; it includes proposal/apply/archive governance, decomposition/risk packet direction, and controlled execution contracts. Core governance is landed; deeper coding-intelligence runtime depth remains an active convergence area.')
md.append('- **Design intelligence**: packetized design contracts and UX productization overlays are landed (XD/XU/B8 support path), with deeper automation and consistency scoring still maturing.')
md.append('- **Obsidian integration**: deterministic canonicalization lane (XK) + memory coupling is landed; automation posture is governed rather than unconstrained.')
md.append('')
md.append('---')
md.append('')
md.append('## C) Latest post-roadmap tranche (current-state adds)')
md.append('From latest durable memory state, additional upgrades/hardening include:')
md.append('- Continuity/control-plane repair tranche reached clean READY with allowed mutation gate and cleared warning/blocker residue.')
md.append('- VAFS-R0 landed: bounded read-only virtual artifact resolver with strict deny and namespace-aware retrieval (`scripts/vafs_resolver.py`, `tests/test_vafs_resolver.py`).')
md.append('- Cron/session-card reconcile policy landed (`dry-run` default + guarded one-shot promotion/demotion contract).')
md.append('- Direct-message masking/reset-watchdog churn fixes landed in runtime-adjacent reply layer.')
md.append('- Freeze-line posture formalized: convergence-first roadmap (truth quiet-state, validation floor, routing/runtime convergence, boringness hardening).')
md.append('')
md.append('---')
md.append('')
md.append('## D) Final answer to “did we upgrade everything vs vanilla?”')
md.append('You upgraded far beyond vanilla across nearly every operating axis: control plane, continuity/failover, verification/evidence, routing/governance, memory/Obsidian/document intelligence, operator UX, release safety, and bounded domain overlays. Remaining work is mainly convergence/depth/boringness hardening, not missing foundational breadth.')

OUT.write_text('\n'.join(md) + '\n', encoding='utf-8')
print(str(OUT))
print(f"lanes={len(lanes)}")

# Marson Pack Import Plan

> For Hermes: Use subagent-driven-development skill to implement this plan task-by-task.

Goal: Import the full Marson operating-layer pack from the provided tarball into the current Hermes repo without losing provenance or silently overwriting current Hermes work.

Architecture: Preserve Marson’s payload structure from extracted_full_runtime at the repo root using the original directory layout: scripts/, docs/ops/, and ops/openclaw/. Treat this as a bulk import wave first, then do integration and adaptation in follow-up slices. Keep the import reviewable and reversible.

Tech Stack: tar extraction, git, Python/JSON/Markdown assets, shell continuity scripts, Hermes repo file layout.

---

## Task 1: Inventory the tar payload and current repo overlap

Objective: Confirm what will be imported and whether path collisions exist.

Files:
- Source tar: `/home/user/openclaw-snapshot-20260412-215839.tar.gz`
- Compare against repo root: `/home/user/.hermes/hermes-agent`

Step 1: Enumerate extracted_full_runtime top-level directories
Run:
`python - <<'PY'
import tarfile
from collections import Counter
p='/home/user/openclaw-snapshot-20260412-215839.tar.gz'
prefix='home/user/.openclaw/workspace/extracted_full_runtime/'
counts=Counter()
with tarfile.open(p,'r:gz') as t:
    for n in t.getnames():
        if n.startswith(prefix):
            rel=n[len(prefix):]
            if rel and not rel.endswith('/'):
                counts[rel.split('/',1)[0]] += 1
print(counts)
PY`
Expected: scripts/, docs/, ops/ appear as the dominant import roots.

Step 2: Check current repo for pre-existing collisions
Run:
`git status --short --branch`
Expected: Understand current dirty state before import.

Step 3: Commit this plan as the checkpoint document

---

## Task 2: Import Marson payload structure directly into repo root

Objective: Bring over the full extracted_full_runtime implementation payload from the tarball.

Files:
- Create/modify under repo root:
  - `scripts/**`
  - `docs/ops/**`
  - `ops/openclaw/**`

Step 1: Extract only extracted_full_runtime subtree into a temporary staging directory
Step 2: Copy staged `scripts/`, `docs/ops/`, and `ops/openclaw/` into repo root preserving paths
Step 3: Record file counts and changed paths

Verification:
- Imported files exist under the expected repo paths
- No unrelated runtime state/log/secrets are imported

---

## Task 3: Verify the import shape

Objective: Make sure the import is reviewable and provenance-preserving.

Files:
- `scripts/**`
- `docs/ops/**`
- `ops/openclaw/**`

Step 1: Run:
`git diff --stat`
Step 2: Run:
`git status --short`
Step 3: Spot-check representative imported files:
- `scripts/knowledge_promotion_queue.py`
- `scripts/session_topology_routing_policy_contract.py`
- `ops/openclaw/continuity/operator_mission_control.sh`
- `docs/ops/knowledge_review_approval_promotion_queue_v1.md`

Expected: Paths preserved; import limited to Marson pack payload.

---

## Task 4: Report import status and remaining integration work

Objective: Separate “files imported” from “features integrated”.

Deliverables:
- concise summary of imported file groups
- note which current Hermes systems now have imported companion assets
- note that integration/adaptation into Hermes runtime still remains as a later wave

Commit:
`git add docs/plans/2026-04-14-marson-pack-import.md scripts docs ops`
`git commit -m "chore: import marson operating-layer pack assets"`

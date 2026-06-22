#!/usr/bin/env python3
"""setup_identity_loop.py — deploy Friday's identity self-improvement loop.

Idempotent one-shot installer that wires the loop into a live ``~/.hermes/``
runtime. Run it once on the host where Friday actually lives:

    python3 scripts/setup_identity_loop.py

What it does (each step is skip-if-already-present):
  1. Copy ``identity_ledger.py`` + ``improvement_queue.py`` and the digest
     wrapper -> ``~/.hermes/scripts/``.
  2. Copy ``friday-identity-reflection.md`` -> ``~/.hermes/cron/prompts/``.
  3. Seed ``~/.hermes/PREFERENCES.md`` and ``~/.hermes/identity/LEDGER.md`` from
     the committed seed templates (only if they don't exist yet).
  4. Stage the first SOUL.md self-description clause through the queue.
  5. Register three cron jobs (by name, skipped if they already exist):
       - daily   : a no_agent job running ``identity_ledger.py`` (rollup).
       - monthly : an LLM job running the identity-reflection prompt.
       - daily   : a no_agent digest for pending identity proposals.

The PREFERENCES.md injection hook itself is core source (already in the agent);
this script only deploys the runtime half.
"""

from __future__ import annotations

import shutil
import sys
import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from hermes_constants import get_hermes_home  # noqa: E402

_ASSETS = _PROJECT_ROOT / "assets" / "identity"
_SCRIPTS = _PROJECT_ROOT / "scripts"

# Cron job names — used as the idempotency key (skip if a job with this name
# already exists).
_ROLLUP_JOB_NAME = "friday-identity-ledger-rollup"
_REFLECTION_JOB_NAME = "friday-identity-reflection"
_DIGEST_JOB_NAME = "friday-identity-review-digest"
_REVIEW_QUEUE_ORIGIN = {
    "platform": "discord",
    "chat_id": "1512110425389400136",
    "chat_name": "Review Queue",
    "chat_topic": "Review Queue",
    "thread_id": "1512260751514009742",
}

_SELF_DESCRIPTION_MARKER = "Do not recite SOUL.md as a self-description"
_SELF_DESCRIPTION_CLAUSE = f"""
## Self-description discipline

{_SELF_DESCRIPTION_MARKER}. Treat SOUL.md as operating guidance, not
autobiographical evidence. Do not make unbacked identity claims unless the
claim is grounded in ledger receipts, approved PREFERENCES.md entries, or the
current task context. Prefer describing observed behavior and evidence over
identity labels.
""".strip()


def _copy(src: Path, dst: Path, *, executable: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    if executable:
        dst.chmod(0o755)
    print(f"  copied {src.name} -> {dst}")


def _seed(src: Path, dst: Path) -> None:
    if dst.exists():
        print(f"  seed exists, leaving as-is: {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"  seeded {dst}")


def _pending_self_description_proposal(home: Path, soul_path: Path) -> bool:
    qdir = home / "identity" / "queue"
    if not qdir.exists():
        return False
    for meta_path in qdir.glob("*/meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (
            meta.get("status") == "pending_review"
            and meta.get("target") == str(soul_path)
            and "self-description clause" in (meta.get("summary") or "")
        ):
            return True
    return False


def _stage_self_description_clause(home: Path) -> bool:
    soul_path = home / "SOUL.md"
    if not soul_path.exists():
        print(f"  SOUL.md not found, skipping first identity proposal: {soul_path}")
        return False

    current = soul_path.read_text(encoding="utf-8")
    if _SELF_DESCRIPTION_MARKER in current:
        print("  SOUL.md self-description clause already present")
        return False
    if _pending_self_description_proposal(home, soul_path):
        print("  SOUL.md self-description clause proposal already pending")
        return False

    drafts = home / "identity" / "drafts"
    drafts.mkdir(parents=True, exist_ok=True)
    draft = drafts / "SOUL.self-description-clause.md"
    proposed = current.rstrip() + "\n\n" + _SELF_DESCRIPTION_CLAUSE + "\n"
    draft.write_text(proposed, encoding="utf-8")

    from scripts import improvement_queue

    rc = improvement_queue.main([
        "create",
        "--target", str(soul_path),
        "--proposed-file", str(draft),
        "--source", "setup-identity-loop",
        "--risk", "low",
        "--summary", "self-description clause: no SOUL.md recitation or unbacked identity claims",
        "--body",
        "First dogfood proposal for the identity gate: the agent must not "
        "recite SOUL.md as self-description or make unbacked identity claims.",
    ])
    if rc != 0:
        raise RuntimeError(f"failed to stage SOUL.md self-description clause (rc={rc})")
    print("  staged SOUL.md self-description clause for review")
    return True


def _register_cron_jobs(home: Path, *, create_job, existing_names: set[str]) -> None:
    if _ROLLUP_JOB_NAME in existing_names:
        print(f"  cron job already exists: {_ROLLUP_JOB_NAME}")
    else:
        create_job(
            prompt=None,
            schedule="0 3 * * *",  # daily at 03:00
            name=_ROLLUP_JOB_NAME,
            script="identity_ledger.py",
            no_agent=True,
            deliver="local",
        )
        print(f"  created daily cron job: {_ROLLUP_JOB_NAME}")

    if _REFLECTION_JOB_NAME in existing_names:
        print(f"  cron job already exists: {_REFLECTION_JOB_NAME}")
    else:
        prompt_path = home / "cron" / "prompts" / "friday-identity-reflection.md"
        create_job(
            prompt=prompt_path.read_text(encoding="utf-8"),
            schedule="0 4 1 * *",  # monthly, 1st at 04:00
            name=_REFLECTION_JOB_NAME,
            deliver="local",
        )
        print(f"  created monthly cron job: {_REFLECTION_JOB_NAME}")

    if _DIGEST_JOB_NAME in existing_names:
        print(f"  cron job already exists: {_DIGEST_JOB_NAME}")
    else:
        create_job(
            prompt=(
                "Script-only identity improvement queue digest. It prints only "
                "when pending identity proposals need human approval; empty "
                "stdout means silent."
            ),
            schedule="12 8 * * *",
            name=_DIGEST_JOB_NAME,
            script="identity_improvement_queue_digest.sh",
            no_agent=True,
            deliver="origin",
            origin=dict(_REVIEW_QUEUE_ORIGIN),
        )
        print(f"  created daily review digest cron job: {_DIGEST_JOB_NAME}")


def main() -> int:
    home = get_hermes_home()
    print(f"Deploying identity loop into {home}")

    # 1. scripts
    _copy(_SCRIPTS / "identity_ledger.py", home / "scripts" / "identity_ledger.py",
          executable=True)
    _copy(_SCRIPTS / "improvement_queue.py", home / "scripts" / "improvement_queue.py",
          executable=True)
    _copy(_SCRIPTS / "identity_improvement_queue_digest.sh",
          home / "scripts" / "identity_improvement_queue_digest.sh",
          executable=True)

    # 2. cron prompt
    _copy(_ASSETS / "friday-identity-reflection.md",
          home / "cron" / "prompts" / "friday-identity-reflection.md")

    # 3. seeds (only if absent)
    _seed(_ASSETS / "PREFERENCES.seed.md", home / "PREFERENCES.md")
    _seed(_ASSETS / "LEDGER.seed.md", home / "identity" / "LEDGER.md")

    # 4. first gated proposal
    _stage_self_description_clause(home)

    # 5. cron jobs
    try:
        from cron.jobs import create_job, list_jobs
    except Exception as e:
        print(f"  WARNING: could not import cron.jobs ({e}); skipping job "
              f"registration. Create the jobs manually with `hermes cron`.")
        return 0

    existing = {j.get("name") for j in list_jobs(include_disabled=True)}
    _register_cron_jobs(home, create_job=create_job, existing_names=existing)

    print(
        "\nDone. Next steps:\n"
        "  • Review the staged SOUL.md self-description clause:\n"
        "      python3 ~/.hermes/scripts/improvement_queue.py list --status pending_review\n"
        "      python3 ~/.hermes/scripts/improvement_queue.py approve <id>\n"
        "  • Confirm the rollup sees completed work:\n"
        "      python3 ~/.hermes/scripts/identity_ledger.py rollup\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

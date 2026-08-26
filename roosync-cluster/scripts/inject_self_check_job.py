"""inject_self_check_job.py
Ajoute le job hermes-self-check (couche 2 du PROPOSAL 26/08) dans
/opt/data/cron/jobs.json. Idempotent : si l'id existe déjà, ne fait rien.

Usage (depuis le host, dans le container) :
    docker cp inject_self_check_job.py hermes:/tmp/
    docker exec hermes python3 /tmp/inject_self_check_job.py
"""

import json
import sys
import shutil
from pathlib import Path

JOBS_PATH = "/opt/data/cron/jobs.json"
PROMPT_PATH = "/opt/data/scripts/hermes-self-check.prompt.md"

JOB_ID = "selfcheck-couche2-20260826"


def main():
    prompt = Path(PROMPT_PATH).read_text(encoding="utf-8")
    p = Path(JOBS_PATH)
    if not p.exists():
        print(f"ERROR: {JOBS_PATH} missing", file=sys.stderr)
        sys.exit(1)
    doc = json.loads(p.read_text(encoding="utf-8"))

    # Idempotence
    for j in doc.get("jobs", []):
        if j.get("id") == JOB_ID:
            print(f"Job {JOB_ID} already present — skip")
            return

    new_job = {
        "id": JOB_ID,
        "name": "hermes-self-check",
        "prompt": prompt,
        "skills": [],
        "skill": None,
        "model": None,
        "provider": None,
        "base_url": None,
        "script": None,
        "no_agent": False,
        "context_from": None,
        "schedule": {
            "kind": "cron",
            "expr": "7 11 */12 * *",  # à 11:07 et 23:07 — 30 min après les tours
            "display": "07 11 */12 * *",
        },
        "schedule_display": "07 11 */12 * *",
        "repeat": {
            "times": None,
            "completed": 0,
        },
        "enabled": True,
        "state": "scheduled",
        "paused_at": None,
        "paused_reason": None,
        "created_at": "2026-08-26T17:40:00+00:00",
        "next_run_at": None,
        "last_run_at": None,
        "last_status": None,
        "last_error": None,
        "last_delivery_error": None,
        "deliver": "telegram:-1003904676273",
        "origin": None,
        "workdir": None,
        "fire_claim": None,
        "failure_streak": 0,
    }

    doc.setdefault("jobs", []).append(new_job)
    # Valide la re-sérialisation
    j2 = json.dumps(doc, ensure_ascii=False, indent=2)
    json.loads(j2)

    # Backup + écriture atomique
    shutil.copyfile(JOBS_PATH, JOBS_PATH + ".bak-pre-selfcheck")
    tmp = JOBS_PATH + ".tmp"
    Path(tmp).write_text(j2, encoding="utf-8")
    Path(tmp).rename(JOBS_PATH)

    print(f"OK: {JOB_ID} injected ({len(prompt)} chars prompt)")


if __name__ == "__main__":
    main()

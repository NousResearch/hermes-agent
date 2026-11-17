"""Create or update the educator's cron jobs from cron-jobs.yaml.

Run by an ADMIN on the host, never by the agent:

    HERMES_HOME=~/.hermes/profiles/educator \\
        .venv/bin/python deploy/educator/apply_cron_jobs.py values.yaml [--dry-run]

- fills the {{…}} placeholders from values.yaml (the same file as the config);
- refuses unless HERMES_HOME is the educator profile and matches EDUCATOR_HOME;
- refuses unless both scripts are installed in <profile>/scripts/ and are
  byte-identical to this repo's copies, and the gateway's Python can import
  everything they import (the scheduler's own preflight);
- creates each job by name, or updates the existing one in place;
- prints the job ids (put release-watch's into cost_budget.essential_jobs)
  and the `hermes own claim` commands to run next.
"""
from __future__ import annotations


import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
JOBS_FILE = HERE / "cron-jobs.yaml"
PLACEHOLDER = re.compile(r"\{\{([A-Z0-9_]+)\}\}")
# Fields this script owns on each job (everything else is left alone).
FIELDS = ("schedule", "prompt", "script", "workdir", "deliver", "enabled_toolsets",
          "required_toolsets", "discretionary")


def fill(text: str, values: dict) -> str:
    missing = sorted(set(PLACEHOLDER.findall(text)) - set(values))
    if missing:
        raise ValueError(f"values.yaml is missing: {', '.join(missing)}")
    return PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), text)


def load_jobs(values: dict) -> list[dict]:
    data = yaml.safe_load(fill(JOBS_FILE.read_text(encoding="utf-8"), values)) or {}
    jobs = data.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("cron-jobs.yaml has no jobs")
    for job in jobs:
        unknown = set(job) - set(FIELDS) - {"name"}
        if unknown:
            raise ValueError(f"{job.get('name')}: unknown fields {sorted(unknown)}")
    return jobs


def preflight(home: Path, values: dict, jobs: list[dict]) -> list[str]:
    problems = []
    want = Path(str(values.get("EDUCATOR_HOME") or "")).expanduser()
    if home.resolve() != want.resolve():
        problems.append(f"HERMES_HOME ({home}) is not EDUCATOR_HOME ({want})")
    if home.name != "educator" or home.parent.name != "profiles":
        problems.append(f"HERMES_HOME ({home}) is not a profiles/educator directory")
    from cron.capability_preflight import script_import_problems
    for source in sorted((HERE / "scripts").glob("*.py")):  # the jobs' scripts + educator_wake.py
        installed = home / "scripts" / source.name
        if not installed.is_file():
            problems.append(f"{installed} is missing — copy deploy/educator/scripts/*.py there")
        elif installed.read_bytes() != source.read_bytes():
            problems.append(f"{installed} differs from {source} — re-copy it")
    for job in jobs:
        name = job["script"]
        issue = script_import_problems(name)
        if issue:
            problems.append(issue)
        workdir = Path(job["workdir"])
        if not workdir.is_dir():
            problems.append(f"workdir {workdir} does not exist")
    problems += capability_problems(home, jobs)
    return problems


def capability_problems(home: Path, jobs: list[dict]) -> list[str]:
    """What the scheduler's capability preflight would refuse: a toolset the
    job needs that cron strips or the profile's config does not offer."""
    cfg_path = home / "config.yaml"
    if not cfg_path.is_file():
        return [f"{cfg_path} is missing — write the profile config first (ROLLOUT.md step 4)"]
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    from cron.capability_preflight import evaluate_job_capabilities
    out = []
    for job in jobs:
        report = evaluate_job_capabilities(job, cfg, None, include_enabled=True,
                                           check_availability=False)
        out += [f"{job['name']}: {p}" for p in report.problems]
    return out


def apply(jobs: list[dict], dry_run: bool) -> dict[str, str]:
    from cron import jobs as cron_jobs
    existing = {j.get("name"): j for j in cron_jobs.list_jobs(include_disabled=True)}
    ids: dict[str, str] = {}
    for spec in jobs:
        name = spec["name"]
        current = existing.get(name)
        if dry_run:
            print(f"would {'update' if current else 'create'} {name}")
            ids[name] = current["id"] if current else "<new>"
            continue
        if current is None:
            created = cron_jobs.create_job(
                prompt=spec["prompt"], schedule=spec["schedule"], name=name,
                deliver=spec["deliver"], script=spec["script"],
                enabled_toolsets=spec["enabled_toolsets"], workdir=spec["workdir"])
            job_id = created["id"]
            extra = {k: spec[k] for k in ("required_toolsets", "discretionary")}
            cron_jobs.update_job(job_id, extra)
            print(f"created {name}: {job_id}")
        else:
            job_id = current["id"]
            cron_jobs.update_job(job_id, {k: spec[k] for k in FIELDS if k in spec})
            print(f"updated {name}: {job_id}")
        ids[name] = job_id
    return ids


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if a != "--dry-run"]
    if len(args) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    values = yaml.safe_load(Path(args[0]).read_text(encoding="utf-8")) or {}
    from hermes_constants import get_hermes_home
    home = Path(get_hermes_home())
    try:
        jobs = load_jobs(values)
    except ValueError as e:
        print(f"apply_cron_jobs: {e}", file=sys.stderr)
        return 1
    problems = preflight(home, values, jobs)
    if problems:
        print("apply_cron_jobs: refusing:\n  - " + "\n  - ".join(problems), file=sys.stderr)
        return 1
    ids = apply(jobs, dry_run="--dry-run" in argv)
    admin = values.get("ADMIN_SLACK_USER_ID")
    print("\nNext:")
    print(f"  1. set cost_budget.essential_jobs to [{ids.get('release-watch')}] (RELEASE_WATCH_JOB_ID)")
    for name, job_id in ids.items():
        print(f"  2. hermes -p educator own claim cron:{job_id} --user {admin} "
              f"--name \"{values.get('ADMIN_NAME')}\"   # {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

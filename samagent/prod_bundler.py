"""Pre-Production -> Production Release Gate & Bundler (`samagent/prod_bundler.py`).

Ensures that a user testing locally in VS Code and the SamAgent Local Platform can only promote
their application to a production release bundle when ALL 7 Pre-Production Deployment Gate checks
(L0–L4 verification, OWASP Top-10 security/IDOR/RBAC probes, zero secret leaks, and valid local dev
artifacts) are green.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from samagent.conductor.verify import VerificationRunner
from samagent.ide_bridge import (
    evaluate_pre_production_gate,
    get_workspace_git_status_and_diff,
    list_workspace_files,
)
from samagent.spec.models import SpecDocument


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def list_production_releases(project_dir: Path) -> List[Dict[str, Any]]:
    rel_root = Path(project_dir).resolve() / ".samagent" / "releases"
    if not rel_root.exists():
        return []
    out: List[Dict[str, Any]] = []
    for d in sorted([p for p in rel_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
        mf = d / "RELEASE_MANIFEST.json"
        if mf.exists():
            try:
                out.append(json.loads(mf.read_text(encoding="utf-8")))
            except Exception:
                pass
    return out


def promote_to_production(
    project_dir: Path,
    *,
    release_tag: Optional[str] = None,
) -> Dict[str, Any]:
    """Run fresh L0–L4 + OWASP verification on *project_dir* and, if all 7 Pre-Prod checks pass,
    generate production deployment artifacts (Dockerfile, docker-compose.prod.yml, .env.example,
    and .samagent/releases/<release_id>/RELEASE_MANIFEST.json).
    """
    root = Path(project_dir).resolve()
    spec = SpecDocument.load(root)
    verifier = VerificationRunner(root, spec)
    report = verifier.run_all()
    deliverable_snapshot = {
        "goal": spec.goal,
        "stack": spec.stack,
        "verification": report.to_dict(),
    }
    gate = evaluate_pre_production_gate(root, deliverable_snapshot)
    if not gate["ready_for_production"]:
        return {
            "promoted": False,
            "reason": "Pre-Production Deployment Gate blocked promotion",
            "blockers": gate["blockers"],
            "pre_prod_gate": gate,
            "verification": report.to_dict(),
        }

    ts = int(time.time())
    rel_id = release_tag or f"rel_{ts}"
    rel_dir = root / ".samagent" / "releases" / rel_id
    rel_dir.mkdir(parents=True, exist_ok=True)

    # 1. Production Dockerfile (non-root user, healthcheck)
    dockerfile = root / "Dockerfile"
    dockerfile.write_text(
        """FROM python:3.11-slim
WORKDIR /srv/app
RUN useradd --create-home --shell /bin/bash appuser
COPY app/ /srv/app/app/
COPY .samagent/contract/ /srv/app/.samagent/contract/
RUN chown -R appuser:appuser /srv/app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --retries=3 CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["python3", "app/main.py", "--serve", "--host", "0.0.0.0", "--port", "8000"]
""",
        encoding="utf-8",
    )

    # 2. docker-compose.prod.yml
    compose_file = root / "docker-compose.prod.yml"
    compose_file.write_text(
        """version: "3.9"
services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - APP_ENV=production
    restart: unless-stopped
""",
        encoding="utf-8",
    )

    # 3. .env.example
    env_example = root / ".env.example"
    env_example.write_text(
        """# SamAgent Verified Production Environment Template
APP_ENV=production
PORT=8000
SQLITE_DB_PATH=app/db/app.sqlite3
""",
        encoding="utf-8",
    )

    # 4. Signed Release Manifest with SHA-256 file hashes & git state
    git_info = get_workspace_git_status_and_diff(root)
    files_meta = list_workspace_files(root)
    checksums: Dict[str, str] = {}
    for f in files_meta:
        p = root / f["rel_path"]
        if p.exists() and p.is_file():
            checksums[f["rel_path"]] = _sha256_file(p)

    manifest = {
        "release_id": rel_id,
        "created_at": ts,
        "goal": spec.goal,
        "stack": spec.stack,
        "workspace": str(root),
        "git_branch": git_info.get("branch", "main"),
        "git_commit": git_info.get("head_commit", "HEAD"),
        "uncommitted_files_at_release": len(git_info.get("changed_files") or []),
        "pre_prod_gate": gate,
        "verification_summary": report.to_dict(),
        "artifacts": [
            "Dockerfile",
            "docker-compose.prod.yml",
            ".env.example",
        ],
        "sha256_checksums": checksums,
    }
    manifest_path = rel_dir / "RELEASE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "promoted": True,
        "release_id": rel_id,
        "manifest_path": str(manifest_path),
        "manifest": manifest,
        "pre_prod_gate": gate,
    }

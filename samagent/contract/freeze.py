"""Contract freeze, two-layer ownership guard, and Contract Change Requests (CCR) (05-final-plan.md §4–5)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import fnmatch
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple
import yaml

from samagent.spec.models import SpecDocument


@dataclass
class ContractVersion:
    version: int
    sha256: str
    frozen_at: float
    ccr_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContractChangeRequest:
    id: str
    from_module: str
    summary: str
    target_file: str  # e.g. "openapi.yaml" | "db/schema.sql" | "types.ts"
    new_content: str
    impacted_modules: List[str] = field(default_factory=list)


@dataclass
class OwnershipVerdict:
    allowed: bool
    module: str
    violating_paths: List[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _compute_contract_sha256(contract_dir: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(contract_dir.rglob("*")):
        if p.is_file() and p.name != "version.json":
            rel = p.relative_to(contract_dir).as_posix()
            h.update(rel.encode("utf-8") + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def freeze_contract(project_dir: Path, spec: SpecDocument) -> ContractVersion:
    """Freeze .samagent/contract/ (openapi.yaml, db/schema.sql, types.ts, design-tokens.json, ownership.yaml)."""
    contract_dir = Path(project_dir) / ".samagent" / "contract"
    db_dir = contract_dir / "db"
    db_dir.mkdir(parents=True, exist_ok=True)

    # 1. OpenAPI contract from stories
    paths_obj: Dict[str, Any] = {}
    for s in spec.stories:
        route_methods = paths_obj.setdefault(s.route, {})
        route_methods[s.method.lower()] = {
            "operationId": f"{s.id.lower()}_{s.as_role.lower()}",
            "summary": f"[{s.id}] As {s.as_role}: {s.can}",
            "x-accept": s.accept,
            "x-role": s.as_role,
            "security": [{"sessionAuth": []}] if s.auth_required else [],
            "responses": {
                "200": {"description": s.accept},
                "201": {"description": "Created"},
                "401": {"description": "Unauthenticated"},
                "403": {"description": "Forbidden (RBAC / IDOR violation)"},
            },
        }
    openapi_doc = {
        "openapi": "3.1.0",
        "info": {"title": spec.goal, "version": "1.0.0"},
        "paths": paths_obj,
    }
    (contract_dir / "openapi.yaml").write_text(
        yaml.safe_dump(openapi_doc, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    # 2. Database schema
    schema_sql = """-- Frozen DB Schema (.samagent/contract/db/schema.sql)
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('visitor', 'member', 'admin')),
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL CHECK (length(trim(title)) > 0),
    description TEXT NOT NULL DEFAULT '',
    capacity INTEGER NOT NULL DEFAULT 20,
    created_by TEXT NOT NULL,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS bookings (
    id TEXT PRIMARY KEY,
    item_id TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    owner_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'confirmed',
    created_at INTEGER NOT NULL,
    UNIQUE(item_id, owner_id)
);
"""
    (db_dir / "schema.sql").write_text(schema_sql, encoding="utf-8")

    # 3. Shared TypeScript types
    types_ts = """// Frozen Shared Types (.samagent/contract/types.ts)
export type Role = 'visitor' | 'member' | 'admin';

export interface Item {
  id: string;
  title: string;
  description: string;
  capacity: number;
  created_by: string;
}

export interface Booking {
  id: string;
  item_id: string;
  owner_id: string;
  status: 'confirmed' | 'cancelled';
}
"""
    (contract_dir / "types.ts").write_text(types_ts, encoding="utf-8")

    # 4. Design tokens
    design_tokens = {
        "colors": {
            "bg": "#0b0f17",
            "surface": "#111827",
            "text": "#f9fafb",
            "accent": "#3b82f6",
            "success": "#10b981",
            "danger": "#ef4444",
        },
        "radius": "8px",
        "font": "Inter, system-ui, sans-serif",
    }
    (contract_dir / "design-tokens.json").write_text(
        json.dumps(design_tokens, indent=2) + "\n", encoding="utf-8"
    )

    # 5. Ownership map (module -> owned globs; contract & acceptance reserved for conductor)
    ownership_map: Dict[str, List[str]] = {
        "conductor": [".samagent/contract/**", ".samagent/acceptance/**", ".samagent/spec.yaml", ".samagent/brief.md"],
    }
    for m in spec.modules:
        ownership_map[m.name] = list(m.owned_globs)
    (contract_dir / "ownership.yaml").write_text(
        yaml.safe_dump({"modules": ownership_map}, sort_keys=False),
        encoding="utf-8",
    )

    digest = _compute_contract_sha256(contract_dir)
    ver = ContractVersion(version=1, sha256=digest, frozen_at=time.time(), ccr_history=[])
    (contract_dir / "version.json").write_text(
        json.dumps(ver.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return ver


def load_ownership_map(project_dir: Path) -> Dict[str, List[str]]:
    own_path = Path(project_dir) / ".samagent" / "contract" / "ownership.yaml"
    if not own_path.exists():
        return {}
    data = yaml.safe_load(own_path.read_text(encoding="utf-8")) or {}
    return dict(data.get("modules") or {})


def _normalize_rel_path(path_str: str, project_dir: Optional[Path] = None) -> str:
    p = Path(path_str)
    if p.is_absolute() and project_dir is not None:
        try:
            p = p.resolve().relative_to(Path(project_dir).resolve())
        except Exception:
            pass
    rel = PurePosixPath(p.as_posix()).as_posix().lstrip("./")
    return rel


def _matches_any_glob(rel_path: str, globs: List[str]) -> bool:
    for g in globs:
        g_norm = g.strip().lstrip("./")
        if fnmatch.fnmatch(rel_path, g_norm):
            return True
        if g_norm.endswith("/**") and (
            rel_path == g_norm[:-3] or rel_path.startswith(g_norm[:-2])
        ):
            return True
    return False


def check_path_ownership(
    target_path: str,
    *,
    module_name: str,
    ownership_map: Dict[str, List[str]],
    project_dir: Optional[Path] = None,
) -> OwnershipVerdict:
    """Layer 1 (fast pre_tool_call guard): check if *module_name* may write *target_path*."""
    rel = _normalize_rel_path(target_path, project_dir=project_dir)

    # Workers are strictly forbidden from modifying frozen contract or acceptance tests
    if module_name != "conductor" and (
        rel.startswith(".samagent/contract/")
        or rel.startswith(".samagent/acceptance/")
        or rel in (".samagent/spec.yaml", ".samagent/brief.md")
    ):
        return OwnershipVerdict(
            allowed=False,
            module=module_name,
            violating_paths=[rel],
            reason=(
                f"Worker '{module_name}' cannot modify frozen contract/acceptance file '{rel}'. "
                "Submit a Contract Change Request (CCR) to the conductor instead."
            ),
        )

    if module_name == "conductor":
        return OwnershipVerdict(allowed=True, module=module_name)

    allowed_globs = ownership_map.get(module_name) or []
    if not allowed_globs:
        return OwnershipVerdict(
            allowed=False,
            module=module_name,
            violating_paths=[rel],
            reason=f"Module '{module_name}' has no owned globs in contract/ownership.yaml.",
        )

    if _matches_any_glob(rel, allowed_globs):
        return OwnershipVerdict(allowed=True, module=module_name)

    return OwnershipVerdict(
        allowed=False,
        module=module_name,
        violating_paths=[rel],
        reason=(
            f"Write to '{rel}' denied for module '{module_name}'. "
            f"Owned globs: {allowed_globs}"
        ),
    )


def check_git_diff_ownership(
    worktree_dir: Path,
    *,
    module_name: str,
    ownership_map: Dict[str, List[str]],
    base_ref: str = "HEAD",
) -> OwnershipVerdict:
    """Layer 2 (authoritative post-hoc git diff check before merge)."""
    proc = subprocess.run(
        ["git", "diff", "--name-only", base_ref],
        cwd=str(worktree_dir),
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=15,
    )
    changed = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
    # Also include untracked files outside .samagent/runs or .samagent/ledger.db
    untracked_proc = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=str(worktree_dir),
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=15,
    )
    for line in (untracked_proc.stdout or "").splitlines():
        rel = line.strip()
        if rel and rel not in changed:
            changed.append(rel)

    violations: List[str] = []
    for rel in changed:
        v = check_path_ownership(rel, module_name=module_name, ownership_map=ownership_map)
        if not v.allowed:
            violations.append(rel)

    if violations:
        return OwnershipVerdict(
            allowed=False,
            module=module_name,
            violating_paths=violations,
            reason=f"Post-hoc git diff check rejected worktree '{module_name}': modified unowned paths {violations}",
        )
    return OwnershipVerdict(allowed=True, module=module_name)


def apply_ccr(project_dir: Path, ccr: ContractChangeRequest) -> Tuple[ContractVersion, List[str]]:
    """Conductor-only: apply a ContractChangeRequest, bump contract@vN, and return modules to steer."""
    contract_dir = Path(project_dir) / ".samagent" / "contract"
    ver_path = contract_dir / "version.json"
    if not ver_path.exists():
        raise FileNotFoundError("Cannot apply CCR before initial contract freeze")
    raw_ver = json.loads(ver_path.read_text(encoding="utf-8"))

    target = (contract_dir / ccr.target_file).resolve()
    target.relative_to(contract_dir.resolve())
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(ccr.new_content, encoding="utf-8")

    new_ver_num = int(raw_ver.get("version", 1)) + 1
    new_sha = _compute_contract_sha256(contract_dir)
    history = list(raw_ver.get("ccr_history") or [])
    history.append(
        {
            "ccr_id": ccr.id,
            "version": f"contract@v{new_ver_num}",
            "from_module": ccr.from_module,
            "target_file": ccr.target_file,
            "summary": ccr.summary,
            "impacted_modules": list(ccr.impacted_modules),
            "applied_at": time.time(),
        }
    )
    ver = ContractVersion(
        version=new_ver_num,
        sha256=new_sha,
        frozen_at=time.time(),
        ccr_history=history,
    )
    ver_path.write_text(json.dumps(ver.to_dict(), indent=2) + "\n", encoding="utf-8")
    return ver, list(ccr.impacted_modules)

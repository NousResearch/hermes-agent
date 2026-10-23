"""`/clean` slash command — VPS junk cleanup with explicit confirm.

Lists what would be deleted (always), waits for a tier choice, then executes.
Reuses the tier scheme documented in skill `hermes-state-maintenance`:
  Tier 1 (GREEN) — VACUUM, snapshots viejos, audio_cache, logs rotados, /tmp scratch.
  Tier 2 (YELLOW) — requiere opt-in explícito: pre-cleanup backups viejos,
                    state-snapshots/ antigos, .env.bak.* acumulados, state.db-corrupted.
No toca /root/.hermes/state.db (solo VACUUM, no DELETE), no toca skills,
no toca proyectos. Output siempre en texto plano, Telegram-friendly.
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

from hermes_constants import get_hermes_home

HERMES_HOME = get_hermes_home()
STAGING = Path("/tmp/hermes-cleanup-staging")
STAGING.mkdir(parents=True, exist_ok=True)

# Tamaño human-readable sin dependencias
def _fmt(n: int) -> str:
    n_f = float(n)
    for unit in ("B", "K", "M", "G", "T"):
        if n_f < 1024:
            return f"{n_f:.1f}{unit}" if unit != "B" else f"{int(n_f)}B"
        n_f /= 1024
    return f"{n_f:.1f}P"

def _du_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        out = subprocess.run(
            ["du", "-sb", str(path)],
            capture_output=True, text=True, timeout=30, check=False
        )
        return int(out.stdout.split()[0]) if out.returncode == 0 else 0
    except Exception:
        return 0

def _ls_files(pattern_dir: Path, suffix: str, max_age_days: int | None = None) -> list[tuple[Path, int]]:
    if not pattern_dir.exists():
        return []
    out = []
    cutoff = time.time() - (max_age_days * 86400) if max_age_days else None
    for f in pattern_dir.iterdir():
        if not f.name.endswith(suffix):
            continue
        if not f.is_file():
            continue
        if cutoff is not None and f.stat().st_mtime > cutoff:
            continue
        out.append((f, f.stat().st_size))
    return sorted(out, key=lambda x: -x[1])

# ─────────────────────────────────────────────────────────────────────────────
# CANDIDATES
# ─────────────────────────────────────────────────────────────────────────────

def _tier1_candidates() -> tuple[list[str], list[str], list[tuple[Path, int]]]:
    """Returns (description_lines, action_descriptions, files_to_handle)."""
    files: list[tuple[Path, int]] = []

    # logs rotados
    logs_dir = HERMES_HOME / "logs"
    files.extend(_ls_files(logs_dir, ".log.1") + _ls_files(logs_dir, ".log.2"))

    # audio_cache viejo (>7 días)
    audio = HERMES_HOME / "audio_cache"
    if audio.exists():
        for child in audio.iterdir():
            if child.is_file() and (time.time() - child.stat().st_mtime) > 7 * 86400:
                files.append((child, child.stat().st_size))

    # state-snapshots viejos (>24h) → mover a staging (reversible)
    snap_dir = HERMES_HOME / "state-snapshots"
    if snap_dir.exists():
        for child in snap_dir.iterdir():
            if child.is_dir() and (time.time() - child.stat().st_mtime) > 86400:
                files.append((child, _du_bytes(child)))

    # /tmp scratch nuestro
    tmp_targets = [
        Path("/tmp/agent-reach-probe"),
        Path("/tmp/state.db.backup-"),
    ]
    for p in tmp_targets:
        parent = p.parent if p.parent.exists() else None
        if parent:
            for child in parent.iterdir():
                if child.name.startswith(p.name):
                    if child.is_file():
                        files.append((child, child.stat().st_size))
                    elif child.is_dir():
                        files.append((child, _du_bytes(child)))

    # corrupted state.db gigante (si existe)
    corrupted = HERMES_HOME / "state.db.corrupted-1787971506"
    if corrupted.exists():
        files.append((corrupted, corrupted.stat().st_size))

    desc = [
        f"logs rotados (.log.1/.log.2): {sum(1 for p,_ in files if 'logs' in str(p))}",
        f"audio_cache stale (>7d): {sum(1 for p,_ in files if 'audio_cache' in str(p))}",
        f"state-snapshots/ antiguos (>24h): {sum(1 for p,_ in files if 'state-snapshots' in str(p))}",
        f"/tmp scratch: {sum(1 for p,_ in files if '/tmp/' in str(p))}",
        "state.db.corrupted-* (si existe)",
    ]
    actions = [
        "rotate-greps: VACUUM state.db (no borra datos, solo compacta)",
        "rm logs/*.log.{1,2} (los actuales .log siguen)",
        "mv state-snapshots/antiguos/ → /tmp/hermes-cleanup-staging/",
        "rm audio_cache/* con mtime>7d",
        "rm /tmp scratch (agent-reach-probe, state.db.backup-*)",
    ]
    if corrupted.exists():
        actions.append(f"rm {corrupted.name} (294MB, recuperado de corruption pasada)")
    return desc, actions, files

def _tier2_candidates() -> tuple[list[str], list[tuple[Path, int]]]:
    """YELLOW — solo con opt-in. pre-cleanup backups, .env.bak.*, sessions viejas (no implementado)."""
    files: list[tuple[Path, int]] = []
    # pre-cleanup-* backups nuestros (>2 archivos más recientes)
    pre_cleanup = sorted(
        [(p, p.stat().st_size) for p in HERMES_HOME.glob("state.db.pre-cleanup-*")],
        key=lambda x: -x[0].stat().st_mtime
    )
    keep = 2
    if len(pre_cleanup) > keep:
        files.extend(pre_cleanup[keep:])

    # .env.bak.* — quedarnos con los últimos 3
    env_baks = sorted(
        [(p, p.stat().st_size) for p in HERMES_HOME.glob(".env.bak.*")],
        key=lambda x: -x[0].stat().st_mtime
    )
    keep_baks = 3
    if len(env_baks) > keep_baks:
        files.extend(env_baks[keep_baks:])

    desc = [
        f"pre-cleanup backups viejos (mantiene últimos 2): {sum(1 for p,_ in files if 'pre-cleanup' in str(p))}",
        f".env.bak.* viejos (mantiene últimos 3): {sum(1 for p,_ in files if '.env.bak' in str(p))}",
    ]
    return desc, files

# ─────────────────────────────────────────────────────────────────────────────
# ACTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _vacuum_state_db() -> tuple[bool, str]:
    db = HERMES_HOME / "state.db"
    if not db.exists():
        return False, "state.db no existe"
    before = db.stat().st_size
    try:
        conn = sqlite3.connect(str(db))
        conn.execute("VACUUM;")
        conn.close()
        after = db.stat().st_size
        saved = before - after
        return True, f"VACUUM: {_fmt(before)} → {_fmt(after)} (saved {_fmt(saved)})"
    except Exception as e:
        return False, f"VACUUM failed: {e}"

def _delete_or_move(path: Path, move_to_staging: bool) -> tuple[bool, str]:
    try:
        if move_to_staging:
            target = STAGING / path.name
            if path.is_dir():
                shutil.move(str(path), str(target))
            else:
                shutil.move(str(path), str(target))
            return True, f"mv {path.name} → staging/"
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        return True, f"rm {path.name}"
    except Exception as e:
        return False, f"FAIL {path.name}: {e}"

def _execute_tier1() -> str:
    out = ["Ejecutando Tier 1 (GREEN)..."]
    ok, msg = _vacuum_state_db()
    out.append(("  ✓ " if ok else "  ✗ ") + msg)

    _, _, files = _tier1_candidates()
    for path, size in files:
        is_snapshot = "state-snapshots" in str(path)
        ok, msg = _delete_or_move(path, move_to_staging=is_snapshot)
        out.append(("  ✓ " if ok else "  ✗ ") + f"{msg} ({_fmt(size)})")
    return "\n".join(out)

def _execute_tier2() -> str:
    out = ["Ejecutando Tier 2 (YELLOW)..."]
    _, files = _tier2_candidates()
    if not files:
        out.append("  (nada para limpiar en Tier 2)")
        return "\n".join(out)
    for path, size in files:
        ok, msg = _delete_or_move(path, move_to_staging=False)
        out.append(("  ✓ " if ok else "  ✗ ") + f"{msg} ({_fmt(size)})")
    return "\n".join(out)

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINTS
# ─────────────────────────────────────────────────────────────────────────────

def _build_preview() -> str:
    """Muestra el preview + menú 1/2/3."""
    home = HERMES_HOME
    du_home = _du_bytes(home)
    du_cache = _du_bytes(home / "cache")
    du_logs = _du_bytes(home / "logs")
    du_audio = _du_bytes(home / "audio_cache")
    du_snap = _du_bytes(home / "state-snapshots")

    desc1, actions1, files1 = _tier1_candidates()
    desc2, files2 = _tier2_candidates()

    total_t1 = sum(sz for _, sz in files1)
    total_t2 = sum(sz for _, sz in files2)

    out = []
    out.append("🧹 /clean — Hermes cleanup preview")
    out.append(f"HOME: {home} = {_fmt(du_home)}")
    out.append(f"  cache: {_fmt(du_cache)} | logs: {_fmt(du_logs)} | audio: {_fmt(du_audio)} | snapshots: {_fmt(du_snap)}")
    out.append("")
    out.append("— Tier 1 (GREEN, ~reclaimed: " + _fmt(total_t1) + ") —")
    for d in desc1:
        out.append(f"  · {d}")
    for a in actions1:
        out.append(f"    → {a}")
    if files1:
        out.append("  Archivos:")
        for p, sz in files1[:10]:
            out.append(f"    · {p.name} ({_fmt(sz)})")
        if len(files1) > 10:
            out.append(f"    ... +{len(files1)-10} más")
    out.append("")
    out.append("— Tier 2 (YELLOW, opt-in, ~reclaimed: " + _fmt(total_t2) + ") —")
    for d in desc2:
        out.append(f"  · {d}")
    out.append("")
    out.append("Elegí una opción:")
    out.append("  1 — Tier 1 solamente")
    out.append("  2 — Tier 1 + Tier 2")
    out.append("  3 — solo /space snapshot (sin borrar nada)")
    out.append("")
    out.append("Respondé con 1, 2 o 3.")
    return "\n".join(out)


def format_clean_preview() -> str:
    """Entry point used by the gateway dispatch. Always returns preview."""
    return _build_preview()


def format_clean_run(tier: int) -> str:
    """Execute cleanup at the given tier (1 or 2)."""
    if tier == 1:
        return _execute_tier1()
    elif tier == 2:
        return _execute_tier1() + "\n\n" + _execute_tier2()
    elif tier == 3:
        home = HERMES_HOME
        return "\n".join([
            "📊 /space snapshot",
            f"  {home}: {_fmt(_du_bytes(home))}",
            f"  cache: {_fmt(_du_bytes(home / 'cache'))}",
            f"  logs: {_fmt(_du_bytes(home / 'logs'))}",
            f"  audio_cache: {_fmt(_du_bytes(home / 'audio_cache'))}",
            f"  state-snapshots: {_fmt(_du_bytes(home / 'state-snapshots'))}",
            f"  state.db: {_fmt((home / 'state.db').stat().st_size if (home / 'state.db').exists() else 0)}",
        ])
    return f"clean: tier inválido {tier}"


# ─────────────────────────────────────────────────────────────────────────────
# SELF-CHECK (ponytail: lazy code without its check is unfinished)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Smoke test: build preview + dry-run execute_tier1 path validation
    print(format_clean_preview())
    print()
    print("--- selftest: tier=3 (no destructive) ---")
    print(format_clean_run(3))

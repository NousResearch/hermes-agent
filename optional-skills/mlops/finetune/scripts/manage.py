#!/usr/bin/env python3
"""
Adapter registry management for the finetune pipeline.

Handles status reporting, promotion, rollback, garbage collection,
and full pipeline orchestration.

Usage:
    python manage.py {status,promote,rollback,run,gc} [options]
"""

import argparse
import logging
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import common
from common import (
    ensure_dirs, load_config, load_json, save_json, read_jsonl, logger,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ============================================================================
# Auto-redeploy helpers (HF snapshot detection, GGUF conversion, llama-server
# lifecycle). Used by the redeploy() orchestrator below.
# ============================================================================

def find_base_snapshot(base_model_id: str) -> Optional[Path]:
    """
    Locate the local HuggingFace snapshot directory for a model.

    Given an HF repo ID like "kai-os/Carnice-9b", returns the most recently
    modified snapshot directory under ~/.cache/huggingface/hub/, or None if
    not found locally.
    """
    if not base_model_id or "/" not in base_model_id:
        return None

    org, name = base_model_id.split("/", 1)
    cache_dir = Path("~/.cache/huggingface/hub").expanduser()
    model_dir = cache_dir / f"models--{org}--{name}"
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    snapshots = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    if not snapshots:
        return None
    return max(snapshots, key=lambda p: p.stat().st_mtime)


def convert_adapter_to_gguf(
    adapter_dir: Path,
    base_snapshot: Path,
    converter: Path,
    force: bool = False,
) -> Path:
    """
    Convert a PEFT safetensors adapter to GGUF LoRA format using
    llama.cpp's convert_lora_to_gguf.py.

    Returns the path to the converted GGUF. If the GGUF already exists
    and force=False, returns the existing path without reconversion.

    Raises RuntimeError if the conversion fails.
    """
    output = adapter_dir / "adapter.gguf"
    if output.exists() and not force:
        logger.info("GGUF already exists at %s, skipping conversion", output)
        return output

    if not converter.exists():
        raise RuntimeError(f"Converter not found: {converter}")

    if not base_snapshot.exists():
        raise RuntimeError(f"Base snapshot not found: {base_snapshot}")

    adapter_model_dir = adapter_dir / "adapter_model"
    if not adapter_model_dir.exists():
        raise RuntimeError(f"Adapter model dir not found: {adapter_model_dir}")

    cmd = [
        sys.executable, str(converter),
        "--base", str(base_snapshot),
        "--outfile", str(output),
        str(adapter_model_dir),
    ]
    logger.info("Converting adapter to GGUF: %s", " ".join(cmd))

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"GGUF conversion failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout[-500:]}\n"
            f"stderr: {result.stderr[-500:]}"
        )
    if not output.exists():
        raise RuntimeError(f"Conversion succeeded but output missing: {output}")

    logger.info("GGUF written to %s", output)
    return output


def stop_llama_server(
    pid_file: Optional[Path] = None,
    expected_basename: str = "llama-server",
) -> bool:
    """
    Stop the llama-server we manage, identified strictly by its PID file.

    Servers started outside our control are deliberately left alone — a
    broad pkill-by-name would take down unrelated llama-server instances
    the user runs for other purposes. If the managed server is not running
    (no PID file, stale PID), that is already the desired end state.

    Before signalling, the PID's /proc cmdline must mention
    `expected_basename` (the server executable's basename). PIDs get
    reused; a PID file left over from a dead server must never kill an
    unrelated process. On mismatch the PID file is treated as stale and
    just removed.

    Returns True if a process was signalled, False otherwise.
    """
    if not pid_file or not pid_file.exists():
        return False

    stopped = False
    try:
        pid = int(pid_file.read_text().strip())

        # Identity check: does this PID still belong to our server?
        cmdline = ""
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
            cmdline = raw.replace(b"\0", b" ").decode("utf-8", errors="replace")
        except OSError:
            pass  # process gone (or no /proc) → treated as stale below
        if expected_basename not in cmdline:
            logger.warning(
                "PID %d cmdline %r does not mention %r — stale PID file, "
                "not signalling",
                pid, cmdline.strip(), expected_basename,
            )
            raise ProcessLookupError(f"stale pid file for {pid}")

        os.kill(pid, signal.SIGTERM)
        time.sleep(2)
        try:
            os.kill(pid, 0)  # is it still alive?
            logger.warning("PID %d did not exit on SIGTERM, sending SIGKILL", pid)
            os.kill(pid, signal.SIGKILL)
            time.sleep(1)
        except ProcessLookupError:
            pass  # already exited cleanly
        stopped = True
    except (ValueError, ProcessLookupError, PermissionError) as e:
        logger.debug("PID-file stop failed: %s", e)
    finally:
        try:
            pid_file.unlink()
        except FileNotFoundError:
            pass

    return stopped


def build_server_cmd(command_template: str, lora_path: Path) -> List[str]:
    """
    Expand a configured server command template into an argv list.

    `command_template` is a multi-line string with `%LORA%` as a placeholder
    for the LoRA path. If the template doesn't contain %LORA%, --lora is
    appended automatically. Each token is expanduser'd so documented
    `~/...` paths work (Popen does not expand `~`).
    """
    template = command_template.strip()
    if "%LORA%" in template:
        cmd_str = template.replace("%LORA%", str(lora_path))
    else:
        cmd_str = f"{template} --lora {shlex.quote(str(lora_path))}"

    # Collapse line continuations and split into argv
    cmd_str = " ".join(cmd_str.split())
    return [os.path.expanduser(tok) for tok in shlex.split(cmd_str)]


def start_llama_server(
    command_template: str,
    lora_path: Path,
    pid_file: Optional[Path] = None,
    log_path: Optional[Path] = None,
) -> int:
    """
    Start llama-server in the background with the LoRA loaded.

    See build_server_cmd() for template semantics.

    Returns the PID of the launched server. Writes the PID to pid_file
    if provided, and stdout/stderr to log_path (default
    <finetune-dir>/llama-server.log).
    """
    cmd = build_server_cmd(command_template, lora_path)

    log_path = log_path or (common.FINETUNE_DIR / "llama-server.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting llama-server: %s", " ".join(cmd))
    logger.info("Server log: %s", log_path)

    log_handle = open(log_path, "ab")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        # The child holds its own duplicate of the fd; keeping ours open
        # just leaks a handle in the parent.
        log_handle.close()

    if pid_file:
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(str(proc.pid))

    return proc.pid


def health_check_llama_server(url: str, timeout: int = 30) -> bool:
    """
    Poll the llama-server health endpoint until it responds or timeout.
    Returns True if the server is reachable, False on timeout.
    """
    import urllib.request
    import urllib.error

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if 200 <= resp.status < 300:
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, OSError):
            pass
        time.sleep(1)
    return False


def redeploy(adapter_dir: Optional[Path] = None, force_convert: bool = False) -> bool:
    """
    Convert the active (or specified) adapter to GGUF and restart
    llama-server with it loaded.

    force_convert=True reconverts even if adapter.gguf already exists —
    manual `/finetune redeploy` uses this so a hand-edited LoRA isn't
    silently shadowed by the cached GGUF. Pipeline auto-redeploys keep
    the cache.

    Configurable via the `finetune.serving` config section. Returns True
    if the new server is responsive at the configured health URL.

    On any failure, prints diagnostic info but does NOT fall back —
    this function is meant to be transparent. The caller (run_pipeline
    --with-bench) handles the rollback decision based on the bench
    result, not on the redeploy success alone.
    """
    config = load_config()
    serving_cfg = config.get("serving", {})

    # Resolve the adapter to deploy
    if adapter_dir is None:
        registry = AdapterRegistry()
        active = None
        for entry in registry.registry.get("adapters", []):
            if entry.get("status") == "active":
                active = entry
                break
        if active is None:
            print("redeploy: no active adapter to deploy")
            return False
        adapter_dir = common.ADAPTERS_DIR / active["cluster_id"] / active["version"]
        print(f"redeploy: deploying {active['cluster_id']} {active['version']}")
    else:
        adapter_dir = Path(adapter_dir).expanduser()

    if not (adapter_dir / "adapter_model").exists():
        print(f"redeploy: adapter_model dir missing under {adapter_dir}")
        return False

    # Step 1: Find HF snapshot for the base model
    snapshot_setting = serving_cfg.get("base_model_snapshot", "auto")
    if snapshot_setting == "auto":
        training_cfg = config.get("training", {})
        base_model_id = training_cfg.get("base_model", "")
        snapshot = find_base_snapshot(base_model_id)
        if snapshot is None:
            print(f"redeploy: could not auto-detect HF snapshot for '{base_model_id}'")
            print("         Set finetune.serving.base_model_snapshot explicitly,")
            print("         or run training first so axolotl downloads the model.")
            return False
        print(f"redeploy: using base snapshot {snapshot}")
    else:
        snapshot = Path(snapshot_setting).expanduser()
        if not snapshot.exists():
            print(f"redeploy: configured snapshot does not exist: {snapshot}")
            return False

    # Step 2: Convert the adapter to GGUF
    converter_setting = str(serving_cfg.get("converter", "") or "").strip()
    if not converter_setting:
        print("redeploy: finetune.serving.converter is not configured.")
        print("          Point it at llama.cpp's convert_lora_to_gguf.py, e.g.:")
        print("          finetune.serving.converter: ~/programs/llama.cpp/convert_lora_to_gguf.py")
        return False
    converter = Path(converter_setting).expanduser()

    try:
        gguf_path = convert_adapter_to_gguf(
            adapter_dir, snapshot, converter, force=force_convert,
        )
    except RuntimeError as e:
        print(f"redeploy: GGUF conversion failed: {e}")
        return False
    print(f"redeploy: GGUF ready at {gguf_path}")

    # Step 3: Restart llama-server (only if a server_command is configured)
    server_command = serving_cfg.get("server_command", "").strip()
    if not server_command:
        print("redeploy: no serving.server_command configured.")
        print(f"          Adapter is at {gguf_path} — start llama-server manually with:")
        print(f"          llama-server -m <base.gguf> --lora {gguf_path} ...")
        return True  # GGUF is ready, server start was opt-out

    # Predictable, per-user default paths under the finetune dir — world-
    # writable /tmp defaults are a multi-user security issue.
    pid_setting = str(serving_cfg.get("server_pid_file", "") or "").strip()
    pid_file = (Path(pid_setting).expanduser() if pid_setting
                else common.FINETUNE_DIR / "llama-server.pid")
    log_setting = str(serving_cfg.get("server_log_path", "") or "").strip()
    log_path = (Path(log_setting).expanduser() if log_setting
                else common.FINETUNE_DIR / "llama-server.log")

    # Validate the server executable BEFORE stopping the old server —
    # otherwise a bad command (typo, missing binary) leaves the user with
    # no server at all.
    server_cmd = build_server_cmd(server_command, gguf_path)
    server_exe = server_cmd[0]
    if not (Path(server_exe).exists() or shutil.which(server_exe)):
        print(f"redeploy: server executable not found: {server_exe}")
        print("          Fix finetune.serving.server_command — the running")
        print("          server (if any) was left untouched.")
        return False

    print("redeploy: stopping existing llama-server...")
    stop_llama_server(pid_file, expected_basename=os.path.basename(server_exe))

    print("redeploy: starting llama-server with new LoRA...")
    try:
        pid = start_llama_server(server_command, gguf_path, pid_file, log_path)
    except (FileNotFoundError, OSError) as e:
        print(f"redeploy: failed to start llama-server: {e}")
        return False
    print(f"redeploy: llama-server PID {pid}")

    # Step 4: Health check
    health_url = serving_cfg.get(
        "health_check_url", "http://localhost:8008/v1/models"
    )
    health_timeout = int(serving_cfg.get("health_check_timeout", 30))

    print(f"redeploy: waiting up to {health_timeout}s for {health_url}...")
    if health_check_llama_server(health_url, health_timeout):
        print(f"redeploy: ✓ server is responsive")
        return True

    print(f"redeploy: ✗ server did not respond within {health_timeout}s")
    print(f"          check {log_path} for the failure reason")
    return False


class AdapterRegistry:
    """Manage versioned adapters with promotion, rollback, and status."""

    def __init__(self):
        ensure_dirs()
        self.registry = load_json(common.REGISTRY_PATH, {"adapters": []})

    def _save(self):
        save_json(common.REGISTRY_PATH, self.registry)

    def _find_adapter(self, cluster_id: str, version: str = None) -> Optional[Dict]:
        """Find an adapter entry in the registry."""
        for entry in self.registry.get("adapters", []):
            if entry["cluster_id"] == cluster_id:
                if version is None or entry["version"] == version:
                    return entry
        return None

    def _find_active(self, cluster_id: str) -> Optional[Dict]:
        """Find the active adapter for a cluster."""
        for entry in self.registry.get("adapters", []):
            if entry["cluster_id"] == cluster_id and entry["status"] == "active":
                return entry
        return None

    def register_adapter(
        self,
        cluster_id: str,
        version: str,
        maturity: str,
        dataset_size: int,
        eval_results: Dict = None,
    ) -> Dict:
        """Register a new adapter version (initially as 'trained', not promoted)."""
        import hashlib

        adapter_dir = common.ADAPTERS_DIR / cluster_id / version
        config_path = adapter_dir / "config.yml"

        # Compute hashes for reproducibility
        base_model_hash = ""
        config_hash = ""
        if config_path.exists():
            config_hash = hashlib.sha256(
                config_path.read_bytes()
            ).hexdigest()[:16]

        entry = {
            "cluster_id": cluster_id,
            "cluster_label": "",
            "version": version,
            "status": "trained",
            "maturity": maturity,
            "base_model_hash": base_model_hash,
            "dataset_version": datetime.now().strftime("%Y-%m-%d"),
            "dataset_size": dataset_size,
            "training_config_hash": f"sha256:{config_hash}",
            "eval_results": eval_results or {},
            "trained_at": datetime.now().isoformat(),
            "promoted_at": None,
            "rollback_target": None,
        }

        # Update label from cluster state
        cluster_state = load_json(common.CLUSTER_STATE_PATH, {})
        cluster_info = cluster_state.get("clusters", {}).get(cluster_id, {})
        entry["cluster_label"] = cluster_info.get("label", f"auto:{cluster_id}")

        # Remove any existing entry for this version
        self.registry["adapters"] = [
            a for a in self.registry.get("adapters", [])
            if not (a["cluster_id"] == cluster_id and a["version"] == version)
        ]
        self.registry["adapters"].append(entry)
        self._save()

        logger.info("Registered adapter: %s %s", cluster_id, version)
        return entry

    def promote(self, cluster_id: str, version: str) -> bool:
        """Promote an adapter to active, demoting the previous one."""
        entry = self._find_adapter(cluster_id, version)
        if not entry:
            logger.error("Adapter not found: %s %s", cluster_id, version)
            return False

        # Demote current active
        current = self._find_active(cluster_id)
        if current:
            current["status"] = "previous"
            entry["rollback_target"] = current["version"]

        entry["status"] = "active"
        entry["promoted_at"] = datetime.now().isoformat()

        # Update symlink
        cluster_dir = common.ADAPTERS_DIR / cluster_id
        active_link = cluster_dir / "active"
        if active_link.is_symlink() or active_link.exists():
            active_link.unlink()
        active_link.symlink_to(version)

        self._save()
        logger.info("Promoted %s %s to active", cluster_id, version)
        return True

    def rollback(self, cluster_id: str) -> bool:
        """Roll back to the previous adapter version."""
        current = self._find_active(cluster_id)
        if not current:
            logger.error("No active adapter for cluster %s", cluster_id)
            return False

        target_version = current.get("rollback_target")
        if not target_version:
            logger.error("No rollback target for %s", cluster_id)
            return False

        target = self._find_adapter(cluster_id, target_version)
        if not target:
            logger.error("Rollback target %s not found", target_version)
            return False

        # Swap
        current["status"] = "rolled_back"
        target["status"] = "active"
        target["promoted_at"] = datetime.now().isoformat()

        # Update symlink
        cluster_dir = common.ADAPTERS_DIR / cluster_id
        active_link = cluster_dir / "active"
        if active_link.is_symlink() or active_link.exists():
            active_link.unlink()
        active_link.symlink_to(target_version)

        self._save()
        logger.info("Rolled back %s to %s", cluster_id, target_version)
        return True

    def gc(self, keep_versions: int = 2):
        """
        Garbage collect old adapter versions.

        Per cluster, the protected set is only the ACTIVE version and its
        rollback predecessor (protecting every registered version would
        make gc a no-op, since the pipeline registers everything). Beyond
        that, the `keep_versions` most recent versions are kept; the rest
        are deleted from disk (adapter dirs and all artifacts inside) and
        dropped from the registry.
        """
        removed: List[str] = []
        for cluster_dir in sorted(common.ADAPTERS_DIR.iterdir()):
            if not cluster_dir.is_dir() or cluster_dir.name.startswith("."):
                continue

            versions = sorted(
                (d for d in cluster_dir.iterdir()
                 if d.is_dir() and d.name.startswith("v")),
                key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else 0,
            )

            # Protect only the active version and its rollback predecessor.
            protected = set()
            for entry in self.registry.get("adapters", []):
                if (entry["cluster_id"] == cluster_dir.name
                        and entry.get("status") == "active"):
                    protected.add(entry["version"])
                    if entry.get("rollback_target"):
                        protected.add(entry["rollback_target"])

            # Keep the N most recent non-protected versions.
            candidates = [v for v in versions if v.name not in protected]
            to_keep = protected | {v.name for v in candidates[-keep_versions:]}

            for v_dir in versions:
                if v_dir.name in to_keep:
                    continue
                shutil.rmtree(v_dir)
                removed.append(f"{cluster_dir.name}/{v_dir.name}")
                print(f"GC: removed {cluster_dir.name}/{v_dir.name} "
                      f"(adapter dir + artifacts)")
                # Drop stale registry entries for deleted versions.
                self.registry["adapters"] = [
                    a for a in self.registry.get("adapters", [])
                    if not (a["cluster_id"] == cluster_dir.name
                            and a["version"] == v_dir.name)
                ]

        if removed:
            self._save()
            print(f"GC: removed {len(removed)} version(s): {', '.join(removed)}")
        else:
            print("GC: nothing to remove.")
        return removed

    def status(self) -> str:
        """Generate a status report."""
        lines = []
        lines.append("=" * 60)
        lines.append("  FINETUNE PIPELINE STATUS")
        lines.append("=" * 60)

        # Data stats
        def _total_lines(dir_path: Path, pattern: str) -> int:
            if not dir_path.exists():
                return 0
            total = 0
            for p in dir_path.glob(pattern):
                with open(p, encoding="utf-8") as f:
                    total += sum(1 for _ in f)
            return total

        extracted_count = _total_lines(common.EXTRACTED_DIR, "extract_*.jsonl")
        scored_count = _total_lines(common.SCORED_DIR, "scored_*.jsonl")

        lines.append(f"\n  Data:")
        lines.append(f"    Extracted sessions: {extracted_count}")
        lines.append(f"    Scored sessions:    {scored_count}")

        # Cluster state
        cluster_state = load_json(common.CLUSTER_STATE_PATH, {})
        if cluster_state:
            lines.append(f"\n  Clustering:")
            lines.append(f"    Algorithm:        {cluster_state.get('algorithm', 'n/a')}")
            lines.append(f"    Active clusters:  {cluster_state.get('clusters_active', 0)}")
            lines.append(f"    Noise sessions:   {cluster_state.get('noise_sessions', 0)}")
            lines.append(f"    Last run:         {cluster_state.get('last_run', 'never')}")

            for cid, info in cluster_state.get("clusters", {}).items():
                lines.append(f"    {cid}: {info.get('label', '?')} "
                           f"({info.get('session_count', 0)} sessions, "
                           f"maturity={info.get('maturity', '?')})")

        # Adapters
        adapters = self.registry.get("adapters", [])
        if adapters:
            lines.append(f"\n  Adapters:")
            for a in adapters:
                status_icon = {
                    "active": "[*]",
                    "trained": "[ ]",
                    "previous": "[-]",
                    "rolled_back": "[x]",
                }.get(a["status"], "[?]")
                lines.append(
                    f"    {status_icon} {a['cluster_id']} {a['version']} "
                    f"({a['status']}, {a.get('dataset_size', '?')} examples)"
                )
                if a.get("eval_results"):
                    er = a["eval_results"]
                    lines.append(
                        f"        eval: ppl={er.get('perplexity', '?')}, "
                        f"fmt={er.get('format_compliance', '?')}, "
                        f"task={er.get('task_completion', '?')}"
                    )
        else:
            lines.append("\n  Adapters: none")

        lines.append("=" * 60)
        return "\n".join(lines)


# Bench assets ship inside the skill bundle (bench/ next to scripts/), so
# they resolve identically from a repo checkout and from an installed skill
# (<hermes-home>/skills/mlops/finetune/).
BENCH_ASSETS_DIR = Path(__file__).resolve().parent.parent / "bench"
BENCH_ENV_SCRIPT = BENCH_ASSETS_DIR / "finetune_bench_env.py"
BENCH_DEFAULT_CONFIG = BENCH_ASSETS_DIR / "default.yaml"
def bench_results_dir() -> Path:
    """Resolve at call time so profile changes (HERMES_HOME) take effect."""
    return common.BENCH_DIR / "results"


def run_bench(prompt_bank: str = None) -> Optional[Path]:
    """
    Run the finetune benchmark via subprocess against the env script.

    Returns the path of the NEW result file written under
    <finetune-dir>/bench/results/, or None if the run failed. A failed or
    resultless run never falls back to a pre-existing bench_*.json — gating
    a promotion on stale data is worse than aborting the gate.
    """
    if not BENCH_ENV_SCRIPT.exists():
        logger.error("Bench env script not found: %s", BENCH_ENV_SCRIPT)
        return None
    if not BENCH_DEFAULT_CONFIG.exists():
        logger.error("Bench config not found: %s", BENCH_DEFAULT_CONFIG)
        return None

    bench_results_dir().mkdir(parents=True, exist_ok=True)
    # Snapshot existing results BEFORE launching so we can require a new one.
    pre_existing = set(bench_results_dir().glob("bench_*.json"))

    cmd = [
        sys.executable, str(BENCH_ENV_SCRIPT), "evaluate",
        "--config", str(BENCH_DEFAULT_CONFIG),
    ]
    if prompt_bank:
        cmd.extend(["--env.prompt_bank_path", prompt_bank])

    common.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    bench_log = common.LOGS_DIR / f"bench_{datetime.now():%Y%m%d_%H%M%S}.log"

    print(f"  Running: {' '.join(cmd)}")
    print(f"  Output: {bench_log}")
    try:
        with open(bench_log, "w", encoding="utf-8") as lf:
            result = subprocess.run(
                cmd, cwd=str(BENCH_ASSETS_DIR), check=False,
                stdout=lf, stderr=subprocess.STDOUT, timeout=3600 * 2,
            )
    except subprocess.TimeoutExpired:
        logger.error("Bench timed out after 2h — see %s", bench_log)
        return None
    except Exception as e:
        logger.error("Bench invocation failed: %s", e)
        return None

    after = set(bench_results_dir().glob("bench_*.json"))
    new = after - pre_existing

    if result.returncode != 0 or not new:
        tail = ""
        try:
            tail = bench_log.read_text(encoding="utf-8", errors="replace")[-2000:]
        except OSError:
            pass
        logger.error(
            "Bench run failed (exit %d, %d new result file(s)) — "
            "refusing to reuse a pre-existing result. Output tail:\n%s",
            result.returncode, len(new), tail,
        )
        return None

    return max(new, key=lambda p: p.stat().st_mtime)


def bench_passes(candidate_path: Path, baseline_path: Path = None) -> tuple:
    """
    Compare a candidate bench result against a baseline.

    Returns (passed: bool, summary: str).
    If baseline_path is None, picks the most recent prior result whose
    total_cases matches the candidate's. This avoids comparing a 243-case
    full bench against a 2-case smoke test, which produces nonsense
    "regressions" on metrics that the smoke test simply didn't measure.
    """
    from eval import compare_metrics, verdict, format_report

    candidate_data = load_json(candidate_path)
    candidate_metrics = candidate_data.get("metrics", {})
    candidate_cases = int(candidate_metrics.get("total_cases", 0))

    if baseline_path is None:
        # Find prior result files, sort by mtime newest-first.
        prior_files = sorted(
            (p for p in bench_results_dir().glob("bench_*.json") if p != candidate_path),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        # Filter to baselines whose total_cases matches the candidate's
        # (within a 10% tolerance to allow for prompt-bank growth between
        # runs). Smoke tests with 2 cases vs a 243-case full bench will
        # never match — they get skipped.
        def _comparable(p: Path) -> bool:
            try:
                m = load_json(p).get("metrics", {})
                cases = int(m.get("total_cases", 0))
            except Exception:
                return False
            if cases == 0 or candidate_cases == 0:
                return False
            ratio = cases / candidate_cases
            return 0.9 <= ratio <= 1.1

        comparable = [p for p in prior_files if _comparable(p)]
        if comparable:
            baseline_path = comparable[0]
        elif prior_files:
            # No comparable baseline exists — record this run as a new
            # baseline rather than comparing against an incompatible one.
            baseline_path = None

    if baseline_path is None:
        return True, (
            f"No comparable baseline found "
            f"(need a prior result with ~{candidate_cases} cases). "
            f"Recording this run as the new baseline."
        )

    baseline_data = load_json(baseline_path)
    baseline_metrics = baseline_data.get("metrics", {})

    comparison = compare_metrics(candidate_metrics, baseline_metrics)
    checks = verdict(comparison)
    passed = checks.get("overall", False)
    report = format_report(
        candidate_metrics, baseline_metrics, comparison, checks,
        cluster_id="(pipeline)", version="(latest)",
    )
    # Annotate which baseline file was used so debugging is easier.
    report = (
        f"Baseline file: {baseline_path.name} "
        f"(case count: {baseline_metrics.get('total_cases', '?')})\n"
        + report
    )
    return passed, report


def run_pipeline(dry_run: bool = False, with_bench: bool = False):
    """
    Run the full pipeline: extract → score → cluster → train → register → promote.

    When with_bench=True, the bench env runs after promotion. If it regresses
    against the most recent prior bench result, the new adapters are
    automatically rolled back.
    """
    from extract import SessionExtractor
    from score import QualityScorer
    from cluster import DomainClusterer
    from train import TrainingOrchestrator

    total_steps = 6 if with_bench else 5

    print(f"\n[1/{total_steps}] Extracting sessions...")
    extractor = SessionExtractor()
    sessions = extractor.extract()
    if not sessions:
        sessions = extractor.get_all_extracted()
    print(f"  → {len(sessions)} sessions")

    if not sessions:
        print("No sessions available. Use hermes to generate some conversations first.")
        return

    print(f"\n[2/{total_steps}] Scoring quality...")
    scorer = QualityScorer()
    scored = scorer.score_all(sessions)
    good = sum(1 for s in scored if s.get("scoring", {}).get("bucket") == "good")
    print(f"  → {good} good, {len(scored) - good} other")

    print(f"\n[3/{total_steps}] Discovering domains...")
    # Always cluster over the COMPLETE scored corpus (no sessions arg →
    # cluster() loads every scored_*.jsonl). Feeding only the incremental
    # extract batch here would rebuild cluster state — and rewrite every
    # cluster's train.jsonl — from just that batch, destroying the full
    # training set on the second run. The batch above only drives
    # scoring/logging.
    clusterer = DomainClusterer()
    cluster_state = clusterer.cluster()
    if cluster_state:
        print(f"  → {cluster_state.get('clusters_active', 0)} clusters")
    else:
        print("  → No clusters (data goes to _general)")

    print(f"\n[4/{total_steps}] Training adapters...")
    orchestrator = TrainingOrchestrator()
    trained = orchestrator.train_eligible(dry_run=dry_run)
    print(f"  → Trained {len(trained)} clusters: {trained}")

    if not (trained and not dry_run):
        print(f"\n[5/{total_steps}] Skipping registration (no training or dry run)")
        print("\nPipeline complete.")
        return

    print(f"\n[5/{total_steps}] Registering and promoting adapters...")
    registry = AdapterRegistry()
    promoted: List[Tuple[str, str]] = []
    for cid in trained:
        cluster_dir = common.ADAPTERS_DIR / cid
        versions = sorted(
            (d.name for d in cluster_dir.iterdir()
             if d.is_dir() and d.name.startswith("v")),
            key=lambda x: int(x[1:]) if x[1:].isdigit() else 0,
        )
        if not versions:
            continue
        version = versions[-1]
        cluster_info = (cluster_state or {}).get("clusters", {}).get(cid, {})
        train_path = common.CLUSTERS_DIR / cid / "train.jsonl"
        ds_size = 0
        if train_path.exists():
            with open(train_path, encoding="utf-8") as f:
                ds_size = sum(1 for _ in f)
        registry.register_adapter(
            cluster_id=cid,
            version=version,
            maturity=cluster_info.get("maturity", "established"),
            dataset_size=ds_size,
        )
        registry.promote(cid, version)
        promoted.append((cid, version))
        print(f"  → Promoted {cid} {version}")

    # Optional: auto-redeploy llama-server with the new adapter loaded.
    # When enabled, the bench step below will measure the adapter that's
    # actually being served, not the bare base model.
    serving_cfg = load_config().get("serving", {})
    served: Optional[Tuple[str, str]] = None
    if serving_cfg.get("auto_redeploy") and promoted:
        print(f"\n[5b/{total_steps}] Redeploying llama-server with new adapter...")
        if len(promoted) > 1:
            print(f"  ⚠ WARNING: {len(promoted)} clusters were promoted but only the")
            print("    last one can be served — the bench below measures ONLY the")
            print("    last-deployed adapter. The other promoted adapters are not")
            print("    covered by this gate; bench them individually if needed.")
        cid, version = promoted[-1]  # last promoted adapter wins
        deploy_dir = common.ADAPTERS_DIR / cid / version
        if redeploy(deploy_dir):
            served = (cid, version)
        else:
            print("  ⚠ Redeploy failed. Adapter is promoted but not yet served.")
            print("    The bench will measure the previously-served model.")

    if not with_bench:
        print("\nPipeline complete. Run '/finetune bench' to verify quality.")
        return

    print(f"\n[6/{total_steps}] Running benchmark gate...")
    candidate_path = run_bench()
    if candidate_path is None:
        print("  → Benchmark failed to run. Adapters remain promoted; verify manually.")
        return

    passed, report = bench_passes(candidate_path)
    print(report)

    if passed:
        print("\n  → BENCHMARK PASSED. Adapters remain active.")
        print("\nPipeline complete.")
    else:
        # The bench only measured the served adapter. When we know which
        # adapter that was (auto-redeploy succeeded) and several clusters
        # were promoted, roll back ONLY the served one — the regression
        # says nothing about the adapters that were never served.
        if served and len(promoted) > 1:
            to_rollback = [served]
            print("\n  → BENCHMARK REGRESSED. Rolling back only the served adapter"
                  f" ({served[0]} {served[1]});")
            print("    the other promoted adapters were not measured and remain"
                  " promoted.")
        else:
            to_rollback = promoted
            print("\n  → BENCHMARK REGRESSED. Rolling back promoted adapters...")
        for cid, version in to_rollback:
            if registry.rollback(cid):
                print(f"    Rolled back {cid} (was {version})")
            else:
                print(f"    Could not rollback {cid} — manual intervention required")
        # If we redeployed and now need to roll back, redeploy the previous
        # adapter so the served model matches the active registry entry.
        if serving_cfg.get("auto_redeploy"):
            print("    Redeploying previous adapter to match registry rollback...")
            redeploy()  # picks up whatever's now active after rollback
        print("\nPipeline complete with regression. Investigate the bench report above.")


CRON_SCHEDULE_MAP = {
    "daily": "0 3 * * *",
    "weekly": "0 3 * * 0",
    "biweekly": "0 3 1,15 * *",
    "monthly": "0 3 1 * *",
}


def setup_cron(schedule: str = "weekly"):
    """Set up scheduled retraining via the Hermes cron system."""
    cron_expr = CRON_SCHEDULE_MAP.get(schedule, schedule)

    prompt = (
        "Run the finetune pipeline: extract new sessions, score quality, "
        "update clusters, retrain eligible adapters, evaluate, and promote "
        "if evaluation passes. Report results including adapter versions, "
        "cluster changes, and any evaluation failures."
    )

    try:
        from tools.cronjob_tools import schedule_cronjob
        result = schedule_cronjob(
            prompt=prompt,
            schedule=cron_expr,
            name="finetune-retrain",
        )
        print(f"Cron job created: {result}")
        print(f"Schedule: {schedule} ({cron_expr})")
    except ImportError:
        # Fall back to shell-level cron
        print(f"Hermes cron not available. Add this to your crontab:")
        scripts_dir = Path(__file__).resolve().parent
        print(f"  {cron_expr}  cd {scripts_dir} && python manage.py run")


def main():
    parser = argparse.ArgumentParser(description="Adapter registry management")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Show pipeline status")

    p_promote = sub.add_parser("promote", help="Promote adapter to active")
    p_promote.add_argument("--cluster", required=True)
    p_promote.add_argument("--version", required=True)

    p_rollback = sub.add_parser("rollback", help="Roll back to previous version")
    p_rollback.add_argument("--cluster", required=True)

    p_run = sub.add_parser("run", help="Run full pipeline")
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument(
        "--with-bench", action="store_true",
        help="Run the benchmark gate after promotion. "
             "Auto-rollback adapters that regress vs. the most recent prior result.",
    )

    sub.add_parser("bench", help="Run the finetune benchmark against the active model")

    p_redeploy = sub.add_parser(
        "redeploy",
        help="Convert the active adapter to GGUF and restart llama-server with it loaded",
    )
    p_redeploy.add_argument(
        "--cluster", default=None,
        help="Cluster ID to deploy (default: the currently-active adapter from registry)",
    )
    p_redeploy.add_argument(
        "--version", default=None,
        help="Version to deploy (default: the active version for the cluster)",
    )

    p_gc = sub.add_parser("gc", help="Garbage collect old versions")
    p_gc.add_argument("--keep", type=int, default=2, help="Versions to keep")

    p_cron = sub.add_parser("cron", help="Set up scheduled retraining")
    p_cron.add_argument("schedule", nargs="?", default="weekly",
                        help="Schedule: daily, weekly, biweekly, monthly, or cron expression")

    args = parser.parse_args()

    if args.command == "status":
        registry = AdapterRegistry()
        print(registry.status())

    elif args.command == "promote":
        registry = AdapterRegistry()
        if registry.promote(args.cluster, args.version):
            print(f"Promoted {args.cluster} {args.version}")
        else:
            print("Promotion failed.")
            sys.exit(1)

    elif args.command == "rollback":
        registry = AdapterRegistry()
        if registry.rollback(args.cluster):
            print(f"Rolled back {args.cluster}")
        else:
            print("Rollback failed.")
            sys.exit(1)

    elif args.command == "run":
        run_pipeline(dry_run=args.dry_run, with_bench=args.with_bench)

    elif args.command == "bench":
        result_path = run_bench()
        if result_path:
            passed, report = bench_passes(result_path)
            print(report)
            print(f"\nResult saved to: {result_path}")
            sys.exit(0 if passed else 1)
        else:
            print("Benchmark failed to run.")
            sys.exit(1)

    elif args.command == "redeploy":
        adapter_dir = None
        if args.version and not args.cluster:
            print("redeploy: --version requires --cluster (versions are "
                  "per-cluster). Re-run with --cluster <id> --version "
                  f"{args.version}.")
            sys.exit(2)
        if args.cluster and args.version:
            adapter_dir = common.ADAPTERS_DIR / args.cluster / args.version
        elif args.cluster:
            # Use the active version of the requested cluster
            registry = AdapterRegistry()
            for entry in registry.registry.get("adapters", []):
                if entry.get("cluster_id") == args.cluster and entry.get("status") == "active":
                    adapter_dir = common.ADAPTERS_DIR / args.cluster / entry["version"]
                    break
            if adapter_dir is None:
                print(f"No active adapter for cluster {args.cluster}")
                sys.exit(1)
        # Manual redeploy always reconverts: SKILL.md recommends redeploying
        # after hand-editing the LoRA, and a cached adapter.gguf would
        # silently win otherwise.
        ok = redeploy(adapter_dir, force_convert=True)
        sys.exit(0 if ok else 1)

    elif args.command == "gc":
        registry = AdapterRegistry()
        registry.gc(keep_versions=args.keep)
        print("Garbage collection complete.")

    elif args.command == "cron":
        setup_cron(args.schedule)


if __name__ == "__main__":
    main()

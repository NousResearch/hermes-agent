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

# How long a CLI invocation waits for the coarse pipeline lock (see main())
# before exiting with "another finetune operation is running".
LOCK_TIMEOUT = 30.0


# ============================================================================
# Auto-redeploy helpers (HF snapshot detection, GGUF conversion, llama-server
# lifecycle). Used by the redeploy() orchestrator below.
# ============================================================================

def serving_manifest_path() -> Path:
    """Resolve at call time so profile changes (HERMES_HOME) take effect."""
    return common.FINETUNE_DIR / "serving.json"


def write_serving_manifest(pid: int, health_url: str, adapters: List[Dict]) -> None:
    """Record what the managed llama-server is actually serving.

    The routing plugin only activates on this manifest: llama.cpp can only
    apply per-request LoRA scales to adapters preloaded via ``--lora`` at
    startup, so the plugin must know exactly which adapters this server
    holds and at which positional index. ``adapters`` entries are
    ``{"id": <--lora index>, "cluster": ..., "version": ..., "gguf": ...}``;
    the list shape is the extension point for multi-adapter serving.
    """
    save_json(serving_manifest_path(), {
        "updated_at": datetime.now().isoformat(),
        "server": {"pid": pid, "health_url": health_url},
        "adapters": adapters,
    })


def clear_serving_manifest() -> None:
    """Remove the serving manifest — the managed server is (being) stopped
    or in an unknown state, so nothing must route against it."""
    try:
        serving_manifest_path().unlink()
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.warning("Could not remove serving manifest: %s", e)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    except OSError:
        return False
    return True


def _log_tail(log_path: Optional[Path], limit: int = 2000) -> str:
    if not log_path:
        return ""
    try:
        return log_path.read_text(encoding="utf-8", errors="replace")[-limit:]
    except OSError:
        return ""

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
    timeout: int = 600,
) -> Path:
    """
    Convert a PEFT safetensors adapter to GGUF LoRA format using
    llama.cpp's convert_lora_to_gguf.py.

    Returns the path to the converted GGUF. If the GGUF already exists
    and force=False, returns the existing path without reconversion.

    The converter writes to a temp path that is os.replace()d into place
    only on verified success — adapter.gguf either doesn't exist or is a
    complete artifact, never a truncated file from a killed/timed-out
    conversion (the exists() cache check above trusts it blindly).

    Raises RuntimeError if the conversion fails or times out.
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

    tmp_output = adapter_dir / "adapter.gguf.converting"
    cmd = [
        sys.executable, str(converter),
        "--base", str(base_snapshot),
        "--outfile", str(tmp_output),
        str(adapter_model_dir),
    ]
    logger.info("Converting adapter to GGUF: %s", " ".join(cmd))

    try:
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"GGUF conversion timed out after {timeout}s — no partial "
                f"adapter.gguf was left behind."
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"GGUF conversion failed (exit {result.returncode}):\n"
                f"stdout: {result.stdout[-500:]}\n"
                f"stderr: {result.stderr[-500:]}"
            )
        if not tmp_output.exists():
            raise RuntimeError(
                f"Conversion succeeded but output missing: {tmp_output}"
            )
    except Exception:
        # Any failure path: never leave a partial temp file to confuse a
        # later run (or a human).
        try:
            tmp_output.unlink()
        except FileNotFoundError:
            pass
        raise

    os.replace(tmp_output, output)
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

    Before signalling, some argv entry of the PID must have
    `expected_basename` as its EXACT basename (any position, to cover
    interpreter/env trampolines like ``/usr/bin/env python3
    .../llama-server``). PIDs get reused; a PID file left over from a
    dead server must never kill an unrelated process — and a substring
    test over the whole cmdline would match a recycled PID running e.g.
    ``tail -f .../llama-server.log`` (whose basename
    ``llama-server.log`` does NOT match exactly). On mismatch the PID
    file is treated as stale and just removed.

    Returns True if a process was signalled, False otherwise.
    """
    if not pid_file or not pid_file.exists():
        return False

    stopped = False
    try:
        pid = int(pid_file.read_text().strip())

        # Identity check: does this PID still belong to our server?
        argv: List[str] = []
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
            argv = [
                a.decode("utf-8", errors="replace")
                for a in raw.split(b"\0") if a
            ]
        except OSError:
            pass  # process gone (or no /proc) → treated as stale below
        exe_names = {os.path.basename(a) for a in argv}
        if expected_basename not in exe_names:
            logger.warning(
                "PID %d argv %r has no entry with basename %r — stale PID "
                "file, not signalling",
                pid, argv, expected_basename,
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


def build_server_cmd(command_template: str, lora_path: Optional[Path]) -> List[str]:
    """
    Expand a configured server command template into an argv list.

    `command_template` is a multi-line string with `%LORA%` as a placeholder
    for the LoRA path. If the template doesn't contain %LORA%, --lora is
    appended automatically. The substituted path is shlex-quoted so a path
    with spaces stays one argv token. Each token is expanduser'd so
    documented `~/...` paths work (Popen does not expand `~`).

    `lora_path=None` builds the BASE-MODEL command: the `%LORA%` token and
    an immediately preceding `--lora` flag are stripped (a template without
    %LORA% is used as-is). Used when the bench gate deactivates a regressed
    adapter that has nothing to roll back to.
    """
    template = command_template.strip()

    if lora_path is None:
        tokens = shlex.split(template)
        cleaned: List[str] = []
        for tok in tokens:
            if tok == "%LORA%":
                if cleaned and cleaned[-1] == "--lora":
                    cleaned.pop()
                continue
            cleaned.append(tok)
        return [os.path.expanduser(tok) for tok in cleaned]

    if "%LORA%" in template:
        cmd_str = template.replace("%LORA%", shlex.quote(str(lora_path)))
    else:
        cmd_str = f"{template} --lora {shlex.quote(str(lora_path))}"

    return [os.path.expanduser(tok) for tok in shlex.split(cmd_str)]


def start_llama_server(
    command_template: str,
    lora_path: Optional[Path],
    pid_file: Optional[Path] = None,
    log_path: Optional[Path] = None,
) -> int:
    """
    Start llama-server in the background with the LoRA loaded
    (or without any LoRA when lora_path is None).

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


def health_check_llama_server(url: str, timeout: int = 30,
                              pid: Optional[int] = None) -> bool:
    """
    Poll the llama-server health endpoint until it responds or timeout.

    When `pid` is given, the LAUNCHED process must also still be alive: a
    pre-existing server on the same port answers the health URL happily
    while our freshly started process died on "address already in use" —
    that must be a failure, not a success. Returns True only if the URL
    responds AND (when given) the pid is alive; False otherwise.
    """
    import urllib.request
    import urllib.error

    deadline = time.time() + timeout
    while time.time() < deadline:
        if pid is not None and not _pid_alive(pid):
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if 200 <= resp.status < 300:
                    return pid is None or _pid_alive(pid)
        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, OSError):
            pass
        time.sleep(1)
    return False


def _resolve_server_paths(serving_cfg: Dict) -> Tuple[Path, Path]:
    """PID file and log path for the managed server.

    Predictable, per-user default paths under the finetune dir — world-
    writable /tmp defaults are a multi-user security issue.
    """
    pid_setting = str(serving_cfg.get("server_pid_file", "") or "").strip()
    pid_file = (Path(pid_setting).expanduser() if pid_setting
                else common.FINETUNE_DIR / "llama-server.pid")
    log_setting = str(serving_cfg.get("server_log_path", "") or "").strip()
    log_path = (Path(log_setting).expanduser() if log_setting
                else common.FINETUNE_DIR / "llama-server.log")
    return pid_file, log_path


def _stop_start_and_verify(
    serving_cfg: Dict,
    server_command: str,
    lora_path: Optional[Path],
    adapters: List[Dict],
) -> bool:
    """Restart the managed llama-server and verify it actually came up.

    On success the serving manifest records the new server + `adapters`
    (empty list = base model only). The manifest is cleared the moment the
    old server is stopped, so a failed restart never leaves a manifest
    describing a server that isn't running.
    """
    pid_file, log_path = _resolve_server_paths(serving_cfg)

    # Validate the server executable BEFORE stopping the old server —
    # otherwise a bad command (typo, missing binary) leaves the user with
    # no server at all.
    server_cmd = build_server_cmd(server_command, lora_path)
    server_exe = server_cmd[0]
    if not (Path(server_exe).exists() or shutil.which(server_exe)):
        print(f"redeploy: server executable not found: {server_exe}")
        print("          Fix finetune.serving.server_command — the running")
        print("          server (if any) was left untouched.")
        return False

    print("redeploy: stopping existing llama-server...")
    stop_llama_server(pid_file, expected_basename=os.path.basename(server_exe))
    clear_serving_manifest()

    what = f"LoRA {lora_path}" if lora_path else "base model (no LoRA)"
    print(f"redeploy: starting llama-server with {what}...")
    try:
        pid = start_llama_server(server_command, lora_path, pid_file, log_path)
    except (FileNotFoundError, OSError) as e:
        print(f"redeploy: failed to start llama-server: {e}")
        return False
    print(f"redeploy: llama-server PID {pid}")

    health_url = serving_cfg.get(
        "health_check_url", "http://localhost:8008/v1/models"
    )
    health_timeout = int(serving_cfg.get("health_check_timeout", 30))

    print(f"redeploy: waiting up to {health_timeout}s for {health_url}...")
    if health_check_llama_server(health_url, health_timeout, pid=pid):
        write_serving_manifest(pid, health_url, adapters)
        print("redeploy: ✓ server is responsive")
        return True

    if not _pid_alive(pid):
        print(f"redeploy: ✗ launched server (PID {pid}) exited — likely a "
              "port conflict with a server we don't manage, or a bad flag.")
        tail = _log_tail(log_path)
        if tail:
            print(f"          log tail ({log_path}):")
            for line in tail.splitlines()[-15:]:
                print(f"          | {line}")
    else:
        print(f"redeploy: ✗ server did not respond within {health_timeout}s")
        print(f"          check {log_path} for the failure reason")
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
        actives = [
            e for e in registry.registry.get("adapters", [])
            if e.get("status") == "active"
        ]
        if not actives:
            print("redeploy: no active adapter to deploy")
            return False
        # Deterministic choice across clusters: the MOST RECENTLY PROMOTED
        # active adapter is served. Single-adapter serving is the honest
        # current model — the manifest's adapters list is the extension
        # point for serving several at once.
        active = max(actives, key=lambda e: str(e.get("promoted_at") or ""))
        adapter_dir = common.ADAPTERS_DIR / active["cluster_id"] / active["version"]
        print(f"redeploy: deploying {active['cluster_id']} {active['version']} "
              "(most recently promoted active adapter)")
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
    except (RuntimeError, subprocess.TimeoutExpired) as e:
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

    # Step 4: stop → start → health check → serving manifest.
    # id 0 = positional index of the --lora flag in the server command;
    # single-adapter serving today, the list shape is the multi-adapter
    # extension point.
    adapters = [{
        "id": 0,
        "cluster": adapter_dir.parent.name,
        "version": adapter_dir.name,
        "gguf": str(gguf_path),
    }]
    return _stop_start_and_verify(serving_cfg, server_command, gguf_path, adapters)


def redeploy_base() -> bool:
    """Restart the managed llama-server WITHOUT any adapter.

    Used by the bench gate when a regressed adapter has no previous
    version to roll back to — the regressed adapter must never stay
    loaded, so the server is restarted on the bare base model (the
    template's `--lora %LORA%` segment is stripped).
    """
    serving_cfg = load_config().get("serving", {})
    server_command = serving_cfg.get("server_command", "").strip()
    if not server_command:
        print("redeploy: no serving.server_command configured — no managed "
              "server to restart on the base model.")
        return False
    return _stop_start_and_verify(serving_cfg, server_command, None, [])


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

        if entry.get("status") == "active":
            # Re-promoting the active version would set rollback_target to
            # itself, making a later rollback a no-op loop.
            print(f"{cluster_id} {version} is already active — nothing to do.")
            return True

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

    def deactivate(self, cluster_id: str) -> bool:
        """Deactivate the active adapter WITHOUT promoting a replacement.

        The bench gate uses this when a regressing adapter is the
        cluster's first version — there is no rollback target, but a
        regressed adapter must never stay active (and keep being served /
        routed to). The caller is responsible for redeploying the base
        model afterwards.
        """
        current = self._find_active(cluster_id)
        if not current:
            logger.error("No active adapter for cluster %s", cluster_id)
            return False

        current["status"] = "deactivated"

        cluster_dir = common.ADAPTERS_DIR / cluster_id
        active_link = cluster_dir / "active"
        if active_link.is_symlink() or active_link.exists():
            active_link.unlink()

        self._save()
        logger.info("Deactivated %s %s (no rollback target)",
                    cluster_id, current["version"])
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

            # Keep the N most recent non-protected versions. keep_versions=0
            # must keep none of them — a bare [-0:] slice is the whole list.
            candidates = [v for v in versions if v.name not in protected]
            recent = (
                {v.name for v in candidates[-keep_versions:]}
                if keep_versions > 0 else set()
            )
            to_keep = protected | recent

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
                    "deactivated": "[!]",
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


def accepted_baseline_path() -> Path:
    """The explicit accepted-baseline pointer for the bench gate.

    Only runs that PASSED the gate (or the first-ever run) are written
    here — never a run that regressed and was rolled back. Picking "most
    recent bench_*.json by mtime" would let failures become the next
    baseline, silently ratcheting quality downward.
    """
    return common.BENCH_DIR / "baseline.json"


def update_baseline(candidate_path: Path) -> None:
    """Record a bench result as the accepted baseline.

    Callers must only invoke this after the gate passed (or on the
    first-ever run). A regressed/rolled-back run must never end up here.
    """
    data = load_json(candidate_path)
    save_json(accepted_baseline_path(), {
        "accepted_at": datetime.now().isoformat(),
        "source": candidate_path.name,
        "metrics": data.get("metrics", {}),
    })
    logger.info("Accepted baseline updated from %s", candidate_path.name)


def bench_passes(candidate_path: Path, baseline_path: Path = None) -> tuple:
    """
    Compare a candidate bench result against the ACCEPTED baseline.

    Returns (passed: bool, summary: str). The baseline is always
    bench/baseline.json (see accepted_baseline_path) unless an explicit
    baseline_path is given; when no accepted baseline exists yet, the run
    passes by definition and the caller records it as the new baseline.
    """
    from eval import compare_metrics, verdict, format_report

    candidate_data = load_json(candidate_path)
    candidate_metrics = candidate_data.get("metrics", {})
    candidate_cases = int(candidate_metrics.get("total_cases", 0))

    if baseline_path is None:
        baseline_path = accepted_baseline_path()
    baseline_data = load_json(baseline_path, {}) if baseline_path.exists() else {}
    baseline_metrics = baseline_data.get("metrics", {})

    if not baseline_metrics:
        return True, (
            "No accepted baseline yet (bench/baseline.json). "
            "This run passes by definition and is recorded as the new baseline."
        )

    comparison = compare_metrics(candidate_metrics, baseline_metrics)
    checks = verdict(comparison)
    passed = checks.get("overall", False)
    report = format_report(
        candidate_metrics, baseline_metrics, comparison, checks,
        cluster_id="(pipeline)", version="(latest)",
    )
    # Annotate which baseline was used so debugging is easier.
    baseline_cases = int(baseline_metrics.get("total_cases", 0))
    header = (
        f"Baseline: {baseline_path.name} "
        f"(accepted {baseline_data.get('accepted_at', '?')} "
        f"from {baseline_data.get('source', '?')}, "
        f"{baseline_cases or '?'} cases)\n"
    )
    if candidate_cases and baseline_cases and not (
            0.9 <= candidate_cases / baseline_cases <= 1.1):
        header += (
            f"WARNING: case counts differ a lot ({candidate_cases} candidate "
            f"vs {baseline_cases} baseline) — did a smoke-test prompt bank "
            "sneak into a gated run?\n"
        )
    return passed, header + report


def apply_bench_gate(registry: "AdapterRegistry",
                     promoted: List[Tuple[str, str]],
                     served: Optional[Tuple[str, str]],
                     candidate_path: Path) -> bool:
    """Enforce the bench gate on freshly promoted (and served) adapters.

    Pass → the candidate becomes the accepted baseline (bench/baseline.json).
    Regression → the measured adapter is rolled back — or DEACTIVATED when
    it is a first version with nothing to roll back to — and the server is
    redeployed to match the registry (base model if no adapter remains
    active). A regressed run NEVER updates the accepted baseline, and the
    gate never leaves a regressed adapter serving.

    Returns True iff the gate passed.
    """
    passed, report = bench_passes(candidate_path)
    print(report)

    if passed:
        update_baseline(candidate_path)
        print("\n  → BENCHMARK PASSED. Adapters remain active; this run is "
              "now the accepted baseline.")
        return True

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
        elif registry.deactivate(cid):
            print(f"    No rollback target for {cid} — deactivated {version}; "
                  "the base model will be served")
        else:
            print(f"    Could not roll back or deactivate {cid} — manual "
                  "intervention required")

    # Re-serve whatever the registry now says — never leave the regressed
    # adapter loaded.
    if any(e.get("status") == "active"
           for e in registry.registry.get("adapters", [])):
        print("    Redeploying the previous adapter to match the registry...")
        redeploy()
    else:
        print("    No active adapter remains — restarting llama-server on "
              "the base model...")
        redeploy_base()
    return False


def run_pipeline(dry_run: bool = False, with_bench: bool = False):
    """
    Run the full pipeline: extract → score → cluster → train → register → promote.

    When with_bench=True, the bench env runs after promotion and redeploy.
    If it regresses against the accepted baseline, the served adapter is
    automatically rolled back (or deactivated, for a first version).
    """
    if with_bench:
        # Hard-fail BEFORE any pipeline work: the bench gate is only
        # meaningful when it measures the newly deployed adapter.
        gate_cfg = load_config().get("serving", {})
        if not gate_cfg.get("auto_redeploy") or not str(
                gate_cfg.get("server_command", "") or "").strip():
            print("run --with-bench requires finetune.serving.auto_redeploy: true")
            print("(and a serving.server_command). Without auto-redeploy the bench")
            print("would measure the PREVIOUSLY served model and then gate the NEW")
            print("adapters on that result — a meaningless gate.")
            print("Either enable auto_redeploy in config.yaml, or run without")
            print("--with-bench (ungated promotion) and bench after redeploying.")
            raise SystemExit(2)

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
            if with_bench:
                print("    Aborting the bench gate: it would measure the "
                      "previously-served model,")
                print("    not the new adapter. Fix serving, then run "
                      "'/finetune redeploy' and '/finetune bench'.")
                raise SystemExit(1)
            print("    The bench would measure the previously-served model.")

    if not with_bench:
        print("\nPipeline complete. NOTE: promotion was UNGATED — no benchmark "
              "ran on the new adapters.")
        print("Run '/finetune bench' to verify quality, or use "
              "'run --with-bench' for gated promotion.")
        return

    print(f"\n[6/{total_steps}] Running benchmark gate...")
    candidate_path = run_bench()
    if candidate_path is None:
        print("  → Benchmark failed to run. Adapters remain promoted; verify manually.")
        return

    if apply_bench_gate(registry, promoted, served, candidate_path):
        print("\nPipeline complete.")
    else:
        print("\nPipeline complete with regression. Investigate the bench report above.")


CRON_SCHEDULE_MAP = {
    "daily": "0 3 * * *",
    "weekly": "0 3 * * 0",
    "biweekly": "0 3 1,15 * *",
    "monthly": "0 3 1 * *",
}


CRON_JOB_NAME = "finetune-retrain"


def _import_cronjob_tool():
    """Import the real Hermes cron API (tools.cronjob_tools.cronjob).

    Works when hermes-agent is pip-installed (tools/ and cron/ are
    installed packages) and from a repo checkout (repo root four levels
    above scripts/). Returns None when unavailable — callers must then be
    honest that NO job was scheduled.
    """
    try:
        from tools.cronjob_tools import cronjob
        return cronjob
    except ImportError:
        pass
    try:
        repo_root = Path(__file__).resolve().parents[4]
    except IndexError:
        return None
    if (repo_root / "tools" / "cronjob_tools.py").exists():
        sys.path.insert(0, str(repo_root))
        try:
            from tools.cronjob_tools import cronjob
            return cronjob
        except ImportError:
            pass
    return None


def _cron_prompt(gated: bool) -> str:
    """The self-contained prompt the scheduled agent session will run."""
    if gated:
        return (
            "Run the finetune retraining pipeline with the benchmark gate: "
            "execute `/finetune run --with-bench`. This extracts new "
            "sessions, scores quality, updates clusters, retrains eligible "
            "adapters, promotes and redeploys them, and gates the promotion "
            "on the benchmark — regressions are rolled back automatically. "
            "Report adapter versions, cluster changes, the bench verdict, "
            "and any rollbacks."
        )
    return (
        "Run the finetune retraining pipeline: execute `/finetune run`. "
        "This extracts new sessions, scores quality, updates clusters, "
        "retrains eligible adapters, and promotes them WITHOUT a benchmark "
        "gate (serving.auto_redeploy is off, so the bench cannot measure a "
        "newly deployed adapter). Report adapter versions and cluster "
        "changes, and note explicitly that the promotion was ungated — the "
        "user should run `/finetune bench` after redeploying to verify "
        "quality."
    )


def _print_crontab_fallback(cron_expr: str, run_args: str):
    scripts_dir = Path(__file__).resolve().parent
    print("To schedule retraining with plain cron instead, add to your crontab:")
    print(f"  {cron_expr}  cd {scripts_dir} && python manage.py {run_args}")
    if run_args == "run":
        print("  (promotion on this schedule is UNGATED — enable "
              "finetune.serving.auto_redeploy for a gated 'run --with-bench')")


def setup_cron(schedule: str = "weekly") -> bool:
    """Set up scheduled retraining via the Hermes cron system.

    Creates (or updates, if it already exists) a job named
    'finetune-retrain' through tools.cronjob_tools.cronjob — the same API
    the cron tool uses. Never claims success it didn't achieve: when the
    cron API is unavailable or the call fails, it says so and prints the
    exact crontab line to add manually.
    """
    import json

    cron_expr = CRON_SCHEDULE_MAP.get(schedule, schedule)
    gated = bool(load_config().get("serving", {}).get("auto_redeploy"))
    run_args = "run --with-bench" if gated else "run"
    prompt = _cron_prompt(gated)

    cronjob = _import_cronjob_tool()
    if cronjob is None:
        print("Hermes cron tooling is not importable from this environment — "
              "NO job was created.")
        _print_crontab_fallback(cron_expr, run_args)
        return False

    try:
        # Update in place when the job already exists so repeated
        # `/finetune cron` calls don't stack duplicate jobs.
        existing_id = None
        listing = json.loads(cronjob(action="list"))
        for job in listing.get("jobs", []):
            if job.get("name") == CRON_JOB_NAME:
                existing_id = job.get("job_id")
                break

        if existing_id:
            result = json.loads(cronjob(
                action="update", job_id=existing_id,
                prompt=prompt, schedule=cron_expr,
            ))
        else:
            result = json.loads(cronjob(
                action="create", prompt=prompt, schedule=cron_expr,
                name=CRON_JOB_NAME,
            ))
    except Exception as e:
        print(f"Cron job setup failed: {e} — NO job was created.")
        _print_crontab_fallback(cron_expr, run_args)
        return False

    if not result.get("success"):
        print(f"Cron job setup failed: {result.get('error', 'unknown error')} "
              "— NO job was created.")
        _print_crontab_fallback(cron_expr, run_args)
        return False

    verb = "updated" if existing_id else "created"
    print(f"Cron job '{CRON_JOB_NAME}' {verb}: {schedule} ({cron_expr})")
    print(f"  Each run executes: /finetune {run_args}")
    if not gated:
        print("  NOTE: promotion on this schedule is UNGATED — "
              "serving.auto_redeploy is off,")
        print("        so the bench gate cannot measure a newly deployed "
              "adapter. Enable")
        print("        finetune.serving.auto_redeploy for gated promotion.")
    message = result.get("message")
    if message:
        print(f"  {message}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Adapter registry management")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Show pipeline status")

    p_promote = sub.add_parser("promote", help="Promote adapter to active")
    p_promote.add_argument("--cluster", required=True)
    p_promote.add_argument("--version", required=True)

    p_rollback = sub.add_parser("rollback", help="Roll back to previous version")
    p_rollback.add_argument("--cluster", required=True)

    p_run = sub.add_parser(
        "run",
        help="Run full pipeline (promotion is UNGATED without --with-bench)",
    )
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument(
        "--with-bench", action="store_true",
        help="Run the benchmark gate after promotion+redeploy (requires "
             "serving.auto_redeploy). Auto-rollback adapters that regress "
             "vs. the accepted baseline.",
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

    # Serialize every mutating command behind the coarse pipeline lock:
    # overlapping cron + manual invocations would otherwise race
    # registry.json (last-writer-wins), version allocation, and the server
    # PID file. Coarse on purpose (see common.pipeline_lock) — a second
    # invocation fails fast with a clear message instead of corrupting
    # state. Standalone `python train.py` takes the same lock.
    if args.command in ("run", "promote", "rollback", "redeploy", "gc"):
        try:
            with common.pipeline_lock(timeout=LOCK_TIMEOUT):
                _dispatch(args)
        except TimeoutError as e:
            print(f"Another finetune operation is running — {e}")
            sys.exit(1)
    else:
        _dispatch(args)


def _dispatch(args):
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
            if passed:
                # An accepted (passing or first-ever) run becomes the new
                # baseline; failing runs never move the ratchet.
                update_baseline(result_path)
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
        if not setup_cron(args.schedule):
            sys.exit(1)


if __name__ == "__main__":
    main()

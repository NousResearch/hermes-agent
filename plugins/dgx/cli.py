"""CLI commands for the dgx plugin.

Wires ``hermes dgx <subcommand>``:
  setup     — interactive wizard: configure host, ports, default model/endpoint
  status    — GPU memory, running models, endpoint health
  models    — list models available on Ollama and vLLM
  use       — switch active model (updates config.yaml model.default)
  endpoint  — switch active endpoint (ollama / vllm / litellm)
  pull      — pull a model into Ollama on the DGX (streaming)
  rm        — remove a model from Ollama on the DGX
  ps        — show models currently loaded in GPU memory
  push      — rsync local files to the DGX workspace
  doctor    — comprehensive health check: SSH, GPU, endpoints
  watch     — live nvidia-smi refresh until Ctrl+C
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from plugins.dgx._dgx_config import (
    DEFAULTS,
    DGXNotConfigured,
    ENDPOINT_LABELS,
    apply_endpoint,
    litellm_base,
    load_dgx_config,
    ollama_base,
    save_dgx_config,
    vllm_base,
)


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------

def register_cli(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="dgx_command")

    subs.add_parser(
        "setup",
        help="Interactive wizard: configure DGX host, endpoints, default model",
    )

    subs.add_parser(
        "status",
        help="Show GPU memory usage, running models, and endpoint health",
    )

    use_p = subs.add_parser("use", help="Switch the active model")
    use_p.add_argument("model", nargs="?", default=None,
                       help="Model name (e.g. qwen2.5-coder:latest)")
    use_p.add_argument(
        "--endpoint",
        choices=["ollama", "vllm", "vllm-32b", "litellm"],
        default=None,
        help="Endpoint to use for this model (auto-detected if omitted)",
    )

    ep_p = subs.add_parser(
        "endpoint",
        help="Switch between ollama / vllm / litellm endpoints",
    )
    ep_p.add_argument(
        "name",
        choices=["ollama", "vllm", "vllm-32b", "litellm"],
        help="Endpoint to activate",
    )

    pull_p = subs.add_parser("pull", help="Pull a model into Ollama on the DGX")
    pull_p.add_argument("model", help="Model name (e.g. nemotron3:70b)")

    rm_p = subs.add_parser("rm", help="Remove a model from Ollama on the DGX")
    rm_p.add_argument("model", help="Model name to remove")
    rm_p.add_argument("--force", "-f", action="store_true", help="Skip confirmation prompt")

    subs.add_parser("ps", help="Show models currently loaded in GPU memory")

    push_p = subs.add_parser("push", help="rsync local files to the DGX workspace")
    push_p.add_argument("local", help="Local path to sync")
    push_p.add_argument("remote", nargs="?", default=None,
                        help="Remote destination (default: ~/workspace/)")

    subs.add_parser("doctor", help="Comprehensive health check: SSH, GPU, endpoints")

    watch_p = subs.add_parser("watch", help="Live nvidia-smi refresh until Ctrl+C")
    watch_p.add_argument("--interval", "-n", type=int, default=2,
                         help="Refresh interval in seconds (default: 2)")

    subparser.set_defaults(func=dgx_command)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def dgx_command(args: argparse.Namespace) -> int:
    """Entry point for ``hermes dgx``.

    The URL helpers (ollama_base/vllm_base) raise DGXNotConfigured when no host
    is set, and several subcommands (status, doctor, models, use, route --check)
    call them. Catch it here — one shared point — so an unconfigured install
    prints the "run hermes dgx setup" guidance and exits 1, rather than dumping
    an uncaught traceback (doctor is the worst case: it's the very command users
    run to diagnose an unconfigured box).
    """
    try:
        return _dgx_dispatch(args)
    except DGXNotConfigured as e:
        print(e)
        return 1


def _dgx_dispatch(args: argparse.Namespace) -> int:
    sub = getattr(args, "dgx_command", None)
    if not sub:
        print("usage: hermes dgx {setup,status,use,endpoint,pull,rm,ps,push,doctor,watch}")
        return 2
    if sub == "setup":
        return _cmd_setup()
    if sub == "status":
        return _cmd_status()
    if sub == "use":
        if not args.model:
            print("usage: hermes dgx use <model>")
            return 2
        return _cmd_use(model=args.model, endpoint=getattr(args, "endpoint", None))
    if sub == "endpoint":
        return _cmd_endpoint(name=args.name)
    if sub == "pull":
        return _cmd_pull(model=args.model)
    if sub == "rm":
        return _cmd_rm(model=args.model, force=getattr(args, "force", False))
    if sub == "ps":
        return _cmd_ps()
    if sub == "push":
        return _cmd_push(local=args.local, remote=args.remote)
    if sub == "doctor":
        return _cmd_doctor()
    if sub == "watch":
        return _cmd_watch(interval=getattr(args, "interval", 2))
    print(f"unknown subcommand: {sub}")
    return 2


# ---------------------------------------------------------------------------
# HTTP helpers (no extra deps — stdlib only)
# ---------------------------------------------------------------------------

def _get_json(url: str, timeout: int = 5) -> Tuple[Optional[Dict], Optional[str]]:
    """GET *url*, return (parsed_json, None) or (None, error_string)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read()), None
    except urllib.error.URLError as e:
        return None, str(e.reason)
    except Exception as e:
        return None, str(e)


def _check_endpoint(url: str, timeout: int = 4) -> Tuple[bool, str]:
    """Return (reachable, status_string)."""
    _, err = _get_json(url, timeout=timeout)
    return (err is None), (err or "ok")


# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------

def _ssh_run(user: str, host: str, cmd: str, timeout: int = 10) -> Tuple[bool, str]:
    """Run *cmd* on host via SSH. Returns (ok, output_or_error)."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=6", "-o", "BatchMode=yes",
             f"{user}@{host}", cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, (result.stderr.strip() or f"exit {result.returncode}")
    except subprocess.TimeoutExpired:
        return False, "ssh timed out"
    except FileNotFoundError:
        return False, "ssh not found"
    except Exception as e:
        return False, str(e)


def _ssh_stream(user: str, host: str, cmd: str, timeout: int = 300) -> int:
    """Run *cmd* on host via SSH, streaming stdout/stderr to the terminal.

    Returns the remote exit code. Used for long-running commands like
    ``ollama pull`` where real-time progress output matters.
    """
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=6", "-o", "BatchMode=yes",
             f"{user}@{host}", cmd],
            timeout=timeout,
        )
        return result.returncode
    except subprocess.TimeoutExpired:
        print("(ssh timed out)")
        return 1
    except FileNotFoundError:
        print("ssh not found — is OpenSSH installed?")
        return 1
    except Exception as e:
        print(f"ssh error: {e}")
        return 1


# ---------------------------------------------------------------------------
# hermes dgx pull
# ---------------------------------------------------------------------------

def _cmd_pull(model: str) -> int:
    dgx = load_dgx_config()
    print(f"Pulling {model} on dgx1 ({dgx['host']}) ...")
    rc = _ssh_stream(dgx["ssh_user"], dgx["host"], f"ollama pull {model}")
    if rc != 0:
        print(f"pull failed (exit {rc})")
    return rc


# ---------------------------------------------------------------------------
# hermes dgx rm
# ---------------------------------------------------------------------------

def _cmd_rm(model: str, force: bool = False) -> int:
    dgx = load_dgx_config()
    if not force:
        try:
            ans = input(f"Remove {model} from {dgx['host']}? [y/N] ").strip().lower()
        except EOFError:
            ans = ""
        if ans not in ("y", "yes"):
            print("aborted")
            return 0
    ok, out = _ssh_run(dgx["ssh_user"], dgx["host"], f"ollama rm {model}")
    if ok:
        print(f"removed {model}")
        return 0
    print(f"rm failed: {out}")
    return 1


# ---------------------------------------------------------------------------
# hermes dgx ps
# ---------------------------------------------------------------------------

def _cmd_ps() -> int:
    dgx = load_dgx_config()
    ok, out = _ssh_run(dgx["ssh_user"], dgx["host"], "ollama ps", timeout=10)
    if not ok:
        print(f"(SSH unavailable: {out})")
        return 1
    lines = out.strip().splitlines()
    # ollama ps prints a header even when empty; detect "nothing loaded" by
    # checking whether there are any data rows after the header.
    data_lines = [l for l in lines[1:] if l.strip()] if len(lines) > 1 else []
    if not data_lines:
        print("No models currently loaded in GPU memory.")
        return 0
    # Pass through ollama's own formatting — it already aligns columns nicely.
    for line in lines:
        print(f"  {line}")
    return 0


# ---------------------------------------------------------------------------
# hermes dgx push
# ---------------------------------------------------------------------------

def _cmd_push(local: str, remote: Optional[str]) -> int:
    dgx = load_dgx_config()
    remote_path = remote or "~/workspace/"
    dest = f"{dgx['ssh_user']}@{dgx['host']}:{remote_path}"
    print(f"Pushing {local}")
    print(f"     → {dest}")
    try:
        result = subprocess.run(
            ["rsync", "-avz", "--progress", local, dest],
            timeout=300,
        )
        return result.returncode
    except FileNotFoundError:
        print("rsync not found — install rsync to use this command")
        return 1
    except subprocess.TimeoutExpired:
        print("rsync timed out")
        return 1
    except Exception as e:
        print(f"push failed: {e}")
        return 1


# ---------------------------------------------------------------------------
# hermes dgx doctor
# ---------------------------------------------------------------------------

_CHECK_OK  = "✓"
_CHECK_FAIL = "✗"


def _cmd_doctor() -> int:
    dgx = load_dgx_config()
    host = dgx["host"]
    user = dgx["ssh_user"]

    print("hermes dgx doctor")
    print("─────────────────")

    failures: List[str] = []

    # --- SSH ---
    ok, out = _ssh_run(user, host, "echo ok", timeout=8)
    sym = _CHECK_OK if ok else _CHECK_FAIL
    print(f"  SSH         {user}@{host}  {sym}")
    if not ok:
        print(f"              ({out})")
        failures.append("ssh")

    # --- GPU ---
    gpu_ok, gpu_out = _ssh_run(
        user, host,
        "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu "
        "--format=csv,noheader,nounits",
        timeout=10,
    )
    sym = _CHECK_OK if gpu_ok else _CHECK_FAIL
    print(f"  GPU         {sym}")
    if gpu_ok and gpu_out:
        for line in gpu_out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                idx, name, used, total = parts[0], parts[1], parts[2], parts[3]
                try:
                    free_gb = (int(total) - int(used)) / 1024
                    print(f"              GPU {idx}  {name}  {free_gb:.0f} GB free")
                except ValueError:
                    print(f"              GPU {idx}  {name}  {used}/{total} MiB")
    elif not gpu_ok:
        failures.append("gpu")

    # --- Ollama ---
    obase = ollama_base(dgx)
    data, err = _get_json(f"{obase}/api/tags", timeout=4)
    if data is not None:
        n = len(data.get("models", []))
        print(f"  Ollama      {obase}  {_CHECK_OK}  ({n} models)")
    else:
        print(f"  Ollama      {obase}  {_CHECK_FAIL}  ({err})")
        failures.append("ollama")

    # --- vLLM ---
    vbase = vllm_base(dgx)
    data, err = _get_json(f"{vbase}/v1/models", timeout=4)
    if data is not None:
        models = [m["id"] for m in data.get("data", [])]
        print(f"  vLLM        {vbase}  {_CHECK_OK}  ({', '.join(models) or 'no models'})")
    else:
        print(f"  vLLM        {vbase}  {_CHECK_FAIL}  ({err})")
        failures.append("vllm")

    # --- LiteLLM (advisory only — key required) ---
    lbase = litellm_base(dgx)
    lm_ok, lm_msg = _check_endpoint(f"{lbase}/health", timeout=4)
    sym = _CHECK_OK if lm_ok else _CHECK_FAIL
    note = "" if lm_ok else "  (key required — see ~/.hermes/.env)"
    print(f"  LiteLLM     {lbase}  {sym}{note}")

    print()
    # SSH failure is always fatal; all inference endpoints failing is fatal
    inference_ok = not ({"ollama", "vllm"} <= set(failures))
    critical_ok = "ssh" not in failures and inference_ok
    if critical_ok:
        print("All critical checks passed.")
    else:
        print("One or more critical checks FAILED — see above.")
    return 0 if critical_ok else 1


# ---------------------------------------------------------------------------
# hermes dgx watch
# ---------------------------------------------------------------------------

_NVIDIA_SMI_QUERY = (
    "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu "
    "--format=csv,noheader,nounits"
)


def _print_gpu_lines(out: str) -> None:
    """Render nvidia-smi csv output as formatted GPU rows."""
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            idx, name, used, total, util = parts[0], parts[1], parts[2], parts[3], parts[4]
            try:
                bar_pct = int(used) / max(int(total), 1)
                bar = "█" * int(bar_pct * 20) + "░" * (20 - int(bar_pct * 20))
                print(f"  GPU {idx}  {name}")
                print(f"  [{bar}] {used}/{total} MiB  ({util}% util)")
            except ValueError:
                print(f"  GPU {idx}  {name}  mem={used}/{total} MiB  util={util}%")


def _cmd_watch(interval: int = 2) -> int:
    """Live-refresh nvidia-smi every *interval* seconds until Ctrl+C."""
    dgx = load_dgx_config()
    host = dgx["host"]
    user = dgx["ssh_user"]

    try:
        while True:
            ok, out = _ssh_run(user, host, _NVIDIA_SMI_QUERY, timeout=10)
            print("\033[2J\033[H", end="")  # clear screen, move cursor home
            print(f"DGX Spark GPU  {host}  —  Ctrl+C to stop\n")
            if ok and out:
                _print_gpu_lines(out)
            else:
                print(f"  (SSH unavailable: {out})")
            time.sleep(interval)
    except KeyboardInterrupt:
        print()
        return 0


# ---------------------------------------------------------------------------
# hermes dgx status
# ---------------------------------------------------------------------------

def _cmd_status() -> int:
    dgx = load_dgx_config()
    host = dgx["host"]
    user = dgx["ssh_user"]
    active = dgx.get("active_endpoint", "ollama")

    print(f"DGX Spark  {host}  (active endpoint: {active})")
    print()

    # --- GPU via nvidia-smi over SSH ---
    print("GPU memory")
    ok, out = _ssh_run(
        user, host,
        "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu "
        "--format=csv,noheader,nounits",
    )
    if ok and out:
        _print_gpu_lines(out)
    else:
        print(f"  (SSH unavailable: {out})")
    print()

    # --- Ollama ---
    obase = ollama_base(dgx)
    data, err = _get_json(f"{obase}/api/tags")
    if data is not None:
        models = [m["name"] for m in data.get("models", [])]
        print(f"Ollama  {obase}  ✓  ({len(models)} models loaded)")
        for m in models:
            marker = " ◀ active" if (active == "ollama" and m.startswith(
                dgx.get("default_model", "").split(":")[0])) else ""
            print(f"  {m}{marker}")
    else:
        print(f"Ollama  {obase}  ✗  ({err})")
    print()

    # --- vLLM ---
    vbase = vllm_base(dgx)
    data, err = _get_json(f"{vbase}/v1/models")
    if data is not None:
        models = [m["id"] for m in data.get("data", [])]
        print(f"vLLM    {vbase}  ✓  ({len(models)} models loaded)")
        for m in models:
            marker = " ◀ active" if active == "vllm" else ""
            print(f"  {m}{marker}")
    else:
        print(f"vLLM    {vbase}  ✗  ({err})")
    print()

    # --- LiteLLM ---
    lbase = litellm_base(dgx)
    ok, msg = _check_endpoint(f"{lbase}/health")
    sym = "✓" if ok else "✗"
    marker = " ◀ active" if active == "litellm" else ""
    print(f"LiteLLM {lbase}  {sym}  ({msg}){marker}")

    return 0


# ---------------------------------------------------------------------------
# hermes dgx use <model>
# ---------------------------------------------------------------------------

def _cmd_use(model: str, endpoint: Optional[str] = None) -> int:
    from hermes_cli.config import load_config, save_config

    dgx = load_dgx_config()

    # Auto-detect endpoint from which service has the model, if not specified
    if endpoint is None:
        endpoint = dgx.get("active_endpoint", "ollama")
        obase = ollama_base(dgx)
        data, _ = _get_json(f"{obase}/api/tags")
        if data is not None:
            ollama_models = [m["name"] for m in data.get("models", [])]
            model_root = model.split(":")[0]
            if any(m.startswith(model_root) for m in ollama_models):
                endpoint = "ollama"

    dgx["default_model"] = model
    apply_endpoint(dgx, endpoint)

    # Also update model.default
    cfg = load_config()
    if not isinstance(cfg.get("model"), dict):
        cfg["model"] = {}
    cfg["model"]["default"] = model
    save_config(cfg)

    print(f"Active model : {model}")
    print(f"Endpoint     : {endpoint}  ({ENDPOINT_LABELS.get(endpoint, endpoint)})")
    return 0


# ---------------------------------------------------------------------------
# hermes dgx endpoint <name>
# ---------------------------------------------------------------------------

def _cmd_endpoint(name: str) -> int:
    dgx = load_dgx_config()
    apply_endpoint(dgx, name)
    label = ENDPOINT_LABELS.get(name, name)
    print(f"Switched to {name}  ({label})")
    _print_model_summary(dgx, name)
    return 0


def _print_model_summary(dgx: Dict[str, Any], endpoint: str) -> None:
    from hermes_cli.config import load_config
    cfg = load_config()
    model = cfg.get("model", {})
    print(f"  base_url : {model.get('base_url', '(not set)')}")
    print(f"  provider : {model.get('provider', '(not set)')}")
    print(f"  model    : {model.get('default', '(not set)')}")


# ---------------------------------------------------------------------------
# hermes dgx setup
# ---------------------------------------------------------------------------

def _prompt(label: str, default: Any) -> str:
    val = input(f"  {label} [{default}]: ").strip()
    return val if val else str(default)


def _probe_ollama(host: str, port: int) -> Tuple[bool, List[str]]:
    url = f"http://{host}:{port}/api/tags"
    data, err = _get_json(url, timeout=4)
    if data is None:
        return False, []
    models = [m["name"] for m in data.get("models", [])]
    return True, models


def _probe_vllm(host: str, port: int) -> Tuple[bool, List[str]]:
    url = f"http://{host}:{port}/v1/models"
    data, err = _get_json(url, timeout=4)
    if data is None:
        return False, []
    models = [m["id"] for m in data.get("data", [])]
    return True, models


def _probe_litellm(host: str, port: int) -> bool:
    url = f"http://{host}:{port}/health"
    ok, _ = _check_endpoint(url, timeout=4)
    return ok


def _cmd_setup() -> int:
    current = load_dgx_config()

    print()
    print("hermes dgx setup")
    print("────────────────")
    print("Configure your DGX Spark inference endpoints.")
    print("Press Enter to accept the current value shown in brackets.")
    print()

    # --- Host / SSH ---
    print("DGX Spark host")
    host = _prompt("IP address", current.get("host", DEFAULTS["host"]))
    ssh_user = _prompt("SSH user", current.get("ssh_user", DEFAULTS["ssh_user"]))
    print()

    # --- Ports ---
    print("Endpoint ports")
    ollama_port = int(_prompt("Ollama port", current.get("ollama_port", DEFAULTS["ollama_port"])))
    vllm_port = int(_prompt("vLLM port", current.get("vllm_port", DEFAULTS["vllm_port"])))
    print()

    # --- Probe endpoints ---
    print("Probing endpoints...")
    ollama_ok, ollama_models = _probe_ollama(host, ollama_port)
    vllm_ok, vllm_models = _probe_vllm(host, vllm_port)

    if ollama_ok:
        summary = f"({len(ollama_models)} models: {', '.join(ollama_models[:3])}{'...' if len(ollama_models) > 3 else ''})"
        print(f"  Ollama  http://{host}:{ollama_port}  ✓  {summary}")
    else:
        print(f"  Ollama  http://{host}:{ollama_port}  ✗  (unreachable — check host/port or VPN)")

    if vllm_ok:
        print(f"  vLLM    http://{host}:{vllm_port}  ✓  ({', '.join(vllm_models)})")
    else:
        print(f"  vLLM    http://{host}:{vllm_port}  ✗  (unreachable)")
    print()

    # --- LiteLLM (optional) ---
    print("LiteLLM proxy (optional — HA pool across all GPU nodes)")
    litellm_host = _prompt("LiteLLM host", current.get("litellm_host", DEFAULTS["litellm_host"]))
    litellm_port = int(_prompt("LiteLLM port", current.get("litellm_port", DEFAULTS["litellm_port"])))
    litellm_ok = _probe_litellm(litellm_host, litellm_port)
    if litellm_ok:
        print(f"  LiteLLM http://{litellm_host}:{litellm_port}  ✓")
    else:
        print(f"  LiteLLM http://{litellm_host}:{litellm_port}  ✗  (unreachable — key required; set OPENAI_API_KEY in ~/.hermes/.env)")
    print()

    # --- Probe vLLM 32B (port 30881) ---
    vllm_32b_port = int(_prompt("vLLM 32B port", current.get("vllm_32b_port", DEFAULTS["vllm_32b_port"])))
    vllm_32b_ok, vllm_32b_models = _probe_vllm(host, vllm_32b_port)
    if vllm_32b_ok:
        print(f"  vLLM 32B http://{host}:{vllm_32b_port}  ✓  ({', '.join(vllm_32b_models)})")
    else:
        print(f"  vLLM 32B http://{host}:{vllm_32b_port}  ✗  (not ready yet — model may still be downloading)")
    print()

    # --- Choose default endpoint ---
    available = []
    if ollama_ok:
        available.append("ollama")
    if vllm_ok:
        available.append("vllm")
    if vllm_32b_ok:
        available.append("vllm-32b")
    if litellm_ok:
        available.append("litellm")

    current_ep = current.get("active_endpoint", "ollama")
    if not available:
        print("WARNING: no endpoints are reachable. Saving config anyway.")
        print("         Check your VPN (WireGuard) or DGX host/port settings.")
        chosen_ep = current_ep
    else:
        print("Default endpoint")
        for i, ep in enumerate(available, 1):
            marker = " (current)" if ep == current_ep else ""
            print(f"  {i}) {ep:<8} — {ENDPOINT_LABELS[ep]}{marker}")
        while True:
            raw = input(f"  Choice [{'1' if available else '?'}]: ").strip()
            if not raw and available:
                chosen_ep = available[0]
                break
            try:
                chosen_ep = available[int(raw) - 1]
                break
            except (ValueError, IndexError):
                print("  Enter a number from the list above.")
    print()

    # --- Choose default model ---
    endpoint_models: List[str] = []
    if chosen_ep == "ollama":
        endpoint_models = ollama_models
    elif chosen_ep in ("vllm", "vllm-32b"):
        endpoint_models = vllm_models

    all_choices: List[Tuple[str, str]] = [(m, "") for m in endpoint_models]

    current_model = current.get("default_model", DEFAULTS["default_model"])
    if all_choices:
        print("Default model")
        for i, (m, note) in enumerate(all_choices, 1):
            marker = " (current)" if m == current_model else ""
            print(f"  {i}) {m}{marker}{note}")
        print(f"  Or type a model name directly.")
        raw = input(f"  Choice [{current_model}]: ").strip()
        if not raw:
            chosen_model = current_model
        elif raw.isdigit() and 1 <= int(raw) <= len(all_choices):
            chosen_model = all_choices[int(raw) - 1][0]
        else:
            chosen_model = raw
    else:
        chosen_model = _prompt("Default model", current_model)
    print()

    # --- Save ---
    dgx = {
        "host": host,
        "ssh_user": ssh_user,
        "ollama_port": ollama_port,
        "vllm_port": vllm_port,
        "litellm_host": litellm_host,
        "litellm_port": litellm_port,
        "active_endpoint": chosen_ep,
        "default_model": chosen_model,
    }
    save_dgx_config(dgx)
    apply_endpoint(dgx, chosen_ep)

    # Update model.default too
    from hermes_cli.config import load_config, save_config
    cfg = load_config()
    if not isinstance(cfg.get("model"), dict):
        cfg["model"] = {}
    cfg["model"]["default"] = chosen_model
    save_config(cfg)

    print("Configuration saved.")
    print()
    print(f"  DGX host  : {host}")
    print(f"  Endpoint  : {chosen_ep}  ({ENDPOINT_LABELS.get(chosen_ep, chosen_ep)})")
    print(f"  Model     : {chosen_model}")
    print()
    print("Next steps:")
    print("  hermes dgx status    — verify GPU and endpoint health")
    print("  hermes dgx models    — browse available models")
    print("  hermes               — start chatting")
    return 0


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(prog="hermes dgx")
    register_cli(parser)
    ns = parser.parse_args()
    sys.exit(dgx_command(ns))

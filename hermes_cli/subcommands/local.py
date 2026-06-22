"""
`hermes local` — manage and inspect the built-in Hermes Native LLM engine
(llama.cpp powered, zero external Ollama/LM Studio required).

Commands:
  hermes local status
  hermes local list
  hermes local suggest [--speed]
  hermes local download <model> [--quant Q4_K_M]
  hermes local unload
  hermes local info <model>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()


def build_local_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "local",
        help="Hermes Native (built-in) local LLM management (llama.cpp)",
        description="Manage the zero-config native LLM server that powers provider=hermes-local.",
    )
    sub = p.add_subparsers(dest="local_command")

    # status
    sub.add_parser("status", help="Show whether a native model server is running + system resources")

    # list
    sub.add_parser("list", aliases=["ls"], help="List catalog models + any .gguf files you have dropped")

    # suggest
    sp = sub.add_parser("suggest", help="Show recommended models for this machine")
    sp.add_argument("--speed", action="store_true", help="Optimize suggestions for speed over quality")
    sp.add_argument("--ctx", type=int, default=8192, help="Target context length for estimation")

    # download
    dp = sub.add_parser("download", help="Download a catalog model (or specific quant)")
    dp.add_argument("model", help="Model key e.g. gemma4-12b or ministral3-8b")
    dp.add_argument("--quant", help="Specific quant (Q4_K_M, Q5_K_M, ...)")

    # unload / stop
    sub.add_parser("unload", aliases=["stop"], help="Stop the currently running native server")

    # info
    ip = sub.add_parser("info", help="Show details + estimated memory for a model")
    ip.add_argument("model")

    p.set_defaults(func=cmd_local)


def cmd_local(args: Any) -> int:
    # Lazy import so the rest of hermes doesn't pull heavy stuff until needed
    from hermes_cli.native_llm import (
        get_system_resources,
        list_available_local_models,
        list_downloaded_models,
        suggest_models,
        ensure_model,
        stop_server,
        get_server_base_url,
        SUPPORTED_MODELS,
        compute_model_footprint,
    )

    cmd = getattr(args, "local_command", None) or "status"

    if cmd in (None, "status"):
        from hermes_cli.native_llm import get_native_status
        st = get_native_status()
        res = st["resources"]
        console.print("[bold]Hermes Native LLM[/bold]")
        if st["running"]:
            console.print(f"  Server: [green]running[/green] → {st['base_url']}")
            if st.get("current_model"):
                cm = st["current_model"]
                console.print(f"  Model: {cm.get('path', '?')}  ({cm.get('size_gb', '?')} GB)")
        else:
            console.print("  Server: [dim]not running[/dim]")
        console.print(f"  RAM: {res['ram_available_gb']:.1f}/{res['ram_total_gb']:.1f} GB  |  VRAM: {res['vram_gb']:.1f} GB")
        console.print(f"  GPU: {res['gpu_name']}")
        if res.get("has_npu"):
            console.print(f"  NPU detected: {res.get('npu_name')}")
        console.print(f"  Local GGUF files: {st.get('downloaded_count', 0)}  (arbitrary user files: {'yes' if st.get('has_arbitrary') else 'no'})")
        return 0

    if cmd in ("list", "ls"):
        rows = list_available_local_models()
        t = Table(title="Hermes Native Models")
        t.add_column("Key")
        t.add_column("Display")
        t.add_column("Est GB (8k)")
        t.add_column("Status")
        t.add_column("Notes", style="dim")
        for r in rows:
            status = "downloaded" if r.get("downloaded") else "download needed"
            if r.get("is_arbitrary"):
                status = "user GGUF"
            t.add_row(
                r["key"],
                r["display"],
                str(r.get("estimated_load_gb", "?")),
                status,
                r.get("notes", "")[:50],
            )
        console.print(t)
        console.print("\nTip: drop any .gguf into the models dir and it will appear automatically.")
        return 0

    if cmd == "suggest":
        res = get_system_resources()
        sugs = suggest_models(resources=res, prefer_speed=bool(args.speed), target_ctx=getattr(args, "ctx", 8192))
        console.print("[bold]Suggested models for your hardware[/bold]")
        for s in sugs:
            console.print(
                f"  [cyan]{s['display']}[/cyan]  ({s['quant']})  ~{s['estimated_gb']} GB   "
                f"q:{s['quality']} s:{s['speed']}   {s['reason']}"
            )
        return 0

    if cmd == "download":
        key = args.model
        quant = getattr(args, "quant", None)
        if key not in SUPPORTED_MODELS and not (key.endswith(".gguf") or key.startswith("local:")):
            # allow user to try arbitrary by name if they know a real repo later
            console.print(f"[yellow]Warning:[/yellow] '{key}' not in built-in catalog. "
                          "You can still use any .gguf you manually place in the models directory.")
        console.print(f"Ensuring {key} {quant or ''} ...")
        try:
            p = ensure_model(key, quant=quant, progress=lambda frac, msg: console.print(f"  {msg}"))
            console.print(f"[green]Ready:[/green] {p}")
        except Exception as e:
            console.print(f"[red]Download failed:[/red] {e}")
            return 1
        return 0

    if cmd in ("unload", "stop"):
        stop_server()
        console.print("Native server stopped (if it was running).")
        return 0

    if cmd == "info":
        key = args.model
        meta = SUPPORTED_MODELS.get(key)
        if not meta:
            console.print("Unknown catalog key. You can still 'hermes local list' to see what you have on disk.")
            return 0
        res = get_system_resources()
        for q in meta.get("quants", [meta.get("default_quant", "Q4_K_M")]):
            est = compute_model_footprint(meta, q)
            fit = "fits well" if (res["vram_gb"] + res["ram_available_gb"]) > est * 1.1 else "tight / may need quant"
            full = "full GPU likely" if res["vram_gb"] >= est - 1.5 else "partial or CPU"
            console.print(f"{meta['display']} @ {q}: ~{est} GB  | {fit}  | {full}")
        console.print(meta.get("notes", ""))
        return 0

    # default
    console.print("Use `hermes local --help` for subcommands (status, list, suggest, download, unload, info).")
    return 0
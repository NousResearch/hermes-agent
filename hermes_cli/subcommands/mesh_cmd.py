"""``hermes mesh`` — Global Internet GPU DePIN & Swarm Mesh.

Discover, share, and delegate compute turns to GPU-enabled Hermes nodes across
the public internet without requiring local WiFi or LAN proximity.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional


def _format_vram(mb: int) -> str:
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb} MB"


def mesh_join(args: argparse.Namespace) -> int:
    from gateway.global_mesh import GlobalMeshCoordinator

    coordinator = GlobalMeshCoordinator(rendezvous_url=args.rendezvous)
    offer_gpu = False if args.no_gpu else (True if args.offer_gpu or args.vram or args.backend or args.device else None)

    node = coordinator.init_local_node(
        endpoint=args.endpoint or "mesh://direct",
        offer_gpu=offer_gpu,
        vram_mb=args.vram,
        backend=args.backend,
        device_name=args.device,
    )

    # Attempt to sync with rendezvous server
    synced, discovered_peers = coordinator.sync_rendezvous(timeout_s=5.0)

    print(f"\033[32mSuccessfully registered node with Hermes Global Mesh!\033[0m")
    print(f"Node ID:       {node.node_id}")
    print(f"Rendezvous:    {coordinator.rendezvous_url} {'[SYNCED]' if synced else '[STANDALONE/OFFLINE]'}")
    print(f"Public Key:    {node.public_key}")
    if node.gpu:
        print(f"GPU Backend:   {node.gpu.compute_backend.upper()} ({node.gpu.device_name})")
        print(f"Total VRAM:    {_format_vram(node.gpu.vram_mb)}")
    else:
        print(f"GPU Compute:   Disabled (Client-only mode - no physical accelerator detected)")
    if discovered_peers:
        print(f"Swarm Peers:   Discovered {len(discovered_peers)} active peers from rendezvous")
    return 0


def mesh_status(args: argparse.Namespace) -> int:
    from gateway.global_mesh import GlobalMeshCoordinator

    coordinator = GlobalMeshCoordinator()
    coordinator.init_local_node()
    summary = coordinator.get_mesh_summary()

    print(f"=== Hermes Global DePIN Mesh Status ===")
    print(f"Local Node ID:      {summary['node_id']}")
    print(f"Rendezvous Relay:   {summary['rendezvous']}")
    print(f"Active Swarm Peers: {summary['active_peers']} / {summary['total_peers']} total")
    print(f"Swarm Total VRAM:   {_format_vram(summary['cluster_vram_mb'])}")
    print(f"Swarm Free VRAM:    {_format_vram(summary['cluster_free_vram_mb'])}")
    if summary["local_gpu"]:
        lg = summary["local_gpu"]
        print(f"Local Hardware:     {lg['device_name']} [{lg['compute_backend'].upper()}] - {_format_vram(lg['free_vram_mb'])} free")
    else:
        print(f"Local Hardware:     Client-only (no GPU advertised)")
    return 0


def mesh_peers(args: argparse.Namespace) -> int:
    from gateway.global_mesh import GlobalMeshCoordinator

    coordinator = GlobalMeshCoordinator()
    peers = list(coordinator.peers.values())
    if not peers:
        print("No remote peers discovered in global mesh. Run `hermes mesh join` to sync.")
        return 0

    print(f"{'NODE ID':<20} {'BACKEND':<10} {'VRAM':<12} {'REPUTATION':<12} {'MODELS'}")
    print("-" * 75)
    for p in peers:
        gpu_info = p.gpu
        b_end = gpu_info.compute_backend if gpu_info else "none"
        vram_str = _format_vram(gpu_info.free_vram_mb) if gpu_info else "0 MB"
        rep_str = f"{gpu_info.reputation_score * 100:.1f}%" if gpu_info else "N/A"
        models_str = ", ".join(gpu_info.supported_models[:2]) if gpu_info and gpu_info.supported_models else "general"
        print(f"{p.node_id:<20} {b_end:<10} {vram_str:<12} {rep_str:<12} {models_str}")
    return 0


def mesh_delegate(args: argparse.Namespace) -> int:
    from gateway.global_mesh import GlobalMeshCoordinator

    coordinator = GlobalMeshCoordinator()
    print(f"Broadcasting compute request to Hermes Global Swarm (target: {args.model}, min VRAM: {_format_vram(args.min_vram)})...")
    receipt = coordinator.delegate_task(
        payload=args.prompt,
        target_model=args.model,
        min_vram_mb=args.min_vram,
    )

    if receipt.status == "completed":
        print(f"\033[32m[+] Execution Confirmed from Swarm Node: {receipt.executor_id}\033[0m")
        print(f"Latency: {receipt.latency_ms} ms | Tokens: {receipt.tokens}")
        print(f"Proof Signature: {receipt.proof_sig[:24]}...")
        print("-" * 60)
        print(receipt.output)
        return 0
    else:
        print(f"\033[31m[-] Delegation failed: {receipt.error_message}\033[0m")
        return 1


def mesh_prune(args: argparse.Namespace) -> int:
    from gateway.global_mesh import GlobalMeshCoordinator

    coordinator = GlobalMeshCoordinator()
    pruned = coordinator.prune_inactive_peers(timeout_seconds=args.timeout)
    print(f"Pruned {pruned} inactive peers from global mesh registry.")
    return 0


def build_mesh_parser(subparsers: argparse._SubParsersAction) -> None:
    mesh_parser = subparsers.add_parser(
        "mesh",
        help="Decentralized Global Internet GPU & Swarm Mesh",
        description="Share, discover, and delegate compute over the global internet mesh.",
    )
    mesh_subparsers = mesh_parser.add_subparsers(dest="mesh_action")

    # join
    p_join = mesh_subparsers.add_parser("join", help="Join the global DePIN mesh and announce hardware")
    p_join.add_argument("--rendezvous", default="https://mesh.hermes.ai/rendezvous", help="Rendezvous discovery server URL")
    p_join.add_argument("--endpoint", default="mesh://direct", help="Publicly routable or relay endpoint")
    p_join.add_argument("--offer-gpu", action="store_true", help="Explicitly offer GPU compute")
    p_join.add_argument("--no-gpu", action="store_true", help="Join in client-only mode (do not offer GPU compute)")
    p_join.add_argument("--vram", type=int, default=None, help="Available VRAM in megabytes (auto-detected if omitted)")
    p_join.add_argument("--backend", default=None, choices=["cuda", "rocm", "mps", "vulkan", "cpu"], help="Hardware acceleration backend (auto-detected if omitted)")
    p_join.add_argument("--device", default=None, help="Human-readable accelerator device name (auto-detected if omitted)")
    p_join.set_defaults(func=mesh_join)

    # status
    p_status = mesh_subparsers.add_parser("status", help="Inspect local and global swarm cluster status")
    p_status.set_defaults(func=mesh_status)

    # peers
    p_peers = mesh_subparsers.add_parser("peers", help="List active mesh peers and GPU capabilities")
    p_peers.set_defaults(func=mesh_peers)

    # delegate
    p_delegate = mesh_subparsers.add_parser("delegate", help="Delegate a compute task or prompt across the global mesh")
    p_delegate.add_argument("--prompt", required=True, help="Prompt or task payload to execute")
    p_delegate.add_argument("--model", default="hermes-3-8b", help="Target model family")
    p_delegate.add_argument("--min-vram", type=int, default=8192, help="Minimum free VRAM required in MB")
    p_delegate.set_defaults(func=mesh_delegate)

    # prune
    p_prune = mesh_subparsers.add_parser("prune", help="Prune unreachable or timed-out mesh peers")
    p_prune.add_argument("--timeout", type=float, default=300.0, help="Heartbeat inactivity threshold in seconds")
    p_prune.set_defaults(func=mesh_prune)

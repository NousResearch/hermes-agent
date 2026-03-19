"""
COSMOS GPU Benchmark — Validates PRO node hardware specs.

Runs deterministic GPU compute benchmarks to verify:
  - GPU meets minimum VRAM (24GB+)
  - GPU meets minimum compute (80+ TFLOPS FP16)
  - Latency to core nodes is under 30ms

Usage:
    python -m cosmos.network.cosmos_benchmark
"""
import time
import asyncio
import socket
from typing import Dict, Optional, Tuple
from loguru import logger

from .cosmos_config import PRO_NODE_REQUIREMENTS, CORE_NODES


def benchmark_gpu() -> Dict:
    """
    Run GPU benchmark. Returns spec report.

    Measures:
      - VRAM total (GB)
      - FP16 TFLOPS (estimated via matrix multiply timing)
      - GPU model name
      - CUDA compute capability
    """
    result = {
        "gpu_model": "unknown",
        "vram_gb": 0,
        "tflops_fp16": 0.0,
        "compute_cap": "0.0",
        "passes": False,
        "reason": "",
    }

    try:
        import torch
        if not torch.cuda.is_available():
            result["reason"] = "No CUDA GPU available"
            return result

        props = torch.cuda.get_device_properties(0)
        result["gpu_model"] = torch.cuda.get_device_name(0)
        result["vram_gb"] = props.total_mem / (1024 ** 3)
        result["compute_cap"] = f"{props.major}.{props.minor}"

        # Estimate FP16 TFLOPS via timed matrix multiply
        size = 4096
        a = torch.randn(size, size, device="cuda", dtype=torch.float16)
        b = torch.randn(size, size, device="cuda", dtype=torch.float16)

        # Warmup
        for _ in range(3):
            torch.mm(a, b)
        torch.cuda.synchronize()

        # Benchmark
        runs = 10
        start = time.perf_counter()
        for _ in range(runs):
            torch.mm(a, b)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # FLOPS = 2 * N^3 per matmul (multiply-add)
        flops_per_run = 2 * (size ** 3)
        total_flops = flops_per_run * runs
        tflops = (total_flops / elapsed) / 1e12
        result["tflops_fp16"] = round(tflops, 2)

        # Check requirements
        reqs = PRO_NODE_REQUIREMENTS
        if result["vram_gb"] < reqs["min_vram_gb"]:
            result["reason"] = f"VRAM {result['vram_gb']:.1f}GB < {reqs['min_vram_gb']}GB minimum"
        elif result["tflops_fp16"] < reqs["min_tflops_fp16"]:
            result["reason"] = f"Compute {result['tflops_fp16']:.1f} TFLOPS < {reqs['min_tflops_fp16']} minimum"
        else:
            result["passes"] = True

    except Exception as e:
        result["reason"] = str(e)

    return result


async def measure_latency(host: str, port: int, samples: int = 10) -> float:
    """
    Measure TCP connection latency to a host.
    Returns average round-trip time in milliseconds.
    """
    times = []

    for _ in range(samples):
        start = time.perf_counter()
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=5.0,
            )
            elapsed = (time.perf_counter() - start) * 1000  # ms
            writer.close()
            await writer.wait_closed()
            times.append(elapsed)
        except (ConnectionError, asyncio.TimeoutError, OSError):
            times.append(9999.0)

        await asyncio.sleep(0.1)

    return sum(times) / len(times) if times else 9999.0


async def run_full_benchmark() -> Dict:
    """
    Run complete PRO node qualification benchmark.

    Returns:
        Dict with gpu_report, latency_results, and overall pass/fail
    """
    logger.info("Starting COSMOS PRO node benchmark...")

    # GPU benchmark
    gpu_report = await asyncio.get_event_loop().run_in_executor(None, benchmark_gpu)
    logger.info(f"GPU: {gpu_report['gpu_model']} | {gpu_report['vram_gb']:.1f}GB VRAM | {gpu_report['tflops_fp16']:.1f} TFLOPS")

    # Latency to core nodes
    latency_results = {}
    for name, cfg in CORE_NODES.items():
        lat = await measure_latency(cfg.host, cfg.port)
        latency_results[name] = round(lat, 2)
        logger.info(f"Latency to {name}: {lat:.2f}ms")

    best_latency = min(latency_results.values()) if latency_results else 9999.0

    # Overall pass/fail
    passes = gpu_report["passes"] and best_latency <= PRO_NODE_REQUIREMENTS["max_latency_ms"]
    reason = ""
    if not gpu_report["passes"]:
        reason = gpu_report["reason"]
    elif best_latency > PRO_NODE_REQUIREMENTS["max_latency_ms"]:
        reason = f"Latency {best_latency:.1f}ms > {PRO_NODE_REQUIREMENTS['max_latency_ms']}ms"

    return {
        "gpu": gpu_report,
        "latency": latency_results,
        "best_latency_ms": best_latency,
        "passes": passes,
        "reason": reason,
    }


if __name__ == "__main__":
    import json
    result = asyncio.run(run_full_benchmark())
    print(json.dumps(result, indent=2))
    if result["passes"]:
        print("\nBenchmark PASSED — this node qualifies for COSMOS PRO.")
    else:
        print(f"\nBenchmark FAILED: {result['reason']}")

"""PTY Bridge to WebSocket Loop Benchmark.

This script benchmarks the CPU utilization (via loop iterations) and worst-case
data detection latency of various sleep policies (fixed sleep vs exponential backoff).

Why we use a Mock/Simulated PTY Bridge & WebSockets here:
1. Windows Platform Compatibility: Spawning a real POSIX PtyBridge requires Unix-only
   APIs (fcntl, termios, ptyprocess) and raises PtyUnavailableError on native Windows.
   Mocking the bridge allows the benchmark to run natively on Windows dev environments.
2. Deterministic Latency Measurement: Testing against a real active subprocess introduces
   thread scheduling jitter and CPU contention. Mocking the timing timeline allows us to
   trigger data availability at the exact microsecond the reader task goes to sleep,
   yielding repeatable and mathematically precise worst-case latency figures.
"""

import asyncio
import time
import concurrent.futures
import sys
import random

class MockBridge:
    def __init__(self, trigger_time: float = 999999.0, read_timeout: float = 0.2, mock_block: bool = False):
        self.trigger_time = trigger_time
        self.read_timeout = read_timeout
        self.mock_block = mock_block
        self.data_returned = False
        self.stop_requested = False

    def read(self, timeout=0.2):
        if self.stop_requested:
            return None
        
        now = time.perf_counter()
        if now >= self.trigger_time:
            if not self.data_returned:
                self.data_returned = True
                return b"data"
            else:
                return None
        
        if self.mock_block:
            # Simulate blocking select
            time_to_wait = min(timeout, self.trigger_time - now)
            if time_to_wait > 0:
                time.sleep(time_to_wait)
                # Check again after sleeping
                now_after = time.perf_counter()
                if now_after >= self.trigger_time:
                    if not self.data_returned:
                        self.data_returned = True
                        return b"data"
                    else:
                        return None
            return b""
        else:
            # Immediate return (simulates select returning immediately or non-blocking)
            return b""

class MockWebSocket:
    def __init__(self):
        self.detected_time = None

    async def send_bytes(self, chunk: bytes):
        if chunk == b"data" and self.detected_time is None:
            self.detected_time = time.perf_counter()

# Loop A: Fixed sleep
async def run_fixed_loop(bridge, ws, executor, sleep_duration=0.05, read_timeout=0.2, custom_sleep_fn=None):
    loop = asyncio.get_running_loop()
    iterations = 0
    idle_iterations = 0
    
    while True:
        iterations += 1
        chunk = await loop.run_in_executor(executor, bridge.read, read_timeout)
        if chunk is None:
            break
        if not chunk:
            idle_iterations += 1
            if custom_sleep_fn:
                await custom_sleep_fn(sleep_duration, iterations)
            else:
                await asyncio.sleep(sleep_duration)
            continue
        try:
            await ws.send_bytes(chunk)
        except Exception:
            break
    return iterations, idle_iterations

# Loop B: Backoff sleep
async def run_backoff_loop(bridge, ws, executor, start_sleep=0.005, cap_sleep=0.5, read_timeout=0.2, custom_sleep_fn=None):
    loop = asyncio.get_running_loop()
    iterations = 0
    idle_iterations = 0
    backoff = start_sleep
    
    while True:
        iterations += 1
        chunk = await loop.run_in_executor(executor, bridge.read, read_timeout)
        if chunk is None:
            break
        if not chunk:
            idle_iterations += 1
            if custom_sleep_fn:
                await custom_sleep_fn(backoff, iterations)
            else:
                await asyncio.sleep(backoff)
            backoff = min(cap_sleep, backoff * 2)
            continue
        backoff = start_sleep
        try:
            await ws.send_bytes(chunk)
        except Exception:
            break
    return iterations, idle_iterations

async def stop_after_delay(bridge, delay):
    await asyncio.sleep(delay)
    bridge.stop_requested = True

async def run_cpu_test(loop_type, mock_block, cap=0.5, sleep_duration=0.05, start_sleep=0.005):
    bridge = MockBridge(trigger_time=999999.0, read_timeout=0.2, mock_block=mock_block)
    ws = MockWebSocket()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        # Start background stopper
        stop_task = asyncio.create_task(stop_after_delay(bridge, 5.0))
        
        # Measure CPU time of the process
        cpu_start = time.process_time()
        real_start = time.perf_counter()
        
        if loop_type == "fixed":
            iters, idle_iters = await run_fixed_loop(bridge, ws, executor, sleep_duration=sleep_duration, read_timeout=0.2)
        else:
            iters, idle_iters = await run_backoff_loop(bridge, ws, executor, start_sleep=start_sleep, cap_sleep=cap, read_timeout=0.2)
            
        cpu_end = time.process_time()
        real_end = time.perf_counter()
        
        await stop_task
        
        cpu_used = cpu_end - cpu_start
        real_used = real_end - real_start
        return iters, idle_iters, cpu_used, real_used

async def run_latency_test(loop_type, mock_block, cap=0.5, sleep_duration=0.05, start_sleep=0.005):
    bridge = MockBridge(trigger_time=999999.0, read_timeout=0.2, mock_block=mock_block)
    ws = MockWebSocket()
    
    trigger_now = False
    
    async def custom_sleep_fn(delay, current_iteration):
        nonlocal trigger_now
        # For backoff loop, trigger when we hit the cap (or near it)
        # For fixed loop, trigger on the 5th iteration
        if loop_type == "backoff":
            if delay >= cap - 1e-9:
                trigger_now = True
        else:
            if current_iteration >= 5:
                trigger_now = True
                
        if trigger_now:
            bridge.trigger_time = time.perf_counter()
            trigger_now = False
            
        await asyncio.sleep(delay)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        if loop_type == "fixed":
            await run_fixed_loop(bridge, ws, executor, sleep_duration=sleep_duration, read_timeout=0.2, custom_sleep_fn=custom_sleep_fn)
        else:
            await run_backoff_loop(bridge, ws, executor, start_sleep=start_sleep, cap_sleep=cap, read_timeout=0.2, custom_sleep_fn=custom_sleep_fn)
            
    latency = ws.detected_time - bridge.trigger_time if ws.detected_time and bridge.trigger_time else 0.0
    return latency

async def main():
    print("======================================================================")
    print("PTY Bridge to WS Loop Benchmark")
    print("======================================================================")
    print("Simulating a 5.0-second idle period, followed by a data chunk arrival.")
    print("Measuring CPU process time (ms), total iterations, and worst-case latency (ms).\n")

    caps = [0.05, 0.1, 0.2, 0.3, 0.5]
    start_sleep = 0.005
    fixed_sleep = 0.05

    for mock_block in [False, True]:
        mode_str = "Blocking Timeout Mode (select timeout = 0.2s)" if mock_block else "Immediate Return Mode (non-blocking / 0 timeout select)"
        print(f"\n--- Running Mode: {mode_str} ---")
        
        # 1. Run Baseline (Fixed Sleep 0.05s)
        print("Running Baseline (Fixed Sleep 0.05s)...")
        base_iters, base_idle_iters, base_cpu, base_real = await run_cpu_test("fixed", mock_block, sleep_duration=fixed_sleep)
        base_latency = await run_latency_test("fixed", mock_block, sleep_duration=fixed_sleep)
        
        # Windows process_time can be 0.0 if execution is very light; handle gracefully
        base_cpu_ms = base_cpu * 1000.0
        base_latency_ms = base_latency * 1000.0
        
        print(f"Baseline Results: Iterations={base_iters}, CPU Time={base_cpu_ms:.1f}ms, Worst-case Latency={base_latency_ms:.1f}ms")
        
        # 2. Run Backoff Caps
        results = []
        for cap in caps:
            print(f"Running Backoff (start={start_sleep}s, cap={cap}s)...")
            iters, idle_iters, cpu_used, real_used = await run_cpu_test("backoff", mock_block, cap=cap, start_sleep=start_sleep)
            latency = await run_latency_test("backoff", mock_block, cap=cap, start_sleep=start_sleep)
            
            cpu_used_ms = cpu_used * 1000.0
            latency_ms = latency * 1000.0
            
            # Calculations
            iter_reduction_pct = (base_iters - iters) / base_iters * 100.0
            
            # For CPU time, because of Windows timer coarseness, we also compare iteration reduction
            # which is a direct and noise-free indicator of loop CPU overhead.
            cpu_reduction_pct = 0.0
            if base_cpu > 0:
                cpu_reduction_pct = (base_cpu - cpu_used) / base_cpu * 100.0
            else:
                # Fallback to iteration reduction if process_time is too coarse/0
                cpu_reduction_pct = iter_reduction_pct
                
            latency_increase_ms = latency_ms - base_latency_ms
            
            results.append({
                "cap": cap,
                "iterations": iters,
                "cpu_ms": cpu_used_ms,
                "latency_ms": latency_ms,
                "iter_reduction_pct": iter_reduction_pct,
                "cpu_reduction_pct": cpu_reduction_pct,
                "latency_increase_ms": latency_increase_ms
            })
            
        # 3. Print Comparison Table
        print("\n| Configuration | Iterations | Iter. Reduction (%) | CPU Time (ms) | Est. CPU Reduction (%) | Worst-case Latency (ms) | Latency Increase (ms) |")
        print("|---|---|---|---|---|---|---|")
        print(f"| Fixed Sleep (0.05s) [Baseline] | {base_iters} | 0.0% | {base_cpu_ms:.1f} | 0.0% | {base_latency_ms:.1f} | 0.0 |")
        for r in results:
            print(f"| Backoff (cap={r['cap']}s) | {r['iterations']} | {r['iter_reduction_pct']:.1f}% | {r['cpu_ms']:.1f} | {r['cpu_reduction_pct']:.1f}% | {r['latency_ms']:.1f} | {r['latency_increase_ms']:.1f} |")

if __name__ == "__main__":
    asyncio.run(main())

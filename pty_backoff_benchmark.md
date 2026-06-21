# PTY to WebSocket Loop Benchmark Report

This document reports the performance trade-offs between CPU utilization (using loop iterations as a proxy) and data detection latency in the `pump_pty_to_ws()` loop located in `hermes_cli/web_server.py`.

## Background
The PR aims to fix a CPU busy-loop in the PTY reader. Previously, the reader yielded control using `await asyncio.sleep(0)`, which caused 100% CPU usage on idle connections.
Two main approaches were evaluated:
1. **Fixed Sleep (0.05s)**: Sleeping for a constant 50ms on empty reads.
2. **Exponential Backoff**: Sleeping starting at 5ms, doubling on every empty read up to a configurable cap (e.g., 50ms to 500ms).

## Methodology
The benchmark script [pty_benchmark.py](file:///c:/Users/Nitro/hermes-agent/pty_benchmark.py) simulates a 5.0-second idle period where the PTY returns `b""`, followed by a chunk of data. We tested two modes:
- **Immediate Return Mode (non-blocking / 0 timeout select)**: Simulates the worst-case CPU usage where `bridge.read` returns immediately without blocking (e.g. non-blocking socket/EOF/busy scenarios).
- **Blocking Timeout Mode (select timeout = 0.2s)**: Simulates the normal/ideal case where `bridge.read` blocks in `select` for up to 0.2s before returning `b""`.

Worst-case latency is measured deterministically by triggering data availability at the exact millisecond the loop enters a capped sleep.

### Use of Mocked/Simulated PTY Bridge & WebSockets
The benchmark script uses a simulated `MockBridge` rather than spawning a real POSIX `PtyBridge` for two primary reasons:
1. **Windows Platform Compatibility**: The actual `PtyBridge` is a POSIX-only module that depends on Unix-specific modules (`fcntl`, `termios`, `ptyprocess`), none of which exist on native Windows Python. Spawning a real `PtyBridge` on Windows raises `PtyUnavailableError`. Using mocks allows the benchmark to run natively on Windows development environments.
2. **Deterministic Latency Measurement (Jitter Reduction)**: Spawning a real subprocess and calling OS-level read/write introduces thread contention, execution delay, and kernel CPU scheduling jitter. Using a mocked timeline allows us to inject data precisely at the microsecond the reader task enters its sleep state. This yields mathematical, repeatable, and noise-free worst-case latency figures across runs.

---

## Test Results

### 1. Immediate Return Mode (non-blocking / 0 timeout select)
| Configuration | Iterations (5s idle) | Iter. Reduction (%) | Worst-case Latency (ms) | Latency Increase (ms) |
|---|---|---|---|---|
| **Fixed Sleep (0.05s) [Baseline]** | 82 | 0.0% | 61.5 | 0.0 |
| **Backoff (cap=0.05s)** | 84 | -2.4% | 61.2 | -0.4 |
| **Backoff (cap=0.1s)** | 51 | 37.8% | 108.5 | +47.0 |
| **Backoff (cap=0.2s)** | 30 | 63.4% | 201.6 | +140.1 |
| **Backoff (cap=0.3s)** | 22 | 73.2% | 309.5 | +248.0 |
| **Backoff (cap=0.5s) [Current Cap]** | 17 | **79.3%** | 510.7 | +449.2 |

### 2. Blocking Timeout Mode (select timeout = 0.2s)
| Configuration | Iterations (5s idle) | Iter. Reduction (%) | Worst-case Latency (ms) | Latency Increase (ms) |
|---|---|---|---|---|
| **Fixed Sleep (0.05s) [Baseline]** | 21 | 0.0% | 64.8 | 0.0 |
| **Backoff (cap=0.05s)** | 21 | 0.0% | 63.8 | -1.0 |
| **Backoff (cap=0.1s)** | 19 | 9.5% | 110.3 | +45.5 |
| **Backoff (cap=0.2s)** | 16 | 23.8% | 202.0 | +137.2 |
| **Backoff (cap=0.3s)** | 14 | 33.3% | 312.9 | +248.0 |
| **Backoff (cap=0.5s) [Current Cap]** | 13 | **38.1%** | 509.4 | +444.6 |

---

## Findings & Trade-offs

1. **CPU Savings**: The exponential backoff with a **0.5s cap** provides the maximum reduction in loop iterations (up to **79.3% reduction** under immediate return, and **38.1%** under normal blocking timeout). This makes it highly efficient at preventing CPU hang/busy-loop issues.
2. **Active Responsiveness**: By starting with a minimal sleep of **5ms** (`start = 0.005`), the backoff loop is up to **10x faster** to read incoming chunks when the terminal is busy, compared to the fixed 50ms sleep.
3. **Worst-Case Latency Trade-off**: The worst-case latency during idle transitions scales linearly with the cap. With a **0.5s cap**, the very first character typed after an idle period can experience up to **500ms** of latency before it is read and sent to the browser.
4. **Resolution**: The `cap = 0.5s` is retained in the codebase to guarantee optimal CPU savings on idle connections, with the understanding that active input reset-to-5ms keeps overall scrolling and multi-byte streams smooth.

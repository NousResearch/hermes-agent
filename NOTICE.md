# Third-Party Notices

Hermes Agent is licensed under Apache-2.0. This file records third-party
sources whose code has been ported (逐行翻译, line-by-line translation)
into this repository, and their original licenses.

Ported code retains a per-file SPDX header identifying the upstream file
and license. Full upstream license texts are stored under `LICENSES/`.

---

## BaiLongma (MIT License)

- **Upstream:** https://github.com/xiaoyuanda666-ship-it/BaiLongma
- **Copyright:** © 2026 xiaoyuanda666-ship-it
- **License:** MIT (see `LICENSES/BaiLongma-MIT.txt`)

The BaiLongma project (a Node.js/Electron desktop AI agent) contributed
architectural designs and reference implementations that were ported to
Python for Hermes. All ported files carry an `Adapted from BaiLongma`
SPDX header pointing at the original `src/**/*.js` source.

### Ported modules

| Hermes file | Upstream source | Notes |
| --- | --- | --- |
| `agent/heartbeat.py` | `src/runtime/consciousness-loop.js` | Autonomous L2 tick loop |
| `agent/heartbeat_policy.py` | `src/runtime/tick-policy.js` | Heartbeat prompt policy |
| `agent/keywords.py` | `src/memory/keywords.js` | Zero-dep CN/EN n-gram extraction |
| `agent/threads.py` | `src/memory/threads.js` | Thread model: read-time temperature, no stack |
| `agent/thread_classifier.py` | `src/memory/thread-classifier.js` | LLM arbiter for weak-signal thread merges |
| `agent/memory_consolidator.py` | `src/memory/consolidator.js` + `src/memory/consolidation-loop.js` | Round-robin memory consolidation (merge / downgrade / skip); dependency-injected store + LLM |
| `agent/self_evolution.py` | `src/memory/self-evolution.js` | Self-evolution ledger for actionable memories (procedure / constraint / policy / lesson); prompt renderer split off from injection (see file docstring for the caching-hazard note) |

(Additional entries will be appended as more BaiLongma modules are
ported.)

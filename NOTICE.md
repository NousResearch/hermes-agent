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
| `agent/prefetch.py` | `src/prefetch/runner.js` | Background prefetch runner (framework only; no built-in tasks) |

(Additional entries will be appended as more BaiLongma modules are
ported.)

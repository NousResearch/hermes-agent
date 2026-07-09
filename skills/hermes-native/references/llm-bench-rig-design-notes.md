# llm-bench-rig Reference Design Notes

Source: `C:\æ\hermes-fork\llm-bench-rig`

## Reference Patterns to Adopt

- `lib/progress.Progress` atomic `_tmp + rename` state machine: `init` → `done` / `error`
- `gate_and_run.sh` native hard gate: server health check → tool-call probe → run-only-if-pass
- `bench.py` layered run flow: speed then quality, with VRAM sampling during execution
- `export.py` HTML/PNG card pipeline from structured results

## Hermes Native Delta

| axis | llm-bench-rig | Hermes Native target |
|---|---|---|
| host | standalone Python bench + bash service | VS Code webview + local bridge |
| concurrency | queued model benchmarks | single runtime state per surface |
| telemetry | JSON progress files | webview log + VS Code status |
| gateway | HTTP health + tool-call gate | WebGPU detect + engine init gate |
| export | Jinja HTML + Playwright PNG | VS Code webview surface only |
| failure mode | `progress.fail(error)` + skip harness | `ENGINE_FAILED` / `INFERENCE_FAILED` in webview |

## Adoption Constraints

- Do not import external bench toolchains into the VS Code surface.
- Reuse only the gating and progress semantics, not the benchmark binaries.
- Keep the existing single-instance Hermes Native webview as the only surface.

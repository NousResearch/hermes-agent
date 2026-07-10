---
name: cuda
description: "Use when building NVIDIA/CUDA compute surfaces for Hermes — the >_n: NemoClaw operator grammar, nvidia-smi host bridges, GPU telemetry tiles, and local CUDA toolchain (nvcc) workflows. Covers the VS Code webview file:// pitfall, WebGPU fallback, and PyTorch/CuPy probe patterns. Pioneer skill: Hermes ships no CUDA skill; this defines the surface."
version: 1.0.0
author: Yæl Méndez
license: MIT
metadata:
  hermes:
    tags: [cuda, nvidia, gpu, nemo-claw, compute-surface, webview, nvidia-smi, operator-grammar, sovereign-mesh]
    related_skills: [sovereign-vscode-surface-dev, hermes-surface-development, hermes-operator-grammar, optimizing-attention-flash]
---

# CUDA — NVIDIA Compute Surface for Hermes

## Overview

Hermes ships **no CUDA/NVIDIA skill** — this is the pioneer. It formalizes the `>_n:` operator
slot (`>_n: → nvidia (NemoClaw) → compute` from AGENTS.md) as a concrete, agent-addressable
compute surface: a sovereign GPU node you can mount, probe, and drive from a VS Code webview.

The pattern has three layers:
1. **Grammar** — `>_n:` routes to NVIDIA compute; the host command is `remoteUse.nvidia`.
2. **Surface** — a single gold/void-independent, green-NVIDIA-themed HTML tile dashboard
   (`nvidia-tiles.html`) mounted in the local surface hub (Computation tab).
3. **Bridge** — a VS Code extension command that shells out to `nvidia-smi` and streams JSON
   back to the webview via `postMessage`. This is the only way to read real GPU state inside a
   webview (browser `fetch` to a local port works only if a bridge is running).

Victus (HP Victus 15-fa2xxx) is the reference node: i5-13420H, RTX 3050 6GB, **nvcc 13.3
installed**, driver 592.27. The skill is written so any NVIDIA node drops in by changing the
`nvidia-smi` query.

## When to Use

- Building or extending a `>_n:` / NemoClaw compute surface for Hermes.
- Wiring `nvidia-smi` GPU telemetry into a VS Code webview (or any local HTML surface).
- Probing PyTorch/CuPy availability or running a CUDA sanity bench on a local node.
- Asked to "map machine capabilities" — the GPU tile is part of the Victus capability map.
- Don't use for: cloud GPU clusters (see `lambda-labs-gpu-cloud`, `modal-serverless-gpu`),
  pure ML training recipes (see `mlops/training/*`), or attention optimization (`optimizing-attention-flash`).

## Quick start

Mount the surface from the hub, or open it directly:

```bash
# in VS Code command palette:
Remote Use: NVIDIA Compute Surface     # -> remoteUse.nvidia
# or from the surface hub: Computation tab -> NVIDIA Tiles
```

The tile auto-refreshes GPU telemetry on mount by posting `{command:'remoteUse.nvidia', action:'smi'}`
to the host. The host runs:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu,temperature.gpu \
  --format=csv,noheader,nounits
```

and posts back `{command:'remoteUse.nvidia', action:'smi', data:{ name, driver, memory_used, memory_total, utilization_gpu, temperature_gpu }}`
(`memory_used`/`memory_total` are bytes; the surface divides by 1024³ for GB).

## Workflow 1: Add a GPU telemetry tile to a surface hub

Checklist:
- [ ] Step 1: Confirm `nvidia-smi` works on the node (it must be on PATH)
- [ ] Step 2: Add the tile HTML + a `refreshGpu()` that posts to the host (or fetches a bridge)
- [ ] Step 3: Add the host bridge in `extension.ts` (`remoteUse.nvidia` → `nvidia-smi`)
- [ ] Step 4: Register the command in `package.json` + `context.subscriptions`
- [ ] Step 5: Render in headless Edge + verify tiles, then compile + pytest

**Step 1 — verify the data shape (do this BEFORE writing the parser):**
```bash
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu,temperature.gpu \
  --format=csv,noheader,nounits
# -> NVIDIA GeForce RTX 3050 6GB Laptop GPU, 592.27, 761, 6144, 0, 47
#    6 comma-separated fields; split on ',' and trim. NaN guard on [N/A].
```

**Step 3 — host bridge (TS, in `vscode-remote-use/src/extension.ts`):**
```ts
const nvidiaCmd = vscode.commands.registerCommand('remoteUse.nvidia', async () => {
  const repo = getHermesRepoPath();
  const hub = `${repo}/templates/surfaces/nvidia-tiles.html`;
  const panel = vscode.window.createWebviewPanel('remoteUseNvidia', 'Remote Use: NVIDIA Compute Surface',
    vscode.ViewColumn.One, { enableScripts: true, retainContextWhenHidden: true,
      localResourceRoots: [vscode.Uri.file(`${repo}/templates/surfaces`)] });
  panel.webview.html = require('fs').readFileSync(hub, 'utf8');
  panel.webview.onDidReceiveMessage(async (message: any) => {
    const { exec } = require('child_process');
    const run = (cmd: string) => new Promise<string>((res) =>
      exec(cmd, { maxBuffer: 1024*1024 }, (e:any,o:string,err:string)=>res((e?err:o)||'')));
    if (message?.action === 'smi') {
      const out = await run('nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits');
      const p = out.split(',').map((s:string)=>s.trim());
      if (p.length >= 6) panel.webview.postMessage({ command:'remoteUse.nvidia', action:'smi', data: {
        name:p[0], driver:p[1], memory_used:parseFloat(p[2])*1048576, memory_total:parseFloat(p[3])*1048576,
        utilization_gpu:parseFloat(p[4]), temperature_gpu:parseFloat(p[5]) }});
    }
  });
});
```

## Workflow 2: Probe the local CUDA stack

```bash
# toolkit
nvcc --version | tail -2                 # -> release 13.3
# PyTorch / CuPy (host-side, not webview)
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import cupy; print(cupy.cuda.runtime.runtimeGetVersion())"
```

Bridge the torch probe into the surface the same way as `smi` (see `references/host-bridge.md`).

## Workflow 3: WebGPU fallback (in-browser, no host)

When the surface runs in a plain browser (no VS Code host), use the client GPU directly:
```js
if (navigator.gpu) {
  const adapter = await navigator.gpu.requestAdapter();
  if (adapter) { /* WebGPU available — WebLLM/Qwen3 path */ }
}
```

This is how Hermes Native (WebLLM/WebGPU, Qwen3-0.6B) runs without CUDA — distinct from the
`nvidia-smi` host bridge. Keep both: WebGPU for in-browser inference, CUDA host bridge for
telemetry/bench on the real GPU.

## Common Pitfalls

1. **VS Code webviews block `file://` iframes.** A surface that mounts another local HTML via
   `src="file:///..."` renders in Edge but is **blank inside VS Code**. Fix: strip `file://`
   in the served HTML and let the host convert paths with `panel.webview.asWebviewUri(...)`.
   (Same fix applied to the whole surface hub — see `sovereign-vscode-surface-dev`.)

2. **`[N/A]` in `nvidia-smi --query-compute-apps=used_memory`.** `parseFloat('[N/A]')` is `NaN`
   and renders as `NaN MB`. Guard with `isFinite(x) ? (x/1048576).toFixed(0)+'MB' : 'n/a'`.

3. **Field count drift.** `nvidia-smi --query-gpu=...` returns exactly the columns you ask for,
   comma-separated, no header (with `noheader`). Always assert `parts.length >= N` before indexing.

4. **Host command not registered.** A webview `postMessage` with an unregistered command silently
   no-ops. Add the command to BOTH `package.json` `contributes.commands` AND
   `context.subscriptions.push(...)` in `extension.ts` or the surface stays dead.

5. **MSVC linker not on PATH (Windows).** `nvcc`/CUDA compile fine, but native `.exe`/Tauri
   linking needs `vcvars64.bat` first. The linker exists via VS 2022 BuildTools — just not on PATH.
   (Victus: BuildTools 14.44.35207 at `C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/`.)

## Verification Checklist

- [ ] `nvidia-smi --query-gpu=...` returns the exact field count the parser expects
- [ ] Host bridge posts `{command:'remoteUse.nvidia', action:'smi', data:{...}}` with byte-sized memory
- [ ] Surface renders in headless Edge (all tiles + dock visible, green NVIDIA theme)
- [ ] `npm run compile` (tsc) exits 0; `package.json` valid
- [ ] `pytest secret_source_bridge/tests.py tests/hermes_runtime/test_computer_use.py` green
- [ ] No secrets hardcoded; `[N/A]` memory renders as `n/a`, not `NaN`
- [ ] Grammar wired: `>_n:` → `remoteUse.nvidia` → `pc://mesh/victus/local`

## References

- `references/host-bridge.md` — full `extension.ts` `remoteUse.nvidia` handler (smi + procs + torch)
- `templates/nvidia-tiles.html` — the shipped green-NVIDIA 6-tile surface (GPU / Compute Path / Processes / Bench / Routes / Envelope)
- `../hermes-native/references/victus-capability-map.md` — verified Victus hardware/toolchain facts

## Resources

- NVIDIA `nvidia-smi` docs: https://nvidia.github.io/nvidia-settings/
- CUDA toolkit (nvcc): https://developer.nvidia.com/cuda-toolkit
- WebGPU: https://developer.mozilla.org/docs/Web/API/WebGPU_API
- Hermes operator grammar: `hermes-operator-grammar` skill

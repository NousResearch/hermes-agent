---
name: gradio-app-development
description: Build and deploy JIT Gradio apps, daggr loops, and Spaces.
version: 0.1.0
author: Jonathan Korstad (Jpalmer95), Hermes Agent
license: Apache-2.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [gradio, daggr, huggingface, spaces, zerogpu, workflows, mcp, mlops]
    related_skills: [comfyui, modal]
---

# Gradio JIT Application & Modular Pipeline Skill

Build, verify, and deploy **Just-In-Time (JIT) Gradio 6 applications** — locally or as Hugging Face Spaces — both **standalone** and **with modular `daggr` or `gr.Workflow` pipelines** wired in as callable steps, batch loops, or embedded visual DAG canvases.

## When to Use

- User needs a custom interactive web tool or API on the fly (image/video editing, 3D mesh inspection/conversion, TTS/music synthesis, data/code transformation, multi-step AI workflows).
- Wrapping Python functions, local GPU models, or remote HF Spaces into a human UI + agent-callable MCP server (`mcp_server=True`).
- Embedding or looping a **`daggr` pipeline** (`Graph`, `GradioNode`, `FnNode`, `InferenceNode`, `InputNode`) inside a broader `gr.Blocks` or `gradio.Server` application.
- Deploying a local Gradio app or `daggr` workflow to a Hugging Face Space using `HF_TOKEN` (CPU, ZeroGPU, or dedicated GPU).
- **Don't use for:** static HTML-only sites with no Python backend (use static GitHub Pages deployment instead) or pure headless cron scripts that never need a UI, API, or DAG canvas.

## Prerequisites

- Python `>= 3.10` with Gradio 6 and companion libraries:
  `terminal(command="pip install 'gradio>=6.27.0' 'daggr>=0.8.0' 'gradio_client>=1.0.0' 'huggingface_hub>=1.0.0' pillow", timeout=180)`
- Optional `HF_TOKEN` environment variable (from https://huggingface.co/settings/tokens) for calling ZeroGPU Spaces or deploying to Hugging Face Spaces.

## Authoritative Live Documentation

When unsure of a Gradio 6 component signature, ZeroGPU rule, or `daggr` API, fetch the live docs with `web_extract`:

### Gradio 6
- **Docs & Component Reference:** https://www.gradio.app/docs
- **Guides Index:** https://www.gradio.app/guides
- **Gradio 6 Migration Guide (breaking changes):** https://www.gradio.app/guides/gradio-6-migration-guide
- **Blocks & Event Listeners:** https://www.gradio.app/guides/blocks-and-event-listeners
- **Controlling Layout (Row, Column, Tab, Accordion, Group):** https://www.gradio.app/guides/controlling-layout
- **State in Blocks (`gr.State`):** https://www.gradio.app/guides/state-in-blocks
- **Dynamic Apps with `@gr.render`:** https://www.gradio.app/guides/dynamic-apps-with-render-decorator
- **Custom HTML & JS Components (`gr.HTML`):** https://www.gradio.app/guides/custom-HTML-components
- **Server Mode (`gradio.Server` FastAPI + custom SPA):** https://www.gradio.app/guides/server-mode
- **Native `gr.Workflow` (Node-based DAG canvas in Gradio 6):** https://www.gradio.app/guides/workflows
- **Building an MCP Server (`mcp_server=True`, `@app.mcp.tool`):** https://www.gradio.app/guides/building-mcp-server-with-gradio
- **Python `gradio_client` Guide:** https://www.gradio.app/guides/getting-started-with-the-python-client
- **JavaScript `@gradio/client` Guide:** https://www.gradio.app/guides/getting-started-with-the-js-client
- **3D Model Component (`gr.Model3D`):** https://www.gradio.app/guides/how-to-use-3D-model-component
- **Streaming Outputs & Progress Bars:** https://www.gradio.app/guides/streaming-outputs · https://www.gradio.app/guides/progress-bars
- **Gradio Skills CLI (`gradio skills add`):** https://www.gradio.app/guides/Gradio-skills-for-ai-coding-assistants

### Hugging Face Spaces & `huggingface_hub`
- **Gradio Spaces Guide:** https://huggingface.co/docs/hub/spaces-sdks-gradio
- **Spaces Configuration Reference (`README.md` YAML frontmatter):** https://huggingface.co/docs/hub/spaces-config-reference
- **ZeroGPU Dynamic GPU Allocation (`@spaces.GPU`):** https://huggingface.co/docs/hub/spaces-zerogpu
- **Manage Spaces Programmatically (`HfApi`, hardware, secrets, ZeroGPU quota):** https://huggingface.co/docs/huggingface_hub/guides/manage-spaces
- **Spaces GPU Hardware Tiers:** https://huggingface.co/docs/hub/spaces-gpus
- **Spaces Sign-In with HF (`hf_oauth: true`):** https://huggingface.co/docs/hub/spaces-oauth

### `daggr` & Companion Ecosystem
- **`daggr` Repository & Docs:** https://github.com/gradio-app/daggr *(Note: upstream archived `daggr` in Aug 2026 in favor of `gr.Workflow`, while `daggr 0.8.0` remains actively used for programmatic code-first DAGs and `SequentialExecutor` sub-pipelines)*
- **Pre-Built `daggr` Pipeline Templates:** https://github.com/Jpalmer95/daggr-pipelines
- **Daggr Studio (Verified Bricks & Self-Healing Canvas):** https://github.com/Jpalmer95/daggr-studio · https://huggingface.co/spaces/jkorstad/daggr-studio

## Architecture Decision Tree

| Mode (`--template`) | When to Choose | Entrypoint / Stack |
|---|---|---|
| **1. Standalone Gradio App** (`blocks`) | Single-tool or multi-tab utility (image/video filter, 3D converter, audio synth, Python calculator/plotter) where a standard form/tab UI is clearest. | `gr.Blocks()` + `demo.launch(mcp_server=True)` |
| **2. Gradio App + Modular `daggr` Sub-Pipeline** (`daggr-modular`) | You want a clean, guided `gr.Blocks` UI (tabs, galleries, batch controls, custom state) that **calls or loops a `daggr.Graph`** under the hood via `SequentialExecutor` and optionally launches the `daggr` canvas (`--canvas`). | `gr.Blocks()` calling `SequentialExecutor(graph, hf_token=token)` inside callbacks or batch loops |
| **3. Hybrid Gradio UI + Live `daggr` Visual Canvas** (`daggr-hybrid`) | Users need both a high-level control UI **and** the interactive node-by-node `daggr` DAG canvas in the same process/Space to inspect or re-run individual steps. | `FastAPI` + `gr.mount_gradio_app` at `/app` + `DaggrServer(graph).app` at `/` |
| **4. Native Gradio 6 `gr.Workflow`** (`workflow`) | You want Gradio 6's built-in drag-and-drop canvas (`workflow.json` + `bind={...}`) with automatic dual App/Canvas toggle (`?ui=app` / `?ui=canvas`). | `gr.Workflow(graph="workflow.json", bind={...}).launch()` |

## How to Run

Use the bundled helper script `scripts/jit_gradio_space.py` via `terminal`:

```python
# 1. Inspect a remote Space API before wiring it
terminal(command="python3 scripts/jit_gradio_space.py inspect-space black-forest-labs/FLUX.1-schnell", timeout=60)

# 2. Scaffold a JIT app (templates: blocks | daggr-modular | daggr-hybrid | workflow)
terminal(command="python3 scripts/jit_gradio_space.py scaffold ./my-jit-tool --title 'Sprite & 3D Studio' --template daggr-modular", timeout=30)

# 3. Verify syntax, README YAML frontmatter, imports, and run a headless smoke test
terminal(command="python3 scripts/jit_gradio_space.py verify ./my-jit-tool", timeout=60)

# 4. Optional: Deploy to Hugging Face Spaces with HF_TOKEN
terminal(command="python3 scripts/jit_gradio_space.py deploy ./my-jit-tool --repo-id username/sprite-3d-studio --hardware cpu-basic --forward-hf-token", timeout=120)
```

## Quick Reference

| Command | Action |
|---|---|
| `python3 scripts/jit_gradio_space.py inspect-space <owner/space>` | Print live endpoints, required inputs, defaults, and return types |
| `python3 scripts/jit_gradio_space.py scaffold <dir> --template blocks` | Scaffold standalone Gradio 6 `gr.Blocks` + MCP server + ZeroGPU shim |
| `python3 scripts/jit_gradio_space.py scaffold <dir> --template daggr-modular` | Scaffold `gr.Blocks` app that runs/loops a modular `daggr.Graph` via `SequentialExecutor` |
| `python3 scripts/jit_gradio_space.py scaffold <dir> --template daggr-hybrid` | Scaffold hybrid FastAPI server mounting `gr.Blocks` at `/app` and `DaggrServer` at `/` |
| `python3 scripts/jit_gradio_space.py scaffold <dir> --template workflow` | Scaffold native Gradio 6 `gr.Workflow` app with `bind` + `edges` |
| `python3 scripts/jit_gradio_space.py verify <dir>` | AST check + HF README frontmatter check + module import + smoke execution |
| `python3 scripts/jit_gradio_space.py deploy <dir> --repo-id <id>` | Create/update HF Space, vendor local modules, set secrets/hardware, check runtime |

## Procedure

1. **Choose the architecture mode** from the decision tree (`blocks`, `daggr-modular`, `daggr-hybrid`, or `workflow`).
   - *Completion criterion:* Mode selected and target directory path chosen.
2. **Inspect any external Hugging Face Spaces** that the app or `daggr` nodes will call using `inspect-space <owner/space>`.
   - *Completion criterion:* Every remote Space's `api_name`, required parameter names, and return tuple/dict structure verified from live `view_api()` output.
3. **Scaffold the application** with `scripts/jit_gradio_space.py scaffold <dir> --template <mode> --title "<Title>"` and customize `app.py` and `requirements.txt` using `read_file` and `patch` / `write_file`.
   - *Completion criterion:* `app.py`, `requirements.txt`, and `README.md` exist on disk with typed functions, docstrings, explicit `api_name`s, and Gradio 6 `launch()` parameters.
4. **Verify locally** with `scripts/jit_gradio_space.py verify <dir>`.
   - *Completion criterion:* Command exits `0` with `[1/3] Syntax OK`, `[2/3] README.md HF Spaces YAML frontmatter OK`, and `[3/3] Module import OK` + smoke run output.
5. **Launch locally or deploy to Hugging Face Spaces** (when requested by the user and `HF_TOKEN` is available) using `scripts/jit_gradio_space.py deploy <dir> --repo-id <user/space>`.
   - *Completion criterion:* Local server responds on its port, or `deploy` prints the live `https://huggingface.co/spaces/<user>/<space>` URL and runtime stage (`BUILDING` / `RUNNING`).

## Domain Recipes & Modular `daggr` Looping

### Adding & Looping a Modular `daggr.Graph` Inside `gr.Blocks`
Construct a `daggr.Graph` (`InputNode`, `FnNode`, `GradioNode`, `InferenceNode`) and execute it inside any Gradio callback or batch loop via `daggr.executor.SequentialExecutor`:

```python
import os
import gradio as gr
from daggr import Graph, InputNode, FnNode
from daggr.executor import SequentialExecutor

def build_modular_pipeline() -> Graph:
    inputs = InputNode(
        name="Inputs",
        ports={
            "prompt": gr.Textbox(label="Prompt", value="stylized crystal sword"),
            "style": gr.Dropdown(["stylized", "pixel-art", "realistic"], value="stylized"),
        },
    )
    enricher = FnNode(
        fn=lambda prompt, style: f"{prompt}, {style} game asset, centered, clean silhouette",
        inputs={"prompt": inputs.prompt, "style": inputs.style},
        outputs={"enhanced_prompt": gr.Textbox(label="Enhanced Prompt")},
    )
    return Graph(name="Asset Prep Module", nodes=[inputs, enricher])

PIPELINE_GRAPH = build_modular_pipeline()

def run_pipeline_once(prompt: str, style: str) -> tuple[str, dict]:
    ex = SequentialExecutor(PIPELINE_GRAPH, hf_token=os.environ.get("HF_TOKEN") or None)
    order = PIPELINE_GRAPH.get_execution_order()
    for idx, node_name in enumerate(order):
        ex.execute_node(node_name, {"prompt": prompt, "style": style} if idx == 0 else None)
    enhanced = ex.results.get(order[-1], {}).get("enhanced_prompt", "")
    return enhanced, ex.results

def run_pipeline_batch(prompts_multiline: str, style: str, progress=gr.Progress()):
    items = [line.strip() for line in prompts_multiline.splitlines() if line.strip()]
    results = []
    for idx, item in enumerate(items):
        progress((idx, max(1, len(items))), desc=f"Running daggr ({idx+1}/{len(items)}): {item[:30]}")
        enhanced, trace = run_pipeline_once(item, style)
        results.append({"item": item, "enhanced_prompt": enhanced, "trace": trace})
        yield results  # Streams incremental updates to Gradio UI
```

See `references/daggr-in-gradio-patterns.md` for caching/resuming partial `ex.results` across edits and mounting `DaggrServer` alongside `gr.Blocks`. See `references/live-docs-and-api-cheatsheet.md` for Gradio 6 component shapes across image/video (`gr.ImageEditor`, `gr.Gallery`, `gr.Video`), 3D (`gr.Model3D`), audio (`gr.Audio`), and code/plots (`gr.Code`, `gr.Plot`, `gr.Dataframe`).

## Pitfalls

1. **Gradio 6 `Blocks` vs `launch()` parameters:** In Gradio 6, `theme=`, `css=`, `css_paths=`, `js=`, `head=`, and `head_paths=` belong on `demo.launch(...)`, **not** `gr.Blocks(...)`.
2. **Gradio 6 `show_api` removal:** `show_api` was removed in Gradio 6. On `launch()`, use `footer_links=["api", "gradio", "settings"]`. On event listeners (`.click()`), use `api_visibility="public" | "undocumented" | "private"`.
3. **`gradio_client` uses `token=`, not `hf_token=`:** In `gradio_client >= 1.0` / Gradio 6, pass `Client(space_id, token=os.getenv("HF_TOKEN"))`.
4. **File handling difference between `daggr` and `gr.Workflow`:**
   - In **`daggr`**, files pass between nodes as **file path strings** on disk. Multi-return Spaces return tuples `(orig, final)`, so set `postprocess=lambda _, final: final`.
   - In **Gradio 6 `gr.Workflow`**, bound Python `fn` nodes receive media as dicts (`{"path": ..., "url": ...}`) and must return `{"path": path, "url": f"/gradio_api/file={encode_file_path(path)}", "is_file": True}`.
5. **ZeroGPU import & model-loading order:** On ZeroGPU Spaces, `import spaces` **must** precede `import torch` or any CUDA initialization, and models must be moved `.to("cuda")` at **module scope** (not inside `@spaces.GPU`), or every call wastes GPU quota transferring weights.
6. **Token safety on public Spaces:** Never accept `hf_token` as an input parameter on queued `@app.api()` endpoints whose call history is publicly visible. Prefer `hf_oauth: true` in `README.md`, Space secrets (`os.environ["HF_TOKEN"]`), or non-queued FastAPI routes.
7. **Vendor local shared modules before uploading a Space:** Each HF Space repo is isolated. Pass `--vendor <path>` to `scripts/jit_gradio_space.py deploy` when your app imports local helper files outside its own directory.

## Verification

- [ ] `python3 scripts/jit_gradio_space.py verify <app_dir>` exits `0` (AST syntax, `README.md` HF YAML frontmatter, module import, and smoke test all pass).
- [ ] All remote HF Space endpoints wired in `GradioNode` or `gradio_client.Client` were verified with `inspect-space` and wrapped in `try/except` fallback logic.
- [ ] Gradio 6 styling args (`theme=`, `css=`, `js=`, `head=`) are passed to `demo.launch()`, not `gr.Blocks()`.
- [ ] If deployed to HF Spaces: `deploy` returns the Space URL and `get_space_runtime` confirms a valid build/run stage.

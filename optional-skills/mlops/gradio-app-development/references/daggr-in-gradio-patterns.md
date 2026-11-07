# Modular `daggr` Pipelines Inside Gradio Applications

This reference covers the four architectural ways to combine **`daggr`** (`Graph`, `GradioNode`, `FnNode`, `InferenceNode`, `InputNode`) with a **Gradio 6 application**, so you can plug `daggr` pipelines into any JIT app as modular components, loop them over batches, cache intermediate node outputs, or mount the live visual DAG canvas alongside your UI.

---

## 1. Calling a `daggr.Graph` as a Modular Step Inside `gr.Blocks`

Use `SequentialExecutor(graph, hf_token=token)` from `daggr.executor` to execute any `daggr.Graph` in-process inside a standard Gradio event callback.

### Key Rules for `SequentialExecutor`
1. **Input values go ONLY to the `InputNode`:** Pass the user's input dict (`{"prompt": "...", ...}`) when executing the `InputNode`, and pass `None` to all downstream nodes (they pull their inputs from upstream edges in `ex.results`).
2. **Do not hardcode `"Inputs"` as the node name:** `daggr` suffixes colliding `InputNode` names (`Inputs_1`). Detect `InputNode`s by `type(node).__name__ == "InputNode"` or by checking the first node in `graph.get_execution_order()`.
3. **`ex.results` is read/write:** After execution, `ex.results` holds `{node_name: {output_port: value}}`. You can store `ex.results` in `gr.State()` and restore it on a subsequent run to **skip already-computed upstream nodes**!

### Complete Pattern: Resumable Multi-Step Pipeline in `gr.Blocks`

```python
import os
import tempfile
from pathlib import Path
import gradio as gr
from daggr import Graph, InputNode, FnNode
from daggr.executor import SequentialExecutor
from PIL import Image, ImageOps


def build_sprite_pipeline() -> Graph:
    """A 3-step modular daggr pipeline: Input -> Concept Gen (Fn/Space) -> Postprocess."""
    inp = InputNode(
        name="Inputs",
        ports={
            "prompt": gr.Textbox(label="Prompt", value="fire ant warrior"),
            "border_px": gr.Slider(0, 32, value=8, step=2, label="Border Padding"),
        },
    )

    def synthesize_draft(prompt: str) -> str:
        # Can call a remote Space via gradio_client or generate locally
        tmp_dir = Path(tempfile.gettempdir()) / "jit_sprites"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        out_path = tmp_dir / f"draft_{abs(hash(prompt)) % 100000}.png"
        img = Image.new("RGBA", (256, 256), (40, 20, 80, 255))
        img.save(out_path)
        return str(out_path)

    step_draft = FnNode(
        fn=synthesize_draft,
        inputs={"prompt": inp.prompt},
        outputs={"draft_image": gr.Image(label="Draft Image")},
    )

    def pad_sprite(draft_image: str, border_px: int = 8) -> str:
        # Remember: daggr passes files as filepath strings!
        img = Image.open(draft_image).convert("RGBA")
        if int(border_px) > 0:
            img = ImageOps.expand(img, border=int(border_px), fill=(255, 180, 0, 255))
        tmp_dir = Path(tempfile.gettempdir()) / "jit_sprites"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        out_path = tmp_dir / f"padded_{Path(draft_image).stem}_{int(border_px)}.png"
        img.save(out_path)
        return str(out_path)

    step_pad = FnNode(
        fn=pad_sprite,
        inputs={"draft_image": step_draft.draft_image, "border_px": inp.border_px},
        outputs={"final_sprite": gr.Image(label="Final Sprite")},
    )

    return Graph(name="Modular Sprite Pipeline", nodes=[inp, step_draft, step_pad])


GRAPH = build_sprite_pipeline()


def execute_graph_modular(
    prompt: str,
    border_px: int,
    cached_state: dict | None,
    only_rerun_last: bool = False,
    progress=gr.Progress(),
):
    """Run the daggr graph, optionally reusing cached upstream results when only tweaking step 2."""
    token = os.environ.get("HF_TOKEN")
    ex = SequentialExecutor(GRAPH, hf_token=token)
    order = GRAPH.get_execution_order()

    if only_rerun_last and cached_state and cached_state.get("results"):
        # Restore previous node outputs and only update InputNode + final node!
        ex.results = dict(cached_state["results"])
        ex.execute_node(order[0], {"prompt": prompt, "border_px": int(border_px)})
        progress(0.5, desc=f"Re-running only {order[-1]}...")
        ex.execute_node(order[-1], None)
    else:
        for i, node_name in enumerate(order):
            progress((i, len(order)), desc=f"Executing daggr node: {node_name}")
            if i == 0:
                ex.execute_node(node_name, {"prompt": prompt, "border_px": int(border_px)})
            else:
                ex.execute_node(node_name, None)

    draft_path = ex.results.get(order[1], {}).get("draft_image")
    final_path = ex.results.get(order[2], {}).get("final_sprite")
    new_state = {"results": ex.results}
    return draft_path, final_path, ex.results, new_state
```

---

## 2. Looping a `daggr` Pipeline (Workaround for Scatter/Gather)

Because `daggr 0.8.0`'s experimental `.each` / `.all()` scatter-gather has bugs, **always loop either**:
1. **Outside `daggr`** in your Gradio event handler (calling `SequentialExecutor` once per item and `yield`-ing updated galleries/tables to the Gradio UI), OR
2. **Inside a single `FnNode`** that loops over a list and calls `gradio_client.Client` or your helper function per item, returning a manifest dict/JSON.

```python
def batch_loop_daggr(prompts_text: str, border_px: int, progress=gr.Progress()):
    """Loop a daggr Graph over N prompts and stream a live Gallery + Manifest to Gradio."""
    prompts = [p.strip() for p in prompts_text.splitlines() if p.strip()]
    gallery_items = []
    manifest = []
    for idx, p in enumerate(prompts):
        progress((idx, len(prompts)), desc=f"[{idx+1}/{len(prompts)}] {p}")
        _, final_path, node_trace, _ = execute_graph_modular(p, border_px, None, False)
        if final_path:
            gallery_items.append((final_path, p))
        manifest.append({"index": idx, "prompt": p, "output": final_path, "trace": node_trace})
        yield gallery_items, manifest
```

---

## 3. Embedding the Live `daggr` Visual Canvas Inside a Gradio App

When you want users to have **both** your custom Gradio UI (`gr.Blocks` or `gradio.Server`) **and** the interactive node-by-node `daggr` visual canvas in the same process/Space:

### Option A: Canvas at Root `/`, Gradio Blocks UI at `/app` (Simplest, Zero Proxy Needed)
`DaggrServer(graph).app` requests `/assets/*`, `/theme.css`, `/api/*`, `/ws/*`, and `/file/*` as **root-absolute paths**. Mounting `gr.Blocks` at `/app` first and mounting `DaggrServer` at `/` last works out of the box:

```python
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import gradio as gr
from daggr.server import DaggrServer

app = FastAPI()

# 1. Mount your Gradio Blocks UI at /app (Gradio internally serves /app/)
demo = build_gradio_blocks()
app = gr.mount_gradio_app(app, demo, path="/app")

@app.get("/app")
async def redirect_app():
    return RedirectResponse("/app/", status_code=307)

# 2. Mount the DaggrServer ASGI app at / LAST so it owns root assets & websocket
daggr_server = DaggrServer(PIPELINE_GRAPH)
app.mount("/", daggr_server.app)

# Run with: uvicorn app:app --host 0.0.0.0 --port 7860
```

### Option B: Custom App at `/`, `daggr` Canvas at `/canvas/` via `PrefixProxy`
If your main Gradio/SPA UI must own `/`, mount a prefix-rewriting ASGI proxy at `/canvas` and inject a small JS shim that prefixes same-origin `WebSocket`/`fetch`/`EventSource` calls. Compare `parsed.host === window.location.host` (not `origin`, because `wss://` vs `https://` differs in scheme). See https://github.com/Jpalmer95/daggr-studio for the reference `PrefixProxy` implementation.

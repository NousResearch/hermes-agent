#!/usr/bin/env python3
"""
JIT Gradio & Modular daggr Space Helper
========================================
CLI utility for Hermes agents to:
  1. `inspect-space` : Introspect a live Hugging Face Space's Gradio API endpoints & parameters
  2. `scaffold`      : Generate a ready-to-run JIT Gradio app directory (4 templates: blocks, daggr-modular, daggr-hybrid, workflow)
  3. `verify`        : Syntax-check, import-check, and smoke-test a local Gradio/daggr app directory
  4. `deploy`        : Create/update a Hugging Face Space, vendor local modules, set secrets/hardware, and verify runtime

Usage examples:
  python3 jit_gradio_space.py inspect-space black-forest-labs/FLUX.1-schnell
  python3 jit_gradio_space.py scaffold ./my-app --title "Audio & Sprite Lab" --template daggr-modular
  python3 jit_gradio_space.py verify ./my-app
  python3 jit_gradio_space.py deploy ./my-app --repo-id username/audio-sprite-lab --hardware cpu-basic --forward-hf-token
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


def _load_hermes_env() -> None:
    """Load ~/.hermes/.env into os.environ if keys like HF_TOKEN are not already set."""
    hermes_home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
    env_file = hermes_home / ".env"
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = val


def _readme_frontmatter(
    title: str,
    sdk: str = "gradio",
    emoji: str = "⚡",
    color_from: str = "indigo",
    color_to: str = "blue",
    short_description: str = "JIT Gradio application built by Hermes Agent",
    tags: list[str] | None = None,
    hf_oauth: bool = True,
) -> str:
    tags = tags or ["gradio", "daggr", "mcp-server", "hermes-agent"]
    tag_lines = "\n".join(f"  - {t}" for t in tags)
    port_line = "app_port: 7860\n" if sdk == "docker" else "app_file: app.py\n"
    oauth_line = "hf_oauth: true\n" if hf_oauth else ""
    return (
        f"---\n"
        f"title: {title}\n"
        f"emoji: {emoji}\n"
        f"colorFrom: {color_from}\n"
        f"colorTo: {color_to}\n"
        f"sdk: {sdk}\n"
        f"python_version: \"3.12\"\n"
        f"{port_line}"
        f"{oauth_line}"
        f"short_description: {short_description[:60]}\n"
        f"tags:\n"
        f"{tag_lines}\n"
        f"---\n\n"
        f"# {emoji} {title}\n\n"
        f"{short_description}\n\n"
        f"## Local Execution\n\n"
        f"```bash\n"
        f"pip install -r requirements.txt\n"
        f"python app.py\n"
        f"```\n"
    )


TEMPLATE_BLOCKS = '''"""
{title} — Standalone Gradio 6 App (Human UI + MCP Server + API)
"""
from __future__ import annotations
import os
import gradio as gr

try:
    import spaces  # Must be imported before CUDA initialization on ZeroGPU
    GPU = spaces.GPU
except ImportError:
    def GPU(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda fn: fn


@GPU(duration=60)
def run_task(prompt: str, intensity: float = 0.75) -> tuple[str, dict]:
    """Execute the core JIT task and return human-readable output + structured metadata.

    Args:
        prompt: Input instruction or text payload.
        intensity: Processing strength between 0.0 and 1.0.
    """
    summary = f"Processed '{{prompt}}' with intensity={{intensity:.2f}}"
    metadata = {{
        "prompt": prompt,
        "intensity": intensity,
        "status": "ok",
    }}
    return summary, metadata


def build_ui() -> gr.Blocks:
    with gr.Blocks(title={title!r}) as demo:
        gr.Markdown(f"# ⚡ {title}\\n{short_description}")
        with gr.Row():
            with gr.Column(scale=1):
                inp_prompt = gr.Textbox(label="Input / Prompt", value="Sample task", lines=3)
                inp_intensity = gr.Slider(0.0, 1.0, value=0.75, step=0.05, label="Intensity")
                btn = gr.Button("Run", variant="primary")
            with gr.Column(scale=1):
                out_summary = gr.Textbox(label="Result Summary", lines=3)
                out_meta = gr.JSON(label="Structured Metadata")

        btn.click(
            fn=run_task,
            inputs=[inp_prompt, inp_intensity],
            outputs=[out_summary, out_meta],
            api_name="run_task",
        )
    return demo


demo = build_ui()
gradio_app = demo

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        mcp_server=True,
        theme=gr.themes.Soft(),
    )
'''


TEMPLATE_DAGGR_MODULAR = '''"""
{title} — Gradio 6 App with Modular daggr Pipeline (Single Run, Batch Loop & Canvas Mode)
"""
from __future__ import annotations
import argparse
import os
import tempfile
from pathlib import Path
import gradio as gr
from PIL import Image, ImageDraw
from daggr import Graph, InputNode, FnNode
from daggr.executor import SequentialExecutor


def build_daggr_module() -> Graph:
    """Construct a modular daggr DAG that can be invoked inside Gradio callbacks or launched as a canvas."""
    inputs = InputNode(
        name="Inputs",
        ports={{
            "prompt": gr.Textbox(label="Prompt", value="neon cybernetic owl emblem"),
            "palette": gr.Dropdown(["cyber-cyan", "solar-gold", "emerald-matrix"], value="cyber-cyan"),
            "border_px": gr.Slider(0, 32, value=12, step=4, label="Border Width"),
        }},
    )

    def step_expand_spec(prompt: str, palette: str = "cyber-cyan") -> dict:
        colors = {{
            "cyber-cyan": ((12, 20, 44), (0, 240, 255)),
            "solar-gold": ((36, 18, 8), (255, 195, 0)),
            "emerald-matrix": ((8, 30, 16), (0, 230, 118)),
        }}
        bg, fg = colors.get(palette, ((20, 20, 30), (200, 200, 255)))
        return {{
            "prompt": prompt.strip(),
            "enhanced_prompt": f"{{prompt.strip()}}, {{palette}} palette, crisp vector-style icon",
            "bg_rgb": list(bg),
            "fg_rgb": list(fg),
        }}

    spec_node = FnNode(
        fn=step_expand_spec,
        inputs={{"prompt": inputs.prompt, "palette": inputs.palette}},
        outputs={{"spec": gr.JSON(label="Expanded Spec")}},
    )

    def step_render_preview(spec: dict, border_px: int = 12) -> str:
        # In daggr, file-based ports pass file path strings on disk!
        bg = tuple(spec.get("bg_rgb", [20, 20, 30]))
        fg = tuple(spec.get("fg_rgb", [0, 240, 255]))
        img = Image.new("RGB", (384, 384), bg)
        draw = ImageDraw.Draw(img)
        b = max(2, int(border_px))
        draw.rectangle([b, b, 384 - b, 384 - b], outline=fg, width=max(2, b // 2))
        draw.ellipse([96, 96, 288, 288], outline=fg, width=4)
        label = str(spec.get("prompt", "JIT"))[:28]
        draw.text((32, 336), label, fill=fg)
        out_dir = Path(tempfile.gettempdir()) / "jit_daggr_modular"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"preview_{{abs(hash((label, str(fg), b))) % 1000000}}.png"
        img.save(out_path)
        return str(out_path)

    render_node = FnNode(
        fn=step_render_preview,
        inputs={{"spec": spec_node.spec, "border_px": inputs.border_px}},
        outputs={{"preview_image": gr.Image(label="Rendered Asset", type="filepath")}},
    )

    return Graph(name={title!r}, nodes=[inputs, spec_node, render_node])


PIPELINE_GRAPH = build_daggr_module()


def execute_daggr_pipeline(
    prompt: str,
    palette: str = "cyber-cyan",
    border_px: int = 12,
) -> tuple[str | None, dict, dict]:
    """Run the modular daggr Graph headlessly via SequentialExecutor and return outputs + node trace."""
    token = os.environ.get("HF_TOKEN") or None
    ex = SequentialExecutor(PIPELINE_GRAPH, hf_token=token)
    order = PIPELINE_GRAPH.get_execution_order()

    for idx, node_name in enumerate(order):
        if idx == 0:
            ex.execute_node(
                node_name,
                {{"prompt": prompt, "palette": palette, "border_px": int(border_px)}},
            )
        else:
            ex.execute_node(node_name, None)

    spec_out = {{}}
    image_out = None
    for node_res in ex.results.values():
        if isinstance(node_res, dict):
            if "spec" in node_res:
                spec_out = node_res["spec"]
            if "preview_image" in node_res:
                image_out = node_res["preview_image"]

    return image_out, spec_out, ex.results


def execute_daggr_batch(
    prompts_multiline: str,
    palette: str = "cyber-cyan",
    border_px: int = 12,
    progress=gr.Progress(),
):
    """Loop the modular daggr pipeline over a batch of prompts and stream Gallery + JSON manifest."""
    items = [line.strip() for line in prompts_multiline.splitlines() if line.strip()]
    gallery: list[tuple[str, str]] = []
    manifest: list[dict] = []
    for i, item in enumerate(items):
        progress((i, max(1, len(items))), desc=f"Running daggr pipeline ({{i+1}}/{{len(items)}}): {{item[:32]}}")
        img_path, spec_out, _ = execute_daggr_pipeline(item, palette, int(border_px))
        if img_path:
            gallery.append((img_path, item))
        manifest.append({{"index": i, "prompt": item, "image": img_path, "spec": spec_out}})
        yield gallery, manifest


def build_ui() -> gr.Blocks:
    with gr.Blocks(title={title!r}) as demo:
        gr.Markdown(
            f"# ⚡ {title}\\n{short_description}\\n\\n"
            f"**Modular `daggr` Graph:** `{{PIPELINE_GRAPH.name}}` "
            f"(execution order: `{{' → '.join(PIPELINE_GRAPH.get_execution_order())}}`)"
        )

        with gr.Tab("Single Run (daggr Module)"):
            with gr.Row():
                with gr.Column(scale=1):
                    p_in = gr.Textbox(label="Prompt", value="neon cybernetic owl emblem", lines=2)
                    pal_in = gr.Dropdown(
                        ["cyber-cyan", "solar-gold", "emerald-matrix"],
                        value="cyber-cyan",
                        label="Palette",
                    )
                    b_in = gr.Slider(0, 32, value=12, step=4, label="Border Width")
                    run_btn = gr.Button("Run Modular Pipeline", variant="primary")
                with gr.Column(scale=1):
                    img_out = gr.Image(label="Rendered Output", type="filepath")
                    spec_out = gr.JSON(label="Intermediate Spec")
                    trace_out = gr.JSON(label="Full daggr Node Trace (ex.results)")

            run_btn.click(
                fn=execute_daggr_pipeline,
                inputs=[p_in, pal_in, b_in],
                outputs=[img_out, spec_out, trace_out],
                api_name="execute_daggr_pipeline",
            )

        with gr.Tab("Batch Loop (Multi-Item daggr Loop)"):
            with gr.Row():
                with gr.Column(scale=1):
                    batch_in = gr.Textbox(
                        label="Prompts (one per line)",
                        value="fire elemental crest\\nquantum compass icon\\ncrystal forge hammer",
                        lines=5,
                    )
                    b_pal_in = gr.Dropdown(
                        ["cyber-cyan", "solar-gold", "emerald-matrix"],
                        value="solar-gold",
                        label="Palette",
                    )
                    b_border_in = gr.Slider(0, 32, value=12, step=4, label="Border Width")
                    batch_btn = gr.Button("Loop Pipeline Over Batch", variant="primary")
                with gr.Column(scale=1):
                    batch_gallery = gr.Gallery(label="Batch Outputs", columns=3)
                    batch_manifest = gr.JSON(label="Batch Manifest")

            batch_btn.click(
                fn=execute_daggr_batch,
                inputs=[batch_in, b_pal_in, b_border_in],
                outputs=[batch_gallery, batch_manifest],
                api_name="execute_daggr_batch",
            )

    return demo


demo = build_ui()
gradio_app = demo
graph = PIPELINE_GRAPH

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--canvas", action="store_true", help="Launch the interactive daggr DAG canvas directly")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    args = parser.parse_args()

    if args.canvas:
        PIPELINE_GRAPH.launch(server_name="0.0.0.0", server_port=args.port)
    else:
        demo.launch(
            server_name="0.0.0.0",
            server_port=args.port,
            mcp_server=True,
            theme=gr.themes.Soft(),
        )
'''


TEMPLATE_DAGGR_HYBRID = '''"""
{title} — Hybrid Gradio UI (/app) + Interactive daggr Canvas (/) in One Server
"""
from __future__ import annotations
import os
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import gradio as gr
from daggr import Graph, InputNode, FnNode
from daggr.server import DaggrServer
from daggr.executor import SequentialExecutor


def build_graph() -> Graph:
    inp = InputNode(name="Inputs", ports={{"text": gr.Textbox(label="Input Text", value="Hello Hybrid Gradio + daggr")}})
    step = FnNode(
        fn=lambda text: f"[Processed by daggr] {{text.upper()}}",
        inputs={{"text": inp.text}},
        outputs={{"result": gr.Textbox(label="Result")}},
    )
    return Graph(name={title!r}, nodes=[inp, step])


PIPELINE_GRAPH = build_graph()


def run_task(text: str) -> tuple[str, dict]:
    ex = SequentialExecutor(PIPELINE_GRAPH)
    order = PIPELINE_GRAPH.get_execution_order()
    ex.execute_node(order[0], {{"text": text}})
    for name in order[1:]:
        ex.execute_node(name, None)
    out = ex.results.get(order[-1], {{}}).get("result", "")
    return out, ex.results


def build_blocks() -> gr.Blocks:
    with gr.Blocks(title={title!r}) as demo:
        gr.Markdown(
            f"# ⚡ {title} (Gradio Surface)\\n"
            f"[Open Live daggr Canvas at `/`](/)"
        )
        inp = gr.Textbox(label="Input Text", value="Hello Hybrid Gradio + daggr")
        btn = gr.Button("Run via SequentialExecutor", variant="primary")
        out = gr.Textbox(label="Result")
        trace = gr.JSON(label="Node Trace")
        btn.click(run_task, [inp], [out, trace], api_name="run_task")
    return demo


demo = build_blocks()
app = FastAPI(title={title!r})
app = gr.mount_gradio_app(app, demo, path="/app")


@app.get("/app")
async def redirect_app() -> RedirectResponse:
    return RedirectResponse("/app/", status_code=307)


daggr_server = DaggrServer(PIPELINE_GRAPH)
app.mount("/", daggr_server.app)
graph = PIPELINE_GRAPH

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
'''


TEMPLATE_WORKFLOW = '''"""
{title} — Native Gradio 6 gr.Workflow App (Dual Canvas & App View)
"""
from __future__ import annotations
import os
import gradio as gr


def clean_text(text: str) -> str:
    """Normalize whitespace and strip leading/trailing punctuation."""
    return " ".join(text.strip().split())


def summarize_spec(text: str) -> str:
    """Produce a structured JIT summary string."""
    words = text.split()
    return f"[{{len(words)}} words] {{' '.join(words[:25])}}"


def run_task(text: str) -> str:
    """Run the two-stage workflow chain directly."""
    return summarize_spec(clean_text(text))


if hasattr(gr, "Workflow"):
    demo = gr.Workflow(
        bind={{
            "Clean Text": clean_text,
            "Summarize Spec": summarize_spec,
        }},
        edges=[("Clean Text", "Summarize Spec")],
        graph="workflow.json",
    )
else:
    # Graceful fallback for Gradio < 6.27 where gr.Workflow is not yet present
    with gr.Blocks(title={title!r}) as demo:
        gr.Markdown(f"# ⚡ {title}\\n*(Fallback Blocks view — upgrade to `gradio>=6.27.0` for `gr.Workflow` canvas)*")
        inp = gr.Textbox(label="Input Text", value="Hello workflow")
        out = gr.Textbox(label="Summary")
        inp.submit(run_task, [inp], [out], api_name="run_task")

gradio_app = demo

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
'''


def cmd_inspect_space(args: argparse.Namespace) -> int:
    _load_hermes_env()
    try:
        from gradio_client import Client
    except ImportError:
        print("ERROR: gradio_client is not installed in this Python environment.", file=sys.stderr)
        return 1

    token = args.token or os.environ.get("HF_TOKEN") or None
    print(f"Inspecting Space: {args.space_id} ...")
    client = Client(args.space_id, token=token)
    info = client.view_api(return_format="dict")
    if args.json:
        print(json.dumps(info, indent=2))
        return 0

    named = info.get("named_endpoints", {})
    print(f"\nFound {len(named)} named endpoints on {args.space_id}:\n")
    for ep_name, ep_data in named.items():
        params = ep_data.get("parameters", [])
        returns = ep_data.get("returns", [])
        param_strs = [
            f"{p.get('parameter_name', '?')}: {p.get('python_type', {}).get('type', '?')}"
            + (f" = {p.get('parameter_default')!r}" if p.get("parameter_has_default") else " (required)")
            for p in params
        ]
        ret_strs = [
            f"{r.get('label', '?')}: {r.get('python_type', {}).get('type', '?')}"
            for r in returns
        ]
        print(f"  api_name={ep_name!r}")
        print(f"    inputs : {', '.join(param_strs) if param_strs else '(none)'}")
        print(f"    outputs: {', '.join(ret_strs) if ret_strs else '(none)'}")
        print()
    return 0


def cmd_scaffold(args: argparse.Namespace) -> int:
    out_dir = Path(args.target_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    title = args.title
    short_desc = args.short_description
    sdk = "gradio"

    if args.template == "daggr-modular":
        app_code = TEMPLATE_DAGGR_MODULAR.format(title=title, short_description=short_desc)
        reqs = ["gradio>=6.0.0", "daggr>=0.8.0", "gradio_client>=1.0.0", "huggingface_hub>=1.0.0", "Pillow"]
    elif args.template == "daggr-hybrid":
        app_code = TEMPLATE_DAGGR_HYBRID.format(title=title, short_description=short_desc)
        reqs = ["gradio>=6.0.0", "daggr>=0.8.0", "fastapi", "uvicorn", "huggingface_hub>=1.0.0"]
    elif args.template == "workflow":
        app_code = TEMPLATE_WORKFLOW.format(title=title, short_description=short_desc)
        reqs = ["gradio>=6.27.0", "gradio_client>=1.0.0", "huggingface_hub>=1.0.0"]
    else:
        app_code = TEMPLATE_BLOCKS.format(title=title, short_description=short_desc)
        reqs = ["gradio>=6.0.0", "gradio_client>=1.0.0", "huggingface_hub>=1.0.0", "spaces", "Pillow"]

    (out_dir / "app.py").write_text(app_code, encoding="utf-8")
    (out_dir / "requirements.txt").write_text("\n".join(reqs) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(
        _readme_frontmatter(
            title=title,
            sdk=sdk,
            short_description=short_desc,
        ),
        encoding="utf-8",
    )

    print(f"Scaffolded JIT Gradio app ({args.template}) at: {out_dir}")
    print(f"  - {out_dir / 'app.py'}")
    print(f"  - {out_dir / 'requirements.txt'}")
    print(f"  - {out_dir / 'README.md'}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    target_dir = Path(args.target_dir).resolve()
    app_file = target_dir / args.app_file
    if not app_file.exists():
        print(f"ERROR: {app_file} does not exist.", file=sys.stderr)
        return 1

    # 1. AST syntax check
    source = app_file.read_text(encoding="utf-8")
    try:
        ast.parse(source, filename=str(app_file))
        print(f"[1/3] Syntax OK: {app_file}")
    except SyntaxError as exc:
        print(f"ERROR: SyntaxError in {app_file}: {exc}", file=sys.stderr)
        return 1

    # 2. Check README frontmatter if present
    readme = target_dir / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        if text.startswith("---\n") and "\n---\n" in text[4:]:
            print("[2/3] README.md HF Spaces YAML frontmatter OK")
        else:
            print("[2/3] WARNING: README.md is missing HF Spaces YAML frontmatter (--- ... ---)")
    else:
        print(f"[2/3] WARNING: No README.md found in {target_dir}")

    # 3. Smoke-import app.py and inspect demo / graph objects
    sys.path.insert(0, str(target_dir))
    spec = importlib.util.spec_from_file_location("jit_app_under_test", str(app_file))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)

    has_demo = hasattr(mod, "demo") or hasattr(mod, "gradio_app") or hasattr(mod, "app")
    has_graph = hasattr(mod, "graph") or hasattr(mod, "PIPELINE_GRAPH")
    print(f"[3/3] Module import OK (demo/app={has_demo}, daggr_graph={has_graph})")

    if hasattr(mod, "execute_daggr_pipeline"):
        img_out, _spec_out, trace = mod.execute_daggr_pipeline("smoke test icon", "cyber-cyan", 8)
        print(f"      Modular daggr smoke run OK -> output={img_out}, nodes_ran={list(trace.keys())}")
    elif hasattr(mod, "run_task"):
        res = mod.run_task("smoke test")
        print(f"      Direct task smoke run OK -> {res[0] if isinstance(res, tuple) else res}")

    return 0


def cmd_deploy(args: argparse.Namespace) -> int:
    _load_hermes_env()
    try:
        from huggingface_hub import HfApi, whoami
    except ImportError:
        print("ERROR: huggingface_hub is not installed in this Python environment.", file=sys.stderr)
        return 1

    target_dir = Path(args.target_dir).resolve()
    if not (target_dir / "app.py").exists():
        print(f"ERROR: No app.py found in {target_dir}", file=sys.stderr)
        return 1

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not found in --token, environment, or ~/.hermes/.env.", file=sys.stderr)
        return 1

    api = HfApi(token=token)
    user_info = whoami(token=token)
    username = user_info.get("name")

    repo_id = args.repo_id
    if "/" not in repo_id:
        repo_id = f"{username}/{repo_id}"

    print(f"Deploying {target_dir} to HF Space: {repo_id} (sdk={args.sdk}, hardware={args.hardware})...")

    create_kwargs = {
        "repo_id": repo_id,
        "repo_type": "space",
        "space_sdk": args.sdk,
        "private": args.private,
        "exist_ok": True,
    }
    if args.hardware and args.hardware != "zero-gpu":
        create_kwargs["space_hardware"] = args.hardware

    api.create_repo(**create_kwargs)

    with tempfile.TemporaryDirectory() as tmp:
        stage_dir = Path(tmp) / "space"
        shutil.copytree(
            target_dir,
            stage_dir,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".venv", ".git", "outputs"),
        )
        for vendor_item in args.vendor or []:
            vpath = Path(vendor_item).resolve()
            if vpath.is_dir():
                shutil.copytree(vpath, stage_dir / vpath.name, dirs_exist_ok=True)
            elif vpath.is_file():
                shutil.copy2(vpath, stage_dir / vpath.name)

        api.upload_folder(
            repo_id=repo_id,
            repo_type="space",
            folder_path=str(stage_dir),
            commit_message=args.commit_message,
        )

    if args.forward_hf_token:
        api.add_space_secret(repo_id=repo_id, key="HF_TOKEN", value=token)
        print("  Added HF_TOKEN secret to Space.")

    for sec in args.secret or []:
        if "=" in sec:
            k, _, v = sec.partition("=")
            api.add_space_secret(repo_id=repo_id, key=k.strip(), value=v.strip())
            print(f"  Added secret {k.strip()} to Space.")

    runtime = api.get_space_runtime(repo_id=repo_id)
    print(f"Successfully deployed: https://huggingface.co/spaces/{repo_id}")
    print(f"Runtime stage: {runtime.stage} | Hardware: {runtime.hardware}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JIT Gradio & Modular daggr Space Helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_insp = sub.add_parser("inspect-space", help="Introspect a live HF Space's Gradio API")
    p_insp.add_argument("space_id", help="HF Space ID (owner/space) or URL")
    p_insp.add_argument("--token", default=None, help="Optional HF_TOKEN")
    p_insp.add_argument("--json", action="store_true", help="Emit raw JSON API info")
    p_insp.set_defaults(func=cmd_inspect_space)

    p_scaf = sub.add_parser("scaffold", help="Scaffold a JIT Gradio app directory")
    p_scaf.add_argument("target_dir", help="Directory to create")
    p_scaf.add_argument("--title", default="JIT Gradio Studio", help="Application title")
    p_scaf.add_argument(
        "--short-description",
        default="JIT Gradio application with modular daggr pipeline",
        help="Short description for README YAML frontmatter (<=60 chars)",
    )
    p_scaf.add_argument(
        "--template",
        choices=["blocks", "daggr-modular", "daggr-hybrid", "workflow"],
        default="daggr-modular",
        help="Template flavor (default: daggr-modular)",
    )
    p_scaf.set_defaults(func=cmd_scaffold)

    p_ver = sub.add_parser("verify", help="Syntax-check, import-check, and smoke-run a local app")
    p_ver.add_argument("target_dir", help="App directory")
    p_ver.add_argument("--app-file", default="app.py", help="Entrypoint file (default: app.py)")
    p_ver.set_defaults(func=cmd_verify)

    p_dep = sub.add_parser("deploy", help="Deploy local app directory to Hugging Face Spaces")
    p_dep.add_argument("target_dir", help="App directory to upload")
    p_dep.add_argument("--repo-id", required=True, help="Space repo ID (owner/name or name)")
    p_dep.add_argument("--sdk", choices=["gradio", "docker"], default="gradio")
    p_dep.add_argument("--hardware", default="cpu-basic", help="Space hardware flavor")
    p_dep.add_argument("--private", action="store_true", help="Create private Space")
    p_dep.add_argument("--token", default=None, help="HF token (defaults to HF_TOKEN env / ~/.hermes/.env)")
    p_dep.add_argument("--forward-hf-token", action="store_true", help="Set HF_TOKEN as a Space secret")
    p_dep.add_argument("--secret", action="append", help="Set KEY=VALUE Space secret (repeatable)")
    p_dep.add_argument("--vendor", action="append", help="Local file/dir to vendor into the Space")
    p_dep.add_argument("--commit-message", default="Deploy JIT Gradio app via Hermes Agent")
    p_dep.set_defaults(func=cmd_deploy)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

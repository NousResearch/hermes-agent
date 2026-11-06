#!/usr/bin/env python3
"""Stage ledger for the open-source 3D asset pipeline.

Chain, per asset:
    prompt -> image -> mask -> mesh -> rig -> animation

Editing any stage (a new sentence, a painted PNG, a Blender re-export)
marks every downstream stage stale. ``plan`` prints only the phase
commands still needed. ``refresh`` re-hashes files on disk so a manual
overwrite is detected without anyone remembering to tell the tool.

This module does not call the network. Phase scripts stay in their skills.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

STAGES = ("prompt", "image", "mask", "mesh", "rig", "animation")
# animation is local-GPU and optional; plan stops here unless asked
DEFAULT_THROUGH = "rig"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

SKILL_DIR = Path(__file__).resolve().parents[1]
# Sibling phase skills install beside this one (skills/<category>/<name>), so
# resolving from the category directory works in a checkout AND after
# `hermes skills install` — no home-directory assumptions.
CATEGORY_DIR = SKILL_DIR.parent
TRELLIS = CATEGORY_DIR / "trellis-image-to-3d" / "scripts" / "trellis_gen.py"
RIG = CATEGORY_DIR / "skintokens-rigging" / "scripts" / "skintokens_rig.py"
# Optional image generator, set by --flux-script. When unset, the image stage
# is a manual/tool step and `plan` prints a hint instead of a command.
FLUX: Path | None = None
_PIPELINE_PYTHON: str | None = None


def pipeline_python() -> str:
    """Interpreter that can import gradio_client.

    Hermes' own venv often cannot. Phase scripts fail immediately if we
    hand them that interpreter, so prefer one that already has the client.
    """
    global _PIPELINE_PYTHON
    if _PIPELINE_PYTHON:
        return _PIPELINE_PYTHON
    for cand in (sys.executable, "/usr/bin/python3"):
        try:
            probe = subprocess.run(
                [cand, "-c", "import gradio_client"],
                capture_output=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if probe.returncode == 0:
            _PIPELINE_PYTHON = cand
            return cand
    _PIPELINE_PYTHON = sys.executable
    return _PIPELINE_PYTHON


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def downstream(stage: str) -> tuple[str, ...]:
    return STAGES[STAGES.index(stage) + 1 :]


def empty_stage() -> dict:
    return {"status": "missing", "path": "", "sha256": "", "source": "", "updated": ""}


def load(root: Path) -> dict:
    path = root / "project.json"
    if not path.exists():
        sys.exit(f"no project.json in {root} — run: asset_stage.py init {root}")
    return json.loads(path.read_text())


def save(root: Path, project: dict) -> None:
    project["updated"] = now()
    text = json.dumps(project, indent=2) + "\n"
    target = root / "project.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(text)
    tmp.replace(target)


def new_project(name: str, target: str) -> dict:
    return {
        "name": name,
        "target": target,
        "updated": now(),
        "assets": {},
    }


def ensure_asset(project: dict, slug: str) -> dict:
    assets = project.setdefault("assets", {})
    if slug not in assets:
        assets[slug] = {
            "prompt": "",
            "stages": {s: empty_stage() for s in STAGES},
            "notes": "",
        }
    return assets[slug]


def mark_downstream_stale(asset: dict, stage: str) -> None:
    for name in downstream(stage):
        st = asset["stages"][name]
        if st["status"] != "missing":
            st["status"] = "stale"
            st["updated"] = now()


def set_stage(asset: dict, stage: str, path: Path | None, source: str, status: str = "ok") -> None:
    st = asset["stages"][stage]
    st["status"] = status
    st["source"] = source
    st["updated"] = now()
    if path is not None:
        st["path"] = str(path)
        st["sha256"] = sha256(path) if path.is_file() else ""
    mark_downstream_stale(asset, stage)


def prompt_lines(path: Path) -> list[str]:
    lines = []
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def slug_tokens(slug: str) -> list[str]:
    parts = slug.split("_")
    if parts and parts[0].isdigit():
        parts = parts[1:]
    return [p.lower() for p in parts if p and p not in {"ant"}]


def match_prompt(slug: str, prompts: list[str]) -> tuple[str, str]:
    """Return (prompt, note). Prefer the line that shares the most slug tokens.

    Index alignment is unsafe: a prompts file with leading comments used to be
    fed to trellis_gen.py, which counted those comments as prompts and shifted
    every row. Token match repairs that.
    """
    tokens = slug_tokens(slug)
    best, best_n = "", 0
    for prompt in prompts:
        low = prompt.lower()
        n = sum(1 for t in tokens if t in low)
        if n > best_n:
            best, best_n = prompt, n
    if best_n == 0:
        return "", "no prompt line shares this slug's words"
    if tokens and best_n < 1:
        return best, "weak prompt match"
    return best, ""


def canonical(root: Path, slug: str, stage: str) -> Path:
    if stage == "image":
        return root / "images" / f"{slug}.png"
    if stage == "mask":
        return root / "images_masked" / f"{slug}.png"
    if stage == "mesh":
        return root / "assets" / "glb" / f"{slug}.glb"
    if stage == "rig":
        return root / "assets" / "rigged" / f"{slug}_rigged.glb"
    if stage == "animation":
        return root / "assets" / "animated" / f"{slug}_animated.glb"
    raise ValueError(stage)


def write_prompts_file(root: Path, project: dict) -> Path:
    path = root / "prompts" / "resolved.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for slug in sorted(project["assets"]):
        prompt = project["assets"][slug]["prompt"].strip()
        if prompt:
            lines.append(prompt)
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return path


def cmd_init(args) -> None:
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "project.json").exists() and not args.force:
        sys.exit(f"{root / 'project.json'} exists (pass --force to replace)")
    for rel in ("images", "images_masked", "assets/glb", "assets/rigged", "assets/animated", "prompts", "working"):
        (root / rel).mkdir(parents=True, exist_ok=True)
    save(root, new_project(args.name, args.target))
    print(f"initialized {root / 'project.json'}")


def cmd_add(args) -> None:
    root = Path(args.root).resolve()
    project = load(root)
    asset = ensure_asset(project, args.slug)
    asset["prompt"] = args.prompt
    set_stage(asset, "prompt", None, source=args.source or "authored")
    asset["stages"]["prompt"]["status"] = "ok"
    save(root, project)
    write_prompts_file(root, project)
    print(f"added {args.slug}; downstream stale from image")


def cmd_set_prompt(args) -> None:
    root = Path(args.root).resolve()
    project = load(root)
    asset = project["assets"].get(args.slug)
    if not asset:
        sys.exit(f"unknown slug {args.slug}")
    asset["prompt"] = args.prompt
    set_stage(asset, "prompt", None, source="edited")
    asset["stages"]["prompt"]["status"] = "ok"
    save(root, project)
    write_prompts_file(root, project)
    print(f"{args.slug}: prompt updated; image..animation stale")


def cmd_import(args) -> None:
    root = Path(args.root).resolve()
    project = load(root)
    if args.slug not in project["assets"]:
        ensure_asset(project, args.slug)
    src = Path(args.file).resolve()
    if not src.is_file():
        sys.exit(f"not a file: {src}")
    dest = canonical(root, args.slug, args.stage)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src != dest:
        shutil.copyfile(src, dest)
    asset = project["assets"][args.slug]
    set_stage(asset, args.stage, dest, source=args.source or "manual")
    save(root, project)
    print(f"{args.slug}: imported {args.stage} -> {dest}; downstream stale")


def cmd_note(args) -> None:
    root = Path(args.root).resolve()
    project = load(root)
    asset = project["assets"].get(args.slug)
    if not asset:
        sys.exit(f"unknown slug {args.slug}")
    asset["notes"] = args.text
    save(root, project)
    print(f"{args.slug}: note saved")


def adopt_one(root: Path, project: dict, slug: str, prompts: list[str]) -> None:
    asset = ensure_asset(project, slug)
    prompt, note = match_prompt(slug, prompts)
    if prompt and not asset["prompt"]:
        asset["prompt"] = prompt
        asset["stages"]["prompt"] = empty_stage()
        asset["stages"]["prompt"]["status"] = "ok"
        asset["stages"]["prompt"]["source"] = "adopted"
        asset["stages"]["prompt"]["updated"] = now()
    if note:
        asset["notes"] = note
    found = {
        "image": root / "images" / f"{slug}.png",
        "mask": root / "images_masked" / f"{slug}.png",
        "mesh": root / "assets" / "glb" / f"{slug}.glb",
        "rig": root / "assets" / "rigged" / f"{slug}_rigged.glb",
        "animation": root / "assets" / "animated" / f"{slug}_animated.glb",
    }
    # rigged files in the dogfood colony are {slug}_rigged.glb; also accept {slug}.glb
    rig_alt = root / "assets" / "rigged" / f"{slug}.glb"
    if not found["rig"].is_file() and rig_alt.is_file():
        found["rig"] = rig_alt
    for stage, path in found.items():
        if path.is_file():
            st = asset["stages"][stage]
            st["status"] = "ok"
            st["path"] = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
            st["sha256"] = sha256(path)
            st["source"] = st["source"] or "adopted"
            st["updated"] = st["updated"] or now()


def discover_slugs(root: Path) -> list[str]:
    slugs = []
    images = root / "images"
    if images.is_dir():
        for p in sorted(images.iterdir()):
            if p.suffix.lower() in IMAGE_SUFFIXES:
                slugs.append(p.stem)
    return slugs


def cmd_adopt(args) -> None:
    root = Path(args.root).resolve()
    for rel in ("prompts", "working", "assets/animated"):
        (root / rel).mkdir(parents=True, exist_ok=True)
    project = json.loads((root / "project.json").read_text()) if (root / "project.json").exists() else new_project(args.name or root.name, args.target)
    prompts: list[str] = []
    prompt_dir = root / "prompts"
    if prompt_dir.is_dir():
        for txt in sorted(prompt_dir.glob("*.txt")):
            if txt.name == "resolved.txt":
                continue
            prompts.extend(prompt_lines(txt))
    slugs = discover_slugs(root)
    if not slugs:
        sys.exit(f"no images in {root / 'images'}")
    for slug in slugs:
        adopt_one(root, project, slug, prompts)
    save(root, project)
    resolved = write_prompts_file(root, project)
    mismatched = [s for s, a in project["assets"].items() if a.get("notes")]
    print(f"adopted {len(slugs)} assets -> {root / 'project.json'}")
    print(f"resolved prompts (comments stripped, matched by slug words): {resolved}")
    if mismatched:
        print("prompt notes:")
        for s in mismatched:
            print(f"  {s}: {project['assets'][s]['notes']}")


def refresh_asset(root: Path, asset: dict) -> list[str]:
    changed = []
    for stage in STAGES:
        if stage == "prompt":
            continue
        st = asset["stages"][stage]
        if not st.get("path"):
            continue
        path = Path(st["path"])
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            if st["status"] != "missing":
                st["status"] = "missing"
                st["updated"] = now()
                changed.append(f"{stage} missing")
                mark_downstream_stale(asset, stage)
            continue
        digest = sha256(path)
        if st.get("sha256") and digest != st["sha256"]:
            st["sha256"] = digest
            st["status"] = "ok"
            st["source"] = "external-edit"
            st["updated"] = now()
            mark_downstream_stale(asset, stage)
            changed.append(f"{stage} edited on disk")
        elif not st.get("sha256"):
            st["sha256"] = digest
    return changed


def cmd_refresh(args) -> None:
    root = Path(args.root).resolve()
    project = load(root)
    any_change = False
    for slug, asset in project["assets"].items():
        changed = refresh_asset(root, asset)
        if changed:
            any_change = True
            print(f"{slug}: {', '.join(changed)}")
    save(root, project)
    if not any_change:
        print("no on-disk edits since last hash")


def stage_index(name: str) -> int:
    if name not in STAGES:
        sys.exit(f"unknown stage {name}; choose from {', '.join(STAGES)}")
    return STAGES.index(name)


def missing_note(script: Path, skill: str) -> str:
    """Empty when the sibling script resolves; otherwise an actionable hint."""
    if script.exists():
        return ""
    return f"sibling skill not installed — hermes skills install official/gaming/{skill}"


def plan_asset(root: Path, project: dict, slug: str, through: str) -> list[dict]:
    asset = project["assets"][slug]
    limit = stage_index(through)
    steps = []
    target = project.get("target") or "godot"
    for stage in STAGES[: limit + 1]:
        st = asset["stages"][stage]
        if st["status"] == "ok":
            continue
        if stage == "prompt":
            steps.append({
                "stage": "prompt",
                "slug": slug,
                "reason": "prompt missing",
                "cmd": [],
                "hint": "set a prompt with: asset_stage.py set-prompt",
            })
            continue
        dest = canonical(root, slug, stage)
        if stage == "image":
            steps.append({
                "stage": "image",
                "slug": slug,
                "reason": st["status"],
                "cmd": (
                    [pipeline_python(), str(FLUX), "--prompt", asset["prompt"], "-o", str(dest)]
                    if FLUX
                    else []
                ),
                "hint": "generate with the image_generate tool or local ComfyUI, then import --stage image",
            })
        elif stage == "mask":
            src = canonical(root, slug, "image")
            steps.append({
                "stage": "mask",
                "slug": slug,
                "reason": st["status"],
                "cmd": [pipeline_python(), str(Path(__file__).resolve()), "mask-file", str(src), str(dest)],
                "hint": "or paint the alpha in GIMP/Krita and import",
            })
        elif stage == "mesh":
            src = canonical(root, slug, "mask")
            steps.append({
                "stage": "mesh",
                "slug": slug,
                "reason": st["status"],
                "cmd": (
                    [
                        pipeline_python(), str(TRELLIS), str(src),
                        "-o", str(dest),
                        "--name", slug,
                        "--target", target,
                    ]
                    if TRELLIS.exists()
                    else []
                ),
                "hint": missing_note(TRELLIS, "trellis-image-to-3d")
                or "or edit/export a GLB from Blender and import --stage mesh",
            })
        elif stage == "rig":
            src = canonical(root, slug, "mesh")
            steps.append({
                "stage": "rig",
                "slug": slug,
                "reason": st["status"],
                "cmd": (
                    [
                        pipeline_python(), str(RIG), str(src),
                        "-o", str(dest),
                        "--preserve-texture-scale",
                    ]
                    if RIG.exists()
                    else []
                ),
                "hint": missing_note(RIG, "skintokens-rigging")
                or "retry with --voxel-postprocess if SkinTokens returns no files",
            })
        elif stage == "animation":
            steps.append({
                "stage": "animation",
                "slug": slug,
                "reason": st["status"],
                "cmd": [],
                "hint": "optional local-GPU step — install the puppeteer-animation skill for it",
            })
    return steps


def _set_flux(args) -> None:
    """Record --flux-script into the module global used by plan_asset."""
    global FLUX
    if getattr(args, "flux_script", None):
        FLUX = Path(args.flux_script).resolve()


def cmd_plan(args) -> None:
    _set_flux(args)
    root = Path(args.root).resolve()
    project = load(root)
    slugs = [args.slug] if args.slug else sorted(project["assets"])
    all_steps = []
    for slug in slugs:
        if slug not in project["assets"]:
            sys.exit(f"unknown slug {slug}")
        steps = plan_asset(root, project, slug, args.through)
        all_steps.extend(steps)
        if not steps:
            print(f"{slug}: nothing to do through {args.through}")
            continue
        print(f"# {slug}")
        for step in steps:
            print(f"# {step['stage']} ({step['reason']}) — {step['hint']}")
            if step["cmd"]:
                print(" ".join(json.dumps(c) if " " in c else c for c in step["cmd"]))
    if args.json:
        print(json.dumps(all_steps, indent=2))


def cmd_status(args) -> None:
    root = Path(args.root).resolve()
    project = load(root)
    slugs = [args.slug] if args.slug else sorted(project["assets"])
    print(f"{project.get('name')}  target={project.get('target')}  assets={len(project['assets'])}")
    for slug in slugs:
        asset = project["assets"].get(slug)
        if not asset:
            sys.exit(f"unknown slug {slug}")
        letters = []
        for s in STAGES:
            status = asset["stages"][s]["status"]
            letters.append({"ok": "OK", "stale": "STALE", "missing": "MISS", "edited": "EDIT"}.get(status, status)[:5])
        row = "  ".join(f"{s[:5]}:{letters[i]}" for i, s in enumerate(STAGES))
        print(f"{slug}  {row}")
        if asset.get("notes"):
            print(f"    note: {asset['notes']}")
        if args.verbose:
            print(f"    prompt: {asset.get('prompt','')[:140]}")


def cmd_mask_file(args) -> None:
    """rembg via the Python API. The rembg CLI imports gradio and crashes without it."""
    from rembg import remove
    src, dest = Path(args.src), Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(remove(src.read_bytes()))
    print(f"masked {src} -> {dest} ({dest.stat().st_size} bytes)")


def record_ok(root: Path, slug: str, stage: str, path: Path, source: str) -> None:
    project = load(root)
    asset = project["assets"][slug]
    set_stage(asset, stage, path, source=source)
    # set_stage marks downstream stale, which is correct if we are about to run them,
    # but a successful stage should not mark later stages we just finished.
    # Caller records stages in order and re-saves after each, so stale-on-later is fine
    # until the next stage records itself ok.
    save(root, project)


def cmd_run(args) -> None:
    import subprocess
    _set_flux(args)
    root = Path(args.root).resolve()
    project = load(root)
    if args.slug not in project["assets"]:
        sys.exit(f"unknown slug {args.slug}")
    steps = plan_asset(root, project, args.slug, args.through)
    runnable = [s for s in steps if s["cmd"]]
    if not runnable:
        print("nothing to run")
        return
    for step in runnable:
        print(f"== {step['slug']} {step['stage']} ==", flush=True)
        print(" ".join(step["cmd"]), flush=True)
        if args.dry_run:
            continue
        proc = subprocess.run(step["cmd"], check=False)
        if proc.returncode != 0 and step["stage"] == "rig":
            # Translucent/glossy/rocky textures fail the default SkinTokens path.
            # Confirmed on the 10-ant dogfood: --voxel-postprocess rescued them.
            retry = step["cmd"] + ["--voxel-postprocess"]
            print("rig failed; retrying with --voxel-postprocess", flush=True)
            proc = subprocess.run(retry, check=False)
        if proc.returncode != 0:
            sys.exit(f"{step['stage']} failed with exit {proc.returncode}")
        dest = canonical(root, args.slug, step["stage"])
        if not dest.is_file():
            sys.exit(f"{step['stage']} exited 0 but {dest} is missing")
        record_ok(root, args.slug, step["stage"], dest, source="pipeline")
        print(f"recorded {step['stage']} ok", flush=True)
    if args.dry_run:
        print("dry-run only")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="create an empty project folder")
    p.add_argument("root")
    p.add_argument("--name", default="assets")
    p.add_argument("--target", default="godot", choices=["godot-mobile", "godot", "unity", "blender", "film"])
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("add", help="add an asset from a prompt")
    p.add_argument("root")
    p.add_argument("slug")
    p.add_argument("prompt")
    p.add_argument("--source", default="authored")
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("set-prompt", help="replace the prompt; marks image onward stale")
    p.add_argument("root")
    p.add_argument("slug")
    p.add_argument("prompt")
    p.set_defaults(func=cmd_set_prompt)

    p = sub.add_parser("import", help="drop a file in at a stage; marks downstream stale")
    p.add_argument("root")
    p.add_argument("slug")
    p.add_argument("--stage", required=True, choices=["image", "mask", "mesh", "rig", "animation"])
    p.add_argument("--file", required=True)
    p.add_argument("--source", default="manual")
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("note")
    p.add_argument("root")
    p.add_argument("slug")
    p.add_argument("text")
    p.set_defaults(func=cmd_note)

    p = sub.add_parser("adopt", help="build project.json from an existing images/glb/rigged tree")
    p.add_argument("root")
    p.add_argument("--name", default="")
    p.add_argument("--target", default="godot")
    p.set_defaults(func=cmd_adopt)

    p = sub.add_parser("refresh", help="re-hash files; an overwrite marks downstream stale")
    p.add_argument("root")
    p.set_defaults(func=cmd_refresh)

    p = sub.add_parser("status")
    p.add_argument("root")
    p.add_argument("--slug")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("plan", help="print only the commands still needed")
    p.add_argument("root")
    p.add_argument("--slug")
    p.add_argument("--through", default=DEFAULT_THROUGH, choices=STAGES)
    p.add_argument("--json", action="store_true")
    p.add_argument("--flux-script", help="optional image generator taking --prompt/-o")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("run", help="execute the plan for one asset, recording each stage")
    p.add_argument("root")
    p.add_argument("slug")
    p.add_argument("--through", default=DEFAULT_THROUGH, choices=STAGES)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--flux-script", help="optional image generator taking --prompt/-o")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("mask-file", help="internal: rembg one file")
    p.add_argument("src")
    p.add_argument("dest")
    p.set_defaults(func=cmd_mask_file)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

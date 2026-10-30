#!/usr/bin/env python3
"""
puppeteer_animate.py — Animate a rigged GLB with a driving video using
Seed3D/Puppeteer's video-guided optimization.

Pipeline per asset:
  1. rigged GLB -> objs/mesh.obj + objs/rig.txt   (Blender: glb_to_rignet.py)
  2. render first-frame reference views           (utils/render_first_frame.py)
     OR use a provided video directly.
  3. extract frames from input.mp4                (ffmpeg, 10 fps)
  4. optical flow + depth + tracking precompute   (utils/save_flow.py)
  5. skeletal motion optimization                 (optimization.py)
  6. (optional) bake to animated GLB via Blender  (bake_animation.py)

Prereqs: Puppeteer cloned + env built (see SKILL.md), Blender on PATH,
ffmpeg on PATH.

Examples:
  # Full run: convert rig, precompute, optimize
  python3 puppeteer_animate.py goblin_rigged.glb --video goblin_walk.mp4 \
      --puppeteer ~/dev/Puppeteer -o out/goblin/

  # Optimization only (data already prepared)
  python3 puppeteer_animate.py --skip-convert --seq-name goblin \
      --puppeteer ~/dev/Puppeteer -o out/goblin/ --iter 500
"""
import argparse
import os
import shutil
import subprocess
import sys


def run(cmd, cwd=None, env=None):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("glb", nargs="?", help="Rigged GLB (SkinTokens output)")
    p.add_argument("--video", help="Driving video (mp4). If omitted, you must "
                   "generate one from a first frame and place it at "
                   "<out>/input.mp4 before running with --skip-convert.")
    p.add_argument("--puppeteer", default=os.environ.get("PUPPETEER_DIR"),
                   help="Path to the Seed3D/Puppeteer clone (or set PUPPETEER_DIR)")
    p.add_argument("-o", "--out", required=True, help="Output/work dir")
    p.add_argument("--seq-name", default=None,
                   help="Sequence name (default: GLB basename)")
    p.add_argument("--iter", type=int, default=500,
                   help="Optimization iterations (200 = quick draft)")
    p.add_argument("--img-size", type=int, default=960)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--main-renderer", default="front")
    p.add_argument("--additional-renderers", default="back,right,left")
    p.add_argument("--smooth-weight", type=float, default=None)
    p.add_argument("--blender", default="blender")
    p.add_argument("--decimate", type=int, default=0,
                   help="Target face count per mesh during conversion (0 = keep). "
                        "Use ~20000 for TRELLIS outputs: the optimizer OOMs on "
                        "200k-vertex meshes, and decimating after rig.txt exists "
                        "breaks the skin-weight index.")
    p.add_argument("--skip-convert", action="store_true",
                   help="objs/ already prepared; go straight to precompute")
    p.add_argument("--skip-precompute", action="store_true")
    args = p.parse_args()

    if not args.puppeteer:
        sys.exit("Pass --puppeteer /path/to/Puppeteer (or set PUPPETEER_DIR). "
                 "See the skill's Prerequisites.")
    pup = os.path.abspath(os.path.expanduser(args.puppeteer))
    anim = os.path.join(pup, "animation")
    venv_py = os.path.join(pup, ".venv", "bin", "python")
    if not os.path.exists(venv_py):
        sys.exit(f"Puppeteer venv not found at {venv_py} — build it first "
                 f"(see SKILL.md)")

    if args.seq_name is None:
        if not args.glb:
            sys.exit("Need GLB path or --seq-name with --skip-convert")
        args.seq_name = os.path.splitext(os.path.basename(args.glb))[0]

    out = os.path.abspath(args.out)
    seq_dir = out  # out dir IS the sequence dir (objs/, imgs/, input.mp4)
    os.makedirs(seq_dir, exist_ok=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. GLB -> objs/
    if not args.skip_convert:
        if not args.glb:
            sys.exit("--skip-convert requires no positional GLB... use it only "
                     "when objs/ exists")
        cmd = [args.blender, "-b", "--factory-startup", "-noaudio", "--python",
               os.path.join(script_dir, "glb_to_rignet.py"), "--",
               os.path.abspath(args.glb), "-o", seq_dir]
        if args.decimate:
            cmd += ["--decimate", str(args.decimate)]
        run(cmd)

    # 2. video -> input.mp4
    video_dst = os.path.join(seq_dir, "input.mp4")
    if args.video:
        shutil.copy(os.path.abspath(args.video), video_dst)
    if not os.path.exists(video_dst) and not args.skip_precompute:
        sys.exit(f"No input.mp4 in {seq_dir}. Pass --video or drop one there. "
                 f"Tip: render first frames with Puppeteer's "
                 f"utils/render_first_frame.py, animate one with an "
                 f"image-to-video model, save as input.mp4")

    # 3. frames (resized at extraction time — Puppeteer caches flow/track/depth
    # at whatever resolution the frames are, and mismatched res crashes or OOMs)
    imgs = os.path.join(seq_dir, "imgs")
    env = dict(os.environ, PYTHONPATH=".",
               PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256")
    if not args.skip_precompute:
        os.makedirs(imgs, exist_ok=True)
        run(["ffmpeg", "-i", video_dst, "-vf",
             f"fps={args.fps},scale={args.img_size}:{args.img_size}",
             os.path.join(imgs, "frame_%04d.png"), "-y"])

        # 4. flow + depth precompute (depth standalone so VDA-Large's 1.5GB
        # is freed before optimization; patch in our Puppeteer fork makes
        # optimization skip DepthModule when the cache exists)
        run([venv_py, "utils/save_flow.py", "--input_path", os.path.dirname(out),
             "--seq_name", args.seq_name], cwd=anim, env=env)
        run([venv_py, os.path.join(pup, "precompute_depth.py"),
             "--input_path", os.path.dirname(out),
             "--seq_name", args.seq_name, "--img_size", str(args.img_size)],
            cwd=anim, env=env)

    # 5. optimization
    cmd = [venv_py, "optimization.py", "--save_path",
           os.path.join(out, "results"), "--iter", str(args.iter),
           "--input_path", os.path.dirname(out), "--img_size",
           str(args.img_size), "--seq_name", args.seq_name,
           "--save_name", f"{args.seq_name}_anim",
           "--main_renderer", args.main_renderer,
           "--additional_renderer", args.additional_renderers]
    if args.smooth_weight is not None:
        cmd += ["--smooth_weight", str(args.smooth_weight)]
    run(cmd, cwd=anim, env=env)

    print(f"\nDONE. Results under {out}/results/{args.seq_name}_anim*")
    print("Next: preview the rendered video; to bake onto the rig as keyframes "
          "and export an animated GLB, run bake_animation.py (see SKILL.md).")


if __name__ == "__main__":
    main()

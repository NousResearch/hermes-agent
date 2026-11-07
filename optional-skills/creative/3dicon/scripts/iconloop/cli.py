"""icon-loop CLI.

Stages are separate subcommands on purpose. Each one either costs money or
takes minutes, and every run wants a human to look before the next step — the
whole point of the verification stage is that it is allowed to say no.

  still    prompt        -> still.png          (image model)
  animate  still.png     -> render.mp4         (Kling, costs money)
  matte    render.mp4    -> master/*.png       (local CPU, slow, free)
  encode   master/*.png  -> icon.webp          (local, instant)
  verify   anything      -> a report you can act on
  run      all of the above
"""
import argparse
import os
import sys

from . import config, encode, kling, matte, motion as motionlib, providers, verify

# The look is stated here rather than left to the caller, because "a 3D X" on
# its own tends to return a photograph. A photoreal object then inherits
# photoreal motion, and any exaggeration in the animation reads as violence
# rather than as character — the style and the motion have to agree.
STILL_RULES = (
    "A stylised 3D icon, not a photograph: simplified friendly forms, soft "
    "matte surfaces, rounded edges, clean flat colour with gentle shading and "
    "no fine surface texture or photographic detail. "
    "Centred on a fully transparent background, nothing else in frame. "
    "Soft even lighting from the upper left. "
    "No text, no watermark, no border, no ground plane, no cast shadow."
)


def _out(args, *parts):
    os.makedirs(args.out, exist_ok=True)
    return os.path.join(args.out, *parts)


def cmd_still(args):
    prompt = args.prompt if args.raw else f"{args.prompt.strip().rstrip('.')}. {STILL_RULES}"
    n = args.variants
    picked = []
    for i in range(n):
        p = _out(args, f"still_{i + 1}.png" if n > 1 else "still.png")
        model = providers.generate(prompt, p, backend=args.backend)
        print(f"  {p}  ({model})")
        picked.append(p)
    if n > 1:
        print(f"\n{n} variants written. Pick one, rename it to still.png, then:"
              f"\n  python -m iconloop animate --out {args.out} --motion \"...\"")
    return picked


def cmd_animate(args):
    still = args.still or _out(args, "still.png")
    if not os.path.isfile(still):
        sys.exit(f"No still at {still} — run `still` first, or pass --still.")
    clause = motionlib.compose(args.motion, args.preset, args.feel, args.strategy,
                              args.emit, args.energy)
    if args.dry_run:
        # Compose and show, spend nothing. This is what the motion is agreed
        # against: the flags are abstract until you can see what they expand to.
        print(f"\n{clause}\n")
        return
    sent = _out(args, "sent_to_kling.png")
    kling.composite(still, sent)
    print(f"  composited onto backing {kling.BACKING} -> {sent}")
    print(f"  motion: {clause[:70]}...")
    fn = kling.animate_openrouter if args.via == "openrouter" else kling.animate
    model, version = fn(sent, clause, _out(args, "render.mp4"),
                        duration=args.duration, model=args.model)
    print(f"  model: {model}\n  version: {version}")


def cmd_matte(args):
    video = args.video or _out(args, "render.mp4")
    frames = matte.extract(video, _out(args, "src"), size=args.extract)
    print(f"  {len(frames)} source frames")
    A, C = matte.matte(frames, model=args.model)
    master = matte.normalize_framing(A, C, _out(args, "master"), master=args.master)
    print(f"  {len(master)} master frames at {args.master}px -> {_out(args, 'master')}")


def cmd_encode(args):
    master_dir = _out(args, "master")
    frames = encode.load(master_dir)
    if not frames:
        sys.exit(f"No master frames in {master_dir} — run `matte` first.")
    src_fps = args.source_fps

    if args.sweep:
        encode.sweep(frames, src_fps, quality=args.quality)
        print("Pick one, then re-run without --sweep, e.g.:"
              f"\n  python -m iconloop encode --out {args.out} --size 288 --fps {src_fps}\n")
        return

    fps = args.fps or src_fps
    n = len(frames) if abs(fps - src_fps) < 0.01 else max(2, round(len(frames) * fps / src_fps))
    if fps < src_fps:
        print(f"  ! {fps}fps from a {src_fps}fps source: dropping "
              f"{len(frames) - n} of {len(frames)} frames. This is the usual cause "
              f"of judder — encode at the source rate first and compare.")
    out = args.name or "icon"
    p = _out(args, f"{out}.webp")
    b = encode.webp(frames, p, fps, size=args.size, quality=args.quality, n=n)
    print(f"  {p}  {b/1024:.0f} KB  ({n} frames @ {fps}fps, {args.size}px)")
    if args.webm:
        b = encode.webm(master_dir, _out(args, f"{out}.webm"), fps, size=args.size)
        print(f"  {_out(args, out + '.webm')}  {b/1024:.0f} KB (VP9 + alpha)")
    if args.mp4:
        b = encode.mp4_flat(master_dir, _out(args, f"{out}.mp4"), fps, size=args.size, bg=args.mp4_bg)
        print(f"  {_out(args, out + '.mp4')}  {b/1024:.0f} KB (H.264, flattened on {args.mp4_bg})")


def cmd_verify(args):
    master_dir = _out(args, "master")
    ok = True
    if os.path.isdir(master_dir) and os.listdir(master_dir):
        r = verify.motion_report(master_dir)
        verify.print_report("motion", r); ok &= r["ok"]
        r = verify.loop_report(master_dir)
        verify.print_report("loop seam", r); ok &= r["ok"]
        sheet = verify.contact_sheet(master_dir, _out(args, "contact_sheet.png"))
        print(f"  --    sheet      {sheet}")
    for w in sorted(f for f in os.listdir(args.out) if f.endswith(".webp")) if os.path.isdir(args.out) else []:
        r = verify.webp_report(os.path.join(args.out, w))
        verify.print_report(w, r); ok &= r["ok"]
    if not ok:
        sys.exit(1)


def cmd_run(args):
    """Generate the still and STOP.

    Deliberately not a one-shot pipeline. Everything after the still costs
    money and four minutes, and animating art you have not looked at is how
    you pay twice. One still, one look, one decision.
    """
    cmd_still(args)
    print(f"\n  Look at {args.out}/still.png before spending anything on motion.")
    print(f"  next: python -m iconloop --out {args.out} animate --strategy <name>\n")


def main():
    config.load_dotenv()
    config.require_tool("ffmpeg", "install it (it does the frame and video work).")

    p = argparse.ArgumentParser(prog="iconloop", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="out", help="working directory (default: out)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_still(s):
        s.add_argument("--prompt", required=True, help="what the object is")
        s.add_argument("--backend", choices=("openai", "gemini", "openrouter"),
                       default=config.opt("ICONLOOP_IMAGE_BACKEND", "openai"))
        s.add_argument("--variants", type=int, default=1, help="generate N to choose from")
        s.add_argument("--raw", action="store_true", help="send the prompt verbatim")

    def add_animate(s):
        s.add_argument("--motion", help="extra motion detail, in physical language")
        s.add_argument("--strategy", choices=sorted(motionlib.STRATEGIES),
                       help="native (moves by itself) | event (does its job once) "
                            "| part (one piece moves) | surface (light travels)")
        s.add_argument("--energy", choices=("still", "calm", "lively", "playful"),
                       help="how much the object itself may move (default follows "
                            "the strategy; lively is a good starting point)")
        s.add_argument("--emit", action="store_true",
                       help="let the object briefly produce sparks, fragments, droplets")
        s.add_argument("--preset", choices=sorted(motionlib.ARCHETYPES),
                       help="optional shortcut for a common object")
        # not --quality: encode already owns that for WebP compression.
        s.add_argument("--feel", choices=sorted(motionlib.QUALITY),
                       help="how the motion should feel")
        s.add_argument("--still", help="override the still to animate")
        s.add_argument("--duration", type=int, default=5)
        s.add_argument("--via", choices=("replicate", "openrouter"),
                       default=config.opt("ICONLOOP_VIDEO_BACKEND", "openrouter"),
                       help="default openrouter (Kling 3.0); replicate is the fallback, and stops at Kling 2.5")
        s.add_argument("--model", help="video model id for the chosen backend")
        s.add_argument("--dry-run", action="store_true",
                       help="print the composed motion prompt and stop, spending nothing")

    def add_matte(s):
        s.add_argument("--video", help="override the clip to matte")
        s.add_argument("--model", default=matte.DEFAULT_MODEL)
        s.add_argument("--extract", type=int, default=640, help="matting resolution")
        s.add_argument("--master", type=int, default=512, help="master frame size")

    def add_encode(s):
        s.add_argument("--source-fps", type=float, default=24.0)
        s.add_argument("--fps", type=float, help="defaults to the source rate")
        # 384, not 288: an icon drawn at 152pt on a 3x screen needs ~456px, so
        # the old default was already being upscaled on the device it was made
        # for. Quality barely matters by comparison — going from 50 to 90 buys
        # about one level of colour out of 255 for half again the bytes — so
        # the size moves and the quality only nudges.
        s.add_argument("--size", type=int, default=384)
        s.add_argument("--quality", type=int, default=60)
        s.add_argument("--name", help="output basename")
        s.add_argument("--sweep", action="store_true", help="print size vs fps and stop")
        s.add_argument("--webm", action="store_true", help="also write VP9+alpha")
        s.add_argument("--mp4", action="store_true", help="also write flattened H.264")
        s.add_argument("--mp4-bg", default="0xFFFFFF")

    s = sub.add_parser("still"); add_still(s); s.set_defaults(fn=cmd_still)
    s = sub.add_parser("animate"); add_animate(s); s.set_defaults(fn=cmd_animate)
    s = sub.add_parser("matte"); add_matte(s); s.set_defaults(fn=cmd_matte)
    s = sub.add_parser("encode"); add_encode(s); s.set_defaults(fn=cmd_encode)
    s = sub.add_parser("verify"); s.set_defaults(fn=cmd_verify)
    s = sub.add_parser("run", help="generate the still, then stop for approval")
    add_still(s); s.set_defaults(fn=cmd_run)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

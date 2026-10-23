"""Checks that catch the failures this pipeline actually produces.

Every one of these exists because it caught something real:

  - motion_report: a Kling render that looked animated in a contact sheet but
    whose object bounding box was byte-identical in all 122 frames. One moving
    frame out of 121. It would have shipped as a 340KB file indistinguishable
    from a PNG.
  - webp_report: "it has alpha" is not the same as "it has SOFT alpha". A
    binary cutout passes a naive check and looks like a sticker on screen.
  - loop_report: a seam larger than a normal frame step is a visible hitch
    once per loop, and is invisible when you scrub frames by hand.
  - contact_sheet: halos and grey boxes only show against a background. Light
    AND dark, because each hides a different failure.
"""
import glob

import numpy as np
from PIL import Image


def _frames(master_dir):
    return [np.asarray(Image.open(p).convert("RGBA"), dtype=float)
            for p in sorted(glob.glob(f"{master_dir}/*.png"))]


def motion_report(master_dir, min_moving_fraction=0.4, min_mean_step=1.0):
    """Did the model actually animate anything? Measured inside the object only."""
    fs = _frames(master_dir)
    st = np.stack(fs)
    obj = st[..., 3].max(axis=0) > 20
    steps = np.array([np.abs(fs[i][..., :3][obj] - fs[i - 1][..., :3][obj]).mean()
                      for i in range(1, len(fs))])
    moving = int((steps > steps.mean() * 0.2).sum())
    # An absolute floor as well as a relative one. The relative test alone is
    # blind to a dead render: if nothing moves, every step still clears 20% of
    # a near-zero mean and it reports 121/121 frames "moving". A real loop runs
    # a mean step of several units; a static one sits near 0.2.
    ok = (len(steps) and moving / len(steps) >= min_moving_fraction
          and steps.mean() >= min_mean_step)
    return {
        "frames": len(fs), "mean_step": float(steps.mean()), "max_step": float(steps.max()),
        "min_step": float(steps.min()), "moving_frames": moving, "total_steps": len(steps),
        "ok": bool(ok),
        "note": "" if ok else
                "The render is largely static — re-prompt rather than encoding it. "
                "If mean_step is near zero the model returned the input image: "
                "check the prompt does not ask the object to hold its shape.",
    }


def loop_report(master_dir):
    """Compare the wrap-around seam against a typical frame step."""
    fs = _frames(master_dir)
    st = np.stack(fs)
    obj = st[..., 3].max(axis=0) > 20
    steps = np.array([np.abs(fs[i][..., :3][obj] - fs[i - 1][..., :3][obj]).mean()
                      for i in range(1, len(fs))])
    seam = float(np.abs(fs[0][..., :3][obj] - fs[-1][..., :3][obj]).mean())
    return {"seam": seam, "mean_step": float(steps.mean()),
            "ok": bool(seam <= steps.mean() * 1.5),
            "note": "" if seam <= steps.mean() * 1.5 else
                    "The loop jumps at the wrap. Was end_image set to the start image?"}


def webp_report(path):
    """Structural check straight off the bytes: animated, alpha, and soft."""
    d = open(path, "rb").read()
    anmf, alph = d.count(b"ANMF"), d.count(b"ALPH")
    im = Image.open(path)
    im.seek(im.n_frames // 2)
    a = np.asarray(im.convert("RGBA"), dtype=float)[..., 3]
    partial = int(((a > 4) & (a < 251)).sum())
    ok = anmf > 1 and alph == anmf and partial > 200
    return {
        "bytes": len(d), "frames": anmf, "alpha_chunks": alph,
        "vp8x_flags": hex(d[20]) if len(d) > 20 else "?",
        "partial_alpha_px": partial, "ok": bool(ok),
        "note": "" if ok else
                "Expected animation + per-frame alpha with soft edges; "
                "a binary cutout will read as a sticker.",
    }


def contact_sheet(master_dir, out_path, light=(251, 250, 244), dark=(38, 48, 31), cols=5):
    """Composite evenly spaced frames on a light and a dark ground."""
    paths = sorted(glob.glob(f"{master_dir}/*.png"))
    picks = [paths[int(round(i * len(paths) / cols)) % len(paths)] for i in range(cols)]
    frames = [Image.open(p).convert("RGBA") for p in picks]
    w = 144
    sheet = Image.new("RGB", (w * cols, w * 2))
    for row, bg in ((0, light), (1, dark)):
        for i, f in enumerate(frames):
            b = Image.new("RGBA", f.size, bg + (255,))
            b.alpha_composite(f)
            sheet.paste(b.convert("RGB").resize((w, w), Image.LANCZOS), (i * w, row * w))
    sheet.save(out_path)
    return out_path


# One line per check. The full numbers are still returned by each report for
# anything that wants them; dumping them all to stdout buried the one line that
# mattered, and a pass needs no evidence — only a failure does.
KEY_FIELDS = {
    "motion": ("frames", "mean_step"),
    "loop seam": ("seam", "mean_step"),
}


def print_report(title, r):
    if r.get("ok"):
        fields = KEY_FIELDS.get(title, ("frames", "bytes"))
        bits = [f"{k} {r[k]:.2f}" if isinstance(r.get(k), float) else f"{k} {r[k]}"
                for k in fields if k in r]
        print(f"  ok    {title:10s} {' · '.join(bits)}")
        return
    print(f"  FAIL  {title}")
    for k, v in r.items():
        if k in ("ok", "note"):
            continue
        print(f"          {k}: {v:.3f}" if isinstance(v, float) else f"          {k}: {v}")
    if r.get("note"):
        print(f"          {r['note']}")

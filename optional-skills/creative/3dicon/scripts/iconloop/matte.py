"""Turn the rendered clip back into frames with true alpha.

The important idea: because the clip was composited onto a backing colour we
chose, we do not have to *estimate* the foreground the way a general matting
model must. We can solve for it exactly.

For a pixel the model observed as F, with recovered alpha a over a backing BG:

    F = a*C + (1-a)*BG   =>   C = (F - (1-a)*BG) / a

That single line is worth more than a better matting model. Measured against
ground truth on a glossy 3D icon it cut edge colour error from 26.9 to 15.9,
because the backing is known rather than guessed.

Two things learned the hard way, both encoded here:
  - Do NOT temporally smooth alpha to "reduce flicker". The object moves, so
    smoothing smears a semi-transparent ghost ring around it. Measured flicker
    on a locked-camera render never justified it.
  - The soft contact shadow does not survive. Every matting model treats a
    semi-transparent shadow as background. Invisible on dark surfaces, visible
    on light ones. See docs/limitations.md.
"""
import glob
import os
import subprocess

import numpy as np
from PIL import Image

# isnet-general-use, benchmarked against ground-truth alpha on 3D icon art:
#   isnet 0.0043 MAE | birefnet-general 0.0041 but 7-30x slower
#   bria-rmbg 0.0045 | u2net 0.0050, with visible background leak
# Speed at equal accuracy is why isnet is the default.
DEFAULT_MODEL = "isnet-general-use"


def extract(video, out_dir, size=640):
    os.makedirs(out_dir, exist_ok=True)
    for old in glob.glob(f"{out_dir}/*.png"):
        os.remove(old)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video,
                    "-vf", f"scale={size}:{size}", f"{out_dir}/%04d.png"], check=True)
    return sorted(glob.glob(f"{out_dir}/*.png"))


def matte(frame_paths, model=DEFAULT_MODEL, progress=True):
    """Return (alpha[N,H,W] in 0..1, colour[N,H,W,3] uint8)."""
    from rembg import new_session, remove
    session = new_session(model)
    A, C = [], []
    for i, p in enumerate(frame_paths):
        im = Image.open(p).convert("RGB")
        F = np.asarray(im, dtype=np.float32)
        # Measure the backing per frame from the corners rather than trusting
        # the constant: the model drifts a level or two across a clip.
        bg = np.concatenate([F[:14, :14].reshape(-1, 3), F[:14, -14:].reshape(-1, 3),
                             F[-14:, :14].reshape(-1, 3), F[-14:, -14:].reshape(-1, 3)]).mean(0)
        a = np.asarray(remove(im, session=session), dtype=np.float32)[..., 3] / 255.0
        safe = np.maximum(a, 1e-3)[..., None]
        col = np.clip((F - (1 - a[..., None]) * bg) / safe, 0, 255)
        col = np.where(a[..., None] > 0.004, col, bg)
        A.append(a)
        C.append(col.astype(np.uint8))
        if progress and i % 25 == 0:
            print(f"  matte {i}/{len(frame_paths)}", flush=True)
    return np.stack(A), np.stack(C)


# The render arrives at 1440px. Mastering at 384 threw away most of it, and
# since the encode cannot add detail back, every downstream size was capped by
# a number chosen for nothing but convenience. 512 keeps enough that a 384px
# or 456px output is a genuine downscale rather than an upscale, and the master
# frames are intermediate files that never ship.
def normalize_framing(A, C, out_dir, master=512, target=None):
    """Write square RGBA masters, the object sized and centred consistently.

    The crop is computed from the UNION of the bounding boxes across the whole
    loop, not per frame and not their mean. Per-frame boxes make the object
    breathe against the canvas edge, which is more distracting than the motion
    itself. The mean is worse in a different way: it is an average of frames
    that may include the object at its largest, so the widest moment of the
    loop overflows the canvas and is cropped. The union is the only choice that
    guarantees every frame fits.

    target is (width, height, cx, cy) as fractions of the canvas. Default keeps
    the object at ~74% of the frame with a little headroom below centre, which
    suits an icon that will sit above a label.
    """
    tw, th, tcx, tcy = target or (0.74, 0.74, 0.5, 0.488)
    boxes = []
    for i in range(len(A)):
        ys, xs = np.nonzero(A[i] > 0.5)
        boxes.append([xs.min(), xs.max(), ys.min(), ys.max()])
    boxes = np.array(boxes, float)
    b = np.array([boxes[:, 0].min(), boxes[:, 1].max(),
                  boxes[:, 2].min(), boxes[:, 3].max()])
    ow, oh = b[1] - b[0] + 1, b[3] - b[2] + 1
    ocx, ocy = (b[0] + b[1]) / 2, (b[2] + b[3]) / 2
    # max, not min. min sizes the canvas to whichever dimension needs less
    # room, so anything markedly taller than it is wide (or the reverse) is
    # cropped at both ends — and it never showed up, because a roughly square
    # icon is the case where the two agree.
    canvas = max(ow / tw, oh / th)
    x0, y0 = ocx - tcx * canvas, ocy - tcy * canvas

    os.makedirs(out_dir, exist_ok=True)
    for old in glob.glob(f"{out_dir}/*.png"):
        os.remove(old)
    # The crop box can start outside the frame: sizing the canvas to fit the
    # larger dimension means a wide or tall object needs more room than the
    # source has on the other axis, and PIL refuses a negative box offset. Pad
    # onto a transparent canvas instead of cropping, which is the same result
    # and cannot go out of bounds.
    side = int(np.ceil(canvas))
    for i in range(len(A)):
        rgba = np.dstack([C[i], (A[i] * 255).clip(0, 255)]).astype(np.uint8)
        src = Image.fromarray(rgba, "RGBA")
        pad = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        pad.paste(src, (int(round(-x0)), int(round(-y0))))
        pad.resize((master, master), Image.LANCZOS).save(f"{out_dir}/{i:04d}.png")
    return sorted(glob.glob(f"{out_dir}/*.png"))

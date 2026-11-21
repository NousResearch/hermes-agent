#!/usr/bin/env python3
"""Generate TEXT-FREE background graphics for Persian decks.

Persian text must never be baked into raster images (matplotlib cannot shape
it); overlays go into the pptx as native RTL text boxes.

  python make_graphics.py --kind pattern --out pattern.png \
      [--w 13.333 --h 7.5 --color C9A227 --alpha 0.10 --dpi 150]

kinds:
  pattern  8-point star tessellation (subtle, for dark title/closing slides)
  grid     fine square grid (lighter motif)
"""
import argparse
import sys

import numpy as np


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", help="output PNG path")
    ap.add_argument("--kind", default="pattern", choices=["pattern", "grid"])
    ap.add_argument("--w", type=float, default=13.333)
    ap.add_argument("--h", type=float, default=7.5)
    ap.add_argument("--color", default="C9A227")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args(argv)

    r, g, b = (int(args.color[i:i + 2], 16) / 255 for i in (0, 2, 4))
    col = (r, g, b, args.alpha)

    fig = plt.figure(figsize=(args.w, args.h), dpi=args.dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, args.w)
    ax.set_ylim(0, args.h)
    ax.axis("off")
    fig.patch.set_alpha(0.0)

    if args.kind == "pattern":
        R, ri, S = 0.52, 0.21, 1.45
        for gx in np.arange(-S, args.w + S, S):
            for gy in np.arange(-S, args.h + S, S):
                theta = np.linspace(0, 2 * np.pi, 17)[:-1] + np.pi / 8
                rad = np.where(np.arange(16) % 2 == 0, R, ri)
                xs = gx + rad * np.cos(theta)
                ys = gy + rad * np.sin(theta)
                ax.add_patch(mpatches.Polygon(np.column_stack([xs, ys]), closed=True,
                                              facecolor="none", edgecolor=col, lw=1.1))
                ax.plot([gx], [gy], marker="o", ms=1.6,
                        color=(r, g, b, min(1.0, args.alpha * 1.6)))
    else:
        for gx in np.arange(0, args.w + 1, 0.9):
            ax.axvline(gx, color=col, lw=0.8)
            ax.axhline(gx, color=col, lw=0.8)

    fig.savefig(args.out, transparent=True)
    plt.close(fig)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

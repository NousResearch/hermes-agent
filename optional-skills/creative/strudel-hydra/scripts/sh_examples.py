#!/usr/bin/env python3
"""Gallery of strudel-hydra livesets: push one to a running server, or export
one as a standalone self-contained .html.

Each set pairs strudel audio with audio-reactive hydra visuals (`a.fft[...]`
reads the running sound). Function availability tracks the pinned strudel/hydra
versions in templates/page.html — treat these as starting points to adapt, not
frozen APIs.

Run from this directory:  cd "$SKILL/scripts"
    python3 sh_examples.py --list
    python3 sh_examples.py pulse
    python3 sh_examples.py acid --export acid.html
"""
import argparse
import json
import sys
from pathlib import Path

from sh_client import push_set

TEMPLATE = Path(__file__).resolve().parent.parent / "templates" / "page.html"

SETS = {
    "pulse": {
        "label": "four-on-the-floor pulse + kaleidoscope",
        "audio": (
            'stack('
            '  note("c1*4").s("sawtooth").lpf(320).decay(.12).sustain(0).gain(.8),'
            '  note("~ c4 ~ c4").s("square").decay(.05).sustain(0).gain(.3)'
            ')'
        ),
        "visual": (
            "osc(30, 0.05, 0.9).kaleid(4)"
            ".modulateScale(osc(4).rotate(0.3), () => 0.2 + a.fft[0])"
            ".rotate(0, 0.05).out()"
        ),
    },
    "bells": {
        "label": "euclidean bells + voronoi pixelate",
        "audio": (
            'note("<c5 e5 g5 b5>(3,8)").s("triangle")'
            '.room(.6).decay(.3).sustain(0).gain(.5)'
        ),
        "visual": (
            "voronoi(8, 0.3, 0.2).color(0.9, 0.4, 1)"
            ".modulatePixelate(noise(3), () => 10 + a.fft[1] * 40).out()"
        ),
    },
    "drone": {
        "label": "sawtooth drone + slow kaleidoscope drift",
        "audio": (
            'note("c2").add(note("0,7,12")).s("sawtooth")'
            '.lpf(sine.range(200, 900).slow(8)).gain(.4)'
        ),
        "visual": (
            "osc(4, 0.1, 0.6).kaleid(3).color(0.4, 0.3, 0.9)"
            ".modulateRotate(osc(1), () => 0.2 + a.fft[0]).out()"
        ),
    },
    "acid": {
        "label": "303 acid line + reactive shapes",
        "audio": (
            'note("c2 eb2 g2 c3 bb2 g2 eb2 c2".fast(2)).s("sawtooth")'
            '.lpf(sine.range(300, 1500).fast(4)).resonance(15)'
            '.decay(.1).sustain(.2).gain(.5)'
        ),
        "visual": (
            "osc(40, 0.15, 0.4).kaleid(6).color(1, 0.3, 0.6)"
            ".modulateRotate(osc(2), () => 0.3 + a.fft[0])"
            ".modulateScale(noise(2), 0.2).out()"
        ),
    },
}


def export(name, out_path):
    """Bake a set into a standalone page that plays without the server."""
    s = SETS[name]
    html = TEMPLATE.read_text(encoding="utf-8")
    inject = "<script>window.__SET__ = " + json.dumps(s) + ";</script>\n</body>"
    if "</body>" not in html:
        raise SystemExit("template has no </body> to inject before")
    html = html.replace("</body>", inject, 1)
    Path(out_path).write_text(html, encoding="utf-8")
    print(f"exported '{name}' -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="strudel-hydra liveset gallery")
    ap.add_argument("name", nargs="?", help="set to push (see --list)")
    ap.add_argument("--list", action="store_true", help="list gallery sets")
    ap.add_argument("--base", default="http://127.0.0.1:8765", help="server base URL")
    ap.add_argument("--export", metavar="FILE", help="write a standalone .html instead of pushing")
    args = ap.parse_args()

    if args.list or not args.name:
        for k, v in SETS.items():
            print(f"{k:8} {v['label']}")
        if not args.name:
            return
        return

    if args.name not in SETS:
        sys.exit(f"unknown set '{args.name}'. Try: {', '.join(SETS)}")

    if args.export:
        export(args.name, args.export)
        return

    s = SETS[args.name]
    print(json.dumps(push_set(args.base, s.get("audio"), s.get("visual"), s["label"])))


if __name__ == "__main__":
    main()

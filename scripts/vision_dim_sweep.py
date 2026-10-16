#!/usr/bin/env python3
"""Print the vision resizer's limit behaviour for this working tree and, optionally, a base commit.

Every number in the pull request's tables comes from this script or from
``tests/tools/test_vision_dimension_cap_fit.py``. Fixtures are synthesised with Pillow, so nothing is
read from the developer's disk.

    python scripts/vision_dim_sweep.py --base 848fc0ad8b
    python scripts/vision_dim_sweep.py --base origin/main --quick

``--base`` loads that commit's ``tools/vision_tools.py`` from git into a temporary module and runs the
same cases against it, so the base and the working tree are measured side by side in one process.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import io
import itertools
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

KIB = 1024
MIB = 1024 * KIB
CAP = 1568  # ``vision_tools._EMBED_MAX_DIMENSION``

# The (dimension cap, byte budget) pairs the callers pass, plus one cap-less combination no caller
# passes (the cap-less paths budget in MiB) kept as an edge probe. Budgets are counted in base64
# characters of the data URL, so 256 KB means 262,144 characters.
CALLER_PARAMS = (
    ("native embed, default budget", CAP, 256 * KIB),
    ("cap-less, default budget (no caller passes this)", None, 256 * KIB),
    ("aux pre-flight (no cap)", None, 20 * MIB),
    ("provider-rejection retry", None, 5 * MIB),
    ("compression shrink", None, 4 * MIB),
    ("clamped budget floor", CAP, 64 * KIB),
)

CASES = (
    ("1569x1177 (one pixel over the cap)", (1569, 1177), "JPEG", MIB, CAP),
    ("4000x3000", (4000, 3000), "JPEG", MIB, CAP),
    ("3200x2400 noise", (3200, 2400), "JPEG", 380 * KIB, CAP),
    ("2000x1500 PNG", (2000, 1500), "PNG", 8 * MIB, CAP),
    ("2048x1536 noise, cap 800", (2048, 1536), "JPEG", 200 * KIB, 800),
    ("2048x1536 noise, cap 1024", (2048, 1536), "JPEG", 200 * KIB, 1024),
    ("2048x1536 noise, cap 1280", (2048, 1536), "JPEG", 200 * KIB, 1280),
    ("2048x1536 noise, cap 1568", (2048, 1536), "JPEG", 200 * KIB, CAP),
    ("60000x64 strip", (60000, 64), "JPEG", 256 * KIB, CAP),
    ("4000x1 (force_jpeg)", (4000, 1), "JPEG", 256 * KIB, CAP),
    ("3000x2000 PNG, 1.5 KB budget", (3000, 2000), "PNG", 1500, CAP),
    ("200x80, 200-character budget", (200, 80), "JPEG", 200, None),
)

# Cap multiples in five aspect ratios plus the edges of the cap, for the bound sweep. Sources stay
# modest (long edge 4000) so the sweep is fast; a cap above the source's own edge simply leaves it
# untouched, which is part of the claim.
BOUND_SHAPES = ((4, 3), (3, 4), (1, 1), (4, 1), (1, 4))
BOUND_CAPS = (1, 64, CAP - 1, CAP, CAP + 1, 2 * CAP, 3 * CAP, 4 * CAP, 8 * CAP)
BOUND_LONG_EDGE = 4000


def build(path: Path, size: tuple[int, int], fmt: str) -> Path:
    """A noisy fixture, incompressible enough that a byte budget binds before the cap does."""
    if not path.exists():
        image = Image.new("RGB", size)
        rng = random.Random(7)
        image.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256))
                       for _ in range(size[0] * size[1])])
        image.save(path, fmt, **({"quality": 95} if fmt == "JPEG" else {}))
    return path


def load_base(ref: str):
    """The resizer as of *ref*, imported from a temporary module, or None."""
    if not ref:
        return None
    blob = subprocess.run(["git", "show", f"{ref}:tools/vision_tools.py"], cwd=ROOT,
                          capture_output=True, text=True, timeout=120, check=False)
    if blob.returncode != 0:
        raise SystemExit(f"cannot read tools/vision_tools.py at {ref}: {blob.stderr.strip()}")
    target = Path(tempfile.mkdtemp(prefix="vision-sweep-")) / "vision_tools_base.py"
    target.write_text(blob.stdout, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("vision_tools_base", target)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {target} as a module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call(module, path: Path, budget: int, cap: int | None, mime: str = "image/jpeg",
         force_jpeg: bool = True) -> dict:
    """One resizer call, with the encoder and resize counters the tables quote."""
    counters = {"encodes": 0, "resizes": 0}
    real_save = Image.Image.save
    real_resize = Image.Image.resize

    def counted_save(self, fp, *args, **kwargs):
        counters["encodes"] += 1
        return real_save(self, fp, *args, **kwargs)

    def counted_resize(self, *args, **kwargs):
        counters["resizes"] += 1
        return real_resize(self, *args, **kwargs)

    Image.Image.save, Image.Image.resize = counted_save, counted_resize
    started = time.perf_counter()
    outcome = {"dims": None, "payload": None, "error": "", "seconds": 0.0}
    try:
        url = module._resize_image_for_vision(path, mime, max_base64_bytes=budget,
                                              max_dimension=cap, force_jpeg=force_jpeg)
        outcome["dims"] = list(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).size)
        outcome["payload"] = len(url)
    except OSError as exc:  # libjpeg refuses some degenerate shapes; reported as a cell
        outcome["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        Image.Image.save, Image.Image.resize = real_save, real_resize
    outcome["seconds"] = round(time.perf_counter() - started, 2)
    outcome.update(counters)
    return outcome


def row(label: str, cells: list[str]) -> str:
    return f"| {label:<34} | " + " | ".join(cells) + " |"


def cell(outcome: dict) -> str:
    if outcome["error"]:
        return outcome["error"][:64]
    dims = outcome["dims"]
    return f"{dims[0]}x{dims[1]}, {outcome['payload']} chars, {outcome['encodes']} enc"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="", help="commit/ref to measure against (git show <ref>:...)")
    parser.add_argument("--quick", action="store_true", help="the case rows only")
    args = parser.parse_args()

    import tools.vision_tools as head

    workdir = Path(tempfile.mkdtemp(prefix="vision-sweep-fixtures-"))
    arms = [("this tree", head)] + ([("base " + args.base[:10], load_base(args.base))] if args.base else [])
    fixtures = {(size, fmt): build(workdir / f"{size[0]}x{size[1]}.{fmt.lower()}", size, fmt)
                for _label, size, fmt, _b, _c in CASES}

    print(f"# vision resizer sweep: {len(arms)} arm(s), fixtures in {workdir}")
    print(row("case", [f"{name} | dims, payload, encodes" for name, _m in arms]))
    print(row("-" * 34, ["-" * 40 for _ in arms]))
    for label, size, fmt, budget, cap in CASES:
        cells = [cell(call(module, fixtures[(size, fmt)], budget, cap,
                           mime="image/png" if fmt == "PNG" else "image/jpeg",
                           force_jpeg=fmt != "PNG"))
                 for _name, module in arms]
        print(row(label, cells))

    if args.quick:
        return 0

    print("\n## Bound sweep: no returned longest edge above the cap, no distortion beyond one pixel")
    for _name, module in arms:
        over = distorted = calls = 0
        for shape in BOUND_SHAPES:
            long_edge = BOUND_LONG_EDGE
            size = (long_edge if shape[0] >= shape[1] else max(long_edge * shape[0] // shape[1], 1),
                    long_edge if shape[1] > shape[0] else max(long_edge * shape[1] // shape[0], 1))
            source = build(workdir / f"b{size[0]}x{size[1]}.jpg", size, "JPEG")
            for cap in BOUND_CAPS:
                outcome = call(module, source, 64 * MIB, cap)
                calls += 1
                if outcome["dims"] and max(outcome["dims"]) > cap:
                    over += 1
                if outcome["dims"]:
                    # The returned ratio must match the source's within a single pixel at the source's
                    # own width; a percentage test would call unavoidable rounding at tiny results a
                    # distortion.
                    w_out, h_out = outcome["dims"]
                    if abs(w_out / size[0] - h_out / size[1]) * size[0] > 1.0:
                        distorted += 1
        print(f"  {_name:<22} {calls} calls, {over} over the cap, {distorted} distorted")

    print("\n## Budget sweep at cap 1568: the answer never shrinks when the budget grows")
    for _name, module in arms:
        source = fixtures[((2048, 1536), "JPEG")]
        areas = [call(module, source, int(60 * KIB * (10 / 1) ** (i / 31)), CAP)["dims"] for i in range(32)]
        areas = [a[0] * a[1] for a in areas if a]
        inversions = sum(1 for a, b in itertools.pairwise(areas) if b < a)
        print(f"  {_name:<22} 32 budgets 61,440..614,400 chars, {inversions} inversions")

    print("\n## Every caller's own (cap, budget), plus one cap-less edge probe, over every fixture")
    for _name, module in arms:
        over = under = raised = calls = 0
        for label, cap, budget in CALLER_PARAMS:
            for _case, size, fmt, _b, _c in CASES:
                outcome = call(module, fixtures[(size, fmt)], budget, cap,
                               mime="image/png" if fmt == "PNG" else "image/jpeg",
                               force_jpeg=fmt != "PNG")
                calls += 1
                if outcome["error"]:
                    raised += 1
                else:
                    if cap is not None and max(outcome["dims"]) > cap:
                        over += 1
                    if outcome["payload"] > budget:
                        under += 1
        print(f"  {_name:<22} {calls} calls, {over} over the caller's cap, "
              f"{under} over its budget, {raised} exceptions")

    print("\n## Unit costs over a 4x3 cap/budget grid")
    for _name, module in arms:
        worst = (0, "")
        for cap in (800, 1024, CAP):
            for budget in (60 * KIB, 200 * KIB, 600 * KIB):
                outcome = call(module, fixtures[((2048, 1536), "JPEG")], budget, cap)
                if outcome["encodes"] > worst[0]:
                    worst = (outcome["encodes"], f"cap {cap}, budget {budget}: {outcome['dims']}, "
                                                 f"{outcome['seconds']}s")
        print(f"  {_name:<22} worst cell {worst[1]} in {worst[0]} encodes")
    print("\nRun `bash scripts/run_tests.sh tests/tools/test_vision_dimension_cap_fit.py` for the "
          "properties this sweep only samples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

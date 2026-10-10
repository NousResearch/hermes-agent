#!/usr/bin/env python3
"""Manifest-driven fashion batch generator (STEP A recolor + STEP B worn).
Provider: OpenRouter gpt-image family. Idempotent: skips any step whose OUTPUT
already exists, so re-running never re-bills. Copy & adapt the MANIFEST/prompts.
Usage:  python batch_gen.py                # all garments
        python batch_gen.py --stepa        # recolor only
        python batch_gen.py --stepb        # worn only
        python batch_gen.py GARMENT_NAME   # single garment (exact dict key)
"""
import base64, json, os, sys, time, urllib.request

ORKEY = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
MODEL = os.environ.get("IMG_MODEL", "openai/gpt-5.4-image-2")
BASE = os.path.dirname(os.path.abspath(__file__))
IMPOSSIBLE_KEY = "$$$never-matches$$$"  # sentinel => unset => run all

# manifest: garment -> {ghost: rel_path, model: rel_path, swatch: {COLOR: rel_path}}
MANIFEST = {
    # "ALASKA": {
    #   "ghost":  "ALASKA/IMG_5967.jpg",
    #   "model":  "MODELLE/m1.jpg",           # REQUIRED for STEP B (user-provided photo)
    #   "swatch": {"NOCCIOLA-302": "ALASKA/Bazaart_663DFE11.jpeg", "...": "..."},
    # },
}

STEPA_PROMPT = ("Recolor this ghost-mannequin garment to the exact solid color shown in the "
    "reference swatch (reference 2). Keep the garment shape, fabric texture, knit/fur structure, "
    "stitching, zippers, buttons, pockets and labels exactly identical — change only the color. "
    "Same lighting, same neutral grey background, invisible/ghost mannequin product photography. "
    "Do not add or alter any text or logos.")

STEPB_PROMPT = ("The model from the reference image wearing the garment from the reference image, "
    "standing front pose, arms relaxed at sides, neutral light grey background, full body shot, "
    "professional ecommerce fashion photography, clean studio lighting. "
    "Preserve all original garment details exactly as shown: stitching, hardware, zippers, buttons "
    "and labels must remain identical to the reference, do not generate any text, logos or writing "
    "on buttons, zippers or labels, keep all branding elements blank and anonymous. "
    "Keep the model's body proportions anatomically natural and identical to the reference "
    "photo: the legs must be a realistic length relative to the torso, with no elongation or "
    "vertical stretching of the figure, and the garment must keep its true length relative to "
    "the body exactly as in the ghost/reference image, do not lengthen the coat, do not stretch "
    "the model's height, keep the head-to-body ratio natural, do not slim or stretch the limbs.")

# Poses: each STEP B variant appends its own pose clause to STEPB_PROMPT.
POSES = {
    "front":       "standing front pose, arms relaxed at sides",
    "34":          "three-quarter angle pose, arms relaxed",
    "back":        "back-facing pose, standing upright",
    "detail":      "close-up on the garment fabric and construction",
    "bust34":      "framed as a close three-quarter bust shot from the waist up, the lower body and legs cropped out of frame, slight three-quarter angle, arms relaxed",
    "editorial":   "a dynamic editorial fashion pose, weight shifted, one hand in coat pocket or adjusting the collar, confident stance",
}
# All poses share the SAME neutral studio background (editorial is a pose, NOT a
# location): the scene is never changed to a street/outdoor setting.
STUDIO_SCENE = ("neutral light grey background, full body shot, professional ecommerce "
    "fashion photography, clean studio lighting")

# Surrounding-outfit guard (MANDATORY on every STEP B): the model's base layers must be
# coherent with the garment — elegant, modern, casual. NO shorts, NO skirts. The reference
# garment itself must never be altered.
OUTFIT_GUARD = ("The model's surrounding outfit (base layers under/around the garment) must be "
    "elegant, modern and casual and coherent with the garment worn — refined knitwear, a "
    "turtleneck or long-sleeve top, and full-length tailored trousers or jeans. Never use "
    "shorts and never use skirts. The reference garment itself must remain exactly identical "
    "to the ghost reference — do not add, remove or alter the garment, its shape, details, "
    "color or its relation to the body.")

# Outfit consistency: for one colorway the SAME surrounding outfit must appear across all
# three poses (front/bust34/editorial). Same top, same trousers, same shoes in every shot.
OUTFIT_CONSISTENCY = ("For this colorway, keep the SAME surrounding outfit across all generated "
    "poses (front, bust34 and editorial): the model wears the exact same top, same trousers "
    "and same shoes in every shot. Do not change the base layers from one pose to the next. "
    "The reference garment itself must remain exactly identical to the ghost reference in "
    "every pose.")

def data_url(path):
    b = open(path, "rb").read()
    ext = "png" if path.lower().endswith(".png") else "jpeg"
    return f"data:image/{ext};base64," + base64.b64encode(b).decode()

def _resize_to(path, size):
    """Downsample the image at `path` to exactly `size` (W,H) px, in place, as PNG."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    # resize keeping aspect, then centre-crop to exact size (source is already 4:5)
    im = im.resize(size, Image.LANCZOS)
    im.save(path, "PNG")

# Final output must be exactly 800x1000 px (4:5). The API generates at aspect 4:5
# (e.g. ~1024x1280); we then downsample to the exact target so every file is pixel-identical.
OUTPUT_SIZE = (800, 1000)

def gen(prompt, refs, out, aspect="4:5", quality="high", retries=3):
    payload = {"model": MODEL, "prompt": prompt, "n": 1, "quality": quality,
               "aspect_ratio": aspect,
               "input_references": [{"type": "image_url", "image_url": {"url": r}} for r in refs]}
    last = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/images",
                data=json.dumps(payload).encode(),
                headers={"Authorization": f"Bearer {ORKEY}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.load(resp)
            img = base64.b64decode(data["data"][0]["b64_json"])
            with open(out, "wb") as f:
                f.write(img)
            _resize_to(out, OUTPUT_SIZE)
            return os.path.getsize(out), data.get("usage", {}).get("cost")
        except Exception as e:
            last = e
            print(f"    [retry {attempt}] {e}", flush=True)
            time.sleep(5 * attempt)
    raise RuntimeError(f"FAILED {out}: {last}")

# selector: exact garment names, else run all; flags for step A/B only
_want = [a for a in sys.argv[1:] if not a.startswith("--")]
ovr = None if not _want else {x: x for x in _want}
only_stepa, only_stepb = "--stepa" in sys.argv, "--stepb" in sys.argv

summary = []
for capo, cfg in MANIFEST.items():
    if ovr is not None and capo not in ovr:
        continue
    ghost = f"{BASE}/{cfg['ghost']}"
    modela = f"{BASE}/{cfg.get('model', IMPOSSIBLE_KEY)}"
    print(f"\n========== {capo} ==========", flush=True)
    for colore, swatchf in cfg["swatch"].items():
        g_out = f"{BASE}/{capo}/ghost_{colore}.png"
        w_out = f"{BASE}/{capo}/indossato_front_{colore}.png"
        print(f"\n--- {capo} / {colore} ---", flush=True)
        # STEP A: skip if ghost output already exists
        if not only_stepb and not os.path.exists(g_out):
            try:
                n, c = gen(STEPA_PROMPT, [data_url(ghost), data_url(f"{BASE}/{swatchf}")], g_out)
                print(f"  STEP A ok ${c}", flush=True); summary.append(("A", capo, colore, c))
            except Exception as e:
                print(f"  STEP A FAIL {colore}: {e}", flush=True)
        else:
            print("  STEP A: skip", flush=True)
        # STEP B: skip if ALL worn outputs exist or model missing
        if only_stepa:
            print("  STEP B: skip (--stepa)", flush=True)
        elif not os.path.exists(g_out):
            print("  STEP B skip: ghost missing", flush=True)
        elif not os.path.exists(modela):
            print(f"  STEP B skip: model photo missing ({modela})", flush=True)
        else:
            for posa in cfg.get("poses", ["front", "bust34", "editorial"]):
                w_out = f"{BASE}/{capo}/indossato_{posa}_{colore}.png"
                if os.path.exists(w_out):
                    print(f"  STEP B {posa}: skip", flush=True)
                    continue
                pose_clause = POSES.get(posa, POSES["front"])
                scene = STUDIO_SCENE  # all poses share the same studio background
                prompt = f"{STEPB_PROMPT} {pose_clause}, {scene}. {OUTFIT_GUARD} {OUTFIT_CONSISTENCY}."
                try:
                    n, c = gen(prompt, [data_url(modela), data_url(g_out)], w_out)
                    print(f"  STEP B {posa} ok ${c}", flush=True); summary.append(("B", capo, colore, posa, c))
                except Exception as e:
                    print(f"  STEP B {posa} FAIL {colore}: {e}", flush=True)

tot = sum((c or 0) for *_, c in summary)
print("\n===== RIEPILOGO =====")
for row in summary:
    if row[0] == "A":
        s, capo, colore, c = row
        print(f"  STEP A {capo} {colore}: ${c}")
    else:
        s, capo, colore, posa, c = row
        print(f"  STEP B {capo} {colore} [{posa}]: ${c}")
print(f"COSTO TOTALE: ${tot:.4f}")
print("DONE")
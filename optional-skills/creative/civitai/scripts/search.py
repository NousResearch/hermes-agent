"""civitai search: models / images / top-models / top-images."""
import argparse
import os
import sys

from _common import (
    api_get, meili_search, parse_browsing_level,
    fmt_int, truncate, die, emit_json, slim_model,
)

MODEL_TYPES = [
    "Checkpoint", "LORA", "LoCon", "DoRA", "TextualInversion", "Hypernetwork",
    "Controlnet", "Poses", "AestheticGradient", "Wildcards", "MotionModule",
    "VAE", "Upscaler", "Workflows", "Detection", "Other",
]
PERIODS = ["AllTime", "Year", "Month", "Week", "Day"]
MODEL_SORTS = [
    "Most Downloaded", "Highest Rated", "Most Collected", "Most Comments",
    "Most Tipped", "Newest", "Oldest",
]
IMAGE_SORTS = [
    "Most Reactions", "Most Comments", "Most Collected", "Newest", "Oldest",
]
REST_SORTS = {"Highest Rated", "Most Downloaded", "Newest"}


def _has_key():
    return bool(os.environ.get("CIVITAI_API_KEY"))


def _g(args, k, d=None):
    return getattr(args, k, d)


def _print_list(items, fmt, header):
    if not items:
        print("No results.")
        return
    print(f"{header} · {len(items)}\n")
    print("\n\n".join(fmt(i, x) for i, x in enumerate(items, 1)) + "\n")


def _fmt_model(idx, h):
    metrics = h.get("metrics") or h.get("stats") or {}
    user = h.get("user") or h.get("creator") or {}
    version = h.get("version") or ((h.get("modelVersions") or [{}])[0])
    trig = h.get("triggerWords") or version.get("trainedWords") or []
    tags = h.get("tags") or []

    out = [
        f"{idx}. #{h.get('id', '?')} \"{h.get('name', '?')}\" by {user.get('username', '?')}",
        f"   {h.get('type', '?')} · base: {version.get('baseModel', '?')} · "
        f"↓ {fmt_int(metrics.get('downloadCount', 0))} · "
        f"★ {fmt_int(metrics.get('thumbsUpCount', 0))}",
    ]
    if trig:
        out.append(f"   trigger: {', '.join(str(w) for w in trig[:5])}")
    if tags:
        tn = [t.get("name", "?") if isinstance(t, dict) else str(t) for t in tags[:5]]
        out.append(f"   tags: {', '.join(tn)}")
    return "\n".join(out)


def _fmt_image(idx, im):
    iid = im.get("id", "?")
    user = im.get("username") or (im.get("user") or {}).get("username", "?")
    stats = im.get("stats") or {}
    react = sum(stats.get(k, 0) for k in ("heartCount", "likeCount", "laughCount", "cryCount"))
    meta = im.get("meta") or {}

    def _names(seq, n):
        return [t.get("name") if isinstance(t, dict) else str(t)
                for t in (seq or [])[:n]]

    sub = []
    for label, key in (("Tools", "tools"), ("Technique", "techniques")):
        names = _names(im.get(key), 3)
        if names:
            sub.append(f"{label}: {', '.join(names)}")
    sub.append(f"Base: {im.get('baseModel', '?')}")

    out = [
        f"{idx}. Image #{iid} by {user} · {fmt_int(react)} reactions",
        "   " + " · ".join(sub),
    ]
    if meta.get("prompt"):
        out.append(f"   Prompt: {truncate(meta['prompt'], 300)}")
    if meta.get("negativePrompt"):
        out.append(f"   Negative: {truncate(meta['negativePrompt'], 200)}")
    p = [f"{k}: {meta[k]}"
         for k in ("steps", "sampler", "cfgScale", "seed")
         if meta.get(k) is not None]
    if p:
        out.append("   " + " · ".join(p))

    loras = []
    for r in (meta.get("resources") or []):
        if (r.get("type") or "").lower() in ("lora", "locon", "dora"):
            w = r.get("weight")
            loras.append(f"{r.get('name', '?')}" + (f" ({w})" if w is not None else ""))
    if loras:
        out.append(f"   LoRAs: {', '.join(loras[:5])}")
    out.append(f"   URL: https://civitai.com/images/{iid}")
    return "\n".join(out)


def _nsfw_flag(args):
    """Resolve --nsfw into the tri-state expected by the API helpers.

    True  → include NSFW (requires a valid CIVITAI_API_KEY)
    False → forced SFW (either user passed nothing, or --nsfw without a key)
    None  → defer to server default

    --nsfw is the single unified opt-in across all four subcommands; there
    is no --sfw-only because the default already excludes NSFW unless the
    user asks for it.
    """
    if _g(args, "nsfw"):
        if _has_key():
            return True
        print("warning: --nsfw without CIVITAI_API_KEY — SFW results",
              file=sys.stderr)
        return False
    return None


def _do_models(args):
    rest_only = bool(_g(args, "ids") or _g(args, "favorites"))
    nsfw = _nsfw_flag(args)
    q = _g(args, "query")

    if q and not rest_only:
        r = meili_search(
            q,
            types=_g(args, "type"),
            base_model=_g(args, "base_model"),
            tag=_g(args, "tag"),
            username=_g(args, "username"),
            sort=args.sort,
            nsfw=nsfw,
            limit=args.limit,
        )
        if args.json:
            emit_json(r)
            return
        _print_list(r.get("hits", []), _fmt_model, "Models (Meilisearch)")
        if r.get("hits"):
            print(f"— {fmt_int(r.get('estimatedTotalHits', 0))} estimated · "
                  f"{r.get('processingTimeMs', '?')}ms (Meilisearch)")
        return

    sort = args.sort if args.sort in REST_SORTS else "Most Downloaded"
    bm = _g(args, "base_model")
    raw_ids = _g(args, "ids")
    params = {
        "limit":      min(args.limit, 100),
        "types":      _g(args, "type"),
        "sort":       sort,
        "period":     args.period,
        "username":   _g(args, "username"),
        "tag":        _g(args, "tag"),
        "favorites":  _g(args, "favorites") or None,
        "ids":        raw_ids,
        "nsfw":       nsfw,
        "baseModels": [bm] if bm else None,
    }
    data = api_get("models", params)

    # Detect Civitai's silent NSFW-drop when --ids was used. The REST endpoint
    # quietly excludes NSFW models from the response when nsfw isn't true,
    # without erroring or flagging which IDs were dropped. Diff requested vs
    # returned and surface the gap so the agent can choose to retry with --nsfw.
    if raw_ids:
        try:
            requested = {int(x.strip()) for x in raw_ids.split(",") if x.strip()}
        except ValueError:
            requested = set()
        returned = {m.get("id") for m in (data.get("items") or []) if m.get("id")}
        missing = requested - returned
        if missing and requested:
            sample = ",".join(str(x) for x in sorted(missing)[:8])
            more = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
            if not _g(args, "nsfw"):
                print(
                    f"warning: {len(missing)}/{len(requested)} requested IDs "
                    f"missing from response — Civitai silently drops NSFW "
                    f"models when --nsfw isn't set. Add --nsfw to fetch all. "
                    f"Missing: {sample}{more}",
                    file=sys.stderr,
                )
            else:
                print(
                    f"warning: {len(missing)}/{len(requested)} requested IDs "
                    f"missing from response (deleted, private, or unavailable). "
                    f"Missing: {sample}{more}",
                    file=sys.stderr,
                )

    if args.json:
        # Slim each model — raw REST payloads can hit 600 KB+ for a 16-ID
        # batch fetch and blow past the terminal transport cap.
        items = [slim_model(m) for m in (data.get("items") or [])]
        emit_json({"items": items, "metadata": data.get("metadata") or {}})
        return
    _print_list(data.get("items", []), _fmt_model, "Models (REST)")


def _do_images(args):
    mvid = _g(args, "model_version_id")
    mid = _g(args, "model_id")
    if mid and not mvid:
        m = api_get(f"models/{mid}")
        versions = m.get("modelVersions") or []
        if not versions:
            die(f"model {mid} has no versions")
        mvid = versions[0]["id"]
        print(f"# resolved model {mid} → version {mvid}", file=sys.stderr)

    # --nsfw is the unified shorthand across all four subcommands. On image
    # commands it desugars to --browsing-level X,XXX when no explicit
    # --browsing-level was given. Explicit --browsing-level always wins so
    # power users can still filter by individual levels (e.g. R only).
    bl = _g(args, "browsing_level")
    if not bl and _g(args, "nsfw"):
        bl = "X,XXX"
    if bl and not _has_key() and parse_browsing_level(bl) & 24:
        print("warning: NSFW without CIVITAI_API_KEY — downgrading to R",
              file=sys.stderr)
        bl = "R"

    params = {
        "modelVersionId": mvid,
        "username":       _g(args, "username"),
        "sort":           args.sort,
        "period":         args.period,
        "limit":          min(args.limit, 200),
        "hasMeta":        True if _g(args, "has_meta") else None,
        "type":           _g(args, "content_type"),
        "browsingLevel":  parse_browsing_level(bl) if bl else None,
        "tag":            _g(args, "tag"),
        "baseModel":      _g(args, "base_model"),
    }
    data = api_get("images", params)
    if args.json:
        emit_json(data)
        return
    _print_list(data.get("items", []), _fmt_image, "Images")


def _common_args(x, image=False):
    x.add_argument("--limit", "-n", type=int, default=5 if image else 10)
    x.add_argument("--json", action="store_true")
    x.add_argument("--period", choices=PERIODS, default="Month")


def main():
    p = argparse.ArgumentParser(prog="search.py", description="Search Civitai")
    sub = p.add_subparsers(dest="cmd", required=True)

    pm = sub.add_parser("models",
                        help="Search models (Meilisearch w/ query, REST otherwise)")
    pm.add_argument("--query", "-q")
    pm.add_argument("--base-model", "-b")
    pm.add_argument("--username", "-u")
    pm.add_argument("--tag")
    pm.add_argument("--type", "-t", action="append", choices=MODEL_TYPES,
                    help="Repeat for multiple")
    pm.add_argument("--ids", help="Comma-separated model IDs (REST only)")
    pm.add_argument("--sort", choices=MODEL_SORTS, default="Most Downloaded")
    pm.add_argument("--nsfw", action="store_true")
    pm.add_argument("--favorites", action="store_true")
    _common_args(pm)

    pi = sub.add_parser("images",
                        help="Browse images (auto-resolves --model-id to latest version)")
    pi.add_argument("--model-id", type=int)
    pi.add_argument("--model-version-id", type=int)
    pi.add_argument("--username")
    pi.add_argument("--tag")
    pi.add_argument("--base-model")
    pi.add_argument("--browsing-level", help='Bitmask: "PG", "X,XXX", "all"')
    pi.add_argument("--nsfw", action="store_true",
                    help="Shorthand for --browsing-level X,XXX (overridden if "
                         "--browsing-level is also passed)")
    pi.add_argument("--content-type", choices=["image", "video"])
    pi.add_argument("--sort", choices=IMAGE_SORTS, default="Most Reactions")
    pi.add_argument("--has-meta", action="store_true")
    _common_args(pi, image=True)

    pt = sub.add_parser("top-models",
                        help="Top models by sort+period (Meilisearch wrapper)")
    pt.add_argument("--type", "-t", action="append", choices=MODEL_TYPES)
    pt.add_argument("--base-model", "-b")
    pt.add_argument("--nsfw", action="store_true")
    pt.add_argument("--sort", choices=MODEL_SORTS, default="Most Downloaded")
    _common_args(pt)

    pti = sub.add_parser("top-images", help="Top images by sort+period")
    pti.add_argument("--sort", choices=IMAGE_SORTS, default="Most Reactions")
    pti.add_argument("--content-type", choices=["image", "video"])
    pti.add_argument("--browsing-level",
                     help='Bitmask: "PG", "X,XXX", "all"')
    pti.add_argument("--nsfw", action="store_true",
                     help="Shorthand for --browsing-level X,XXX (overridden if "
                          "--browsing-level is also passed)")
    pti.add_argument("--tag")
    pti.add_argument("--base-model")
    pti.add_argument("--has-meta", action="store_true")
    _common_args(pti, image=True)

    args = p.parse_args()
    if args.cmd in ("models", "top-models"):
        _do_models(args)
    else:
        _do_images(args)


if __name__ == "__main__":
    main()
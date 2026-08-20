"""civitai show: model / version / hash / prompts."""
import argparse
import sys

from _common import (
    api_get, fmt_size, fmt_int, truncate, die, emit_json,
    slim_model, slim_version, slim_prompt_image,
)

PROMPT_SORTS = [
    "Most Reactions", "Most Comments", "Most Collected", "Newest", "Oldest",
]


def _do_model(args):
    m = api_get(f"models/{args.model_id}")
    if args.json:
        emit_json(slim_model(m))
        return

    mid = m.get("id", "?")
    versions = m.get("modelVersions") or []
    base = (versions[0] if versions else {}).get("baseModel", "?")
    tags = m.get("tags") or []
    tn = ", ".join(
        t.get("name", "?") if isinstance(t, dict) else str(t)
        for t in tags[:8]
    )

    creator = (m.get("creator") or {}).get("username", "?")
    print(f"Model #{mid} \"{m.get('name', '?')}\" by {creator}")
    print(f"Type: {m.get('type', '?')} · Base: {base} · Tags: {tn}")
    print("\nVersions (newest first):")
    for v in versions:
        files = v.get("files") or []
        pf = next((f for f in files if f.get("primary")),
                  files[0] if files else {})
        primary = " primary" if pf.get("primary") else ""
        print(f"  #{v.get('id', '?'):<8} {v.get('name', '?'):<25} — "
              f"{fmt_size(pf.get('sizeKB'))}{primary}")

    desc = truncate(m.get("description") or "", 500)
    if desc:
        print(f"\nDescription: {desc}")
    print(f"URL: https://civitai.com/models/{mid}")


def _do_version(args):
    v = api_get(f"model-versions/{args.version_id}")
    if args.json:
        emit_json(slim_version(v))
        return

    vid = v.get("id", "?")
    mid = v.get("modelId", "?")
    pm = v.get("model") or {}
    trained = v.get("trainedWords") or []

    print(f"Version #{vid} — \"{v.get('name', '?')}\"")
    print(f"Parent: #{mid} \"{pm.get('name', '?')}\"")
    print(f"Base: {v.get('baseModel', '?')} · Trigger: "
          f"{', '.join(str(w) for w in trained[:8]) or '(none)'}")

    for f in (v.get("files") or []):
        flag = " [PRIMARY]" if f.get("primary") else ""
        meta = f.get("metadata") or {}
        fmt = f.get("type") or meta.get("format", "?")
        fp = meta.get("fp", "")
        hashes = f.get("hashes") or {}

        size_line = f"  Size: {fmt_size(f.get('sizeKB'))} · Format: {fmt}"
        if fp:
            size_line += f" · {fp}"
        if meta.get("size"):
            size_line += f" · {meta['size']}"

        print(f"\nFile: {f.get('name', '?')}{flag}")
        print(size_line)
        for h in ("SHA256", "AutoV2", "CRC32", "BLAKE3"):
            if hashes.get(h):
                print(f"  {h}: {hashes[h]}")
        print(f"  Scan: pickle={f.get('pickleScanResult', '?')} · "
              f"virus={f.get('virusScanResult', '?')}")
        print(f"  URL: {f.get('downloadUrl', '?')}")


def _do_hash(args):
    h = args.hash.strip().upper()
    v = api_get(f"model-versions/by-hash/{h}")
    if args.json:
        emit_json(v)
        return

    vid = v.get("id", "?")
    mid = v.get("modelId", "?")
    pm = v.get("model") or {}
    print(f"Match: {h}")
    print(f"Version #{vid} \"{v.get('name', '?')}\"")
    print(f"Parent: #{mid} \"{pm.get('name', '?')}\" · "
          f"Type: {pm.get('type', '?')} · Base: {v.get('baseModel', '?')}")
    print(f"URL: https://civitai.com/models/{mid}?modelVersionId={vid}")


def _do_prompts(args):
    m = api_get(f"models/{args.model_id}")
    versions = m.get("modelVersions") or []
    if not versions:
        die(f"model {args.model_id} has no versions")
    vid = versions[0]["id"]
    print(f"# resolved model {args.model_id} → version {vid}", file=sys.stderr)

    data = api_get("images", {
        "modelVersionId": vid,
        "sort":           args.sort,
        "limit":          min(args.limit, 50),
        "hasMeta":        True,
    })
    if args.json:
        emit_json({
            "modelId":   args.model_id,
            "modelName": m.get("name"),
            "versionId": vid,
            "items":     [slim_prompt_image(im) for im in (data.get("items") or [])],
        })
        return

    items = data.get("items") or []
    if not items:
        print("No images with meta.")
        return

    print(f"Prompts for #{args.model_id} \"{m.get('name', '?')}\" · "
          f"{len(items)} images\n")
    for i, im in enumerate(items, 1):
        meta = im.get("meta") or {}
        iid = im.get("id", "?")
        user = im.get("username") or (im.get("user") or {}).get("username", "?")
        stats = im.get("stats") or {}
        react = sum(stats.get(k, 0)
                    for k in ("heartCount", "likeCount", "laughCount", "cryCount"))

        print(f"{i}. Image #{iid} by {user} · {fmt_int(react)} reactions")
        if meta.get("prompt"):
            print(f"   Prompt: {truncate(meta['prompt'], 500)}")
        if meta.get("negativePrompt"):
            print(f"   Negative: {truncate(meta['negativePrompt'], 300)}")
        p = [f"{k}: {meta[k]}"
             for k in ("steps", "sampler", "cfgScale", "seed")
             if meta.get(k) is not None]
        if p:
            print("   " + " · ".join(p))

        loras = []
        for r in (meta.get("resources") or []):
            if (r.get("type") or "").lower() in ("lora", "locon", "dora"):
                w = r.get("weight")
                loras.append(f"{r.get('name', '?')}"
                             + (f" ({w})" if w is not None else ""))
        if loras:
            print(f"   LoRAs: {', '.join(loras[:5])}")
        print(f"   URL: https://civitai.com/images/{iid}\n")


def main():
    p = argparse.ArgumentParser(
        prog="show.py",
        description="Show Civitai model/version/hash/prompts",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pm = sub.add_parser("model", help="Full model details with versions")
    pm.add_argument("model_id", type=int)
    pm.add_argument("--json", action="store_true")

    pv = sub.add_parser("version", help="Version details with file hashes + scans")
    pv.add_argument("version_id", type=int)
    pv.add_argument("--json", action="store_true")

    ph = sub.add_parser("hash", help="Reverse-lookup by SHA256/AutoV2/CRC32/BLAKE3")
    ph.add_argument("hash")
    ph.add_argument("--json", action="store_true")

    pp = sub.add_parser("prompts", help="Mine working prompts from images of a model")
    pp.add_argument("model_id", type=int)
    pp.add_argument("--sort", choices=PROMPT_SORTS, default="Most Reactions")
    pp.add_argument("--limit", "-n", type=int, default=5)
    pp.add_argument("--json", action="store_true")

    args = p.parse_args()
    dispatch = {
        "model":   _do_model,
        "version": _do_version,
        "hash":    _do_hash,
        "prompts": _do_prompts,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
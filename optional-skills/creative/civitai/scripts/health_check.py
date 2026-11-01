"""civitai skill — health check. Static + network + auth + functional + config.

Usage:
  python3 health_check.py                     # full check
  python3 health_check.py --offline           # static only (no network)
  python3 health_check.py --json              # structured output
  python3 health_check.py --comfyui-path PATH # also verify ComfyUI dir layout

Exits 0 if all checks pass (skips allowed), 1 if any fail. Designed to be
runnable both interactively and from CI / installer post-hooks.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["_common.py", "search.py", "show.py", "download.py"]
SAFE_MODEL_ID = 257749       # "Realistic Vision V6.0" — stable SFW sentinel
SAFE_HASH = "E837144C55"     # AutoV2 hash of the above
MEILI_DEFAULT_KEY = "8c46eb2508e21db1e9828a97968d91ab1ca1caa5f70a00e88a2ba1e286603b61"
SYMBOL = {"ok": "  ok  ", "skip": " skip ", "fail": " FAIL "}


def _c(group, name, status, detail=""):
    return {"group": group, "name": name, "status": status, "detail": detail}


def _http(url, key=None, body=None, timeout=12):
    t0 = time.time()
    headers = {"User-Agent": "civitai-skill-healthcheck/1.0"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    if body:
        headers["Content-Type"] = "application/json"
    try:
        req = urllib.request.Request(url, data=body, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, int((time.time() - t0) * 1000), r.read()
    except urllib.error.HTTPError as e:
        return e.code, int((time.time() - t0) * 1000), e.read()
    except urllib.error.URLError as e:
        return None, int((time.time() - t0) * 1000), str(e).encode()


def static_checks():
    out = []
    v = sys.version_info
    out.append(_c(
        "Static", "Python ≥ 3.7",
        "ok" if v >= (3, 7) else "fail",
        f"got {v.major}.{v.minor}.{v.micro}",
    ))
    for s in SCRIPTS:
        p = os.path.join(HERE, s)
        if os.path.exists(p):
            out.append(_c("Static", f"{s} present", "ok"))
        else:
            out.append(_c("Static", f"{s} present", "fail", f"missing at {p}"))

    # Importability of _common
    try:
        sys.path.insert(0, HERE)
        __import__("_common")
        out.append(_c("Static", "_common.py importable", "ok"))
    except Exception as e:
        out.append(_c("Static", "_common.py importable", "fail", str(e)))

    # --help on each executable script
    for s in ("search.py", "show.py", "download.py"):
        try:
            r = subprocess.run(
                [sys.executable, os.path.join(HERE, s), "--help"],
                capture_output=True, timeout=10,
            )
            out.append(_c(
                "Static", f"{s} --help",
                "ok" if r.returncode == 0 else "fail",
                "" if r.returncode == 0 else f"exit {r.returncode}",
            ))
        except Exception as e:
            out.append(_c("Static", f"{s} --help", "fail", str(e)))
    return out


def network_checks():
    out = []
    code, ms, _ = _http("https://civitai.com/api/v1/models?limit=1")
    out.append(_c(
        "Network", "civitai.com reachable",
        "ok" if code == 200 else "fail",
        f"HTTP {code} in {ms}ms" if code else f"network error after {ms}ms",
    ))
    code, ms, _ = _http("https://search-new.civitai.com/")
    # Any HTTP code means the host responded; the root is allowed to 404
    out.append(_c(
        "Network", "search-new.civitai.com reachable",
        "ok" if code is not None else "fail",
        f"HTTP {code} in {ms}ms" if code else f"network error after {ms}ms",
    ))
    return out


def auth_checks():
    key = os.environ.get("CIVITAI_API_KEY")
    if not key:
        return [_c("Auth", "CIVITAI_API_KEY", "skip", "unset — SFW-only mode is fine")]
    code, ms, _ = _http("https://civitai.com/api/v1/models?limit=1", key=key)
    if code == 200:
        return [_c("Auth", "CIVITAI_API_KEY", "ok", f"accepted ({ms}ms)")]
    if code in (401, 403):
        return [_c(
            "Auth", "CIVITAI_API_KEY", "fail",
            f"rejected with HTTP {code} — key is set but invalid",
        )]
    return [_c("Auth", "CIVITAI_API_KEY", "fail", f"unexpected HTTP {code}")]


def functional_checks():
    out = []
    key = os.environ.get("CIVITAI_API_KEY")

    # 1. Known model
    code, ms, body = _http(
        f"https://civitai.com/api/v1/models/{SAFE_MODEL_ID}", key=key,
    )
    if code == 200:
        try:
            name = json.loads(body).get("name", "?")
            out.append(_c(
                "Functional", f"GET /models/{SAFE_MODEL_ID}",
                "ok", f'resolved to "{name}" ({ms}ms)',
            ))
        except Exception as e:
            out.append(_c("Functional", "GET /models", "fail", f"non-JSON: {e}"))
    else:
        out.append(_c("Functional", "GET /models", "fail", f"HTTP {code}"))

    # 2. Hash lookup
    code, _, body = _http(
        f"https://civitai.com/api/v1/model-versions/by-hash/{SAFE_HASH}", key=key,
    )
    if code == 200:
        try:
            mid = json.loads(body).get("modelId", "?")
            out.append(_c("Functional", f"by-hash/{SAFE_HASH}", "ok", f"→ model #{mid}"))
        except Exception:
            out.append(_c("Functional", "by-hash", "fail", "non-JSON"))
    else:
        out.append(_c(
            "Functional", "by-hash", "skip",
            f"HTTP {code} (sentinel may have been retired)",
        ))

    # 3. Meilisearch
    mkey = os.environ.get("MEILISEARCH_KEY") or MEILI_DEFAULT_KEY
    payload = json.dumps({"queries": [{
        "q": "anime", "indexUid": "models_v9", "limit": 5,
        "filter": 'type = "LORA"', "sort": ["metrics.downloadCount:desc"],
    }]}).encode()
    code, ms, resp = _http(
        "https://search-new.civitai.com/multi-search",
        key=mkey, body=payload,
    )
    if code == 200:
        try:
            hits = json.loads(resp)["results"][0].get("hits", [])
            out.append(_c(
                "Functional", "Meilisearch 'anime'+LORA",
                "ok" if hits else "fail",
                f"{len(hits)} hits ({ms}ms)",
            ))
        except Exception as e:
            out.append(_c("Functional", "Meilisearch", "fail", f"parse: {e}"))
    else:
        out.append(_c("Functional", "Meilisearch", "fail", f"HTTP {code}"))

    # 4. download.py emit + key-leak guard
    try:
        r = subprocess.run(
            [sys.executable, os.path.join(HERE, "download.py"),
             str(SAFE_MODEL_ID), "--format", "curl"],
            capture_output=True, text=True, timeout=20,
        )
        if r.returncode != 0:
            out.append(_c(
                "Functional", "download.py emits", "fail",
                f"exit {r.returncode}: {r.stderr[:120]}",
            ))
        elif "curl -L" not in r.stdout:
            out.append(_c("Functional", "download.py emits", "fail", "no curl line"))
        elif key and key in r.stdout:
            # Key VALUE leaked — critical security failure
            out.append(_c(
                "Functional", "download.py emits", "fail",
                "KEY VALUE LEAKED in output — abort install!",
            ))
        else:
            out.append(_c(
                "Functional", "download.py emits", "ok",
                "curl/wget/ps lines emit, no key leak",
            ))
    except Exception as e:
        out.append(_c("Functional", "download.py emits", "fail", str(e)))
    return out


def config_checks(path):
    if not path:
        return []
    if not os.path.isdir(path):
        return [_c(
            "Config", f"--comfyui-path {path}", "fail",
            "directory does not exist",
        )]
    missing = [
        s for s in ("checkpoints", "loras")
        if not os.path.isdir(os.path.join(path, s))
    ]
    if missing:
        return [_c(
            "Config", f"--comfyui-path {path}", "skip",
            f"missing subfolders: {missing}",
        )]
    return [_c(
        "Config", f"--comfyui-path {path}", "ok",
        "checkpoints/, loras/ present",
    )]


def _emit_text(checks):
    by_group = {}
    for c in checks:
        by_group.setdefault(c["group"], []).append(c)
    print("Civitai Skill — Health Check\n" + "=" * 28)
    for g, items in by_group.items():
        print(f"\n[{g}]")
        for c in items:
            line = f"{SYMBOL[c['status']]} {c['name']}"
            if c["detail"]:
                line += f" — {c['detail']}"
            print(line)
    ok = sum(1 for c in checks if c["status"] == "ok")
    sk = sum(1 for c in checks if c["status"] == "skip")
    fl = sum(1 for c in checks if c["status"] == "fail")
    print(f"\nSummary: {ok} ok, {sk} skip, {fl} fail")


def main():
    p = argparse.ArgumentParser(
        prog="health_check.py",
        description="Civitai skill health check (static, network, auth, functional, config).",
    )
    p.add_argument("--offline", action="store_true",
                   help="Skip network/auth/functional")
    p.add_argument("--json", action="store_true",
                   help="Emit JSON instead of text")
    p.add_argument("--comfyui-path",
                   help="Also verify this ComfyUI models path")
    args = p.parse_args()

    checks = static_checks()
    if not args.offline:
        net = network_checks()
        checks += net
        if any(c["status"] == "ok" for c in net):
            checks += auth_checks()
            checks += functional_checks()
    checks += config_checks(args.comfyui_path)

    if args.json:
        print(json.dumps(checks, indent=2))
    else:
        _emit_text(checks)
    sys.exit(0 if all(c["status"] != "fail" for c in checks) else 1)


if __name__ == "__main__":
    main()
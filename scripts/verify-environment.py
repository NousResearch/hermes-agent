#!/usr/bin/env python3
"""verify-environment.py — 校验某版本的 agent 环境是否在 COS 上齐全(github-free 不变量的闸)。

读 apexnodes-environment.yaml,对给定 runtime commit,HEAD-check 每个 COS 自托管产物
(runtime tarball / uv per-triple / PortableGit per-arch)是否 200。缺任一 → 非零退出。
**发版 / 翻 is_default 前跑它**:防 hc-448 类(发的版本环境没全上 COS → 大陆装机回落 github → 炸)。

用法:
  python3 scripts/verify-environment.py --commit <full-sha> [--manifest PATH] [--json]
"""
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request


def head(url, timeout=10):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0


def http_get(url, timeout=15):
    """(status, body_bytes). status 0 / body b'' on any transport error."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception:
        return 0, b""


def bundle_host(m):
    """Bundle objects live under the COS host ROOT (bundle/ is a sibling of the
    runtime/ prefix), so strip the cos_base's trailing /runtime — same derivation
    as apex-bundle-install.cjs::deriveCosHost."""
    return re.sub(r"/runtime$", "", m["cos_base"].rstrip("/"), flags=re.I)


def load_manifest(path):
    try:
        import yaml
    except ImportError:
        sys.exit("verify-environment: 需要 PyYAML (pip install pyyaml)")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def static_source_checks(manifest_path):
    """hc-476: cheap static regression guards for runtime pull points that
    have no HTTP endpoint of their own to HEAD-check (unlike cos_artifacts /
    cn_registry, this isn't network reachability — it's "does the consumer
    still read the env this manifest promises it does"). Intentionally NOT
    the Phase-2 "flag any new unregistered foreign URL" gate the file header
    mentions — that needs to enumerate every hardcoded literal across
    install.sh/.ps1/node-bootstrap.sh, which is a bigger lift than one ticket;
    this only guards the specific regression hc-476 just fixed.
    """
    repo_root = os.path.dirname(os.path.abspath(manifest_path))
    checks = []

    node_bootstrap = os.path.join(repo_root, "scripts", "lib", "node-bootstrap.sh")
    try:
        src = open(node_bootstrap, encoding="utf-8").read()
        ok = "HERMES_NODE_DIST_BASE" in src
    except OSError as e:
        ok = False
        node_bootstrap = f"{node_bootstrap} ({e})"
    checks.append(("node-bootstrap.sh honors HERMES_NODE_DIST_BASE", ok, node_bootstrap))

    # hc-472 followup: uv.lock records the ACTUAL registry each package was
    # resolved against, so the CN mirror default index (UV_DEFAULT_INDEX etc,
    # apexnodes_apply_cn_mirror_env / Set-ApexCnMirrorEnv) is a DIFFERENT
    # identity than the lock expects — `uv sync --extra all --locked` then
    # refuses outright ("lockfile needs to be updated") and every CN-mirror
    # install silently fell through past hash verification into the
    # unverified `uv pip install` fallback tiers, every time. install.sh /
    # install.ps1 must sanitize that index env for the --locked calls
    # specifically (see _uv_sync_locked / Invoke-UvSyncLocked) — this guards
    # the fix from silently regressing back to the raw, unsanitized call.
    install_sh = os.path.join(repo_root, "scripts", "install.sh")
    try:
        src = open(install_sh, encoding="utf-8").read()
        ok = (
            "_uv_sync_locked" in src
            and "-u UV_DEFAULT_INDEX" in src
            and "_uv_sync_locked --check" in src
            and "if _uv_sync_locked; then" in src
        )
    except OSError as e:
        ok = False
        install_sh = f"{install_sh} ({e})"
    checks.append(("install.sh sanitizes mirror index env for uv sync --locked", ok, install_sh))

    install_ps1 = os.path.join(repo_root, "scripts", "install.ps1")
    try:
        src = open(install_ps1, encoding="utf-8").read()
        ok = (
            "function Invoke-UvSyncLocked" in src
            and '$env:UV_DEFAULT_INDEX = $null' in src
            and "Invoke-UvSyncLocked -Check" in src
        )
    except OSError as e:
        ok = False
        install_ps1 = f"{install_ps1} ({e})"
    checks.append(("install.ps1 sanitizes mirror index env for uv sync --locked", ok, install_ps1))

    return checks


def cos_urls(m, commit):
    """yield (label, url|None) — 每个必须存在的 COS 产物。"""
    base = m["cos_base"].rstrip("/")
    for art in m.get("cos_artifacts", []):
        name = art["name"]
        by = art.get("keyed_by")
        if by == "commit":
            if not commit:
                yield (name, None)
                continue
            yield (f"{name}[{commit[:10]}]", f"{base}/{art['key'].format(commit=commit)}")
        elif by == "triple":
            for t in art["triples"]:
                key = art["key"].format(triple=t["triple"], ext=t["ext"])
                yield (f"{name}[{t['triple']}]", f"{base}/{key}")
        elif by == "arch":
            ver = art["version"]
            for a in art["arches"]:
                key = art["key"].format(version=ver, arch=a)
                yield (f"{name}[{a}]", f"{base}/{key}")


def check_bundles(m, key, only_os=None):
    """hc-472 E1: yield (label, status, detail) for the prebuilt runtime bundles.
    Per platform (win-x64 / mac-arm64 / mac-x64) for a given key: HEAD-200 on
    manifest.json + the .tar.gz + its .sha256, and sha SELF-CONSISTENCY (the
    manifest's archive.sha256 == the .sha256 sidecar's first field). This is the
    is_default-flip hard gate (hc-471). `only_os` (e.g. {"win"}) narrows to a
    phase's acceptance legs (P1 = win-only, design §7); default = all platforms.
    status: OK | MISS(code) | MISMATCH | SKIP.
    """
    b = m.get("bundles")
    if not b:
        yield ("bundles[section]", "MISS(none)", "apexnodes-environment.yaml 无 bundles 段")
        return
    if not key:
        yield ("bundles[--bundle-key]", "SKIP", "--check-bundles 需配 --bundle-key <sha12>")
        return
    host = bundle_host(m)
    fw = b["framework"]
    sc = b.get("sha_self_consistency") or {}
    plats = [p for p in b.get("platforms", []) if not only_os or p["os"] in only_os]
    if not plats:
        yield ("bundles[--bundle-os]", "SKIP", f"没有平台匹配 --bundle-os={sorted(only_os or [])}")
        return
    for plat in plats:
        osn, arch = plat["os"], plat["arch"]
        base = f"{host}/" + b["prefix"].format(framework=fw, key=key, os=osn, arch=arch)
        for tmpl in b.get("objects", []):
            name = tmpl.format(key=key, os=osn, arch=arch)
            code = head(f"{base}/{name}")
            good = 200 <= code < 300
            yield (f"bundle[{osn}-{arch}]/{name}", "OK" if good else f"MISS({code})", f"{base}/{name}")
        if sc:
            label = f"bundle[{osn}-{arch}]/sha-self-consistency"
            man_url = f"{base}/" + sc["manifest"].format(key=key, os=osn, arch=arch)
            sha_url = f"{base}/" + sc["sha_sidecar"].format(key=key, os=osn, arch=arch)
            mcode, mbody = http_get(man_url)
            scode, sbody = http_get(sha_url)
            if not (200 <= mcode < 300 and 200 <= scode < 300):
                yield (label, f"MISS({mcode}/{scode})", "manifest.json 或 .sha256 取不到")
                continue
            try:
                ref = json.loads(mbody.decode("utf-8"))
                for part in sc.get("manifest_field", "archive.sha256").split("."):
                    ref = ref[part]
                sidecar = sbody.decode("utf-8").split()[0].strip().lower()
                if str(ref).strip().lower() == sidecar:
                    yield (label, "OK", f"{sidecar[:16]}…")
                else:
                    yield (label, "MISMATCH", f"manifest {str(ref)[:16]}… != sidecar {sidecar[:16]}…")
            except Exception as e:  # any parse/shape error = not self-consistent
                yield (label, "MISMATCH", f"解析失败: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", help="runtime 完整 SHA(runtime-source tarball 用)")
    ap.add_argument("--manifest", default=os.path.join(os.path.dirname(__file__), "..", "apexnodes-environment.yaml"))
    ap.add_argument("--check-bundles", action="store_true", help="额外校验 bundle/ 前缀的预构建 runtime bundle(hc-472),需配 --bundle-key")
    ap.add_argument("--bundle-key", help="bundle key(runtime sha12);配 --check-bundles")
    ap.add_argument("--bundle-os", help="逗号分隔,限定平台(如 win 只校验 P1 验收腿);默认全平台")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    only_os = {s.strip() for s in args.bundle_os.split(",") if s.strip()} if args.bundle_os else None

    m = load_manifest(args.manifest)
    results = []
    ok = True
    for name, url in cos_urls(m, args.commit):
        if url is None:
            results.append((name, "SKIP", "缺 --commit,无法校验 runtime tarball"))
            # A bundle-only run legitimately omits --commit; don't fail on the skip.
            if not args.check_bundles:
                ok = False
            continue
        code = head(url)
        good = 200 <= code < 300
        results.append((name, "OK" if good else f"MISS({code})", url))
        ok = ok and good

    reg = [(r["name"], head(r["mirror"].split("/simple")[0])) for r in m.get("cn_registry", [])]

    static = static_source_checks(args.manifest)
    ok = ok and all(s_ok for _, s_ok, _ in static)

    bundles = list(check_bundles(m, args.bundle_key, only_os)) if args.check_bundles else []
    ok = ok and all(s == "OK" for _, s, _ in bundles)

    if args.json:
        out = {
            "ok": ok,
            "cos": [{"name": n, "status": s, "url": u} for n, s, u in results],
            "cn_registry": [{"name": n, "http": c} for n, c in reg],
            "static_checks": [{"name": n, "ok": s_ok, "detail": d} for n, s_ok, d in static],
        }
        if args.check_bundles:
            out["bundles"] = {
                "key": args.bundle_key,
                "checks": [{"name": n, "status": s, "detail": d} for n, s, d in bundles],
            }
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(f"=== COS 产物 (commit={args.commit or '<none>'}) ===")
        for n, s, _ in results:
            print(f"  [{s:>9}] {n}")
        print("=== cn_registry 可达性(仅参考)===")
        for n, c in reg:
            print(f"  [{c}] {n}")
        print("=== 静态回归守卫(源码断言,纳入闸)===")
        for n, s_ok, d in static:
            print(f"  [{'OK' if s_ok else 'FAIL':>9}] {n} ({d})")
        if args.check_bundles:
            print(f"=== 预构建 bundle (key={args.bundle_key or '<none>'},纳入闸)===")
            for n, s, _ in bundles:
                print(f"  [{s:>9}] {n}")
        na = m.get("needs_audit", [])
        if na:
            print(f"=== needs_audit ({len(na)}) — 源未确认,未纳入闸 ===")
            for x in na:
                print(f"  - {x['name']}: {x.get('note', '')}")
        print(f"\nRESULT: {'PASS — 环境已全镜像' if ok else 'FAIL — 上面有缺失镜像'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

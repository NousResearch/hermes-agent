"""civitai download: emit curl/wget/PowerShell commands for ComfyUI.

The agent is responsible for forwarding `civitai.comfyui_path` (from the skill
activation message) as `--comfyui-path`. This script does NOT read config.yaml
itself — see SKILL.md Recipe 5.

Untrusted strings (filenames, URLs, target paths from the API) are escaped:
  - bash (curl/wget):   shlex.quote() — wraps with single quotes, neutralizing $ and `
  - PowerShell:         literal single-quoted strings with '' doubling
The $CIVITAI_API_KEY (or $env:CIVITAI_API_KEY) reference is the ONLY token left
unquoted, so the shell still expands it at run time.
"""
import argparse
import shlex

from _common import api_get, fmt_size, die, emit_json

COMFYUI_FOLDER_MAP = {
    "Checkpoint":        "checkpoints",
    "LORA":              "loras",
    "LoCon":             "loras",
    "DoRA":              "loras",
    "TextualInversion":  "embeddings",
    "Hypernetwork":      "hypernetworks",
    "Controlnet":        "controlnet",
    "VAE":               "vae",
    "Upscaler":          "upscale_models",
    "Poses":             "poses",
    "AestheticGradient": "aesthetic_gradients",
    "MotionModule":      "animatediff_motion_lora",
    "Wildcards":         "wildcards",
}


def _ps_quote(s):
    """Wrap a string as a PowerShell single-quoted literal (no interpolation).

    PowerShell escapes a literal single quote by doubling it: 'don''t'.
    """
    return "'" + str(s).replace("'", "''") + "'"


def main():
    p = argparse.ArgumentParser(
        prog="download.py",
        description="Emit shell commands to download a Civitai model into ComfyUI.",
    )
    p.add_argument("model_id", type=int)
    p.add_argument("--version-id", type=int,
                   help="Specific version (default: latest)")
    p.add_argument(
        "--comfyui-path",
        default="",
        help=("ComfyUI models root, e.g. D:/ComfyUI/models. The agent should "
              "pass the configured civitai.comfyui_path here when set; "
              "otherwise output uses a <COMFYUI_PATH> placeholder."),
    )
    p.add_argument("--format", choices=["all", "curl", "wget", "ps"],
                   default="all")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    m = api_get(f"models/{args.model_id}")
    mtype = m.get("type", "Other")
    name = m.get("name", "?")
    versions = m.get("modelVersions") or []
    if not versions:
        die(f"model {args.model_id} has no versions")

    if args.version_id:
        v = next((x for x in versions if x.get("id") == args.version_id), None)
        if v is None:
            die(f"version {args.version_id} not found on model {args.model_id}")
    else:
        v = versions[0]

    vid = v.get("id")
    files = v.get("files") or []
    if not files:
        die(f"version {vid} has no files")

    subfolder = COMFYUI_FOLDER_MAP.get(mtype, "other")
    base = (args.comfyui_path or "<COMFYUI_PATH>").rstrip("/\\").replace("\\", "/")

    payload = []
    for f in files:
        target = f"{base}/{subfolder}/{f.get('name', '?')}"
        payload.append({
            "filename":    f.get("name", "?"),
            "size_kb":     f.get("sizeKB"),
            "primary":     bool(f.get("primary")),
            "target":      target,
            "url":         f.get("downloadUrl", "?"),
            "hashes":      f.get("hashes") or {},
            "pickle_scan": f.get("pickleScanResult", "?"),
            "virus_scan":  f.get("virusScanResult", "?"),
        })

    if args.json:
        emit_json({
            "model_id":     args.model_id,
            "name":         name,
            "type":         mtype,
            "version_id":   vid,
            "version_name": v.get("name", "?"),
            "subfolder":    subfolder,
            "files":        payload,
        })
        return

    print(f"Download commands for #{args.model_id} \"{name}\"")
    print(f"Version #{vid} \"{v.get('name', '?')}\" · type: {mtype} · "
          f"subfolder: {subfolder}/")
    if not args.comfyui_path:
        print("# (no --comfyui-path provided — using placeholder <COMFYUI_PATH>; "
              "agent should forward civitai.comfyui_path when configured)")
    print()

    # The auth header is hardcoded (no API data) — we WANT shell variable
    # expansion of $CIVITAI_API_KEY, so we leave it unquoted/unescaped.
    auth_bash = '"Authorization: Bearer $CIVITAI_API_KEY"'

    for f in payload:
        flag = " [PRIMARY]" if f["primary"] else ""
        print(f"# {f['filename']}{flag} — {fmt_size(f['size_kb'])}")
        if f["hashes"].get("AutoV2"):
            print(f"# AutoV2: {f['hashes']['AutoV2']}")
        if f["hashes"].get("SHA256"):
            print(f"# SHA256: {f['hashes']['SHA256']}")
        print(f"# Scan: pickle={f['pickle_scan']} · virus={f['virus_scan']}")

        url, tgt = f["url"], f["target"]
        url_sh = shlex.quote(url)
        tgt_sh = shlex.quote(tgt)
        url_ps = _ps_quote(url)
        tgt_ps = _ps_quote(tgt)

        if args.format in ("all", "curl"):
            print(f"curl -L -H {auth_bash} -o {tgt_sh} {url_sh}")
        if args.format in ("all", "wget"):
            print(f"wget --header={auth_bash} -O {tgt_sh} {url_sh}")
        if args.format in ("all", "ps"):
            print(f'Invoke-WebRequest -Uri {url_ps} '
                  f'-Headers @{{"Authorization"="Bearer $env:CIVITAI_API_KEY"}} '
                  f'-OutFile {tgt_ps}')
        print()


if __name__ == "__main__":
    main()
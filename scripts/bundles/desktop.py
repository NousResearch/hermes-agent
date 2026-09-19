"""Build a complete desktop bundle with the shared Python payload tools."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run(argv: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("bundle: " + subprocess.list2cmdline(argv), flush=True)
    subprocess.run(argv, cwd=cwd, env=env, check=True)


def capture(argv: list[str], repo: Path) -> str:
    return subprocess.check_output(argv, cwd=repo, text=True, encoding="utf-8").strip()


def release_version(repo: Path, tag: str) -> str:
    from scripts.termux.deb_version import channel_for_tag

    channel_for_tag(tag)  # shared release tag grammar, not a second version parser
    version = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8-sig"))["project"]["version"]
    if "-canary." not in tag and tag != "v" + version:
        raise ValueError(f"tag {tag} does not match project version {version}")
    return tag[1:]


def npm_command(node: str) -> list[str]:
    # Dependency preparation and packaging must resolve the same npm identity.
    return [node, str(ROOT / "scripts/build/node-deps.mjs"), "--npm"]


def build(repo: Path, tag: str | None, variant: str, builder_args: list[str],
          commit_build: str | None = None) -> None:
    from scripts.bundles.desktop_prepare import BuildRequest, prepare
    from scripts.releases.bundle_env import decode
    request = BuildRequest.create(repo, tag=tag, commit=commit_build, variant=variant,
                                  work=repo / ".build/desktop-job", cache=repo / ".cache/desktop-inputs",
                                  bundle_env=decode(os.environ.get("HERMES_BUNDLE_ENV_JSON", "")))
    build_prepared(prepare(request), builder_args)


def build_prepared(path: Path, builder_args: list[str], variant: str | None = None) -> None:
    from scripts.bundles.desktop_prepare import PreparedDesktop
    from scripts.bundles.desktop_inputs import build_lock
    prepared = PreparedDesktop.load(path)
    with build_lock(prepared.request.source):
        _build_prepared(prepared, builder_args, variant)


def _build_prepared(prepared, builder_args: list[str], variant: str | None) -> None:
    prepared.validate()
    from scripts.bundles.desktop_inputs import build_environment, packaging_environment, select_variant, validate_builder_identity
    from scripts.bundles.native import finish_native

    request = prepared.request
    validate_builder_identity(request, builder_args)
    if request.channel_request is not None and not request.target.startswith(("darwin-", "win32-")):
        raise ValueError("channel builds require a supported native macOS or Windows target")
    variant = select_variant(prepared, variant)
    repo, node = request.source, str(prepared.node)
    env = build_environment(prepared, variant, os.environ)
    desktop = repo / "apps/desktop"
    targets = {"win32": ["--win", "msix"], "darwin": ["--mac", "dmg", "zip"], "linux": ["--linux", "AppImage"]}[sys.platform]
    package_args = ["--prepared", str(prepared.packager), "--native-deps", str(prepared.native),
                    *targets, f"-c.extraMetadata.version={request.version}"]
    run([node, "scripts/run-electron-builder.mjs", "--validate-only", *package_args, *builder_args],
        cwd=desktop, env=env)
    run([node, "scripts/build/node-deps.mjs", "--source", str(repo), "--reuse", "--no-install",
         "--native-toolchain", prepared.native_toolchain,
         *[arg for name in request.workspaces() for arg in ("--workspace", name)]], cwd=repo, env=env)
    products = repo / "apps/desktop/build/products"
    icons = products / "icons"
    run([str(prepared.icon_python), "-I", str(repo / "scripts/generate_icons.py"),
         "--source", str(repo), "--out", str(icons)], cwd=repo, env=env)
    if variant != "light":
        run([node, "scripts/build/tui.mjs", "--source", str(repo), "--out", str(products / "tui")], cwd=repo, env=env)
        run([node, "scripts/build/web.mjs", "--source", str(repo), "--icons", str(products / "icons"),
             "--out", str(products / "web")], cwd=repo, env=env)
        if prepared.payload is None:
            raise ValueError("payload dependencies were not prepared")
        if finish_native(prepared.payload, {"tui": products / "tui", "web": products / "web"}):
            raise RuntimeError("prepared payload assembly failed")
    from scripts.bundles.desktop_prepare import require_source
    require_source(repo, request.commit)
    shutil.copytree(icons / "apps/desktop/assets", desktop / "assets", dirs_exist_ok=True)
    run([node, "scripts/write-build-stamp.mjs"], cwd=desktop, env=env)
    run([node, "scripts/build/desktop.mjs", "--source", str(repo), "--icons", str(icons),
         "--stamp", str(desktop / "build/install-stamp.json"), "--native-deps", str(prepared.native),
         "--out", str(desktop / "dist")], cwd=repo, env=env)
    # Windows file-version and MSIX build-number policy remains with its packager.
    version_args = []
    if sys.platform == "win32":
        if request.channel_request is not None:
            metadata = {"file": request.channel_request["windowsVersion"], "build": None}
        elif request.tag is None:
            # The plain version needs no canary build-number override.
            metadata = {"file": None, "build": None}
        else:
            script = "const w=require('./apps/desktop/scripts/windows-file-version.mjs');const m=require('./scripts/msix-shared.mjs');console.log(JSON.stringify({file:w.windowsFileVersion(process.argv[1]),build:process.argv[2]!=='store'&&process.argv[1].includes('-canary.')?m.canaryBuildMinutes(process.argv[1],process.cwd()):null}))"
            metadata = json.loads(capture([node, "-e", script, request.tag, variant], repo))
        env.pop("BUILD_NUMBER", None)
        if metadata["build"] is not None and variant != "store":
            env["BUILD_NUMBER"] = str(metadata["build"])
        if metadata["file"]:
            version_args = [f'-c.extraMetadata.shortVersion={metadata["file"]}', f'-c.extraMetadata.shortVersionWindows={metadata["file"]}']
    require_source(repo, request.commit)
    run([node, "scripts/run-electron-builder.mjs", *package_args, *version_args, *builder_args], cwd=desktop,
        env=packaging_environment(env, os.environ, request.target))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=False,
                        help="Release tag (vX.Y.Z / canary), exclusive with --commit/--channel-request")
    parser.add_argument("--commit", dest="commit_build", default=None,
                        help="Commit-only build: exact full 40-char SHA the checkout is at; "
                             "version comes from the target pyproject, no tag is referenced")
    parser.add_argument("--channel-request", type=Path, help="Immutable admitted channel request JSON")
    parser.add_argument("--variant", choices=["bundled", "store", "light"])
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("builder_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    builder_args = [v for v in args.builder_args if v != "--"]
    try:
        if args.prepared:
            if args.tag or args.commit_build or args.channel_request or args.prepare_only or args.work or args.cache:
                parser.error("--prepared supplies the complete build request")
            build_prepared(args.prepared, builder_args, args.variant)
        else:
            from scripts.bundles.desktop_prepare import BuildRequest, prepare
            from scripts.releases.bundle_env import decode
            request = BuildRequest.create(args.repo, tag=args.tag, commit=args.commit_build,
                                          variant=args.variant or "bundled",
                                          work=args.work or args.repo / ".build/desktop-job",
                                          cache=args.cache or args.repo / ".cache/desktop-inputs",
                                          bundle_env=decode(os.environ.get("HERMES_BUNDLE_ENV_JSON", "")),
                                          channel_request=json.loads(args.channel_request.read_text(encoding="utf-8-sig"))
                                          if args.channel_request else None)
            if args.prepare_only and builder_args:
                parser.error("builder arguments belong to the build phase")
            result = prepare(request)
            if args.prepare_only:
                print(result)
            else:
                build_prepared(result, builder_args)
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"desktop build: {exc}\n")


if __name__ == "__main__":
    main()

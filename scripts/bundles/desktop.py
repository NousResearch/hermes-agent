"""Build a complete desktop bundle with the shared Python payload tools."""
from __future__ import annotations

import argparse
import json
import os
import re
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
    # npm.cmd needs cmd.exe; Node's CLI accepts argv directly, including spaces.
    npm = shutil.which("npm")
    if not npm:
        raise FileNotFoundError("npm is required")
    prefix = Path(npm).resolve().parent
    candidates = [prefix / "node_modules/npm/bin/npm-cli.js", prefix.parent / "lib/node_modules/npm/bin/npm-cli.js"]
    for candidate in candidates:
        if candidate.is_file():
            return [node, str(candidate)]
    # POSIX npm is normally a symlink to its CLI file.
    if os.name != "nt":
        return [node, str(Path(npm).resolve())]
    raise FileNotFoundError(f"npm CLI missing beside {npm}")


def build(repo: Path, tag: str | None, variant: str, builder_args: list[str],
          commit_build: str | None = None) -> None:
    from pm.store import current_target
    from scripts.releases.commit_build import require_commit, version_at
    from scripts.releases.bundle_env import decode

    repo = repo.resolve()
    bundle_env = decode(os.environ.get("HERMES_BUNDLE_ENV_JSON", ""))
    if bundle_env and not commit_build:
        raise ValueError("Bundle environment defaults require a commit build")
    if commit_build:
        commit = require_commit(commit_build)
        if tag:
            raise ValueError("Commit builds cannot also select a tag")
        if capture(["git", "rev-parse", "HEAD"], repo) != commit:
            raise ValueError("the build checkout must be at the commit being built")
        version = version_at(repo, commit)
    else:
        version = release_version(repo, tag)
        commit = capture(["git", "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"], repo)
        if capture(["git", "rev-parse", "HEAD"], repo) != commit:
            raise ValueError("the build checkout must be at the release tag")
    node = shutil.which("node")
    if not node or not shutil.which("uv"):
        raise FileNotFoundError("Node and uv are required")
    banner = capture(["uv", "--version"], repo)
    if not re.search(r"[a-z0-9_]+-[a-z0-9]+-[a-z][a-z0-9-]*", banner):
        raise ValueError(f"uv must report its build triple (official uv 0.12+): {banner}")
    npm = npm_command(node)
    env = {**os.environ, "CI": "true", "PYTHONUTF8": "1", "GITHUB_SHA": commit,
           "HERMES_DESKTOP_VARIANT": variant}
    env["HERMES_BUNDLE_ENV_JSON"] = json.dumps(bundle_env, sort_keys=True)
    if commit_build:
        env["HERMES_PAYLOAD_VERSION"] = version
        env["HERMES_BUILD_COMMIT"] = commit
        env.pop("HERMES_PAYLOAD_TAG", None)
        env.pop("GITHUB_REF_NAME", None)
        env.pop("GITHUB_HEAD_REF", None)
    else:
        env.pop("HERMES_BUILD_COMMIT", None)
        env["HERMES_PAYLOAD_TAG"] = tag
    target = current_target()
    node_arch = capture([node, "-p", "process.arch"], repo)
    if node_arch != target.split("-")[1]:
        raise ValueError(f"Node {node_arch} does not match build target {target}")
    if target == "win32-arm64":
        from scripts.build.windows_deps import prepare_windows_environment

        env = prepare_windows_environment(source=repo, state=repo / "apps/desktop/build/.build-deps", env=env)
    workspaces = ["apps/desktop"] + ([] if variant == "light" else ["ui-tui", "web"])
    run([node, "scripts/build/node-deps.mjs", "--source", str(repo),
         *[arg for workspace in workspaces for arg in ("--workspace", workspace)]], cwd=repo, env=env)
    payload = repo / "apps/desktop/build/agent-payload"
    if variant == "light":
        shutil.rmtree(payload, ignore_errors=True)
    else:
        products = repo / "apps/desktop/build/products"
        run([node, "scripts/generate-icons.mjs", "--source", str(repo), "--out", str(products / "icons")], cwd=repo, env=env)
        run([node, "scripts/build/tui.mjs", "--source", str(repo), "--out", str(products / "tui")], cwd=repo, env=env)
        run([node, "scripts/build/web.mjs", "--source", str(repo), "--icons", str(products / "icons"),
             "--out", str(products / "web")], cwd=repo, env=env)
        run([sys.executable, "-m", "scripts.bundles.stage", "--out", str(payload), "--ref", commit,
             "--tui", str(products / "tui"), "--web", str(products / "web")], cwd=repo, env=env)
    desktop = repo / "apps/desktop"
    # Windows file-version and MSIX build-number policy remains with its packager.
    version_args = []
    if sys.platform == "win32":
        if commit_build:
            # The plain version needs no canary build-number override.
            metadata = {"file": None, "build": None}
        else:
            script = "const w=require('./apps/desktop/scripts/windows-file-version.mjs');const m=require('./scripts/msix-shared.mjs');console.log(JSON.stringify({file:w.windowsFileVersion(process.argv[1]),build:process.argv[2]!=='store'&&process.argv[1].includes('-canary.')?m.canaryBuildMinutes(process.argv[1],process.cwd()):null}))"
            metadata = json.loads(capture([node, "-e", script, tag, variant], repo))
        env.pop("BUILD_NUMBER", None)
        if metadata["build"] is not None and variant != "store":
            env["BUILD_NUMBER"] = str(metadata["build"])
        if metadata["file"]:
            version_args = [f'-c.extraMetadata.shortVersion={metadata["file"]}', f'-c.extraMetadata.shortVersionWindows={metadata["file"]}']
    targets = {"win32": ["--win", "msix"], "darwin": ["--mac", "dmg", "zip"], "linux": ["--linux", "AppImage"]}[sys.platform]
    run([*npm, "run", "build"], cwd=desktop, env=env)
    run([*npm, "run", "builder", "--", *targets, f"-c.extraMetadata.version={version}", *version_args, *builder_args], cwd=desktop, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=False,
                        help="Release tag (vX.Y.Z / canary). Required unless --commit is given")
    parser.add_argument("--commit", dest="commit_build", default=None,
                        help="Commit-only build: exact full 40-char SHA the checkout is at; "
                             "version comes from the target pyproject, no tag is referenced")
    parser.add_argument("--variant", choices=["bundled", "store", "light"], default="bundled")
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("builder_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if bool(args.tag) == bool(args.commit_build):
        parser.error("exactly one of --tag or --commit is required")
    build(args.repo, args.tag, args.variant, [v for v in args.builder_args if v != "--"],
          commit_build=args.commit_build)


if __name__ == "__main__":
    main()

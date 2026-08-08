"""Scaffold or validate a `.devcontainer/` directory for a project.

Usage:
    python init.py <target-dir> [--python 3.11] [--node 20] [--features ...]
                         [--vscode-extensions ...] [--dockerfile] [--image ...]
                         [--port 3000,8000] [--post-create "..."]
                         [--force] [--dry-run] [--validate]

Generates a `devcontainer.json` (and optionally a `Dockerfile`) inside
`.devcontainer/` under the target directory. With `--validate`, just
checks the existing config and exits. With `--dry-run`, prints the
config that would be written to stdout instead of touching disk.

Stdlib only. No third-party dependencies.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


PYTHON_IMAGE = "mcr.microsoft.com/devcontainers/python"
NODE_IMAGE = "mcr.microsoft.com/devcontainers/javascript-node"

# Common devcontainer features the user is likely to want, with the
# recommended pin pattern (`:latest` is fine for `--dry-run`; we still
# warn unpinned features on `--validate` so the user adopts pinning).
DEFAULT_FEATURES: dict[str, str] = {
    "ghcr.io/devcontainers/features/common-utils:2": "latest",
    "ghcr.io/devcontainers/features/git:1": "latest",
}

# Field order follows the devcontainer.json schema recommendation. The
# order doesn't change semantics, but Codespaces and the devcontainer
# CLI are pickier about a few fields.
DEVCONTAINER_KEY_ORDER = [
    "name",
    "build",
    "image",
    "features",
    "customizations",
    "forwardPorts",
    "portsAttributes",
    "otherPortsAttributes",
    "mounts",
    "runArgs",
    "runServices",
    "containerUser",
    "containerEnv",
    "remoteUser",
    "remoteEnv",
    "userEnvProbe",
    "updateRemoteUserUID",
    "init",
    "privileged",
    "capAdd",
    "securityOpt",
    "shutdownAction",
    "workspaceMount",
    "workspaceFolder",
    "onCreateCommand",
    "updateContentCommand",
    "postCreateCommand",
    "postStartCommand",
    "postAttachCommand",
    "waitFor",
    "composeContainer",
    "service",
    "runServices",
    "shutdownAction",
]


def parse_features(spec: str | None) -> dict[str, str | dict]:
    """Parse `--features name:version,name2:version2` into a features map.

    Each entry is referenced as `name` (with `:latest` implied) or
    `name:version` for an explicit pin. If the user passes an empty
    string or None, returns the default common-utils + git features.
    """
    if spec is None or spec.strip() == "":
        return {
            "ghcr.io/devcontainers/features/common-utils:2": "latest",
            "ghcr.io/devcontainers/features/git:1": "latest",
        }
    out: dict[str, str | dict] = {}
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        if ":" in token and not token.startswith("ghcr.io/"):
            # Allow `--features docker-in-docker:2` shorthand
            name, version = token.split(":", 1)
            out[f"ghcr.io/devcontainers/features/{name}:{version}"] = "latest"
        else:
            out[f"ghcr.io/devcontainers/features/{token}"] = "latest"
    return out


def parse_csv(spec: str | None) -> list[str]:
    if spec is None:
        return []
    return [s.strip() for s in spec.split(",") if s.strip()]


def build_config(
    *,
    python_version: str | None,
    node_version: str | None,
    features: dict[str, str | dict],
    vscode_extensions: list[str],
    port_list: list[str],
    post_create: str | None,
    image_override: str | None,
    use_dockerfile: bool,
) -> dict:
    """Build a devcontainer.json-shaped Python dict.

    Field selection rules:
    - If --dockerfile: emit `build.dockerfile` (and `build.context`),
      no `image` field. (Mutually exclusive — schema requires one or
      the other.)
    - Else: emit `image` based on --python / --node / --image.
    """
    config: dict = {"name": "Dev Container"}

    if use_dockerfile:
        config["build"] = {"dockerfile": "Dockerfile", "context": ".."}
    else:
        if image_override:
            config["image"] = image_override
        elif python_version:
            config["image"] = f"{PYTHON_IMAGE}:{python_version}-bookworm"
        elif node_version:
            config["image"] = f"{NODE_IMAGE}:{node_version}-bookworm"
        else:
            config["image"] = f"{PYTHON_IMAGE}:3.12-bookworm"

    if features:
        config["features"] = features

    if vscode_extensions:
        config.setdefault("customizations", {}).setdefault("vscode", {})["extensions"] = vscode_extensions

    if port_list:
        config["forwardPorts"] = [int(p) for p in port_list]

    if post_create:
        config["postCreateCommand"] = post_create

    return _reorder_fields(config)


def _reorder_fields(config: dict) -> dict:
    """Return a new dict with fields in the schema-recommended order.

    Unknown fields (e.g. user-added) end up at the end in original
    insertion order, which is the JSON-equivalent of "preserved but
    not promoted."
    """
    seen: set[str] = set()
    ordered: dict = {}
    for key in DEVCONTAINER_KEY_ORDER:
        if key in config:
            ordered[key] = config[key]
            seen.add(key)
    for key, value in config.items():
        if key not in seen:
            ordered[key] = value
    return ordered


def build_dockerfile(*, python_version: str | None, node_version: str | None) -> str:
    """Generate a minimal Dockerfile that pairs with the generated config.

    Uses the same official devcontainer base image as the default
    `image:` field, then adds common dev tooling as `vscode`. The build
    stage runs as root for apt; runtime switches to `vscode` (UID 1000).
    """
    if python_version:
        base = f"{PYTHON_IMAGE}:{python_version}-bookworm"
    elif node_version:
        base = f"{NODE_IMAGE}:{node_version}-bookworm"
    else:
        base = f"{PYTHON_IMAGE}:3.12-bookworm"

    return (
        f"FROM {base}\n"
        "\n"
        "# Common dev tooling. Add project-specific build steps below.\n"
        "USER root\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends \\\n"
        "    build-essential \\\n"
        "    curl \\\n"
        "    git \\\n"
        "    less \\\n"
        "    vim \\\n"
        " && rm -rf /var/lib/apt/lists/*\n"
        "\n"
        "# Drop back to the non-root user the base image provides.\n"
        "USER vscode\n"
        "WORKDIR /workspaces\n"
    )


def write_config(
    target: Path,
    config: dict,
    *,
    use_dockerfile: bool,
    force: bool,
    python_version: str | None,
    node_version: str | None,
) -> None:
    dev_dir = target / ".devcontainer"
    config_path = dev_dir / "devcontainer.json"
    if config_path.exists() and not force:
        raise SystemExit(
            f"refusing to overwrite existing {config_path} (pass --force to override)"
        )
    dev_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    if use_dockerfile:
        dockerfile_path = dev_dir / "Dockerfile"
        if dockerfile_path.exists() and not force:
            raise SystemExit(
                f"refusing to overwrite existing {dockerfile_path} (pass --force to override)"
            )
        dockerfile_path.write_text(
            build_dockerfile(
                python_version=python_version,
                node_version=node_version,
            ),
            encoding="utf-8",
        )


def validate(target: Path) -> int:
    config_path = target / ".devcontainer" / "devcontainer.json"
    if not config_path.is_file():
        print(f"FAIL: no devcontainer.json at {config_path}", file=sys.stderr)
        return 1
    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        print(f"FAIL: invalid JSON in {config_path}: {exc}", file=sys.stderr)
        return 1
    issues: list[str] = []
    if not isinstance(config, dict):
        issues.append("top-level JSON must be an object")
    if isinstance(config, dict):
        if "image" not in config and "build" not in config:
            issues.append("missing 'image' or 'build' — devcontainer needs one")
        if "features" in config and not isinstance(config["features"], dict):
            issues.append("'features' must be an object mapping feature IDs to options")
        for feat_id in config.get("features", {}).keys():
            if not feat_id.startswith("ghcr.io/"):
                issues.append(
                    f"feature {feat_id!r} is not pinned with a registry prefix "
                    f"(expected e.g. 'ghcr.io/devcontainers/features/<name>:<version>')"
                )
            elif ":" not in feat_id.split("/")[-1]:
                issues.append(f"feature {feat_id!r} has no version pin (use e.g. ':latest' or ':2')")
    if issues:
        for line in issues:
            print(f"  - {line}", file=sys.stderr)
        return 1
    print(f"OK: {config_path} looks well-formed")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scaffold or validate a .devcontainer/ directory.",
    )
    parser.add_argument(
        "target",
        help="Directory under which to create .devcontainer/ (or to validate).",
    )
    parser.add_argument("--python", help="Pin a Python version (e.g. 3.11).")
    parser.add_argument("--node", help="Pin a Node version (e.g. 20).")
    parser.add_argument(
        "--features",
        help="Comma-separated features (e.g. 'docker-in-docker,git' or 'docker-in-docker:2,git:1').",
    )
    parser.add_argument(
        "--vscode-extensions",
        help="Comma-separated VS Code extension IDs to install in the container.",
    )
    parser.add_argument("--dockerfile", action="store_true", help="Generate a matching Dockerfile.")
    parser.add_argument("--image", help="Override the base image (e.g. 'mcr.microsoft.com/devcontainers/base:debian-12').")
    parser.add_argument(
        "--port",
        action="append",
        help="Port to forward (repeat for multiple, or comma-separate).",
    )
    parser.add_argument(
        "--post-create",
        help="Command to run after container creation (e.g. 'pip install -r requirements.txt').",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated config to stdout instead of writing.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the existing .devcontainer/devcontainer.json and exit.",
    )
    return parser.parse_args(argv)


def _flatten_ports(port_args: list[str] | None) -> list[str]:
    if not port_args:
        return []
    out: list[str] = []
    for entry in port_args:
        out.extend(parse_csv(entry))
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    target = Path(args.target).expanduser().resolve()

    if args.validate:
        return validate(target)

    if not target.is_dir():
        raise SystemExit(f"target is not a directory: {target}")

    config = build_config(
        python_version=args.python,
        node_version=args.node,
        features=parse_features(args.features),
        vscode_extensions=parse_csv(args.vscode_extensions),
        port_list=_flatten_ports(args.port),
        post_create=args.post_create,
        image_override=args.image,
        use_dockerfile=args.dockerfile,
    )

    if args.dry_run:
        print(json.dumps(config, indent=2))
        return 0

    write_config(
        target,
        config,
        use_dockerfile=args.dockerfile,
        force=args.force,
        python_version=args.python,
        node_version=args.node,
    )
    print(f"wrote {target / '.devcontainer' / 'devcontainer.json'}")
    if args.dockerfile:
        print(f"wrote {target / '.devcontainer' / 'Dockerfile'}")
    print("next: open the repo in VS Code and run 'Dev Containers: Reopen in Container'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

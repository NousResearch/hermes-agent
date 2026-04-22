"""
``hermes skills nacos ...`` subcommands.

Thin wrappers over ``tools.nacos_cli_client`` + ``tools.skills_nacos``:

* ``doctor``  – verify nacos-cli installation + server addr + config file
* ``list``    – list skills in a namespace/group
* ``pull``    – install a skill from Nacos (with conflict matrix)
* ``push``    – zip a local skill and upload to Nacos
* ``sync``    – bulk-pull an entire namespace
* ``login``   – delegate to nacos-cli login (interactive)

The pull command is conflict-aware:

    local absent              -> install
    local present + no lock   -> require --force
    local unchanged (lock ok) -> require --update
    local modified            -> require --force
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Callable, List, Optional

from rich.console import Console
from rich.table import Table

from hermes_constants import get_hermes_home
from tools.nacos_cli_client import (
    NacosCliClient,
    NacosCliError,
    NacosCliNotInstalled,
    NacosSkillEntry,
)
from tools.skills_nacos import (
    NacosSkillSource,
    parse_nacos_identifier,
)

logger = logging.getLogger(__name__)
_console = Console()

_VERSION_RE = re.compile(r"^(version:\s*)(\d+)\.(\d+)\.(\d+)\s*$", re.MULTILINE)


ClientFactory = Callable[[], NacosCliClient]


def _default_client() -> NacosCliClient:
    return NacosCliClient()


# ---------------------------------------------------------------------- doctor

def cmd_doctor(
    *,
    client: Optional[NacosCliClient] = None,
    server_addr: Optional[str] = None,
    console: Optional[Console] = None,
) -> int:
    c = client or _default_client()
    console = console or _console
    server_addr = server_addr if server_addr is not None else os.environ.get("NACOS_SERVER_ADDR")
    ok = True
    if c.check_installed():
        try:
            ver = c.version()
            console.print(f"[green]nacos-cli: installed[/] ({ver})")
        except NacosCliError as e:
            console.print(f"[yellow]nacos-cli installed but version check failed: {e}[/]")
            ok = False
    else:
        console.print("[red]nacos-cli: not installed[/]")
        console.print("  -> install via: [bold]npm i -g @nacos-group/cli[/]")
        ok = False
    if server_addr:
        console.print(f"NACOS_SERVER_ADDR = {server_addr}")
    else:
        console.print("[yellow]NACOS_SERVER_ADDR not set[/]")
        ok = False
    cfg = Path.home() / ".nacos-cli.conf"
    if cfg.exists():
        console.print(f"[green]config:[/] {cfg}")
    else:
        console.print(f"[yellow]config missing:[/] {cfg} (run `hermes skills nacos login`)")
    return 0 if ok else 1


# ---------------------------------------------------------------------- list

def cmd_list(
    *,
    client: Optional[NacosCliClient] = None,
    namespace: str = "public",
    group: str = "hermes-skills",
    query: Optional[str] = None,
    console: Optional[Console] = None,
) -> int:
    c = client or _default_client()
    console = console or _console
    try:
        entries: List[NacosSkillEntry] = c.list_skills(
            namespace=namespace, group=group, query=query
        )
    except NacosCliNotInstalled as e:
        console.print(f"[red]{e}[/]")
        return 1
    except NacosCliError as e:
        console.print(f"[red]nacos list failed:[/] {e}")
        return 1
    if not entries:
        console.print("[dim](no skills found)[/]")
        return 0
    table = Table(title=f"Nacos skills in {namespace}/{group}")
    table.add_column("name", style="cyan")
    table.add_column("version")
    table.add_column("author")
    table.add_column("description")
    for entry in entries:
        desc = entry.description or ""
        table.add_row(entry.name, entry.version, entry.author or "-", desc[:60])
    console.print(table)
    return 0


# ---------------------------------------------------------------------- login

def cmd_login(*, server_addr: Optional[str] = None) -> int:
    """Delegate to nacos-cli login (interactive).

    We never capture stdin/stdout here — let nacos-cli prompt the user
    directly so credentials never touch the hermes process.
    """
    args = ["nacos-cli", "login"]
    if server_addr:
        args.extend(["--server", server_addr])
    try:
        return subprocess.call(args)
    except FileNotFoundError:
        _console.print(
            "[red]nacos-cli not found.[/] install via "
            "[bold]npm i -g @nacos-group/cli[/]"
        )
        return 1


# ---------------------------------------------------------------------- pull

def _dir_hash(path: Path) -> str:
    """SHA256 of a directory's content; matches tools.skills_guard.content_hash convention."""
    h = hashlib.sha256()
    for fp in sorted(path.rglob("*")):
        if fp.is_file():
            rel = fp.relative_to(path).as_posix()
            h.update(rel.encode())
            h.update(b"\x00")
            h.update(fp.read_bytes())
    return "sha256:" + h.hexdigest()


def cmd_pull(
    args,
    *,
    client_factory: Optional[ClientFactory] = None,
    console: Optional[Console] = None,
    installer: Optional[Callable[..., None]] = None,
) -> int:
    """Install a skill from Nacos via the canonical install pipeline.

    Adds a nacos-specific conflict check on top of ``do_install``:
    if the local skill directory hash differs from the recorded lock hash,
    the pull is rejected unless ``--force`` is passed.
    """
    console = console or _console
    factory = client_factory or _default_client
    client = factory()

    source = NacosSkillSource(client=client)
    raw = args.name
    if raw.startswith("nacos://"):
        ident_str = raw
    else:
        ns = args.namespace or source.default_namespace
        group = args.group or source.default_group
        tail = f"@{args.version}" if args.version else ""
        ident_str = f"nacos://{ns}/{group}/{raw}{tail}"

    ident = parse_nacos_identifier(ident_str)

    # --- local-state conflict check (specific to nacos) ---
    # Use module-attribute access so monkeypatching tools.skills_hub.SKILLS_DIR
    # in tests is honored even after this module is imported.
    import tools.skills_hub as _sh

    target = _sh.SKILLS_DIR / ident.name
    lock = _sh.HubLockFile()
    existing = lock.get_installed(ident.name)

    if target.exists():
        current_hash = _dir_hash(target)
        if existing:
            recorded = existing.get("content_hash")
            if recorded is None:
                # Lock entry exists but has no checksum — we cannot verify
                # whether the local copy has been modified.  Require --update
                # or --force so the user makes an explicit choice, and fall
                # through to reinstall when either is set.
                if not args.update and not args.force:
                    console.print(
                        f"[yellow]{ident.name} already installed "
                        "(no checksum recorded);[/] "
                        "use --update to refresh or --force to overwrite"
                    )
                    return 0
            elif recorded != current_hash and not args.force:
                console.print(
                    f"[red]local modifications detected for {ident.name};[/] "
                    "push your changes first or re-run with --force"
                )
                return 2
            elif not args.update and not args.force:
                console.print(
                    f"[yellow]{ident.name} already installed;[/] "
                    "use --update to refresh from nacos"
                )
                return 0
        elif not args.force:
            console.print(
                f"[yellow]{ident.name} exists locally with no nacos lock;[/] "
                "use --force to overwrite"
            )
            return 2

    # --- delegate to the canonical install pipeline ---
    # We use --force semantics for both --update (already-installed re-fetch)
    # and --force (user override).  skip_confirm avoids prompting in a CLI
    # pull — the scanner + policy still decide whether to block.
    if installer is None:
        from hermes_cli.skills_hub import do_install as installer  # type: ignore[assignment]

    try:
        installer(
            ident_str,
            category="",
            force=bool(args.force or args.update),
            console=console,
            skip_confirm=True,
        )
    except NacosCliError as e:
        console.print(f"[red]nacos fetch failed:[/] {e}")
        return 1
    return 0


# ---------------------------------------------------------------------- push

def _bump_version(skill_md: Path, level: str) -> str:
    text = skill_md.read_text()
    m = _VERSION_RE.search(text)
    if not m:
        raise ValueError(f"no semver `version:` line in {skill_md}")
    major, minor, patch = int(m.group(2)), int(m.group(3)), int(m.group(4))
    if level == "major":
        major, minor, patch = major + 1, 0, 0
    elif level == "minor":
        minor, patch = minor + 1, 0
    else:
        patch += 1
    new_v = f"{major}.{minor}.{patch}"
    skill_md.write_text(
        _VERSION_RE.sub(lambda mm: f"{mm.group(1)}{new_v}", text)
    )
    return new_v


def _zip_skill_dir(src: Path, dst: Path) -> None:
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(src.rglob("*")):
            if fp.is_dir():
                continue
            rel = fp.relative_to(src).as_posix()
            parts = rel.split("/")
            if parts[0] in (".git", "__pycache__") or "__pycache__" in parts:
                continue
            zf.write(fp, arcname=rel)


def cmd_push(
    args,
    *,
    client_factory: Optional[ClientFactory] = None,
    console: Optional[Console] = None,
) -> int:
    console = console or _console
    src = get_hermes_home() / "skills" / args.name
    if not src.exists():
        console.print(f"[red]skill not found locally:[/] {src}")
        return 1
    skill_md = src / "SKILL.md"
    if not skill_md.exists():
        console.print(f"[red]missing SKILL.md:[/] {skill_md}")
        return 1
    if args.bump:
        try:
            new_v = _bump_version(skill_md, args.bump)
            console.print(f"[cyan]bumped version -> {new_v}[/]")
        except ValueError as e:
            console.print(f"[red]{e}[/]")
            return 1

    namespace = args.namespace or os.environ.get("NACOS_NAMESPACE", "public")
    group = args.group or "hermes-skills"

    with tempfile.TemporaryDirectory() as td:
        zip_path = Path(td) / f"{args.name}.zip"
        _zip_skill_dir(src, zip_path)
        client = (client_factory or _default_client)()
        try:
            result = client.upload_skill(zip_path, namespace=namespace, group=group)
        except NacosCliError as e:
            console.print(f"[red]nacos upload failed:[/] {e}")
            return 1

    version = result.get("version", "?")
    console.print(
        f"[green]pushed[/] {args.name} v{version} to {namespace}/{group}"
    )
    return 0


# ---------------------------------------------------------------------- sync

def cmd_sync(
    args,
    *,
    client_factory: Optional[ClientFactory] = None,
    console: Optional[Console] = None,
) -> int:
    console = console or _console
    out_dir = str(get_hermes_home() / "skills")
    client = (client_factory or _default_client)()
    try:
        result = client.sync_namespace(
            namespace=args.namespace,
            group=args.group,
            output_dir=out_dir,
        )
    except NacosCliError as e:
        console.print(f"[red]nacos sync failed:[/] {e}")
        return 1
    synced = result.get("synced", [])
    skipped = result.get("skipped", [])
    console.print(f"[green]synced:[/] {', '.join(synced) or '(none)'}")
    if skipped:
        console.print(f"[yellow]skipped:[/] {', '.join(skipped)}")
    return 0


# ---------------------------------------------------------------------- argparse

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hermes skills nacos")
    sub = p.add_subparsers(dest="action", required=True)

    sub.add_parser("doctor", help="check nacos-cli + server reachability")

    lst = sub.add_parser("list", help="list skills in a namespace/group")
    lst.add_argument("--namespace", default=os.environ.get("NACOS_NAMESPACE", "public"))
    lst.add_argument("--group", default="hermes-skills")
    lst.add_argument("--query", default=None)

    pull = sub.add_parser("pull", help="install a skill from Nacos")
    pull.add_argument("name")
    pull.add_argument("--namespace", default=None)
    pull.add_argument("--group", default=None)
    pull.add_argument("--version", default=None)
    pull.add_argument("--update", action="store_true")
    pull.add_argument("--force", action="store_true")

    push = sub.add_parser("push", help="upload a local skill to Nacos")
    push.add_argument("name")
    push.add_argument("--namespace", default=None)
    push.add_argument("--group", default="hermes-skills")
    push.add_argument("--bump", choices=["major", "minor", "patch"], default=None)

    sync = sub.add_parser("sync", help="pull every skill in a namespace/group")
    sync.add_argument("--namespace", default=os.environ.get("NACOS_NAMESPACE", "public"))
    sync.add_argument("--group", default="hermes-skills")

    sub.add_parser("login", help="delegate to nacos-cli login")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    if args.action == "doctor":
        return cmd_doctor()
    if args.action == "list":
        return cmd_list(namespace=args.namespace, group=args.group, query=args.query)
    if args.action == "pull":
        return cmd_pull(args)
    if args.action == "push":
        return cmd_push(args)
    if args.action == "sync":
        return cmd_sync(args)
    if args.action == "login":
        return cmd_login()
    parser.error(f"unknown action: {args.action}")
    return 2

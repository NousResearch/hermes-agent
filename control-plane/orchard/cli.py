"""orchard — control-plane CLI.

    orchard provision <id> --mm-user <mm_user_id> [--name "Full Name"]
    orchard deprovision <id>
    orchard list
    orchard serve [--ingress cli|mattermost]
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from .config import Settings
from .provisioner import deprovision, scaffold_home
from .registry import Registry


def _settings(args) -> Settings:
    return Settings.load(args.config)


def cmd_provision(args) -> int:
    s = _settings(args)
    reg = Registry(s.paths.registry_db)
    try:
        emp = reg.add(args.id, args.name or args.id, args.mm_user)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    home = scaffold_home(s, emp)
    print(f"✓ provisioned '{emp.id}' (mm_user={emp.mm_user_id})")
    print(f"  home: {home}  [mode {oct(s.security.home_mode_int)}]")
    print(f"  model: {s.llm.provider}:{s.llm.model} -> {s.llm.base_url}")
    if not s.security.run_as_user:
        print("  ⚠ run_as_user is null: workers share this OS user (dev-only isolation).")
    return 0


def cmd_deprovision(args) -> int:
    s = _settings(args)
    reg = Registry(s.paths.registry_db)
    removed_reg = reg.remove(args.id)
    removed_home = deprovision(s, args.id)
    print(f"{'✓' if (removed_reg or removed_home) else '·'} deprovisioned '{args.id}' "
          f"(registry={removed_reg}, home={removed_home})")
    return 0


def cmd_list(args) -> int:
    s = _settings(args)
    reg = Registry(s.paths.registry_db)
    emps = reg.all()
    if not emps:
        print("no employees provisioned")
        return 0
    print(f"{'ID':<20} {'MM_USER':<24} {'HOME':<8} NAME")
    for e in emps:
        home = s.paths.home_for(e.id)
        print(f"{e.id:<20} {e.mm_user_id:<24} {'yes' if home.exists() else 'MISSING':<8} {e.display_name}")
    return 0


def cmd_serve(args) -> int:
    from .backends import make_backend
    from .router import Router
    from .supervisor import Supervisor

    s = _settings(args)
    reg = Registry(s.paths.registry_db)
    backend = make_backend(s)
    supervisor = Supervisor(s, backend)

    if args.ingress == "cli":
        from .ingress.cli import CLIIngress
        ingress = CLIIngress(default_sender=getattr(args, "as_employee", None) or "tester")
    else:
        from .ingress.mattermost import MattermostIngress
        ingress = MattermostIngress(s.mattermost.url, s.mattermost.token)

    router = Router(s, reg, supervisor, ingress)
    try:
        asyncio.run(router.serve())
    except KeyboardInterrupt:
        print("\nshutting down")
    return 0


def cmd_web(args) -> int:
    from . import api
    s = _settings(args)
    print(f"orchard admin UI on http://{args.host}:{args.port}  (backend={s.backend}, sandbox={s.security.sandbox or 'none'})")
    api.run(s, args.host, args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    p = argparse.ArgumentParser(prog="orchard", description="Hermes multi-tenant control-plane")
    p.add_argument("--config", default="config.yaml", help="path to config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("provision", help="create + lock down an employee workspace")
    sp.add_argument("id")
    sp.add_argument("--mm-user", required=True, help="Mattermost user_id of the employee")
    sp.add_argument("--name", default=None)
    sp.set_defaults(func=cmd_provision)

    sp = sub.add_parser("deprovision", help="remove an employee workspace")
    sp.add_argument("id")
    sp.set_defaults(func=cmd_deprovision)

    sp = sub.add_parser("list", help="list provisioned employees")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("serve", help="run the control-plane")
    sp.add_argument("--ingress", choices=["cli", "mattermost"], default="cli")
    sp.add_argument("--as", dest="as_employee", default=None,
                    help="CLI ingress: treat unprefixed input as this employee's mm_user")
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser("web", help="run the admin web UI + API")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8700)
    sp.set_defaults(func=cmd_web)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

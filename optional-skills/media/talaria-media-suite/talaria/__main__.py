"""CLI entry — python3 -m talaria.

One argparse tree: 10 groups (7 services + library + ops + the top-level
dashboard entry) carrying 68 subs, plus the promoted top-level subs
(status setup discover diagnose queue dashboard version). Exit codes:
0 ok · 1 any TalariaError · 2 usage (argparse default). TalariaError is
caught ONLY here and prints one stderr line prefixed "talaria:" — never
a traceback, never a key (redaction rule in core).
"""

import argparse
import sys

from . import __version__
from . import bazarr, dashboard, jellyfin, library, lidarr, ops, overseerr
from . import prowlarr, radarr, sonarr
from .core import TalariaError

PROG = "python3 -m talaria"
DESCRIPTION = ("talaria — drive a self-hosted *arr + Jellyfin media stack "
               "(Sonarr, Radarr, Lidarr, Prowlarr, Bazarr, Overseerr, "
               "Jellyfin)")

GROUPS = (sonarr, radarr, lidarr, prowlarr, bazarr, overseerr, jellyfin,
          library, ops)


def _version(args):
    print("talaria %s" % __version__)


def build_parser():
    parser = argparse.ArgumentParser(
        prog=PROG, description=DESCRIPTION,
        epilog="Config: TALARIA_HOST + one "
               "<SERVICE>_KEY per service — see `python3 -m talaria setup`.")
    parser.add_argument("--json", action="store_true",
                        help="emit the last API payload as JSON (approx "
                        "subs carry \"approx\": true)")
    parser.add_argument("--timeout", type=float, default=10.0, metavar="N",
                        help="per-request timeout in seconds (default 10)")
    parser.add_argument("--host", metavar="H",
                        help="override the host for this invocation")
    top = parser.add_subparsers(dest="group", metavar="<group>")
    for mod in GROUPS:
        g = top.add_parser(mod.NAME, help=mod.HELP)
        mod.register(g)
    ops.register_promoted(top)          # status setup discover diagnose queue
    dashboard.register_promoted(top)    # dashboard <out.html>
    v = top.add_parser("version", help="print the talaria version")
    v.set_defaults(func=_version)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 2
    try:
        func(args)
    except TalariaError as exc:
        print("talaria: %s" % exc, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("talaria: interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())

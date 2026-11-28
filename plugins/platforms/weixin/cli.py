"""Plugin-owned CLI flags for a single interactive binding attempt."""

from __future__ import annotations

import argparse


def register_cli(parser: argparse.ArgumentParser) -> None:
    subcommands = parser.add_subparsers(dest="weixin_command", required=True)
    login = subcommands.add_parser("login", help="扫码连接微信，并保存账号与访问设置")
    login.add_argument(
        "--prompt-bind-owner", action=argparse.BooleanOptionalAction, default=None,
        help="询问是否绑定本人微信；选择 N 时要求实名授权（默认关闭）",
    )
    login.add_argument(
        "--verify-pin", action=argparse.BooleanOptionalAction, default=None,
        help="请求手机 PIN 校验（默认关闭，实际校验由服务端决定）",
    )
    login.add_argument("--acctweb-base-url", default=None, help="实名授权测试服务地址，仅影响实名接口")
    login.add_argument("--acctweb-proxy", default=None, help="实名授权 HTTP/HTTPS 代理，仅影响实名接口")


def dispatch(args: argparse.Namespace) -> int:
    from .adapter import interactive_setup

    success = interactive_setup(**{
        key: getattr(args, key, None)
        for key in ("prompt_bind_owner", "verify_pin", "acctweb_base_url", "acctweb_proxy")
    })
    return 0 if success else 1

"""统一调试工具 CLI — 整合 GC 日志、线程堆栈、ES 日志分析。"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.es_log_search import EsLogSearcher
from scripts.gc_log_analyzer import GcLogAnalyzer
from scripts.thread_stack_analyzer import ThreadStackAnalyzer


def _config_default(section: str, key: str, fallback=None):
    """从 DEFAULT_THRESHOLDS 读取默认值。"""
    from config.default_thresholds import DEFAULT_THRESHOLDS
    return DEFAULT_THRESHOLDS.get(section, {}).get(key, fallback)


class DebugToolsCLI:
    """统一调试工具 CLI，通过子命令方式使用三个分析器。"""

    def __init__(self):
        self.parser = self._build_parser()

    def _build_parser(self) -> argparse.ArgumentParser:
        """构建参数解析器。"""
        parser = argparse.ArgumentParser(
            prog="debug_tools",
            description="统一调试工具 CLI — GC 日志、线程堆栈、ES 日志分析",
        )
        subparsers = parser.add_subparsers(dest="subcommand", help="子命令")

        # gc 子命令
        gc_parser = subparsers.add_parser("gc", help="GC 日志分析")
        gc_parser.add_argument("--host", required=True, help="目标主机地址")
        gc_parser.add_argument("--ssh-user", default=_config_default("ssh", "default_user", "root"), help="SSH 用户名")
        gc_parser.add_argument("--ssh-key", default=_config_default("ssh", "default_key_path"), help="SSH 私钥路径")
        gc_parser.add_argument("--ssh-port", type=int, default=_config_default("ssh", "default_port", 22), help="SSH 端口")
        gc_parser.add_argument("--gc-log-path", help="GC 日志文件路径")
        gc_parser.add_argument("--start", help="开始时间")
        gc_parser.add_argument("--end", help="结束时间")

        # stack 子命令
        stack_parser = subparsers.add_parser("stack", help="线程堆栈分析")
        stack_parser.add_argument("--host", required=True, help="目标主机地址")
        stack_parser.add_argument("--ssh-user", default=_config_default("ssh", "default_user", "root"), help="SSH 用户名")
        stack_parser.add_argument("--ssh-key", default=_config_default("ssh", "default_key_path"), help="SSH 私钥路径")
        stack_parser.add_argument("--ssh-port", type=int, default=_config_default("ssh", "default_port", 22), help="SSH 端口")
        stack_parser.add_argument("--pid", type=int, help="进程 PID")
        stack_parser.add_argument("--process-name", default=_config_default("jvm", "tomcat_process_name", "catalina"), help="进程名")
        stack_parser.add_argument("--keyword", help="过滤关键字")
        stack_parser.add_argument("--context-lines", type=int, default=_config_default("debug", "stack_context_lines", 10), help="上下文行数")

        # log 子命令
        log_parser = subparsers.add_parser("log", help="ES 日志检索")
        log_parser.add_argument("--es-url", default=_config_default("elk", "elasticsearch_url"), help="Elasticsearch URL")
        log_parser.add_argument("--host-name", required=True, help="主机名")
        log_parser.add_argument("--start", required=True, help="开始时间")
        log_parser.add_argument("--end", required=True, help="结束时间")
        log_parser.add_argument("--keyword", help="搜索关键字")
        log_parser.add_argument("--level", help="日志级别")
        log_parser.add_argument("--size", type=int, default=_config_default("debug", "es_default_size", 100), help="返回条数")

        return parser

    def parse_args(self, argv: List[str]) -> argparse.Namespace:
        """解析命令行参数。"""
        return self.parser.parse_args(argv)

    def _build_gc_config(self, args: argparse.Namespace) -> Dict[str, Any]:
        """从参数构建 GC 分析器配置。"""
        config = {"host": args.host, "ssh_user": args.ssh_user, "ssh_port": args.ssh_port}
        if args.ssh_key:
            config["ssh_key"] = args.ssh_key
        if args.gc_log_path:
            config["gc_log_path"] = args.gc_log_path
        return config

    def _build_stack_config(self, args: argparse.Namespace) -> Dict[str, Any]:
        """从参数构建线程堆栈分析器配置。"""
        config = {"host": args.host, "ssh_user": args.ssh_user, "ssh_port": args.ssh_port}
        if args.ssh_key:
            config["ssh_key"] = args.ssh_key
        if args.pid:
            config["pid"] = args.pid
        if args.process_name:
            config["tomcat_process_name"] = args.process_name
        return config

    def _build_log_config(self, args: argparse.Namespace) -> Dict[str, Any]:
        """从参数构建 ES 日志搜索器配置。"""
        config = {}
        if args.es_url:
            config["elasticsearch_url"] = args.es_url
        return config

    def run(self, argv: List[str]) -> Dict[str, Any]:
        """执行调试工具命令。无子命令时 sys.exit(1)。"""
        args = self.parse_args(argv)

        if not args.subcommand:
            sys.exit(1)

        if args.subcommand == "gc":
            config = self._build_gc_config(args)
            analyzer = GcLogAnalyzer(config)
            return analyzer.analyze(args.start, args.end)

        elif args.subcommand == "stack":
            config = self._build_stack_config(args)
            analyzer = ThreadStackAnalyzer(config)
            return analyzer.analyze(args.keyword, args.context_lines)

        elif args.subcommand == "log":
            config = self._build_log_config(args)
            searcher = EsLogSearcher(config)
            return searcher.search(
                args.host_name, args.start, args.end,
                keyword=args.keyword, level=args.level, size=args.size,
            )

        # 未知子命令
        sys.exit(1)


def main():
    """CLI 入口。"""
    cli = DebugToolsCLI()
    result = cli.run(sys.argv[1:])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

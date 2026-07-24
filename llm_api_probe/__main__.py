"""CLI 入口: 跑全部 / 部分 probe 模块并生成报告。

用法:
    python -m llm_api_probe --config configs/probe.yaml
    python -m llm_api_probe --config configs/probe.yaml --only speed,security
    python -m llm_api_probe --config configs/probe.yaml --provider openai-official
    python -m llm_api_probe --init-config configs/my.yaml    # 生成示例配置
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 让 python -m llm_api_probe 能找到 probes 包 (它和 __main__.py 同级)
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from probes.config import load_config, write_example_config  # noqa: E402
from probes.models import Provider, ProbeResult  # noqa: E402
from probes.report import render_console_table, write_json, write_markdown  # noqa: E402

# 延迟导入 probe 模块
PROBE_REGISTRY = {
    "connectivity": "probes.probe_connectivity",
    "context":     "probes.probe_context",
    "speed":       "probes.probe_speed",
    "stability":   "probes.probe_stability",
    "security":    "probes.probe_security",
    "ability":     "probes.probe_ability",
}


def _run_one_probe(probe_name: str, provider: Provider, model: str, verbose: bool = False, **kwargs) -> ProbeResult:
    import importlib
    mod = importlib.import_module(PROBE_REGISTRY[probe_name])
    # 把 verbose 透传给 probe (probe 自己决定要不要传给 client.call)
    kwargs["verbose"] = verbose
    try:
        return mod.run(provider, model, **kwargs)
    except Exception as e:
        r = ProbeResult(probe=probe_name, provider=provider.name, ok=False)
        r.error = f"{type(e).__name__}: {e}"
        return r


def main() -> int:
    parser = argparse.ArgumentParser(
        description="检测 API key 接入的大模型能力 / 速度 / 稳定性 / 安全性",
    )
    parser.add_argument("--config", "-c", help="provider YAML 配置文件")
    parser.add_argument(
        "--only",
        help=f"只跑指定 probe 模块 (逗号分隔), 可选: {','.join(PROBE_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--provider", "-p",
        action="append",
        help="只跑指定 provider.name (可多次), 默认全部",
    )
    parser.add_argument(
        "--model", "-m",
        help="覆盖 provider.models, 只测这一个 model (用于 ad-hoc)",
    )
    parser.add_argument(
        "--out-dir", "-o",
        default="reports",
        help="报告输出目录 (默认 ./reports)",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="不输出 Markdown 报告",
    )
    parser.add_argument(
        "--init-config",
        metavar="PATH",
        help="生成示例配置文件到 PATH 然后退出",
    )
    parser.add_argument(
        "--api-key",
        action="append",
        default=[],
        metavar="NAME=KEY",
        help="命令行注入 API key, 覆盖 YAML/env (可多次): --api-key Apimart=sk-xxx",
    )
    parser.add_argument(
        "--api-key-stdin",
        action="store_true",
        help="从 stdin 读取 api_key 覆盖 (JSON, {provider_name: key}), "
             "不进任何日志和报告",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="实时打印每个 HTTP 请求 (方便看进度, 知道没卡死)",
    )
    args = parser.parse_args()

    # 生成示例配置
    if args.init_config:
        write_example_config(args.init_config)
        print(f"[+] 已生成示例配置: {args.init_config}")
        return 0

    if not args.config:
        parser.error("必须提供 --config 或 --init-config")

    # 解析 api_key 覆盖 (命令行 > stdin > 环境变量)
    api_key_override: dict[str, str] = {}
    for item in args.api_key:
        if "=" not in item:
            print(f"[!] --api-key 格式错误, 应为 NAME=KEY: {item}", file=sys.stderr)
            return 2
        k, v = item.split("=", 1)
        api_key_override[k.strip()] = v.strip()
    if args.api_key_stdin:
        import json as _json
        raw = sys.stdin.read().strip()
        if raw:
            try:
                api_key_override.update(_json.loads(raw))
            except _json.JSONDecodeError as e:
                print(f"[!] --api-key-stdin JSON 解析失败: {e}", file=sys.stderr)
                return 2

    providers = load_config(args.config, api_key_override=api_key_override)
    if args.provider:
        providers = [p for p in providers if p.name in args.provider]
    if not providers:
        print("[!] 没有匹配的 provider", file=sys.stderr)
        return 2

    selected_probes = (
        [p.strip() for p in args.only.split(",") if p.strip()]
        if args.only
        else list(PROBE_REGISTRY.keys())
    )
    for p in selected_probes:
        if p not in PROBE_REGISTRY:
            print(f"[!] 未知 probe: {p}, 可选: {list(PROBE_REGISTRY.keys())}", file=sys.stderr)
            return 2

    print(f"[+] 加载 {len(providers)} 个 provider, 跑 {len(selected_probes)} 个模块")

    results: list[ProbeResult] = []
    t_start = time.perf_counter()

    for p in providers:
        models_to_test = [args.model] if args.model else (p.models or [])
        if not models_to_test:
            print(f"[!] provider '{p.name}' 没指定 model, 跳过")
            continue
        for model in models_to_test:
            print(f"\n── {p.label} ({p.name}) · {model} ──")
            for probe in selected_probes:
                t0 = time.perf_counter()
                print(f"  [{probe}] ...", end=" ", flush=True)
                r = _run_one_probe(probe, p, model)
                dt = time.perf_counter() - t0
                print(f"{r.ok and 'OK' or 'FAIL'} ({dt:.1f}s)")
                for w in r.warnings:
                    print(f"      ⚠ {w}")
                for f in r.findings:
                    print(f"      · {f}")
                if r.error:
                    print(f"      ✗ {r.error}")
                results.append(r)

    elapsed = time.perf_counter() - t_start

    # 输出
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    # 控制台报告
    print()
    print(render_console_table(providers, results))

    # JSON / Markdown
    write_json(providers, results, out_dir / f"report_{ts}.json")
    if not args.no_markdown:
        write_markdown(providers, results, out_dir / f"report_{ts}.md")

    print(f"\n[+] 报告已保存到 {out_dir}/report_{ts}.{{json,md}}")
    print(f"[+] 总耗时 {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""
Warashibe Reselling — Gateway slash command
/warashibe status|license|reroll|research|shipping|platforms|sop|ledger|profit
"""
from __future__ import annotations
import json
import socket
import subprocess
import sys
import time
try:
    from . import core
except ImportError:  # direct plugin-module smoke tests
    import core

OFFICE_HOST = "127.0.0.1"
OFFICE_PORT = 18765


def _ensure_office_server() -> bool:
    """Start the local HTTP server when the MOA office is not already served."""
    with socket.socket() as sock:
        if sock.connect_ex((OFFICE_HOST, OFFICE_PORT)) == 0:
            return True
    script = core.PLUGIN_DIR / "scripts" / "serve_office.py"
    if not script.exists():
        return False
    kwargs = {"cwd": str(core.PLUGIN_DIR)}
    if sys.platform == "win32":
        kwargs["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
        kwargs["close_fds"] = True
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        **kwargs,
    )
    for _ in range(10):
        time.sleep(0.1)
        with socket.socket() as sock:
            if sock.connect_ex((OFFICE_HOST, OFFICE_PORT)) == 0:
                return True
    return False


def _office_url() -> str:
    if not _ensure_office_server():
        return ""
    return f"http://{OFFICE_HOST}:{OFFICE_PORT}/office.html"

def handle_warashibe(args: str) -> str:
    """Gateway slash handler"""
    parts = args.strip().split(maxsplit=1)
    sub = parts[0] if parts else "status"
    rest = parts[1] if len(parts) > 1 else ""

    if sub == "status":
        r = {
            "platforms": len(core.PLATFORMS),
            "sedori_roles": list(core.SEDORI_ROLES.keys()),
            "shipping_apis": list(core.SHIPPING_APIS.keys()),
            "defaults": core.DEFAULTS,
        }
        return f"```\n{json.dumps(r, ensure_ascii=False, indent=2)}\n```"

    elif sub == "license":
        r = core.generate_license_package(rest or "license_pkg")
        return f"✅ 古物商許可パッケージ生成: {r['count']}ファイル → {r['output_dir']}"

    elif sub in ("moa", "office"):
        url = _office_url()
        if not url:
            return "❌ 3DオフィスHTTPサーバーを起動できませんでした。"
        return (
            "🌐 MOA AI Company 3Dオフィス\n"
            f"{url}\n"
            "Hermes Desktopの右側ペインでこのURLを開いてください。"
        )

    elif sub == "reroll":
        r = core.reroll_ai_employees(dry_run="--execute" not in rest)
        return f"```\n{json.dumps(r, ensure_ascii=False, indent=2)}\n```"

    elif sub == "research":
        import shlex
        opts = shlex.split(rest)
        keyword = ""
        budget = 10000
        for i, o in enumerate(opts):
            if o == "-k" and i+1 < len(opts): keyword = opts[i+1]
            if o == "-b" and i+1 < len(opts): budget = int(opts[i+1])
        if not keyword:
            return "❌ `--keyword` 必須。例: `/warashibe research -k \"Switchソフト\" -b 10000`"
        r = core.generate_research_sheet(keyword, budget)
        return f"✅ 試算シート生成: {r['output_path']}"

    elif sub == "price":
        import shlex
        from .price_research import search_markets
        opts = shlex.split(rest)
        keyword = ""
        platforms = ["mercari", "yahoo_auction"]
        limit = 10
        dry_run = "--live" not in opts
        i = 0
        while i < len(opts):
            if opts[i] in ("-k", "--keyword") and i + 1 < len(opts):
                keyword = opts[i + 1]
                i += 2
            elif opts[i] == "--platforms" and i + 1 < len(opts):
                platforms = [p for p in opts[i + 1].replace(",", " ").split() if p]
                i += 2
            elif opts[i] in ("-l", "--limit") and i + 1 < len(opts):
                limit = int(opts[i + 1])
                i += 2
            else:
                i += 1
        if not keyword:
            return "❌ `--keyword` 必須。例: `/warashibe price -k \"Switch\"`"
        r = search_markets(keyword, platforms, limit, dry_run=dry_run)
        return f"```\n{json.dumps(r, ensure_ascii=False, indent=2)}\n```"

    elif sub == "shipping":
        r = core.verify_shipping_apis()
        return f"```\n{json.dumps(r, ensure_ascii=False, indent=2)}\n```"

    elif sub == "platforms":
        r = core.platform_comparison()
        lines = ["| プラットフォーム | 手数料 | 送料連携 | API | |", "|---|---|---|---|"]
        for k, v in r.items():
            lines.append(f"| {v['name']} | {v['fee_pct']}%+{v['fee_flat']} | {v['shipping']} | {v['api']} |")
        return "\n".join(lines)

    elif sub == "sop":
        r = core.generate_sop_templates(rest or "sop")
        return f"✅ SOPテンプレート生成: {r['files']} → {r['output_dir']}"

    elif sub == "ledger":
        r = core.init_ledger(rest or None)
        return f"✅ 古物台帳初期化: {r['ledger_path']}"

    elif sub == "profit":
        import shlex
        opts = shlex.split(rest)
        buy = sell = 0
        platform = "mercari"
        shipping = 0
        for i, o in enumerate(opts):
            if o == "--buy" and i+1 < len(opts): buy = int(opts[i+1])
            elif o == "--sell" and i+1 < len(opts): sell = int(opts[i+1])
            elif o == "--platform" and i+1 < len(opts): platform = opts[i+1]
            elif o == "--shipping" and i+1 < len(opts): shipping = int(opts[i+1])
        if not buy or not sell:
            return "❌ `--buy` と `--sell` 必須。例: `/warashibe profit --buy 1000 --sell 2000 --platform mercari`"
        r = core.calc_profit(buy, sell, platform, shipping)
        emoji = "✅ GO" if r["go"] else "❌ STOP"
        return f"{emoji} 利益 {r['profit']}円 ({r['profit_rate']*100:.1f}%) / 手数料 {r['platform_fee']}円"

    else:
        return "`/warashibe `<sub> | status license reroll research price shipping platforms sop ledger profit moa`"


def handle_moa(args: str) -> str:
    """Return the local 3D MOA office URL for the Desktop/browser pane."""
    return handle_warashibe("moa")

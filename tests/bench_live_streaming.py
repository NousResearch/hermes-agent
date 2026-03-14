#!/usr/bin/env python3
import asyncio, os, sys, time, argparse, statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gateway.config import Platform, PlatformConfig
from gateway.platforms.telegram import TelegramAdapter
from gateway.stream_consumer import GatewayStreamConsumer

SAMPLE_TOKENS = (
    "Hello ", "there! ", "I ", "am ", "streaming ", "tokens ", "from ",
    "the ", "gateway ", "performance ", "benchmark. ", "This ", "measures ",
    "real ", "Telegram ", "API ", "latency ", "under ", "concurrent ",
    "load. ", "Each ", "session ", "sends ", "a ", "unique ", "message ",
    "to ", "verify ", "delivery. ",
)

def _token_stream(sid, n=150):
    toks = [f"[bench-{sid}] "]
    for i in range(n):
        toks.append(SAMPLE_TOKENS[i % len(SAMPLE_TOKENS)])
    return toks

async def _run_session(adapter, chat_id, sid, loop, scfg, delay=0.03):
    c = GatewayStreamConsumer(adapter=adapter, chat_id=chat_id,
        streaming_cfg=scfg, metadata={}, loop=loop)
    toks = _token_stream(sid)
    t0 = time.monotonic()
    task = asyncio.create_task(c.run_with_timeout(timeout=60))
    t1 = None
    for tok in toks:
        c.on_delta(tok)
        if t1 is None:
            t1 = time.monotonic()
        await asyncio.sleep(delay)
    c.finish()
    await task
    t2 = time.monotonic()
    return {"sid": sid, "sent": c.already_sent, "total": t2 - t0,
            "tokens": len(toks), "chars": sum(len(t) for t in toks),
            "delivery": t2 - (t1 or t0)}

async def run_bench(token, chat_id, counts):
    cfg = PlatformConfig()
    cfg.token = token
    cfg.enabled = True
    a = TelegramAdapter(cfg)
    if not await a.connect():
        print("FATAL: connect failed")
        return
    loop = asyncio.get_event_loop()
    scfg = {"enabled": True, "transport": "auto", "buffer_threshold": 20,
            "edit_interval": 0.15, "cursor": " \u2589"}
    results = {}
    for n in counts:
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  BENCHMARK: {n} concurrent session(s)")
        print(f"{sep}")
        if n > 1:
            print("  Waiting 5s for rate limits...")
            await asyncio.sleep(5)
        tw0 = time.monotonic()
        tasks = [_run_session(a, chat_id, i + 1, loop, scfg) for i in range(n)]
        res = await asyncio.gather(*tasks, return_exceptions=True)
        tw = time.monotonic() - tw0
        ok = [r for r in res if not isinstance(r, Exception) and r["sent"]]
        fail = []
        for r in res:
            if isinstance(r, Exception):
                fail.append(str(r))
            elif not r["sent"]:
                fail.append(f"session {r[sid]}: not sent")
        d = {"wall": tw, "ok": len(ok), "fail": len(fail)}
        if ok:
            tt = [s["total"] for s in ok]
            dd = [s["delivery"] for s in ok]
            d["avg"] = statistics.mean(tt)
            d["max"] = max(tt)
            d["min"] = min(tt)
            d["avg_d"] = statistics.mean(dd)
            d["p95_d"] = sorted(dd)[int(len(dd) * 0.95)] if len(dd) > 1 else dd[0]
        results[n] = d
        print(f"  Wall time:     {tw:.2f}s")
        print(f"  Success/Total: {len(ok)}/{n}")
        if ok:
            _v = d["avg"]
            print(f"  Avg total:     {_v:.2f}s")
            _v = d["min"]
            print(f"  Min total:     {_v:.2f}s")
            _v = d["max"]
            print(f"  Max total:     {_v:.2f}s")
            _v = d["avg_d"]
            print(f"  Avg delivery:  {_v:.2f}s")
            _v = d["p95_d"]
            print(f"  P95 delivery:  {_v:.2f}s")
        for fmsg in fail[:3]:
            print(f"  FAIL: {fmsg}")
    sep2 = "=" * 60; print(f"\n{sep2}\n  SUMMARY\n{sep2}")
    for n, d in results.items():
        s = "PASS" if d["fail"] == 0 else "DEGRADED"
        _a = d.get("avg", 0)
        x = f" avg={_a:.2f}s" if d["ok"] else ""
        _w = d["wall"]; _o = d["ok"]
        print(f"  {n:>2} sessions: wall={_w:.2f}s ok={_o}/{n} [{s}]{x}")
    await a.disconnect()
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--chat-id", required=True)
    p.add_argument("--sessions", default="1,5,10")
    a = p.parse_args()
    tok = os.getenv("TELEGRAM_BOT_TOKEN")
    if not tok:
        import yaml
        cp = os.path.expanduser("~/.hermes/config.yaml")
        if os.path.exists(cp):
            with open(cp) as f:
                cd = yaml.safe_load(f) or {}
            tok = cd.get("platforms", {}).get("telegram", {}).get("token", "")
        if not tok:
            ep = os.path.expanduser("~/.hermes/.env")
            if os.path.exists(ep):
                for ln in open(ep):
                    if ln.startswith("TELEGRAM_BOT_TOKEN="):
                        tok = ln.strip().split("=", 1)[1].strip("\"'")
    if not tok:
        print("FATAL: No token")
        sys.exit(1)
    counts = [int(x) for x in a.sessions.split(",")]
    asyncio.run(run_bench(tok, a.chat_id, counts))

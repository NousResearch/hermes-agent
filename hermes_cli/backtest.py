"""
Gateway-side wrapper for the trader-bot backtest. Invoked by the /backtest slash command.

Usage (from gateway/run.py dispatch arm):
    return await asyncio.to_thread(format_runbacktest, raw_args)

`raw_args` is a single string the user typed after `/backtest`, e.g.:
    YPFD.BA --period 6mo --hold 5 --timeframe 1d

Output goes back to the chat as the command reply. Telegram has a 4096-char limit per
message — if the report is longer we split it into chunks.

ponytail: synchronous, runs under asyncio.to_thread() from the handler. Subprocess
timeout is 90s; longer backtests (e.g. 10y of 1m bars) need to be chunked, add a
separate --chunk flag if Fede ever needs that.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

TRADER_BOT_DIR = Path("/root/projects/trader-bot")
BACKTEST_SCRIPT = TRADER_BOT_DIR / "backtest.py"
DEFAULT_TIMEOUT_S = 90
MAX_MSG_LEN = 3800  # safe under Telegram's 4096 with markdown overhead


def _validate_symbol(s: str) -> str:
    # yfinance tickers: letters, digits, dot, dash, slash, equals, caret. Be lenient.
    if not re.match(r"^[A-Za-z0-9\.\-\/\=\^]{1,16}$", s):
        raise ValueError(f"symbolo inválido: {s!r}")
    return s.upper()


def _parse_args(raw: str) -> list[str]:
    """Split the raw user input into argv tokens, validating the symbol token."""
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("uso: /backtest SYMBOL [--period 1y] [--timeframe 1d] [--hold 5]")
    parts = raw.split()
    _validate_symbol(parts[0])
    return parts


def _run_subprocess(args: list[str]) -> str:
    result = subprocess.run(
        [sys.executable, str(BACKTEST_SCRIPT), *args],
        cwd=str(TRADER_BOT_DIR),
        capture_output=True,
        text=True,
        timeout=DEFAULT_TIMEOUT_S,
        check=False,
    )
    if result.returncode != 0:
        err = (result.stderr or "").strip() or "error desconocido"
        raise RuntimeError(f"backtest.py falló (exit {result.returncode}): {err[:400]}")
    return (result.stdout or "").strip()


def _split_for_chat(text: str, max_len: int = MAX_MSG_LEN) -> list[str]:
    """Split text into Telegram-friendly chunks, breaking at line boundaries when possible."""
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        cut = text.rfind("\n", 0, max_len)
        if cut == -1 or cut < max_len * 0.5:
            cut = max_len
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return chunks


def format_runbacktest(raw_args: str) -> str:
    """Run the backtest and return the report, chunked for chat."""
    args = _parse_args(raw_args)
    out = _run_subprocess(args)
    if not out:
        return "backtest terminó sin output (¿sin datos?)."
    chunks = _split_for_chat(out)
    if len(chunks) == 1:
        return chunks[0]
    header = f"📊 Backtest ({len(chunks)} partes):\n\n"
    return header + chunks[0] + f"\n\n— parte 1/{len(chunks)} —"


if __name__ == "__main__":
    # Quick self-test from CLI: python3 -m hermes_cli.backtest YPFD.BA --period 1y
    import sys as _sys
    print(format_runbacktest(" ".join(_sys.argv[1:]) or "YPFD.BA --period 1y"))
#!/usr/bin/env python3
"""Fake a busy coding session in the terminal.

Scenes cycle in random order, grouped into profiles: `developer` fakes honest
work (an editor typing source, a build, a test run, git activity) and `hacker`
fakes the movie kind (digital rain, an intrusion, a key crack). Every byte the
scenes print is invented — no file is read or written, no command runs,
nothing touches the network. The "targets" are RFC 5737 documentation
addresses and example.* hosts, so the theatre cannot point at anything real.
This is a joke screensaver in the `cmatrix` tradition, not a tool.

    python3 busy_terminal.py                      # hacker profile, until Ctrl-C
    python3 busy_terminal.py --profile developer  # fake coding session
    python3 busy_terminal.py --duration 120       # two minutes, then exit
    python3 busy_terminal.py --scene warroom      # the multi-window poster
    python3 busy_terminal.py --window             # open a new terminal, return now

`--window` is the one thing here that starts a process: it re-launches this
script inside a fresh terminal window and exits. An agent needs it, because a
captured pipe has no TTY to animate and an unbounded run would never return.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence, TextIO

SCENES = ("code", "build", "tests", "git", "matrix", "warroom", "intrusion", "crack")

PROFILES: dict[str, tuple[str, ...]] = {
    "developer": ("code", "build", "tests", "git"),
    "hacker": ("matrix", "warroom", "intrusion", "crack"),
    "mixed": SCENES,
}

RESET = "\033[0m"
BOLD = "\033[1m"
GREY = "\033[38;5;244m"
RED = "\033[38;5;203m"
GREEN = "\033[38;5;114m"
YELLOW = "\033[38;5;180m"
BLUE = "\033[38;5;75m"
MAGENTA = "\033[38;5;176m"
CYAN = "\033[38;5;80m"
ORANGE = "\033[38;5;215m"

CLEAR_SCREEN = "\033[2J\033[H"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"


# ── Console ──────────────────────────────────────────────────────────────────


class Console:
    """The only surface a scene is allowed to touch.

    Output and timing are injected rather than hardcoded to stdout and
    ``time.sleep`` so a test can drive an entire scene against a fake clock and
    capture the frames instead of sleeping through them.
    """

    def __init__(
        self,
        *,
        width: int = 100,
        height: int = 30,
        color: bool = True,
        speed: float = 1.0,
        write: Callable[[str], None] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.width = max(40, width)
        self.height = max(10, height)
        self.color = color
        self.speed = speed if speed > 0 else 1.0
        self._write = write if write is not None else _stdout_writer
        self._sleep = sleep if sleep is not None else time.sleep

    def paint(self, text: str = "") -> None:
        """Emit text with no trailing newline."""
        self._write(text)

    def line(self, text: str = "") -> None:
        self._write(text + "\n")

    def pause(self, seconds: float) -> None:
        """Sleep, scaled by --speed. Never negative, never a busy-wait."""
        self._sleep(max(0.0, seconds) / self.speed)

    def tint(self, text: str, code: str) -> str:
        return f"{code}{text}{RESET}" if self.color else text

    def clear(self) -> None:
        if self.color:
            self._write(CLEAR_SCREEN)


def _stdout_writer(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


# ── Pure formatters ──────────────────────────────────────────────────────────


def progress_bar(done: float, total: float, width: int = 28) -> str:
    """A fixed-width bar. Out-of-range input clamps instead of overflowing."""
    width = max(1, width)
    fraction = 0.0 if total <= 0 else done / total
    fraction = min(1.0, max(0.0, fraction))
    filled = round(fraction * width)

    return "█" * filled + "░" * (width - filled)


def human_bytes(count: float) -> str:
    """Byte count as a short human string (1536 -> '1.5 KiB')."""
    step = 1024.0
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(count) < step or unit == "GiB":
            return f"{count:.0f} {unit}" if unit == "B" else f"{count:.1f} {unit}"
        count /= step

    return f"{count:.1f} GiB"


def next_scene(rng: random.Random, last: str = "", scenes: Sequence[str] = SCENES) -> str:
    """Pick the next scene, never the one that just played.

    Back-to-back repeats are what make a shuffle look broken, so they are
    excluded rather than left to chance. A one-scene catalog still returns
    that scene — the rule yields rather than looping forever.
    """
    options = [scene for scene in scenes if scene != last] or list(scenes)

    return rng.choice(options)


def test_summary(passed: int, failed: int, skipped: int, seconds: float) -> str:
    """The pytest-style tail line. Failures lead when there are any."""
    parts = []
    if failed:
        parts.append(f"{failed} failed")
    parts.append(f"{passed} passed")
    if skipped:
        parts.append(f"{skipped} skipped")

    return f"{', '.join(parts)} in {seconds:.2f}s"


# ── Syntax highlighting ──────────────────────────────────────────────────────

KEYWORDS = {
    "python": {
        "async", "await", "class", "def", "elif", "else", "except", "finally",
        "for", "from", "if", "import", "in", "is", "not", "raise", "return",
        "try", "while", "with", "yield", "None", "True", "False",
    },
    "ts": {
        "async", "await", "const", "export", "function", "if", "import",
        "interface", "let", "new", "return", "type", "useEffect", "useState",
        "from", "null", "true", "false",
    },
    "go": {
        "defer", "err", "for", "func", "if", "import", "nil", "package",
        "range", "return", "struct", "type", "var",
    },
    "rust": {
        "as", "enum", "fn", "impl", "let", "match", "mod", "mut", "pub",
        "return", "self", "struct", "use", "while", "Some", "None", "Ok", "Err",
    },
}

_TOKEN = re.compile(
    r"(?P<comment>#.*$|//.*$)"
    r"|(?P<string>\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*')"
    r"|(?P<number>\b\d+(?:\.\d+)?\b)"
    r"|(?P<word>[A-Za-z_][A-Za-z_0-9]*)"
)


def highlight(line: str, language: str, color: bool = True) -> str:
    """Tint keywords, strings, comments, and numbers. A no-op without color."""
    if not color:
        return line

    keywords = KEYWORDS.get(language, set())

    def paint(match: re.Match[str]) -> str:
        text = match.group(0)
        kind = match.lastgroup
        if kind == "comment":
            return f"{GREY}{text}{RESET}"
        if kind == "string":
            return f"{GREEN}{text}{RESET}"
        if kind == "number":
            return f"{ORANGE}{text}{RESET}"
        if kind == "word" and text in keywords:
            return f"{MAGENTA}{text}{RESET}"

        return text

    return _TOKEN.sub(paint, line)


# ── Content ──────────────────────────────────────────────────────────────────

REPO = "~/work/atlas"

BRANCHES = ("feat/ingest-backoff", "fix/session-leak", "feat/live-query", "chore/deps")

SOURCES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "services/ingest/retry.py",
        "python",
        (
            "import asyncio",
            "import random",
            "from typing import Awaitable, Callable, TypeVar",
            "",
            'T = TypeVar("T")',
            "",
            "",
            "async def with_backoff(",
            "    call: Callable[[], Awaitable[T]],",
            "    attempts: int = 5,",
            "    base: float = 0.25,",
            ") -> T:",
            '    """Retry an awaitable with jittered exponential backoff."""',
            "    last = None",
            "    for attempt in range(attempts):",
            "        try:",
            "            return await call()",
            "        except TransientError as exc:",
            "            last = exc",
            "            delay = base * 2 ** attempt",
            "            # Jitter keeps a thundering herd from resynchronising.",
            "            await asyncio.sleep(delay + random.random() * base)",
            "    raise RetryExhausted(attempts) from last",
        ),
    ),
    (
        "web/src/hooks/use-live-query.ts",
        "ts",
        (
            "import { useEffect, useState } from 'react'",
            "",
            "interface LiveQuery<T> {",
            "  data: T | null",
            "  error: Error | null",
            "  stale: boolean",
            "}",
            "",
            "export function useLiveQuery<T>(key: string): LiveQuery<T> {",
            "  const [data, setData] = useState<T | null>(null)",
            "  const [error, setError] = useState<Error | null>(null)",
            "",
            "  useEffect(() => {",
            "    let cancelled = false",
            "    subscribe(key, next => {",
            "      // A late frame must never overwrite newer intent.",
            "      if (!cancelled) setData(next)",
            "    }).catch(setError)",
            "",
            "    return () => {",
            "      cancelled = true",
            "    }",
            "  }, [key])",
            "",
            "  return { data, error, stale: data === null }",
            "}",
        ),
    ),
    (
        "internal/api/handler.go",
        "go",
        (
            "package api",
            "",
            "import (",
            '    "encoding/json"',
            '    "net/http"',
            '    "time"',
            ")",
            "",
            "func (s *Server) handleSnapshot(w http.ResponseWriter, r *http.Request) {",
            "    ctx, cancel := context.WithTimeout(r.Context(), 3*time.Second)",
            "    defer cancel()",
            "",
            "    snap, err := s.store.Snapshot(ctx, chi.URLParam(r, \"id\"))",
            "    if err != nil {",
            "        s.fail(w, http.StatusBadGateway, err)",
            "        return",
            "    }",
            "",
            "    w.Header().Set(\"Cache-Control\", \"no-store\")",
            "    json.NewEncoder(w).Encode(snap)",
            "}",
        ),
    ),
    (
        "crates/parser/src/lexer.rs",
        "rust",
        (
            "use std::str::Chars;",
            "",
            "pub struct Lexer<'a> {",
            "    chars: Chars<'a>,",
            "    offset: usize,",
            "}",
            "",
            "impl<'a> Lexer<'a> {",
            "    pub fn next_token(&mut self) -> Option<Token> {",
            "        let start = self.offset;",
            "        match self.bump()? {",
            "            c if c.is_whitespace() => self.next_token(),",
            "            // A bare '-' is a minus; '->' is an arrow.",
            "            '-' if self.peek() == Some('>') => Some(Token::Arrow),",
            "            c if c.is_ascii_digit() => Some(self.number(start)),",
            "            _ => Some(Token::Unknown(start)),",
            "        }",
            "    }",
            "}",
        ),
    ),
)

BUILDS = (
    (
        "npm run build",
        "vite v5.4.11 building for production...",
        (
            "web/assets/index-4f21ac.js",
            "web/assets/vendor-9cb0e1.js",
            "web/assets/index-0b7714.css",
        ),
    ),
    (
        "cargo build --release",
        "   Compiling atlas-parser v0.9.3",
        (
            "target/release/atlas",
            "target/release/atlas-parser.rlib",
        ),
    ),
    (
        "docker build -t atlas/ingest:dev .",
        "=> [internal] load build definition from Dockerfile",
        (
            "layer sha256:9c1b4f2a",
            "layer sha256:2fd0aa71",
        ),
    ),
)

TEST_FILES = (
    "tests/ingest/test_backoff.py",
    "tests/api/test_snapshot.py",
    "tests/store/test_lineage.py",
    "tests/web/use-live-query.test.ts",
    "tests/parser/test_lexer.py",
)

COMMIT_SUBJECTS = (
    "fix(ingest): jitter the backoff so retries stop resynchronising",
    "feat(api): cache-bust snapshot responses behind a short timeout",
    "refactor(parser): fold the arrow case into next_token",
    "fix(web): drop a late frame instead of overwriting newer state",
    "test(store): pin the lineage invariant across compression",
)

CI_CHECKS = (
    "lint / ruff",
    "tests / python 3.12",
    "tests / node 22",
    "build / linux-amd64",
    "supply-chain / audit",
)

# The hacker profile's props. Hosts are RFC 2606 example domains and the
# addresses RFC 5737 reserves for documentation — unroutable by construction.
MATRIX_GLYPHS = "ﾊﾐﾋｰｳｼﾅﾓﾆｻﾜﾂｵﾘｱﾎﾃﾏｹﾒｴｶｷﾑﾕﾗｾﾈｽﾀﾇ0123456789Z:･.=*+-<>"
MATRIX_HEAD = "\033[1m\033[38;5;157m"
MATRIX_BODY = "\033[38;5;41m"

TARGETS = (
    ("vault.example.net", "203.0.113.42"),
    ("mainframe.example.org", "198.51.100.17"),
    ("archive.example.com", "203.0.113.208"),
)

PORTS = (
    (22, "ssh"),
    (25, "smtp"),
    (80, "http"),
    (443, "https"),
    (3306, "mysql"),
    (5432, "postgres"),
    (6379, "redis"),
    (8443, "https-alt"),
)

USERS = ("admin", "root", "operator", "dispatch", "jmoriarty", "svc_legacy")

LOOT = (
    "/srv/vault/blueprints.tar.gz",
    "/srv/vault/ledger-2026.db",
    "/opt/mainframe/schematics.zip",
)

PASSWORDS = ("Tr1n1ty-1999", "sw0rdf1sh", "Z10N-4cc3ss", "m0rph3us#7")


# ── Scenes ───────────────────────────────────────────────────────────────────


def type_out(
    console: Console,
    text: str,
    *,
    prefix: str = "",
    language: str = "",
    rng: random.Random,
) -> None:
    """Type a line a character at a time, then repaint it highlighted.

    Highlighting a partial line would recolor tokens as they grow, which reads
    as flicker; typing plain and repainting once at the end looks like an
    editor catching up. The repaint returns to column 0, so it has to redraw
    `prefix` (the line-number gutter) or the code slides left over it.
    """
    if prefix:
        console.paint(prefix)

    for char in text:
        console.paint(char)
        console.pause(rng.uniform(0.004, 0.028) if char != " " else 0.006)

    if language and console.color:
        console.paint("\r" + prefix + highlight(text, language, console.color))
    console.line()


def prompt(console: Console, command: str, rng: random.Random) -> None:
    """The shell prompt plus a typed-out command."""
    console.paint(console.tint(REPO, BLUE) + console.tint(" ❯ ", GREEN))
    for char in command:
        console.paint(char)
        console.pause(rng.uniform(0.012, 0.05))
    console.line()
    console.pause(0.35)


def scene_code(console: Console, rng: random.Random) -> None:
    """An editor pane filling with source, line numbers and all."""
    path, language, lines = rng.choice(SOURCES)

    console.clear()
    console.line(console.tint(f"  {path}", BOLD) + console.tint("   ● unsaved", ORANGE))
    console.line(console.tint("  " + "─" * (console.width - 4), GREY))

    for number, text in enumerate(lines, start=1):
        gutter = console.tint(f"{number:>4} │ ", GREY)
        type_out(console, text, prefix=gutter, language=language, rng=rng)

        # Every so often, stop and stare at it like a person would.
        if rng.random() < 0.12:
            console.pause(rng.uniform(0.4, 1.1))

    console.line()
    console.pause(0.5)
    console.line(console.tint("  saved", GREEN) + console.tint(f"  {path}", GREY))
    console.pause(1.2)


def scene_build(console: Console, rng: random.Random) -> None:
    """A build with staged progress and an artifact table."""
    command, banner, artifacts = rng.choice(BUILDS)

    console.line()
    prompt(console, command, rng)
    console.line(console.tint(banner, GREY))

    total = rng.randint(180, 940)
    step = max(1, total // rng.randint(12, 20))
    done = 0
    while done < total:
        done = min(total, done + step)
        bar = progress_bar(done, total)
        console.paint(f"\r{console.tint(bar, CYAN)} {done}/{total} modules")
        console.pause(rng.uniform(0.05, 0.16))

    console.line()
    console.line()
    for artifact in artifacts:
        size = rng.uniform(12_000, 940_000)
        gzip = size / rng.uniform(2.8, 4.1)
        console.line(
            console.tint(f"  {artifact:<34}", GREY)
            + console.tint(f"{human_bytes(size):>10}", YELLOW)
            + console.tint(f"  │ gzip: {human_bytes(gzip):>9}", GREY)
        )
        console.pause(0.14)

    console.line()
    console.line(console.tint(f"  ✓ built in {rng.uniform(3.2, 24.0):.2f}s", GREEN))
    console.pause(1.4)


def scene_tests(console: Console, rng: random.Random) -> None:
    """A test run that mostly passes, occasionally retries a flake."""
    console.line()
    prompt(console, "pytest -q", rng)

    passed = 0
    failed = 0
    skipped = 0
    for path in rng.sample(TEST_FILES, k=rng.randint(3, len(TEST_FILES))):
        console.paint(console.tint(f"  {path:<38}", GREY))
        for _ in range(rng.randint(6, 26)):
            roll = rng.random()
            if roll < 0.02:
                failed += 1
                console.paint(console.tint("F", RED))
            elif roll < 0.05:
                skipped += 1
                console.paint(console.tint("s", YELLOW))
            else:
                passed += 1
                console.paint(console.tint(".", GREEN))
            console.pause(rng.uniform(0.01, 0.07))
        console.line()

    seconds = rng.uniform(1.8, 19.4)
    console.line()
    summary = test_summary(passed, failed, skipped, seconds)
    console.line(console.tint(f"  {summary}", RED if failed else GREEN))

    if failed:
        console.pause(0.8)
        console.line(console.tint("  rerunning the failure in isolation…", GREY))
        console.pause(rng.uniform(1.0, 2.0))
        console.line(console.tint("  ✓ passed on retry — flake, not a break", GREEN))

    console.pause(1.4)


def scene_git(console: Console, rng: random.Random) -> None:
    """Commit, push, then CI checks going green one at a time."""
    branch = rng.choice(BRANCHES)
    subject = rng.choice(COMMIT_SUBJECTS)
    sha = "".join(rng.choice("0123456789abcdef") for _ in range(7))

    console.line()
    prompt(console, f'git commit -am "{subject}"', rng)
    files = rng.randint(2, 9)
    console.line(console.tint(f"[{branch} {sha}]", GREY) + f" {subject}")
    console.line(
        console.tint(
            f" {files} files changed, "
            f"{rng.randint(18, 240)} insertions(+), {rng.randint(3, 90)} deletions(-)",
            GREY,
        )
    )
    console.pause(0.9)

    prompt(console, f"git push origin {branch}", rng)
    objects = rng.randint(14, 61)
    for label, count in (("Counting objects", objects), ("Compressing objects", objects // 2)):
        for index in range(1, count + 1):
            percent = round(index / count * 100)
            console.paint(f"\r{console.tint(label, GREY)}: {percent:>3}% ({index}/{count})")
            console.pause(0.012)
        console.line(", done.")

    written = rng.uniform(2_000, 96_000)
    console.line(
        console.tint("Writing objects", GREY)
        + f": 100% ({objects}/{objects}), {human_bytes(written)}, done."
    )
    console.pause(0.6)
    console.line(console.tint(f"remote: Resolving deltas: 100% ({objects}/{objects}), done.", GREY))
    console.line("To github.com:atlas/atlas.git")
    console.line(console.tint(f"   {sha}..{sha[::-1]}  {branch} -> {branch}", GREY))
    console.line()

    console.pause(0.8)
    console.line(console.tint("  CI", BOLD))
    for check in rng.sample(CI_CHECKS, k=rng.randint(3, len(CI_CHECKS))):
        console.paint(console.tint(f"    ● {check}", YELLOW))
        console.pause(rng.uniform(0.5, 1.6))
        console.paint("\r" + console.tint(f"    ✓ {check}", GREEN) + "\n")

    console.pause(1.4)


# ── Hacker profile scenes ────────────────────────────────────────────────────


def move_to(row: int, col: int) -> str:
    """ANSI cursor move, 1-based."""
    return f"\033[{row};{col}H"


def fit(text: str, width: int) -> str:
    """Clip or pad to exactly `width` — pane content must never spill out."""
    if width <= 0:
        return ""

    return text[:width].ljust(width)


@dataclass(frozen=True)
class Rect:
    """A window rectangle in 1-based terminal cells, borders included."""

    top: int
    left: int
    width: int
    height: int

    @property
    def bottom(self) -> int:
        return self.top + self.height - 1

    @property
    def right(self) -> int:
        return self.left + self.width - 1

    def contains(self, row: int, col: int) -> bool:
        return self.top <= row <= self.bottom and self.left <= col <= self.right

    def overlaps(self, other: "Rect") -> bool:
        return not (
            self.right < other.left
            or other.right < self.left
            or self.bottom < other.top
            or other.bottom < self.top
        )


@dataclass
class Drop:
    """One falling column of digital rain."""

    col: int
    row: float
    speed: float
    trail: int


def spawn_drop(col: int, height: int, rng: random.Random) -> Drop:
    """A fresh drop, staggered above the screen so columns trickle in.

    Speed stays at or below one cell per tick — faster drops skip rows and
    leave holes in their own trail.
    """
    return Drop(
        col=col,
        row=-rng.uniform(0.0, height),
        speed=rng.uniform(0.35, 1.0),
        trail=rng.randint(4, max(5, height // 2)),
    )


def rain_step(
    drops: list[Drop],
    height: int,
    rng: random.Random,
    avoid: Sequence[Rect] = (),
) -> str:
    """Advance every drop one tick and return the whole frame as one string.

    One write per frame is the point: per-cell writes flicker and swamp the
    terminal with flushes. Cells inside `avoid` are never touched, so the rain
    flows around the war-room windows instead of scribbling through them.
    """

    def open_cell(row: int, col: int) -> bool:
        return 1 <= row <= height and not any(rect.contains(row, col) for rect in avoid)

    parts = [MATRIX_BODY]
    for index, drop in enumerate(drops):
        prev = int(drop.row)
        drop.row += drop.speed
        head = int(drop.row)

        if head != prev:
            if open_cell(head, drop.col):
                parts.append(
                    move_to(head, drop.col)
                    + MATRIX_HEAD + rng.choice(MATRIX_GLYPHS) + RESET + MATRIX_BODY
                )
            # The old head dims into the trail body.
            if open_cell(prev, drop.col):
                parts.append(move_to(prev, drop.col) + rng.choice(MATRIX_GLYPHS))
            tail = head - drop.trail
            if open_cell(tail, drop.col):
                parts.append(move_to(tail, drop.col) + " ")

        if drop.row - drop.trail > height:
            drops[index] = spawn_drop(drop.col, height, rng)

    parts.append(RESET)

    return "".join(parts)


def scene_matrix(console: Console, rng: random.Random) -> None:
    """Digital rain, then a word from the machine."""
    if not console.color:
        # No ANSI means no cursor addressing — degrade to scrolling glyphs.
        for _ in range(rng.randint(24, 40)):
            line = "".join(
                rng.choice(MATRIX_GLYPHS) if rng.random() < 0.28 else " "
                for _ in range(console.width - 2)
            )
            console.line(line)
            console.pause(0.05)

        return

    console.clear()
    drops = [spawn_drop(col, console.height, rng) for col in range(1, console.width, 2)]
    for _ in range(rng.randint(150, 230)):
        console.paint(rain_step(drops, console.height, rng))
        console.pause(0.045)

    console.clear()
    console.pause(0.7)
    console.paint(MATRIX_BODY + BOLD)
    for message in ("wake up…", "your build finished hours ago"):
        console.paint("  ")
        for char in message:
            console.paint(char)
            console.pause(rng.uniform(0.05, 0.14))
        console.line()
        console.pause(0.9)
    console.paint(RESET)
    console.pause(1.2)


# ── The war room: several live windows over the rain ────────────────────────


def warroom_layout(width: int, height: int) -> dict[str, Rect]:
    """Window rectangles scaled to the terminal.

    Panes never overlap each other — none of them re-stamps on the others'
    cadence, so overlap would scribble. The dialog is the one exception: it
    floats over everything because it is re-stamped into every frame, last.
    Panes that would come out too small to read are dropped rather than
    squeezed.
    """
    rects: dict[str, Rect] = {}
    left_w = (width - 6) // 2
    right_w = width - 6 - left_w
    top_h = (height - 5) // 2
    bottom_h = height - 5 - top_h

    candidates = {
        "memdump": Rect(top=2, left=2, width=left_w, height=top_h),
        "uplink": Rect(top=2, left=left_w + 4, width=right_w, height=height - 3),
        "intercept": Rect(top=top_h + 3, left=2, width=left_w, height=bottom_h),
    }
    for name, rect in candidates.items():
        if rect.width >= 14 and rect.height >= 4:
            rects[name] = rect

    dialog_width = min(36, width - 4)
    rects["dialog"] = Rect(
        top=max(1, height // 2 - 2),
        left=max(1, (width - dialog_width) // 2),
        width=dialog_width,
        height=5,
    )

    # The three corner dialogs only exist on generous terminals, in bands the
    # password dialog never reaches, so the four floaters stay readable.
    if width >= 70 and height >= 20:
        rects["alert"] = Rect(top=3, left=width - 28, width=24, height=4)
        rects["proxy"] = Rect(top=height - 5, left=4, width=26, height=4)
        rects["exfil"] = Rect(top=height - 5, left=width - 32, width=26, height=4)

    return rects


@dataclass
class Pane:
    """One live window: a box, a text tone, and a feed that fills it.

    Lines carry their own tone so an accent rolled at append time survives
    every repaint — re-rolling on repaint would make old rows flicker.
    """

    rect: Rect
    title: str
    tone: str
    feed: Callable[[random.Random, int], str]
    period: int
    reveal: int
    lines: list[tuple[str, str]]
    accent: str = ""
    accent_chance: float = 0.0


def roll_tone(rng: random.Random, pane: Pane) -> str:
    """The tone for a freshly appended line — usually the pane's, sometimes
    its accent."""
    if pane.accent and rng.random() < pane.accent_chance:
        return pane.accent

    return pane.tone


def feed_hex(rng: random.Random, width: int) -> str:
    """A memdump row: offset, then as many hex pairs as fit."""
    pairs = max(1, (width - 7) // 3)
    body = " ".join(f"{rng.randrange(256):02x}" for _ in range(pairs))

    return f"{rng.randrange(0x10000):04x}: {body}"


def feed_intercept(rng: random.Random, width: int) -> str:
    return (
        f"pkt {rng.randrange(10000):04d} ▸ 203.0.113.{rng.randrange(1, 255)}:443"
        f" · TLS1.3 · {rng.uniform(0.2, 9.9):.1f} KiB"
    )


def feed_trace(rng: random.Random, width: int) -> str:
    return (
        f"hop {rng.randrange(2, 15):02d}  {rng.uniform(4.0, 240.0):6.1f} ms"
        f"  relay-{rng.randrange(10):02d}.example.net"
    )


def box_stamp(console: Console, rect: Rect, title: str, tone: str) -> str:
    """Paint a window frame with a blank interior, in one string."""
    label = title[: max(0, rect.width - 6)]
    top = "┌─ " + label + " " + "─" * (rect.width - len(label) - 5) + "┐"
    parts = [move_to(rect.top, rect.left) + console.tint(top, tone)]
    for row in range(rect.top + 1, rect.bottom):
        parts.append(
            move_to(row, rect.left)
            + console.tint("│" + " " * (rect.width - 2) + "│", tone)
        )
    parts.append(
        move_to(rect.bottom, rect.left)
        + console.tint("└" + "─" * (rect.width - 2) + "┘", tone)
    )

    return "".join(parts)


def interior_stamp(console: Console, pane: Pane) -> str:
    """Repaint a pane's content area, bottom-aligned like a scrolling log."""
    inner_rows = pane.rect.height - 2
    inner_width = pane.rect.width - 4
    visible = pane.lines[-inner_rows:]
    padded = [("", pane.tone)] * (inner_rows - len(visible)) + visible

    parts = []
    for offset, (line, tone) in enumerate(padded):
        parts.append(
            move_to(pane.rect.top + 1 + offset, pane.rect.left + 2)
            + console.tint(fit(line, inner_width), tone)
        )

    return "".join(parts)


def dialog_stamp(
    console: Console,
    rect: Rect,
    password: str,
    locked: int,
    rng: random.Random,
) -> str:
    """The floating centerpiece: masked characters locking in one by one."""
    granted = locked >= len(password)
    tone = GREEN if granted else CYAN
    title = "ACCESS GRANTED" if granted else "MATCHING PASSWORD"

    cells = []
    for index, char in enumerate(password):
        if index < locked:
            cells.append(console.tint(char, BOLD + GREEN))
        elif rng.random() < 0.14:
            cells.append(console.tint(rng.choice("0123456789ABCDEF"), CYAN))
        else:
            cells.append(console.tint("▪", GREY))

    inner_width = rect.width - 4
    row = " ".join(cells)
    pad = max(0, (inner_width - (len(password) * 2 - 1)) // 2)
    status = f"{locked}/{len(password)} matched" if not granted else "session key accepted"

    return (
        box_stamp(console, rect, title, BOLD + tone if granted else tone)
        + move_to(rect.top + 2, rect.left + 2) + " " * pad + row
        + move_to(rect.top + 3, rect.left + 2)
        + console.tint(fit(status.center(inner_width), inner_width), GREY)
    )


def alert_stamp(console: Console, rect: Rect, age: int) -> str:
    """The red corner alarm: a countdown that reads like consequences."""
    urgent = (age // 6) % 2 == 0
    tone = BOLD + RED if urgent else RED
    seconds = max(0, 45 - age // 18)
    inner = rect.width - 4

    return (
        box_stamp(console, rect, "PERIMETER", tone)
        + move_to(rect.top + 1, rect.left + 2)
        + console.tint(fit("⚠ trace detected", inner), RED)
        + move_to(rect.top + 2, rect.left + 2)
        + console.tint(fit(f"lockout in 00:{seconds:02d}", inner), tone)
    )


def proxy_stamp(console: Console, rect: Rect, age: int) -> str:
    """The green corner status: relay hops securing one by one."""
    hops = min(5, 1 + age // 22)
    chain = " ▸ ".join(f"{relay:02d}" for relay in (3, 7, 12, 19, 22)[:hops])
    secured = hops == 5
    status = "chain secured" if secured else f"{hops}/5 hops secured"
    inner = rect.width - 4

    return (
        box_stamp(console, rect, "PROXY CHAIN", GREEN)
        + move_to(rect.top + 1, rect.left + 2)
        + console.tint(fit(f"relay {chain}", inner), GREEN)
        + move_to(rect.top + 2, rect.left + 2)
        + console.tint(fit(status, inner), BOLD + GREEN if secured else GREY)
    )


def exfil_stamp(console: Console, rect: Rect, age: int, total: float) -> str:
    """The amber corner meter: bytes leaving the building."""
    fraction = min(1.0, age / 90.0)
    inner = rect.width - 4
    bar = progress_bar(fraction, 1.0, max(6, inner - 6))
    counter = f"{human_bytes(total * fraction)} / {human_bytes(total)}"

    return (
        box_stamp(console, rect, "EXFIL", ORANGE)
        + move_to(rect.top + 1, rect.left + 2)
        + console.tint(bar, ORANGE) + console.tint(f" {int(fraction * 100):>3}%", GREY)
        + move_to(rect.top + 2, rect.left + 2)
        + console.tint(fit(counter, inner), GREY)
    )


def scene_warroom(console: Console, rng: random.Random) -> None:
    """The full movie set: rain behind several live panes, dialog on top."""
    if not console.color:
        # No cursor addressing — interleave the feeds as a flat log instead.
        for _ in range(rng.randint(24, 40)):
            roll = rng.random()
            if roll < 0.4:
                console.line(feed_intercept(rng, console.width - 2))
            elif roll < 0.7:
                console.line(feed_hex(rng, console.width - 2))
            else:
                console.line(feed_trace(rng, console.width - 2))
            console.pause(0.06)
        console.line(f"[dialog] password matched ({rng.randint(6, 14)} candidates)")
        console.pause(1.0)

        return

    console.clear()
    layout = warroom_layout(console.width, console.height)
    dialog_rect = layout.pop("dialog")
    alert_rect = layout.pop("alert", None)
    proxy_rect = layout.pop("proxy", None)
    exfil_rect = layout.pop("exfil", None)

    dressing = {
        "memdump": (feed_hex, GREY, RED, 0.22),
        "uplink": (feed_intercept, GREEN, BOLD + GREEN, 0.12),
        "intercept": (feed_trace, CYAN, GREEN, 0.30),
    }
    panes = []
    for index, name in enumerate(sorted(layout)):
        feed, tone, accent, chance = dressing[name]
        panes.append(
            Pane(
                rect=layout[name],
                title=name,
                tone=tone,
                feed=feed,
                period=rng.randint(3, 6),
                reveal=6 + index * rng.randint(6, 10),
                lines=[],
                accent=accent,
                accent_chance=chance,
            )
        )

    drops = [spawn_drop(col, console.height, rng) for col in range(1, console.width, 2)]
    password = rng.choice(PASSWORDS)
    dialog_at = rng.randint(40, 60)
    lock_every = rng.randint(7, 11)
    locked = 0
    alert_at = rng.randint(18, 26)
    proxy_at = rng.randint(28, 38)
    exfil_at = rng.randint(46, 58)
    exfil_total = rng.uniform(2e7, 8e7)
    total = dialog_at + lock_every * len(password) + 24

    for tick in range(total):
        parts = []
        shown = [pane.rect for pane in panes if tick >= pane.reveal]
        for rect, since in ((alert_rect, alert_at), (proxy_rect, proxy_at), (exfil_rect, exfil_at)):
            if rect is not None and tick >= since:
                shown.append(rect)
        if tick >= dialog_at:
            shown.append(dialog_rect)
        parts.append(rain_step(drops, console.height, rng, avoid=shown))

        for pane in panes:
            if tick == pane.reveal:
                parts.append(box_stamp(console, pane.rect, pane.title, pane.tone))
            elif tick > pane.reveal and (tick - pane.reveal) % pane.period == 0:
                pane.lines.append((pane.feed(rng, pane.rect.width - 4), roll_tone(rng, pane)))
                pane.lines = pane.lines[-(pane.rect.height - 2):]
                parts.append(interior_stamp(console, pane))

        # Floaters re-stamp every tick; the password dialog goes last, on top.
        if alert_rect is not None and tick >= alert_at:
            parts.append(alert_stamp(console, alert_rect, tick - alert_at))
        if proxy_rect is not None and tick >= proxy_at:
            parts.append(proxy_stamp(console, proxy_rect, tick - proxy_at))
        if exfil_rect is not None and tick >= exfil_at:
            parts.append(exfil_stamp(console, exfil_rect, tick - exfil_at, exfil_total))
        if tick >= dialog_at:
            if locked < len(password) and tick > dialog_at and (tick - dialog_at) % lock_every == 0:
                locked += 1
            parts.append(dialog_stamp(console, dialog_rect, password, locked, rng))

        console.paint("".join(parts))
        console.pause(0.05)

    console.clear()
    console.pause(0.5)


def scene_intrusion(console: Console, rng: random.Random) -> None:
    """The movie hack: scan, brute-force, ACCESS GRANTED, exfiltrate."""
    host, addr = rng.choice(TARGETS)

    console.line()
    prompt(console, f"./ghost --target {addr} --stealth", rng)
    console.line(console.tint(f"[ghost] resolving target… {host} ({addr})", GREY))
    console.pause(0.6)

    console.paint(console.tint("[ghost] mapping 1024 ports ", GREY))
    for _ in range(rng.randint(18, 30)):
        console.paint(console.tint(".", CYAN))
        console.pause(rng.uniform(0.02, 0.09))
    console.line()

    for port, service in sorted(rng.sample(PORTS, k=rng.randint(3, 5))):
        console.line(
            f"  {port:>5}/tcp  " + console.tint("open", GREEN) + f"   {service}"
        )
        console.pause(0.18)
    console.line(console.tint("[ghost] stack: nginx/1.27 · openssh 9.8 · debian 13", GREY))
    console.pause(0.7)

    console.line(console.tint("[ghost] trying credentials", GREY))
    attempt = rng.randint(120, 400)
    for _ in range(rng.randint(4, 7)):
        attempt += rng.randint(7, 60)
        user = rng.choice(USERS)
        console.paint(
            f"\r  attempt {attempt:04d}  {user:<12} {'•' * 10}  "
            + console.tint("DENIED  ", RED)
        )
        console.pause(rng.uniform(0.2, 0.5))
    attempt += rng.randint(7, 60)
    console.paint(
        f"\r  attempt {attempt:04d}  {'svc_backup':<12} {'•' * 10}  "
        + console.tint("ACCEPTED", GREEN) + "\n"
    )
    console.pause(0.5)

    inner = "   ACCESS GRANTED   "
    console.line()
    console.line(console.tint("  ╔" + "═" * len(inner) + "╗", GREEN))
    console.line(console.tint("  ║" + inner + "║", BOLD + GREEN))
    console.line(console.tint("  ╚" + "═" * len(inner) + "╝", GREEN))
    console.line()
    console.pause(0.9)

    total = rng.uniform(6e6, 9e7)
    done = 0.0
    console.line(console.tint(f"[ghost] pulling {rng.choice(LOOT)}", GREY))
    while done < total:
        done = min(total, done + total * rng.uniform(0.04, 0.12))
        console.paint(
            "\r  " + console.tint(progress_bar(done, total), CYAN)
            + f" {human_bytes(done):>9} / {human_bytes(total)}"
        )
        console.pause(rng.uniform(0.08, 0.2))
    console.line()
    console.line(console.tint("[ghost] wiping session logs… done", GREY))
    console.line(console.tint("connection closed by remote host.", GREY))
    console.pause(1.4)


def scene_crack(console: Console, rng: random.Random) -> None:
    """Hollywood decryption: hex spray, then the key locks in byte by byte."""
    console.line()
    prompt(console, "./decrypt blueprints.tar.gz.aes --gpu", rng)
    console.line(console.tint("[decrypt] cipher: aes-256-gcm · key schedule unknown", GREY))
    console.pause(0.4)

    pairs = min(24, (console.width - 6) // 3)
    for _ in range(rng.randint(4, 8)):
        dump = " ".join(f"{rng.randrange(256):02x}" for _ in range(pairs))
        console.line(console.tint(f"  {dump}", GREY))
        console.pause(rng.uniform(0.05, 0.15))

    console.line(console.tint("[decrypt] brute-forcing key segments", GREY))
    key = [f"{rng.randrange(256):02X}" for _ in range(16)]
    locked = [False] * 16
    order = list(range(16))
    rng.shuffle(order)
    for step, target in enumerate(order, start=1):
        # The trope itself: unlocked bytes flicker, then one locks in.
        for _ in range(rng.randint(2, 5)):
            cells = [
                console.tint(key[i], GREEN) if locked[i]
                else console.tint(f"{rng.randrange(256):02X}", GREY)
                for i in range(16)
            ]
            console.paint("\r  KEY ▸ " + " ".join(cells) + f"  {step - 1:>2}/16")
            console.pause(rng.uniform(0.03, 0.09))
        locked[target] = True
    console.paint(
        "\r  KEY ▸ " + " ".join(console.tint(byte, GREEN) for byte in key) + "  16/16"
    )
    console.line()
    console.pause(0.5)

    console.line(console.tint("[decrypt] key recovered · verifying MAC… ok", GREEN))
    console.line(console.tint("[decrypt] plaintext written to ./blueprints.tar.gz", GREY))
    console.pause(1.4)


SCENE_RUNNERS: dict[str, Callable[[Console, random.Random], None]] = {
    "code": scene_code,
    "build": scene_build,
    "tests": scene_tests,
    "git": scene_git,
    "matrix": scene_matrix,
    "warroom": scene_warroom,
    "intrusion": scene_intrusion,
    "crack": scene_crack,
}


def run(
    console: Console,
    rng: random.Random,
    *,
    scene: str = "",
    scenes: Sequence[str] = PROFILES["hacker"],
    duration: float = 0.0,
    now: Callable[[], float] = time.monotonic,
) -> int:
    """Cycle `scenes` until `duration` elapses. Returns how many ran.

    A pinned `scene` beats the profile's rotation. `duration <= 0` means
    forever, so the caller (not this loop) owns the exit condition — Ctrl-C in
    the CLI, a fixed scene count in a test.
    """
    started = now()
    played = 0
    last = ""
    # Open on the war room when it is in the rotation — that is the
    # multi-window poster, and a random first pick often hides it for minutes.
    pending_opener = "warroom" if (not scene and "warroom" in scenes) else ""

    while True:
        if scene:
            last = scene
        elif pending_opener:
            last = pending_opener
            pending_opener = ""
        else:
            last = next_scene(rng, last, scenes)
        SCENE_RUNNERS[last](console, rng)
        played += 1

        if duration > 0 and now() - started >= duration:
            return played


# ── Launching a visible window ───────────────────────────────────────────────

LINUX_TERMINALS = (
    ("x-terminal-emulator", ("-e",)),
    ("gnome-terminal", ("--",)),
    ("konsole", ("-e",)),
    ("xfce4-terminal", ("-e",)),
    ("alacritty", ("-e",)),
    ("kitty", ("--",)),
    ("xterm", ("-e",)),
)


class NoTerminalError(RuntimeError):
    """No terminal emulator on this machine can host the screensaver."""


def applescript_string(text: str) -> str:
    """Quote text as an AppleScript string literal."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def window_argv(
    command: str,
    platform: str,
    which: Callable[[str], str | None] = shutil.which,
) -> list[str]:
    """Build the argv that runs `command` in a NEW visible terminal window.

    Takes the platform as data rather than reading `sys.platform`, so all three
    branches are checkable from one host.
    """
    if platform == "darwin":
        # Size the new window for the war room (panes + four dialogs). A
        # default 80x24 Terminal clips the corners and looks like one pane.
        quoted = applescript_string(command)
        return [
            "osascript",
            "-e",
            "tell application \"Terminal\"\n"
            f"  set w to do script {quoted}\n"
            "  set number of columns of front window to 140\n"
            "  set number of rows of front window to 42\n"
            "  activate\n"
            "end tell",
        ]

    if platform == "win32":
        return ["cmd", "/c", "start", "", "cmd", "/k", command]

    for emulator, flags in LINUX_TERMINALS:
        if which(emulator):
            # `sh -c` normalises the argument shape across emulators.
            return [emulator, *flags, "sh", "-c", command]

    raise NoTerminalError(
        "no terminal emulator found (tried: "
        + ", ".join(name for name, _ in LINUX_TERMINALS)
        + ")"
    )


def relaunch_command(argv: Sequence[str], script: str = "", python: str = "") -> str:
    """The shell command that re-runs this script without `--window`."""
    parts = [
        python or sys.executable,
        script or str(Path(__file__).resolve()),
        *[arg for arg in argv if arg != "--window"],
    ]

    return " ".join(shlex.quote(part) for part in parts)


def open_in_window(
    argv: Sequence[str],
    *,
    platform: str = sys.platform,
    spawn: Callable[..., object] = subprocess.Popen,
    which: Callable[[str], str | None] = shutil.which,
) -> int:
    """Start the screensaver in its own window and return immediately."""
    command = window_argv(relaunch_command(argv), platform, which)
    spawn(command)

    return 0


# ── CLI ──────────────────────────────────────────────────────────────────────


def supports_color(stream: TextIO, requested: bool) -> bool:
    """Colour when asked for it, the stream is a TTY, and NO_COLOR is unset."""
    if not requested or os.environ.get("NO_COLOR"):
        return False

    return bool(getattr(stream, "isatty", lambda: False)())


def _enable_windows_ansi() -> None:
    """Turn on VT processing so the escapes mean something on Windows."""
    if sys.platform != "win32":
        return

    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        # An old console just gets no colour; the animation still runs.
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="busy_terminal.py",
        description="A joke screensaver that fakes a busy coding session.",
    )
    parser.add_argument(
        "--duration", type=float, default=0.0,
        help="Seconds to run. 0 (default) runs until Ctrl-C.",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Time multiplier. 2 is twice as fast, 0.5 half.",
    )
    parser.add_argument(
        "--profile", choices=sorted(PROFILES), default="hacker",
        help="Scene set: hacker (Hollywood, default), developer (fake work), mixed (both).",
    )
    parser.add_argument(
        "--scene", choices=SCENES, default="",
        help="Play one scene on repeat instead of cycling the profile.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for a reproducible run.")
    parser.add_argument("--no-color", action="store_true", help="Plain text, no ANSI escapes.")
    parser.add_argument(
        "--window", action="store_true",
        help="Open a new terminal window running this, then exit. Use this from an agent.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw)

    if args.window:
        return open_in_window(raw)

    _enable_windows_ansi()

    size = shutil.get_terminal_size(fallback=(100, 30))
    console = Console(
        width=size.columns,
        height=size.lines,
        color=supports_color(sys.stdout, not args.no_color),
        speed=args.speed,
    )
    rng = random.Random(args.seed)

    if console.color:
        console.paint(HIDE_CURSOR)
    try:
        run(
            console,
            rng,
            scene=args.scene,
            scenes=PROFILES[args.profile],
            duration=args.duration,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if console.color:
            console.paint(SHOW_CURSOR + RESET)
        console.line()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

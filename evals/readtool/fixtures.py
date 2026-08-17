"""Hostile-workspace fixture generator for the read-tool eval.

Builds a realistic project workspace containing the "small zoo of hostile
files" every real codebase keeps: a huge lockfile, a single-line minified
bundle, an ever-growing log, an empty config, unicode-mangled filenames,
a FIFO, a lying file extension, and a near-miss filename.

Deterministic: same bytes every call (fixed seed, fixed content).
"""

from __future__ import annotations

import os
import random
import unicodedata
from pathlib import Path

# Ground-truth constants shared with tasks.py graders.
LEFT_PAD_VERSION = "1.3.0"
RETRY_BASE_MS = 250
RETRY_CAP_MS = 30000
LOG_ERROR_REQ_ID = "req-7f3d9"
LOG_ERROR_TS = "2026-08-08T23:41:17Z"
REPORT_LINES = 412
NOTES_BULLET_3 = "rotate the API keys quarterly"
AGENTS_BUILD_CMD = "npm run build:prod"

# The filename the fixture writes (adversarial spelling) vs the spelling a
# prompt/screen would show (clean spelling). The two render IDENTICALLY:
# NARROW NO-BREAK SPACE vs space, RIGHT SINGLE QUOTATION MARK vs it typed
# again, NFD vs NFC accents. (Accent-dropping is a VISIBLE difference and
# deliberately not part of this task — that class belongs to did-you-mean.)
NOTES_NAME_CLEAN = "Meeting notes\u2019 r\u00e9sum\u00e9 3.04 PM.txt"
NOTES_NAME_HOSTILE = unicodedata.normalize(
    "NFD", "Meeting\u202fnotes\u2019 re\u0301sume\u0301 3.04\u202fPM.txt"
)


def build_workspace(dest: str | Path) -> Path:
    """Create the fixture workspace under ``dest``. Returns the root path."""
    root = Path(dest)
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260809)

    _write_package_json(root)
    _write_lockfile(root, rng)
    _write_app_js(root)
    _write_minified_bundle(root, rng)
    _write_server_log(root, rng)
    _write_report(root)
    _write_empty_overrides(root)
    _write_unicode_notes(root)
    _write_agents_md(root)
    _write_fake_txt_png(root, rng)
    _make_fifo(root)
    return root


def _write_package_json(root: Path) -> None:
    (root / "package.json").write_text(
        "{\n"
        '  "name": "demo-api",\n'
        '  "version": "4.2.1",\n'
        '  "scripts": {\n'
        '    "test": "vitest run",\n'
        '    "build": "tsc -p .",\n'
        '    "build:prod": "tsc -p . && node scripts/bundle.js"\n'
        "  },\n"
        '  "dependencies": {\n'
        f'    "left-pad": "{LEFT_PAD_VERSION}",\n'
        '    "express": "5.1.2",\n'
        '    "pino": "10.0.3"\n'
        "  }\n"
        "}\n",
        encoding="utf-8",
    )


def _write_lockfile(root: Path, rng: random.Random) -> None:
    """~80K-line synthetic package-lock.json. The token tarpit."""
    out = [
        "{",
        '  "name": "demo-api",',
        '  "lockfileVersion": 3,',
        '  "packages": {',
    ]
    for i in range(8000):
        name = f"pkg-{i:05d}"
        sha = "".join(rng.choices("0123456789abcdef", k=64))
        out += [
            f'    "node_modules/{name}": {{',
            f'      "version": "{rng.randint(0, 9)}.{rng.randint(0, 20)}.{rng.randint(0, 40)}",',
            f'      "resolved": "https://registry.npmjs.org/{name}/-/{name}.tgz",',
            f'      "integrity": "sha512-{sha}",',
            '      "engines": {',
            '        "node": ">=18"',
            "      },",
            '      "license": "MIT",',
            "      \"dependencies\": {},",
            "    },",
        ]
    out += ["  }", "}"]
    (root / "package-lock.json").write_text("\n".join(out), encoding="utf-8")


def _write_app_js(root: Path) -> None:
    (root / "src").mkdir(exist_ok=True)
    (root / "src" / "app.js").write_text(
        "import express from 'express';\n"
        "import { logger } from './log.js';\n\n"
        "// Exponential backoff with full jitter. Base 250ms, capped at 30s.\n"
        "export function retryDelay(attempt) {\n"
        f"  const base = {RETRY_BASE_MS};\n"
        f"  const cap = {RETRY_CAP_MS};\n"
        "  const exp = Math.min(cap, base * 2 ** attempt);\n"
        "  return Math.floor(Math.random() * exp);\n"
        "}\n\n"
        "export function createApp() {\n"
        "  const app = express();\n"
        "  app.get('/healthz', (_req, res) => res.send('ok'));\n"
        "  return app;\n"
        "}\n",
        encoding="utf-8",
    )


def _write_minified_bundle(root: Path, rng: random.Random) -> None:
    """One ~600KB line. Greps for 'retryDelay' hit this file too."""
    words = [
        "function", "return", "var", "const", "let", "typeof", "void 0",
        "Object.assign", "Promise.resolve", "Array.isArray",
    ]
    parts = [
        "(()=>{\"use strict\";function retryDelay(t){return Math.floor(Math.random()*"
        "Math.min(3e4,250*Math.pow(2,t)))}"
    ]
    while sum(len(p) for p in parts) < 600_000:
        a = rng.choice("abcdefghijklmnopqrstuvwxyz")
        b = rng.randint(0, 99999)
        parts.append(
            f"function {a}{b}(e,n){{return {rng.choice(words)}===e?n:{a}{b}}}"
        )
    parts.append("})();")
    (root / "src" / "app.min.js").write_text("".join(parts), encoding="utf-8")


def _write_server_log(root: Path, rng: random.Random) -> None:
    """~150K lines of INFO noise with one ERROR near the tail."""
    (root / "logs").mkdir(exist_ok=True)
    total = 150_000
    error_at = total - 300
    with (root / "logs" / "server.log").open("w", encoding="utf-8") as fh:
        for i in range(total):
            if i == error_at:
                fh.write(
                    f"{LOG_ERROR_TS} ERROR http request failed "
                    f"request_id={LOG_ERROR_REQ_ID} status=502 upstream=payments "
                    "err=connect ECONNREFUSED 10.0.4.17:8443\n"
                )
                continue
            mm = i % 60
            ss = (i * 7) % 60
            rid = "".join(rng.choices("0123456789abcdef", k=5))
            fh.write(
                f"2026-08-08T2{i % 4}:{mm:02d}:{ss:02d}Z INFO http request ok "
                f"request_id=req-{rid} status=200 dur_ms={rng.randint(2, 90)}\n"
            )


def _write_report(root: Path) -> None:
    (root / "data").mkdir(exist_ok=True)
    lines = [
        f"metric_{i:04d}: value={i * 3} region={'us' if i % 2 else 'eu'}"
        for i in range(1, REPORT_LINES + 1)
    ]
    (root / "data" / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_empty_overrides(root: Path) -> None:
    (root / "config").mkdir(exist_ok=True)
    (root / "config" / "overrides.yaml").write_text("", encoding="utf-8")


def _write_unicode_notes(root: Path) -> None:
    (root / "notes").mkdir(exist_ok=True)
    (root / "notes" / NOTES_NAME_HOSTILE).write_text(
        "Team sync notes\n"
        "- ship the payments retry fix\n"
        "- audit the staging TLS certs\n"
        f"- {NOTES_BULLET_3}\n"
        "- close out the Q3 incident review\n",
        encoding="utf-8",
    )


def _write_agents_md(root: Path) -> None:
    (root / "AGENTS.md").write_text(
        "# demo-api contributor guide\n\n"
        "## Build\n\n"
        f"Production builds run `{AGENTS_BUILD_CMD}` (typecheck + bundle).\n"
        "Dev builds use `npm run build`.\n\n"
        "## Tests\n\n"
        "Run `npm test` (vitest). CI requires green tests before merge.\n",
        encoding="utf-8",
    )


def _write_fake_txt_png(root: Path, rng: random.Random) -> None:
    """A .txt that is actually a PNG. Extension lies; magic bytes don't."""
    payload = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) + bytes(
        rng.getrandbits(8) for _ in range(4096)
    )
    (root / "data" / "data.txt").write_bytes(payload)


def _make_fifo(root: Path) -> None:
    fifo = root / "logs" / "live.pipe"
    if not fifo.exists():
        os.mkfifo(fifo)


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "/tmp/readtool-ws"
    p = build_workspace(target)
    print(f"workspace built at {p}")

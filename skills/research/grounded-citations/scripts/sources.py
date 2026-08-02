#!/usr/bin/env python3
"""Citation ledger for grounded answers and documents.

Owns the ``url -> [n]`` mapping used by the ``grounded-citations`` skill.
Ids are assigned at retrieval time and never change, so a draft's ``[3]``
always resolves to the same page.  The model only ever emits integers the
ledger handed it, which is what makes the citations verifiable.

Subcommands
-----------
  reset                            start a clean ledger
  add URL [URL ...]                register source(s), print their ids
  ingest FILE|-                    register every url found in JSON tool output
  quote ID --text T --from FILE|-  attach verbatim supporting evidence to a source
  list                             show the ledger
  render                           render a Sources block
  verify DRAFT                     check a draft's citations against the ledger

Fact-checking is evidence-backed citation: ``quote`` only accepts text that
literally appears in the fetched page text you point it at, ``verify
--evidence`` requires every cited source to carry at least one such quote, and
``render --style evidence`` prints the quotes under each source so the reader
can check the chain themselves.  Claims from model knowledge are declared with
an ``[unverified]`` marker rather than silently blended in.

Ledger path resolution (first wins):
  --ledger PATH
  $HERMES_CITATION_LEDGER
  $HERMES_HOME/cache/citations/ledger.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _hermes_home import get_hermes_home  # noqa: E402

SCHEMA_VERSION = 1

# A citation marker in prose: [12].  Markdown links ([text](url)) and
# reference-style labels are excluded by requiring digits only and no
# following "(" or ":".
_CITE_RE = re.compile(r"\[(\d{1,4})\](?![(:])")
_SOURCES_HEADER_RE = re.compile(r"^\s*(?:#{1,6}\s*)?(?:\*\*)?sources:?(?:\*\*)?\s*$", re.IGNORECASE)
_SOURCE_LINE_RE = re.compile(r"^\s*\[(\d{1,4})\]\s*[-–:]?\s*(\S+)")
_URL_IN_TEXT_RE = re.compile(r"https?://[^\s\"'<>)\]}]+")
_FENCE_RE = re.compile(r"^\s*(?:```|~~~)")
# Explicit declaration that a claim comes from model knowledge, not a source.
_UNVERIFIED_RE = re.compile(r"\[unverified\]", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Ledger I/O
# ---------------------------------------------------------------------------


def resolve_ledger_path(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get("HERMES_CITATION_LEDGER", "").strip()
    if env:
        return Path(env).expanduser()
    return get_hermes_home() / "cache" / "citations" / "ledger.json"


def normalize_url(url: str) -> str:
    """Canonicalize a URL for ledger identity.

    Strips the fragment and a trailing slash so ``/page``, ``/page/`` and
    ``/page#section`` are one source.  Query strings are significant and are
    kept — they usually select different content.
    """
    u = (url or "").strip()
    if "#" in u:
        u = u.split("#", 1)[0]
    stripped = u.rstrip("/")
    return stripped or u


def load_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": SCHEMA_VERSION, "sources": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise SystemExit(f"error: ledger at {path} is unreadable ({exc}); run `reset` to start over")
    if not isinstance(data, dict) or not isinstance(data.get("sources"), list):
        raise SystemExit(f"error: ledger at {path} has an unexpected shape; run `reset`")
    return data


def save_ledger(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class _LedgerLock:
    """Best-effort cross-process lock (O_EXCL lockfile, stdlib only).

    Parallel subagents can share one ledger via --ledger; without a lock two
    concurrent ``add`` calls can assign the same id.  Falls through after the
    timeout rather than blocking a task forever — a stale lock must never
    wedge the ledger.
    """

    def __init__(self, path: Path, timeout: float = 5.0) -> None:
        self.lock_path = path.with_suffix(path.suffix + ".lock")
        self.timeout = timeout
        self.fd: int | None = None

    def __enter__(self) -> "_LedgerLock":
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self.fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except FileExistsError:
                if time.monotonic() >= deadline:
                    # Assume a stale lock from a crashed run.
                    try:
                        self.lock_path.unlink()
                    except OSError:
                        return self
                    continue
                time.sleep(0.05)

    def __exit__(self, *_exc: object) -> None:
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
        try:
            self.lock_path.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------


def _by_url(sources: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {s["url"]: s for s in sources}


def add_sources(
    path: Path,
    urls: Iterable[str],
    title: str | None = None,
    accessed: str | None = None,
) -> list[dict[str, Any]]:
    """Register urls, returning their ledger entries (existing or new)."""
    urls = [u for u in (str(u).strip() for u in urls) if u]
    if not urls:
        return []
    with _LedgerLock(path):
        data = load_ledger(path)
        sources = data["sources"]
        index = _by_url(sources)
        out: list[dict[str, Any]] = []
        changed = False
        for raw in urls:
            key = normalize_url(raw)
            existing = index.get(key)
            if existing is not None:
                if title and not existing.get("title"):
                    existing["title"] = title
                    changed = True
                out.append(existing)
                continue
            entry = {
                "id": len(sources) + 1,
                "url": key,
                "title": (title or "").strip(),
                "accessed": accessed or time.strftime("%Y-%m-%d"),
            }
            sources.append(entry)
            index[key] = entry
            out.append(entry)
            changed = True
        if changed:
            save_ledger(path, data)
    return out


def urls_from_json(payload: Any) -> list[tuple[str, str]]:
    """Walk arbitrary JSON tool output collecting (url, title) pairs.

    Handles web_search (``data.web[]``), web_extract (``results[]``) and any
    other nesting, in document order, deduped.
    """
    found: list[tuple[str, str]] = []
    seen: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            url = node.get("url") or node.get("link") or node.get("source_url")
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                key = normalize_url(url)
                if key not in seen:
                    seen.add(key)
                    raw_title = node.get("title") or node.get("name") or ""
                    found.append((url, raw_title if isinstance(raw_title, str) else ""))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return found


# ---------------------------------------------------------------------------
# Evidence quotes (fact-checking)
# ---------------------------------------------------------------------------


def _normalize_ws(text: str) -> str:
    """Collapse all whitespace runs to single spaces for verbatim matching."""
    return " ".join((text or "").split())


def quote_in_evidence(quote: str, evidence: str) -> bool:
    """True when ``quote`` appears verbatim (whitespace-insensitively,
    case-insensitively) in the fetched ``evidence`` text."""
    q = _normalize_ws(quote).casefold()
    return bool(q) and q in _normalize_ws(evidence).casefold()


def attach_quote(path: Path, source_id: int, quote: str, evidence: str) -> dict[str, Any]:
    """Attach a verbatim quote to a ledger entry after checking it against
    the evidence text.  Raises SystemExit on unknown id or non-verbatim text —
    a quote the page does not contain is exactly the fabrication this guards
    against."""
    quote = (quote or "").strip()
    if len(_normalize_ws(quote).split()) < 3:
        raise SystemExit("error: quote too short — use at least 3 words of verbatim text")
    if not quote_in_evidence(quote, evidence):
        raise SystemExit(
            "error: quote not found verbatim in the evidence text — "
            "copy the exact wording from the fetched page, do not paraphrase"
        )
    with _LedgerLock(path):
        data = load_ledger(path)
        entry = next((s for s in data["sources"] if s["id"] == source_id), None)
        if entry is None:
            raise SystemExit(f"error: no source [{source_id}] in the ledger")
        quotes = entry.setdefault("quotes", [])
        norm = _normalize_ws(quote).casefold()
        if not any(_normalize_ws(q.get("text", "")).casefold() == norm for q in quotes):
            quotes.append({"text": quote, "added": time.strftime("%Y-%m-%d")})
            save_ledger(path, data)
    return entry


def render_sources(
    sources: list[dict[str, Any]],
    style: str = "markdown",
    only: set[int] | None = None,
) -> str:
    picked = [s for s in sources if only is None or s["id"] in only]
    picked.sort(key=lambda s: s["id"])
    if not picked:
        return ""
    lines: list[str] = []
    if style == "bibtex":
        for s in picked:
            key = f"source{s['id']}"
            title = s.get("title") or s["url"]
            lines.append(
                "@misc{%s,\n  title = {%s},\n  howpublished = {\\url{%s}},\n  note = {Accessed %s}\n}"
                % (key, title, s["url"], s.get("accessed", ""))
            )
        return "\n".join(lines)
    if style == "footnotes":
        for s in picked:
            title = s.get("title")
            suffix = f" — {title}" if title else ""
            lines.append(f"[^{s['id']}]: {s['url']}{suffix}")
        return "\n".join(lines)
    header = "Sources:" if style == "plain" else "## Sources"
    lines.append(header)
    if style != "plain":
        lines.append("")
    for s in picked:
        title = s.get("title")
        suffix = f" — {title}" if title else ""
        lines.append(f"[{s['id']}] {s['url']}{suffix}")
        if style == "evidence":
            for q in s.get("quotes", []):
                lines.append(f'    > "{q.get("text", "")}"')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _split_draft(text: str) -> tuple[str, dict[int, str]]:
    """Split a draft into (prose, sources_block_map).

    The sources block is everything after the last ``Sources`` header; its
    ``[n] url`` lines are parsed out so they aren't counted as prose citations.
    Fenced code blocks are dropped from prose.
    """
    lines = text.splitlines()
    header_idx = -1
    for i, line in enumerate(lines):
        if _SOURCES_HEADER_RE.match(line):
            header_idx = i
    listed: dict[int, str] = {}
    if header_idx >= 0:
        for line in lines[header_idx + 1:]:
            m = _SOURCE_LINE_RE.match(line)
            if m:
                url_match = _URL_IN_TEXT_RE.search(line)
                listed[int(m.group(1))] = url_match.group(0) if url_match else m.group(2)
        body_lines = lines[:header_idx]
    else:
        body_lines = lines
    prose: list[str] = []
    in_fence = False
    for line in body_lines:
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            prose.append(line)
    return "\n".join(prose), listed


def _sentences(prose: str) -> list[str]:
    """Rough sentence split over prose lines, skipping headings and tables."""
    out: list[str] = []
    for line in prose.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("|"):
            continue
        if stripped.startswith(">"):
            stripped = stripped.lstrip("> ").strip()
        for part in re.split(r"(?<=[.!?])\s+", stripped):
            part = part.strip()
            if len(part.split()) >= 4:
                out.append(part)
    return out


def verify_draft(
    draft_path: Path,
    sources: list[dict[str, Any]],
    strict: bool = False,
    min_coverage: float | None = None,
    require_evidence: bool = False,
) -> tuple[int, list[str], list[str]]:
    """Return (exit_code, errors, warnings)."""
    text = draft_path.read_text(encoding="utf-8")
    prose, listed = _split_draft(text)
    by_id = {s["id"]: s for s in sources}

    errors: list[str] = []
    warnings: list[str] = []

    cited = [int(m) for m in _CITE_RE.findall(prose)]
    cited_set = set(cited)

    unknown = sorted(i for i in cited_set if i not in by_id)
    if unknown:
        errors.append(
            "citations not in the ledger (hallucinated or renumbered): "
            + ", ".join(f"[{i}]" for i in unknown)
        )

    if cited_set and not listed:
        errors.append("draft cites sources but has no `Sources:` block — run `render --cited-in`")

    missing_from_block = sorted(cited_set - set(listed)) if listed else []
    if missing_from_block:
        errors.append(
            "cited but absent from the Sources block: "
            + ", ".join(f"[{i}]" for i in missing_from_block)
        )

    for sid, url in sorted(listed.items()):
        entry = by_id.get(sid)
        if entry is None:
            errors.append(f"Sources block lists [{sid}], which is not in the ledger")
            continue
        if normalize_url(url) != entry["url"]:
            errors.append(
                f"Sources block URL for [{sid}] does not match the ledger "
                f"(block: {url} / ledger: {entry['url']}) — re-run `render`"
            )

    extra_in_block = sorted(set(listed) - cited_set)
    if extra_in_block:
        warnings.append(
            "listed in Sources but never cited inline: "
            + ", ".join(f"[{i}]" for i in extra_in_block)
        )

    registered_uncited = sorted(set(by_id) - cited_set)
    if registered_uncited:
        warnings.append(
            "registered in the ledger but not cited in this draft: "
            + ", ".join(f"[{i}]" for i in registered_uncited)
        )

    sentences = _sentences(prose)
    cited_sentences = [s for s in sentences if _CITE_RE.search(s)]
    unverified_sentences = [s for s in sentences if _UNVERIFIED_RE.search(s)]
    covered = [s for s in sentences if _CITE_RE.search(s) or _UNVERIFIED_RE.search(s)]
    coverage = (len(covered) / len(sentences)) if sentences else 0.0
    if min_coverage is not None and sentences and coverage < min_coverage:
        errors.append(
            f"citation coverage {coverage:.0%} is below the required {min_coverage:.0%} "
            f"({len(covered)}/{len(sentences)} sentences cited or marked [unverified])"
        )

    if require_evidence:
        unevidenced = sorted(
            i for i in cited_set if i in by_id and not by_id[i].get("quotes")
        )
        if unevidenced:
            errors.append(
                "cited sources carry no verbatim evidence quote (run `quote` with the "
                "fetched page text): " + ", ".join(f"[{i}]" for i in unevidenced)
            )

    over_cited = [s for s in sentences if len(_CITE_RE.findall(s)) > 3]
    if over_cited:
        warnings.append(f"{len(over_cited)} sentence(s) carry more than 3 citations")

    code = 1 if errors else (1 if (strict and warnings) else 0)
    quoted = sum(1 for s in sources if s.get("quotes"))
    stats = (
        f"{len(sentences)} prose sentence(s), {len(cited_sentences)} cited, "
        f"{len(unverified_sentences)} marked [unverified] ({coverage:.0%} covered), "
        f"{len(cited_set)} distinct source(s) cited, "
        f"{len(by_id)} in ledger ({quoted} with evidence quotes)"
    )
    warnings.insert(0, f"stats: {stats}")
    return code, errors, warnings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_only(spec: str | None) -> set[int] | None:
    if not spec:
        return None
    out: set[int] = set()
    for chunk in spec.replace(" ", "").split(","):
        if not chunk:
            continue
        if "-" in chunk:
            lo, _, hi = chunk.partition("-")
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(chunk))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sources.py", description="Citation ledger for grounded answers and documents."
    )
    parser.add_argument("--ledger", help="ledger file path (overrides env / default)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("reset", help="start a clean ledger")

    p_add = sub.add_parser("add", help="register source url(s), print their ids")
    p_add.add_argument("urls", nargs="+")
    p_add.add_argument("--title", help="title for the source (single-url calls)")
    p_add.add_argument("--accessed", help="access date (default: today)")
    p_add.add_argument("--json", action="store_true", help="emit JSON instead of ids")

    p_ing = sub.add_parser("ingest", help="register every url in JSON tool output")
    p_ing.add_argument("file", help="JSON file, or - for stdin")

    p_q = sub.add_parser("quote", help="attach verbatim supporting evidence to a source")
    p_q.add_argument("id", type=int, help="ledger id of the source the quote supports")
    p_q.add_argument("--text", required=True, help="the exact quote, copied from the page")
    p_q.add_argument(
        "--from",
        dest="evidence",
        required=True,
        help="file with the fetched page text (or - for stdin) the quote must appear in",
    )

    p_list = sub.add_parser("list", help="show the ledger")
    p_list.add_argument("--json", action="store_true")

    p_render = sub.add_parser("render", help="render a Sources block")
    p_render.add_argument(
        "--style", default="markdown", choices=["markdown", "plain", "footnotes", "bibtex", "evidence"]
    )
    p_render.add_argument("--only", help="ids to include, e.g. 1,3,5-7")
    p_render.add_argument("--cited-in", help="include only ids cited in this draft file")

    p_ver = sub.add_parser("verify", help="check a draft's citations against the ledger")
    p_ver.add_argument("draft")
    p_ver.add_argument("--strict", action="store_true", help="treat warnings as failures")
    p_ver.add_argument("--min-coverage", type=float, help="required cited-sentence share, e.g. 0.5")
    p_ver.add_argument(
        "--evidence",
        action="store_true",
        help="require every cited source to carry at least one verbatim quote",
    )

    args = parser.parse_args(argv)
    path = resolve_ledger_path(args.ledger)

    if args.cmd == "reset":
        with _LedgerLock(path):
            save_ledger(path, {"version": SCHEMA_VERSION, "sources": []})
        print(f"ledger reset: {path}")
        return 0

    if args.cmd == "add":
        title = args.title if len(args.urls) == 1 else None
        entries = add_sources(path, args.urls, title=title, accessed=args.accessed)
        if args.json:
            print(json.dumps(entries, indent=2, ensure_ascii=False))
        else:
            for e in entries:
                print(f"[{e['id']}] {e['url']}")
        return 0

    if args.cmd == "ingest":
        raw = sys.stdin.read() if args.file == "-" else Path(args.file).read_text(encoding="utf-8")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"error: input is not valid JSON ({exc})", file=sys.stderr)
            return 2
        pairs = urls_from_json(payload)
        if not pairs:
            print("no urls found in input", file=sys.stderr)
            return 1
        for url, title in pairs:
            entry = add_sources(path, [url], title=title or None)[0]
            print(f"[{entry['id']}] {entry['url']}")
        return 0

    if args.cmd == "quote":
        raw = (
            sys.stdin.read()
            if args.evidence == "-"
            else Path(args.evidence).read_text(encoding="utf-8")
        )
        entry = attach_quote(path, args.id, args.text, raw)
        print(f"[{entry['id']}] evidence attached ({len(entry.get('quotes', []))} quote(s))")
        return 0

    data = load_ledger(path)
    sources = sorted(data["sources"], key=lambda s: s["id"])

    if args.cmd == "list":
        if args.json:
            print(json.dumps(sources, indent=2, ensure_ascii=False))
        elif not sources:
            print(f"ledger is empty: {path}")
        else:
            for s in sources:
                title = f"  {s['title']}" if s.get("title") else ""
                nq = len(s.get("quotes", []))
                mark = f"  ({nq} quote{'s' if nq != 1 else ''})" if nq else ""
                print(f"[{s['id']}] {s['url']}{title}{mark}")
        return 0

    if args.cmd == "render":
        only = _parse_only(args.only)
        if args.cited_in:
            draft = Path(args.cited_in).read_text(encoding="utf-8")
            prose, _ = _split_draft(draft)
            cited = {int(m) for m in _CITE_RE.findall(prose)}
            only = cited if only is None else (only & cited)
        block = render_sources(sources, style=args.style, only=only)
        if not block:
            print("no sources to render", file=sys.stderr)
            return 1
        print(block)
        return 0

    if args.cmd == "verify":
        draft_path = Path(args.draft)
        if not draft_path.is_file():
            print(f"error: no such draft: {draft_path}", file=sys.stderr)
            return 2
        code, errors, warnings = verify_draft(
            draft_path,
            sources,
            strict=args.strict,
            min_coverage=args.min_coverage,
            require_evidence=args.evidence,
        )
        for w in warnings:
            print(f"warn: {w}")
        for e in errors:
            print(f"FAIL: {e}", file=sys.stderr)
        print("citations OK" if code == 0 else "verification failed")
        return code

    return 2


if __name__ == "__main__":
    sys.exit(main())

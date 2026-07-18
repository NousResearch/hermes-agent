#!/usr/bin/env python3
"""Deterministic retrieval over a pinned, MIT-licensed image-prompt corpus.

Upstream corpus: freestylefly/awesome-gpt-image-2 (MIT). The corpus ships
taxonomy, template blocks, and worked cases; this script fetches it once at a
pinned commit, verifies every byte against recorded sha256 digests, and then
serves selection/grounding queries fully offline.

The grounding contract:

    prime    ONLINE. The only network step. Run once before first use.
    ground   PURE and OFFLINE. Resolves a selection into real cases, or
             reports grounded=false. Never touches the network.

A citation is stamped if and only if it resolves. Zero matches is an honest
``grounded: false`` (exit 0). A cold or corrupt cache is a provisioning
failure (exit 1), not a data answer. This split exists because a reference
that is a POINTER (a URL, an uninstalled file) degrades silently to zero
inside a context that cannot fetch, while still being cited -- the model must
consume retrieved text, never fetch it.

No upstream text is embedded in this file. It is fetched, verified, and
cached outside the repository tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ENGINE_NAME = "gpt-image-2-style-library"
UPSTREAM_REPO = "freestylefly/awesome-gpt-image-2"
UPSTREAM_PIN = "a04beebfa3195ef8dfbf1c57da7df9e989c2173b"
UPSTREAM_LICENSE = "MIT"
UPSTREAM_HOME = f"https://github.com/{UPSTREAM_REPO}"

_RAW = "https://raw.githubusercontent.com/{repo}/{pin}/{path}"

# local name -> (upstream path, sha256 at UPSTREAM_PIN).
# These digests are the trust anchor: every read re-hashes the cached bytes and
# compares against this table. A sidecar "downloaded: true" marker is never
# trusted, because a fetch killed midway leaves a truncated file that such a
# marker would bless.
CORPUS_FILES: dict[str, tuple[str, str]] = {
    "cases.json": (
        "data/cases.json",
        "3c88ef3d3c15ca319992fc82f860de6674412fe913a585a50664fc2a687261b3",
    ),
    "style-library.json": (
        "data/style-library.json",
        "80f5cae039d0d6f312f0e2de2c9b3fc8a806640b0d517c120d704a71c5e4aa72",
    ),
    "templates.md": (
        "docs/templates.md",
        "f8e5009821d2099da51e23ab467ec9e938fdab34856e439ac36138c155eec926",
    ),
    "LICENSE": (
        "LICENSE",
        "27a75c48bac29eb78f43c19f75c4e175974c8f1046d848d5562eae1ead2f1176",
    ),
}

CACHE_ENV = "IMAGE_PROMPT_FACTORY_CACHE_DIR"

_DEFAULT_K = 5
_EXEMPLAR_CHAR_CAP = 1200
_EXEMPLAR_TOTAL_BUDGET = 8000
_HTTP_TIMEOUT_S = 60

_W_CATEGORY = 3
_W_STYLE = 2
_W_SCENE = 1
_W_EXAMPLE_CASE = 4

# CJK ideographs plus CJK/fullwidth punctuation. Escaped rather than literal so
# this file stays pure ASCII: it is read and piped by toolchains that default
# to cp1252. Used only to offer an English-exemplar filter; the corpus is
# bilingual by design (this is also why retrieval is a deterministic taxonomy
# filter, not embedding search: an English-only embedder over a majority-CJK
# corpus would be quietly wrong).
_CJK = re.compile("[\u3000-\u9fff\uff00-\uffef]")


class CorpusMissing(RuntimeError):
    """Cache absent, incomplete, or byte-for-byte wrong. Exit 1: provisioning bug."""


class UsageError(RuntimeError):
    """Bad arguments. Exit 1."""


@dataclass(frozen=True)
class Case:
    id: int
    title: str
    prompt: str
    category: str
    styles: tuple[str, ...]
    scenes: tuple[str, ...]
    featured: bool
    source_url: str


@dataclass(frozen=True)
class Template:
    id: str
    category: str
    anchor: str
    styles: tuple[str, ...]
    scenes: tuple[str, ...]
    example_cases: tuple[int, ...]


@dataclass(frozen=True)
class Corpus:
    root: Path
    pin: str
    cases: dict[int, Case]
    templates: dict[str, Template]

    @property
    def template_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.templates))


@dataclass(frozen=True)
class Exemplar:
    id: int
    title: str
    prompt: str
    category: str
    styles: tuple[str, ...]
    scenes: tuple[str, ...]
    source_url: str
    truncated: bool


@dataclass(frozen=True)
class Grounding:
    grounded: bool
    matched: int
    resolved_case_ids: tuple[int, ...]
    unresolved_case_ids: tuple[int, ...]
    exemplars: tuple[Exemplar, ...] = ()
    provenance: dict = field(default_factory=dict)

    def summary(self) -> dict:
        out = {
            "grounded": self.grounded,
            "matched": self.matched,
            "resolved_case_ids": list(self.resolved_case_ids),
            "unresolved_case_ids": list(self.unresolved_case_ids),
        }
        out.update(self.provenance)
        return out

    def full(self) -> dict:
        payload = self.summary()
        payload["exemplars"] = [
            {
                "id": e.id,
                "title": e.title,
                "prompt": e.prompt,
                "category": e.category,
                "styles": list(e.styles),
                "scenes": list(e.scenes),
                "source_url": e.source_url,
                "truncated": e.truncated,
            }
            for e in self.exemplars
        ]
        return payload


def _http_get(url: str) -> bytes:
    """Network seam. Module-level name so tests can monkeypatch it."""
    req = urllib.request.Request(url, headers={"User-Agent": "hermes-image-prompt-factory"})
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:  # noqa: S310
        return resp.read()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cache_root(explicit: "str | Path | None" = None) -> Path:
    """Install-independent by construction: the same cache is found no matter
    which project directory the agent runs from."""
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get(CACHE_ENV)
    if env:
        return Path(env).expanduser()
    return Path.home() / ".hermes" / "cache" / "image-prompt-factory"


def corpus_dir(*, pin: "str | None" = None, cache_dir=None) -> Path:
    return cache_root(cache_dir) / (pin or UPSTREAM_PIN)


def prime(*, pin=None, cache_dir=None, force=None) -> Path:
    """ONLINE. Fetch, verify, and atomically install the corpus."""
    resolved_pin = pin or UPSTREAM_PIN
    force = bool(force)
    target = corpus_dir(pin=resolved_pin, cache_dir=cache_dir)

    if target.is_dir() and not force:
        try:
            require_corpus(pin=resolved_pin, cache_dir=cache_dir)
            return target
        except CorpusMissing:
            pass  # present but wrong: fall through and refetch

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{resolved_pin[:8]}.tmp.", dir=target.parent))
    try:
        for name, (path, expected) in CORPUS_FILES.items():
            raw = _http_get(_RAW.format(repo=UPSTREAM_REPO, pin=resolved_pin, path=path))
            actual = _sha256(raw)
            if resolved_pin == UPSTREAM_PIN and actual != expected:
                raise CorpusMissing(
                    f"{path}: sha256 mismatch at pin {resolved_pin[:8]}\n"
                    f"  expected {expected}\n  actual   {actual}\n"
                    "Refusing to write an unverified corpus."
                )
            (staging / name).write_bytes(raw)

        if target.is_dir():
            shutil.rmtree(target)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def require_corpus(*, pin=None, cache_dir=None) -> Corpus:
    """Decide validity by re-hashing the actual bytes on every call -- never a
    marker file's claim about them."""
    resolved_pin = pin or UPSTREAM_PIN
    root = corpus_dir(pin=resolved_pin, cache_dir=cache_dir)
    hint = (
        f"corpus not provisioned at {root}\n"
        "  run: python scripts/style_corpus.py prime"
    )
    if not root.is_dir():
        raise CorpusMissing(hint)

    for name, (_path, expected) in CORPUS_FILES.items():
        f = root / name
        if not f.is_file():
            raise CorpusMissing(f"{hint}\n  missing file: {name}")
        if resolved_pin == UPSTREAM_PIN:
            actual = _sha256(f.read_bytes())
            if actual != expected:
                raise CorpusMissing(
                    f"{hint}\n  corrupt file: {name}\n"
                    f"  expected {expected}\n  actual   {actual}"
                )

    return _load(root, resolved_pin)


def _load(root: Path, pin: str) -> Corpus:
    raw_cases = json.loads((root / "cases.json").read_text(encoding="utf-8"))
    cases: dict[int, Case] = {}
    for c in raw_cases["cases"]:
        prompt = str(c.get("prompt") or "")
        if not prompt.strip():
            continue  # a case with no prompt cannot ground a citation
        cases[int(c["id"])] = Case(
            id=int(c["id"]),
            title=str(c.get("title") or ""),
            prompt=prompt,
            category=str(c.get("category") or ""),
            styles=tuple(c.get("styles") or ()),
            scenes=tuple(c.get("scenes") or ()),
            featured=bool(c.get("featured")),
            source_url=str(c.get("sourceUrl") or ""),
        )

    raw_lib = json.loads((root / "style-library.json").read_text(encoding="utf-8"))
    templates: dict[str, Template] = {}
    for t in raw_lib.get("templates", ()):
        templates[str(t["id"])] = Template(
            id=str(t["id"]),
            category=str(t.get("category") or ""),
            anchor=str(t.get("templateAnchor") or ""),
            styles=tuple(t.get("styles") or ()),
            scenes=tuple(t.get("scenes") or ()),
            example_cases=tuple(int(x) for x in (t.get("exampleCases") or ())),
        )
    return Corpus(root=root, pin=pin, cases=cases, templates=templates)


def _truncate(prompt: str, cap: int) -> tuple[str, bool]:
    if len(prompt) <= cap:
        return prompt, False
    cut = prompt[:cap]
    nl = cut.rfind("\n")
    if nl > cap // 2:
        cut = cut[:nl]
    return cut.rstrip(), True


def _provenance(corpus: Corpus) -> dict:
    return {
        "prompt_engine": ENGINE_NAME,
        "corpus_pin": corpus.pin,
        "corpus_source": UPSTREAM_HOME,
        "corpus_sha256": CORPUS_FILES["cases.json"][1],
        "license": UPSTREAM_LICENSE,
    }


def select(
    corpus: Corpus,
    *,
    template_id=None,
    category=None,
    styles=None,
    scenes=None,
    case_ids=None,
    lang=None,
    k=None,
) -> Grounding:
    """Deterministic taxonomy retrieval. No embeddings, no LLM.

    Every tunable arrives as a None sentinel and is resolved here, so a test
    or a caller can override it without fighting a cached def-time default.
    Ranking follows the corpus's own documented selection order: category,
    then style tag, then scene tag, then the template's own example cases.
    """
    k = _DEFAULT_K if k is None else int(k)
    lang = (lang or "").lower() or None
    want_styles = {s for s in (styles or ())}
    want_scenes = {s for s in (scenes or ())}

    tpl = corpus.templates.get(template_id) if template_id else None
    if template_id and tpl is None:
        raise UsageError(f"unknown template_id: {template_id}")
    example_ids = set(tpl.example_cases) if tpl else set()

    pool = list(corpus.cases.values())
    if lang == "en":
        pool = [c for c in pool if not _CJK.search(c.prompt)]

    # Cited ids are ANCHORS, not a replacement for ranking: the caller names
    # nearest cases, then taxonomy tops the slate up to k. An unknown id is a
    # normal "cited but absent" outcome, kept distinguishable from "matched
    # zero".
    available = {c.id for c in pool}
    unresolved: list[int] = []
    anchors: list[Case] = []
    seen: set[int] = set()
    for cid in case_ids or ():
        cid = int(cid)
        if cid in available and cid not in seen:
            anchors.append(corpus.cases[cid])
            seen.add(cid)
        elif cid not in available:
            unresolved.append(cid)

    scored = []
    for c in pool:
        if c.id in seen:
            continue
        score = 0
        if category and c.category == category:
            score += _W_CATEGORY
        score += _W_STYLE * len(want_styles.intersection(c.styles))
        score += _W_SCENE * len(want_scenes.intersection(c.scenes))
        if c.id in example_ids:
            score += _W_EXAMPLE_CASE
        if score > 0:
            scored.append((score, c))
    # id is unique and terminal, so the ordering is a total order: stable output.
    scored.sort(key=lambda sc: (-sc[0], not sc[1].featured, sc[1].id))
    ranked = anchors + [c for _score, c in scored]

    exemplars: list[Exemplar] = []
    budget = _EXEMPLAR_TOTAL_BUDGET
    for c in ranked[:k]:
        text, truncated = _truncate(c.prompt, _EXEMPLAR_CHAR_CAP)
        if exemplars and len(text) > budget:
            break
        budget -= len(text)
        exemplars.append(
            Exemplar(
                id=c.id,
                title=c.title,
                prompt=text,
                category=c.category,
                styles=c.styles,
                scenes=c.scenes,
                source_url=c.source_url,
                truncated=truncated,
            )
        )

    grounded = bool(exemplars)
    return Grounding(
        grounded=grounded,
        matched=len(ranked),
        resolved_case_ids=tuple(e.id for e in exemplars),
        unresolved_case_ids=tuple(unresolved),
        exemplars=tuple(exemplars),
        # A citation is stamped only when it resolves. Nothing to cite,
        # nothing stamped.
        provenance=_provenance(corpus) if grounded else {},
    )


def template_body(corpus: Corpus, template_id: str) -> str:
    tpl = corpus.templates.get(template_id)
    if tpl is None:
        raise UsageError(f"unknown template_id: {template_id}")
    doc = (corpus.root / "templates.md").read_text(encoding="utf-8")
    anchor = tpl.anchor or ""
    start = doc.find(f'<a name="{anchor}"></a>')
    if start < 0:
        raise UsageError(f"template anchor not found in templates.md: {anchor!r}")
    nxt = doc.find('<a name="tpl-', start + 1)
    return doc[start : nxt if nxt > 0 else len(doc)].strip()


def _csv(value: "str | None") -> "list[str] | None":
    if not value:
        return None
    return [p.strip() for p in value.split(",") if p.strip()]


def _cmd_ground(args) -> int:
    """Resolve a selection JSON into grounded exemplars, fully offline.

    Reads ``--selection`` (a JSON file the agent wrote after choosing a
    template/taxonomy), writes ``grounding.local.json`` next to it (or at
    ``--out``). The ``.local.json`` suffix marks it agent-eyes-only: its case
    text is third-party MIT material to learn structure from, never to be
    pasted into a pack or redistributed.
    """
    sel_path = Path(args.selection)
    if not sel_path.is_file():
        raise UsageError(f"selection file not found: {sel_path}")
    sel = json.loads(sel_path.read_text(encoding="utf-8"))

    corpus = require_corpus(cache_dir=args.cache_dir)

    ids = sel.get("example_case_ids") or sel.get("case_ids") or None
    if ids:
        ids = [int(re.sub(r"[^0-9]", "", str(i)) or -1) for i in ids]

    grounding = select(
        corpus,
        template_id=sel.get("template_id") or None,
        category=sel.get("category") or None,
        styles=sel.get("style_tags") or None,
        scenes=sel.get("scene_tags") or None,
        case_ids=ids,
        lang=args.lang,
        k=args.k,
    )

    out = Path(args.out) if args.out else sel_path.parent / "grounding.local.json"
    out.write_text(json.dumps(grounding.full(), ensure_ascii=False, indent=2), encoding="utf-8")

    summary = grounding.summary()
    summary["grounding_path"] = str(out)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def _cmd_prime(args) -> int:
    root = prime(pin=args.pin, cache_dir=args.cache_dir, force=args.force)
    corpus = require_corpus(pin=args.pin, cache_dir=args.cache_dir)
    print(
        json.dumps(
            {
                "primed": str(root),
                "pin": corpus.pin,
                "cases": len(corpus.cases),
                "templates": len(corpus.templates),
                "template_ids": list(corpus.template_ids),
                "license": UPSTREAM_LICENSE,
            },
            indent=2,
        )
    )
    return 0


def _cmd_verify(args) -> int:
    corpus = require_corpus(pin=args.pin, cache_dir=args.cache_dir)
    print(json.dumps({"ok": True, "pin": corpus.pin, "cases": len(corpus.cases)}))
    return 0


def _cmd_stats(args) -> int:
    corpus = require_corpus(pin=args.pin, cache_dir=args.cache_dir)
    by_cat: dict[str, int] = {}
    english = 0
    for c in corpus.cases.values():
        by_cat[c.category] = by_cat.get(c.category, 0) + 1
        if not _CJK.search(c.prompt):
            english += 1
    print(
        json.dumps(
            {
                "pin": corpus.pin,
                "cases": len(corpus.cases),
                "english_prompts": english,
                "templates": len(corpus.templates),
                "by_category": dict(sorted(by_cat.items(), key=lambda kv: -kv[1])),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _cmd_select(args) -> int:
    corpus = require_corpus(pin=args.pin, cache_dir=args.cache_dir)
    ids = [int(x) for x in (_csv(args.cases) or ())] or None
    grounding = select(
        corpus,
        template_id=args.template_id,
        category=args.category,
        styles=_csv(args.styles),
        scenes=_csv(args.scenes),
        case_ids=ids,
        lang=args.lang,
        k=args.k,
    )
    payload = grounding.full() if args.full else grounding.summary()
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _cmd_template(args) -> int:
    corpus = require_corpus(pin=args.pin, cache_dir=args.cache_dir)
    print(template_body(corpus, args.template_id))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="style_corpus", description=__doc__.splitlines()[0])
    p.add_argument("--pin", default=None)
    p.add_argument("--cache-dir", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prime", help="ONLINE: fetch + verify + install the corpus")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=_cmd_prime)

    sv = sub.add_parser("verify", help="re-hash the cached bytes")
    sv.set_defaults(func=_cmd_verify)

    ss = sub.add_parser("stats", help="corpus counts")
    ss.set_defaults(func=_cmd_stats)

    sl = sub.add_parser("select", help="deterministic retrieval")
    sl.add_argument("--template-id", default=None)
    sl.add_argument("--category", default=None)
    sl.add_argument("--styles", default=None, help="comma separated")
    sl.add_argument("--scenes", default=None, help="comma separated")
    sl.add_argument("--cases", default=None, help="comma separated ids")
    sl.add_argument("--lang", default=None, choices=["en"])
    sl.add_argument("--k", type=int, default=None)
    sl.add_argument("--full", action="store_true", help="include exemplar bodies")
    sl.set_defaults(func=_cmd_select)

    st = sub.add_parser("template", help="print a template body")
    st.add_argument("template_id")
    st.set_defaults(func=_cmd_template)

    sg = sub.add_parser("ground", help="OFFLINE: resolve a selection JSON into exemplars")
    sg.add_argument("--selection", required=True, help="path to selection JSON")
    sg.add_argument("--out", default=None, help="grounding output path")
    sg.add_argument("--lang", default=None, choices=["en"])
    sg.add_argument("--k", type=int, default=None)
    sg.set_defaults(func=_cmd_ground)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (CorpusMissing, UsageError) as exc:
        print(f"style_corpus: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

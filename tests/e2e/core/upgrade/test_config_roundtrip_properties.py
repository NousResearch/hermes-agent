"""C18 — config and settings persistence round-trip properties.

A seeded property / matrix layer over every surface that writes ``config.yaml`` or ``.env``:

* P1  real load/save: ``save_config(load_config())`` is byte-identical, and a save that changes
      one key changes that key's line(s) only (comments, quoting, unicode, ``${VAR}`` templates,
      explicit-default values, dotted provider names and nulls all survive);
* P2  "set one key" changes exactly one key, on every writer: in-process ``set_config_value`` over
      EVERY documented scalar leaf of ``DEFAULT_CONFIG``, the real ``hermes config set`` CLI
      subprocess, a real ``python -m tui_gateway.entry`` stdio JSON-RPC process (``config.set``),
      and the dashboard/Desktop ``PUT /api/config`` (partial bodies, like Desktop sends);
* P3  a failed read never results in a clobbering write: for every single config read an
      operation performs, a transient ``EMFILE`` on exactly that read, plus unreadable, truncated
      and non-mapping files, must leave every section the operation did not target intact;
* P4  Hermes's ``.env`` loaders are idempotent (self-references, placeholders, quoting), the
      sanitizer is a fixed point, and ``save_env_value`` changes exactly one line;
* P5  every config migration is idempotent from every historical ``_config_version`` and never
      clobbers user-set values;
* P6  every top-level section tolerates ``section: null`` through load, effective-config
      resolution, save, set, TUI RPC, dashboard, migration and the gateway loaders.

Generators are ``random.Random(seed)`` over a fixed seed list (hypothesis is not a dependency);
every assertion message carries the seed / case so a failure is reproducible with ``-k``.

Known-red cells on the current base are ``xfail(strict=True)`` with the issue they reproduce, so
the suite is green today and flips red the moment the bug is fixed (drop the marker then).
"""

from __future__ import annotations

import contextlib
import copy
import difflib
import errno
import io
import json
import os
import queue
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import pytest
import yaml

from tests.e2e.core.upgrade._helpers import WORKTREE, isolated_env

import hermes_cli.config as C
from hermes_cli.config_defaults import DEFAULT_CONFIG

LATEST = int(DEFAULT_CONFIG["_config_version"])

# ─────────────────────────────────────────────────────────────────────────────
# Structural helpers (pure, no production code)
# ─────────────────────────────────────────────────────────────────────────────

_MISSING = object()


def _leaves(tree: Any, prefix: tuple = ()) -> dict[tuple, Any]:
    """Flatten a parsed YAML tree to ``{path: scalar-or-list}``; an empty mapping is a leaf."""
    if isinstance(tree, dict) and tree:
        out: dict[tuple, Any] = {}
        for k, v in tree.items():
            out.update(_leaves(v, prefix + (k,)))
        return out
    return {prefix: tree}


def _changed_paths(before: dict, after: dict) -> set[tuple]:
    a, b = _leaves(before), _leaves(after)
    return {p for p in a.keys() | b.keys() if a.get(p, _MISSING) != b.get(p, _MISSING)
            or type(a.get(p)) is not type(b.get(p))}


def _under(path: tuple, targets: Iterable[tuple]) -> bool:
    """``path`` is a target, lies below one, or is an ancestor that a target write replaced."""
    return any(path[:len(t)] == t or t[:len(path)] == path for t in targets)


def _get(tree: dict, path: tuple, default: Any = _MISSING) -> Any:
    cur: Any = tree
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _set(tree: dict, path: tuple, value: Any) -> None:
    cur = tree
    for p in path[:-1]:
        if not isinstance(cur.get(p), dict):
            cur[p] = {}
        cur = cur[p]
    cur[path[-1]] = value


def _top_level_keys(text: str) -> list[str]:
    return [m.group(1) for m in re.finditer(r"^([^\s#\-][^:\n]*):", text, re.M)]


def _assert_no_duplicate_top_level(text: str, ctx: str) -> None:
    keys = _top_level_keys(text)
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    assert not dupes, f"[{ctx}] duplicated top-level keys {dupes} in:\n{text}"


_NULL_TOKEN = re.compile(r"^(\s*[^#\n]*?:)\s+(?:null|~)(\s+#.*)?$")


def _canon_line(line: str) -> str:
    """ruamel re-emits an untouched ``key: null`` as ``key:`` (same value, cosmetic; see report)."""
    m = _NULL_TOKEN.match(line)
    return (m.group(1) + (" " + m.group(2).strip() if m.group(2) else "")) if m else re.sub(r":\s+#", ": #", line)


def _line_diff(before: str, after: str) -> tuple[set[int], list[list[str]]]:
    """(indices of BEFORE lines that were replaced/deleted, blocks of inserted AFTER lines)."""
    a = [_canon_line(x) for x in before.splitlines()]
    b = [_canon_line(x) for x in after.splitlines()]
    touched: set[int] = set()
    inserted: list[list[str]] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag in ("replace", "delete"):
            touched.update(range(i1, i2))
        if tag in ("replace", "insert") and j2 > j1:
            inserted.append(b[j1:j2])
    return touched, inserted


def _udiff(before: str, after: str) -> str:
    return "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), "before", "after"))


def _assert_one_key_write(ctx: str, before_text: str, after_text: str, targets: set[tuple],
                          target_lines: set[int], *, allow_insert: bool) -> None:
    """Bytes outside the target key's line(s) are identical; a new key is ONE contiguous block."""
    touched, inserted = _line_diff(before_text, after_text)
    stray = touched - target_lines
    assert not stray, (
        f"[{ctx}] a write to {sorted(targets)} rewrote untouched lines "
        f"{sorted(stray)}:\n{_udiff(before_text, after_text)}")
    if not allow_insert:
        assert sum(len(b) for b in inserted) <= max(len(target_lines), 1), (
            f"[{ctx}] a value change inserted extra lines:\n{_udiff(before_text, after_text)}")
    else:
        assert len(inserted) <= 1, f"[{ctx}] new key split across blocks:\n{_udiff(before_text, after_text)}"
    _assert_no_duplicate_top_level(after_text, ctx)


# ─────────────────────────────────────────────────────────────────────────────
# Seeded config generator
# ─────────────────────────────────────────────────────────────────────────────

_YAML11_WORDS = {"y", "n", "yes", "no", "true", "false", "on", "off", "null", "~"}
_PLAIN_OK = re.compile(r"^[A-Za-z_][A-Za-z0-9_\-/]*$")
_TRICKY = (
    "off", "on", "yes", "No", "y", "null", "~", "True",
    "010", "1e3", "0x1F", "1_000", ".5", "+1", "12:30", "2026-09-23", "0o17", ".inf",
    "colon: space", "hash # not a comment", "*star", "&anchor", "!tag", "%pct", "@at", "`tick",
    "  leading space", "trailing space  ", "", "quote \" and ' both", "tab\there", "line1\nline2",
    "a\\b\\c", "C:\\Users\\me\\AppData", "{not: a map}", "[not, a, list]", "- dash", "? what",
)
_UNICODE = ("héllo wörld", "中文配置值", "emoji 😀 ok", "Ünïcödé-ß", "Ελληνικά", "日本語テキスト", "مرحبا")
_WORDS = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet")


def _render_scalar(v: Any, rng: random.Random) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    assert isinstance(v, str)
    if v.startswith("${") and v.endswith("}") and _PLAIN_OK.match(v[2:-1]):
        return v
    if _PLAIN_OK.match(v) and v.lower() not in _YAML11_WORDS:
        return v
    if "\n" in v or "\t" in v or "\\" in v or rng.random() < 0.5:
        return json.dumps(v, ensure_ascii=False)
    return "'" + v.replace("'", "''") + "'"


def _documented_scalar_leaves() -> list[tuple[tuple, Any]]:
    out = []

    def walk(d: dict, pre: tuple) -> None:
        for k, v in d.items():
            p = pre + (k,)
            if isinstance(v, dict):
                if v:
                    walk(v, p)
            elif isinstance(v, (bool, int, float, str)) or v is None:
                out.append((p, v))
    walk(DEFAULT_CONFIG, ())
    return out


DOCUMENTED = _documented_scalar_leaves()
# Generator leaves: nested (section.key[.sub]) documented scalars whose segments hold no literal dot
# and which the loader does not canonicalise into another path.
_GEN_EXCLUDE = {("agent", "max_turns"), ("model", "provider"), ("model", "base_url"), ("model", "api_mode")}
_GEN_LEAVES = [(p, d) for p, d in DOCUMENTED
               if 2 <= len(p) <= 3 and not any("." in s for s in p) and p not in _GEN_EXCLUDE
               and isinstance(DEFAULT_CONFIG.get(p[0]), dict)]


class Case:
    """A generated config: the text as a user would hand-write it plus the parse oracle."""

    def __init__(self, seed: int, text: str, tree: dict, leaf_lines: dict[tuple, int], env: dict[str, str]):
        self.seed, self.text, self.tree, self.leaf_lines, self.env = seed, text, tree, leaf_lines, env

    def lines_of(self, path: tuple) -> set[int]:
        return {self.leaf_lines[path]} if path in self.leaf_lines else set()


def _value_for(default: Any, rng: random.Random, *, long: bool, env: dict[str, str], seed: int) -> Any:
    roll = rng.random()
    if long and roll < 0.35:
        # #119844 shape: a fold point right after a backslash, plus plain long prose.
        return rng.choice([
            "A" * rng.randint(74, 85) + "D:\\CentBrowserPortable " + "B" * 40,
            " ".join(rng.choice(_WORDS) for _ in range(40)) + " \\scripts\\monitor-off-v3.ps1 end",
            "x" * 150 + " " + "C:\\Users\\me\\.hermes\\scripts\\" + "y" * 90,
        ])
    if roll < 0.08:
        return None
    if roll < 0.18:
        return copy.deepcopy(default) if not isinstance(default, (dict, list)) else None
    if isinstance(default, bool):
        return rng.random() < 0.5
    if isinstance(default, int):
        return rng.randint(0, 100000)
    if isinstance(default, float):
        return round(rng.uniform(0, 1000), 3)
    kind = rng.random()
    if kind < 0.35:
        return rng.choice(_TRICKY)
    if kind < 0.55:
        return rng.choice(_UNICODE) + " " + rng.choice(_WORDS)
    if kind < 0.65:
        name = f"C18_REF_{seed}_{len(env)}"
        env[name] = f"secret-{rng.randrange(10**8)}-{seed}"
        return "${" + name + "}"
    return f"{rng.choice(_WORDS)}-{rng.randrange(10**6)}"


def _before_version(text: str, block: str) -> str:
    """Insert ``block`` (newline-terminated) right before the root ``_config_version:`` line."""
    out, n = re.subn(r"^_config_version:", block + "_config_version:", text, count=1, flags=re.M)
    assert n == 1, text
    return out


def gen_case(seed: int, *, long: bool = False, n_sections: tuple = (5, 11), exclude_top: set = frozenset(),
             null_leaves: bool = True) -> Case:
    rng = random.Random(seed)
    env: dict[str, str] = {}
    by_section: dict[str, list] = {}
    for p, d in _GEN_LEAVES:
        if p[0] not in exclude_top:
            by_section.setdefault(p[0], []).append((p, d))
    sections = rng.sample(sorted(by_section), rng.randint(*n_sections))
    tree: dict = {}
    for sec in sections:
        for p, d in rng.sample(by_section[sec], min(len(by_section[sec]), rng.randint(1, 5))):
            if len(p) == 3 and not isinstance(_get(tree, p[:2], {}), dict):
                continue
            if len(p) == 2 and isinstance(_get(tree, p), dict):
                continue
            v = _value_for(d, rng, long=long, env=env, seed=seed)
            if v is None and not null_leaves:
                v = f"{rng.choice(_WORDS)}"
            _set(tree, p, v)
    if long:  # guarantee at least one scalar far past ruamel's default fold width
        sec = sections[0]
        leaf = next(p for p in _leaves(tree) if p[0] == sec)
        _set(tree, leaf, "A" * 76 + "D:\\CentBrowserPortable " + "B" * 60)
    if "providers" not in exclude_top:  # a provider name with a literal dot (#84064 family)
        tree["providers"] = {f"acme.v{seed % 7}": {"base_url": f"http://127.0.0.1:9/v{seed}", "api_mode": "chat_completions"}}
    if "c18_custom_root" not in exclude_top:
        tree["c18_custom_root"] = rng.choice(_UNICODE)
    if "skills" not in exclude_top and rng.random() < 0.6:  # a list the writer must keep element-wise
        tree.setdefault("skills", {})
        if isinstance(tree["skills"], dict):
            tree["skills"]["external_dirs"] = [f"/opt/c18/{rng.choice(_WORDS)}", rng.choice(_UNICODE)]
    tree["_config_version"] = LATEST

    lines: list[str] = [f"# C18 generated config (seed={seed}) — comments must survive every write"]
    leaf_lines: dict[tuple, int] = {}

    def emit(node: dict, depth: int, pre: tuple) -> None:
        for k, v in node.items():
            ind = "  " * depth
            key = k if _PLAIN_OK.match(k) else json.dumps(k)
            if depth == 0 and rng.random() < 0.3:
                lines.append(f"# section {k}: hand-written note ü")
            if isinstance(v, dict):
                lines.append(f"{ind}{key}:")
                emit(v, depth + 1, pre + (k,))
            elif isinstance(v, list):
                lines.append(f"{ind}{key}:")
                for item in v:
                    lines.append(f"{ind}  - {_render_scalar(item, rng)}")
            else:
                note = "  # inline note" if rng.random() < 0.25 and k != "_config_version" else ""
                leaf_lines[pre + (k,)] = len(lines)
                lines.append(f"{ind}{key}: {_render_scalar(v, rng)}{note}")
    emit(tree, 0, ())
    text = "\n".join(lines) + "\n"
    parsed = yaml.safe_load(text)
    assert parsed == tree, f"generator self-check failed for seed {seed}:\n{text}"
    return Case(seed, text, tree, leaf_lines, env)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / production seams
# ─────────────────────────────────────────────────────────────────────────────

def _cfg_path() -> Path:
    return C.get_config_path()


def _reset_config_caches(*, keep_lkg: bool = False) -> None:
    C._RAW_CONFIG_CACHE.clear()
    C._LOAD_CONFIG_CACHE.clear()
    if not keep_lkg:
        C._LAST_EXPANDED_CONFIG_BY_PATH.clear()
    with contextlib.suppress(Exception):
        from tui_gateway import server
        server._cfg_cache = server._cfg_sig = server._cfg_path = None


def _write_file(path: Path, text: str) -> None:
    """Atomic replace (new inode), like every Hermes writer, so no cache can serve stale bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.c18tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.fixture
def home(monkeypatch):
    """The per-test HERMES_HOME the repo conftest sandboxes; config caches reset around the test."""
    hh = Path(os.environ["HERMES_HOME"])
    hh.mkdir(parents=True, exist_ok=True)
    assert str(_cfg_path()).startswith(str(hh))
    _reset_config_caches()
    with contextlib.suppress(Exception):
        from tui_gateway import server
        monkeypatch.setattr(server, "_hermes_home", hh)
    yield hh
    _reset_config_caches()


def _install(case: Case, monkeypatch) -> Path:
    for k, v in case.env.items():
        monkeypatch.setenv(k, v)
    path = _cfg_path()
    _write_file(path, case.text)
    _reset_config_caches()
    return path


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


class ReadFaults:
    """Counts every read of config.yaml through the lowest-level YAML read seam
    (``hermes_cli.config.fast_safe_load`` — every config.yaml reader in hermes_cli.config,
    ``read_user_config_raw`` and therefore tui_gateway go through it) and fails chosen reads with
    a transient ``EMFILE`` on an otherwise intact file."""

    def __init__(self, monkeypatch, config_path: Path):
        self.path = os.path.abspath(str(config_path))
        self.count = 0
        self.fail_at: set[int] = set()
        real = C.fast_safe_load

        def wrapped(stream):
            name = getattr(stream, "name", None)
            if isinstance(name, str) and os.path.abspath(name) == self.path:
                self.count += 1
                if self.count in self.fail_at:
                    raise OSError(errno.EMFILE, "Too many open files (C18 injected)")
            return real(stream)
        monkeypatch.setattr(C, "fast_safe_load", wrapped)

    def arm(self, *ks: int) -> None:
        self.count, self.fail_at = 0, set(ks)


# ─────────────────────────────────────────────────────────────────────────────
# P1 — real load/save round-trip
# ─────────────────────────────────────────────────────────────────────────────

P1_SEEDS = list(range(101, 113))
P1_LONG_SEEDS = [201, 202, 203, 204]
_XFAIL_119844 = pytest.mark.xfail(
    strict=True,
    reason="#119844 (open): ruamel round-trip writer has no `width`, so every save re-folds long "
           "scalars (untouched keys change bytes) and a fold after a backslash inserts a space into "
           "the stored value. Fix: `yaml_rt.width = 2**31 - 1` in utils._roundtrip_load.")


def _noop_save_roundtrip(case: Case, monkeypatch) -> None:
    path = _install(case, monkeypatch)
    loaded = C.load_config()
    for p, v in _leaves(case.tree).items():  # ${VAR} expansion is the only load-time rewrite
        want = case.env.get(v[2:-1], v) if isinstance(v, str) and v.startswith("${") else v
        assert _get(loaded, p) == want, f"[seed={case.seed}] load_config()[{p}] = {_get(loaded, p)!r}, file says {v!r}"
    with _quiet():
        C.save_config(loaded)
    after = _read(path)
    assert yaml.safe_load(after) == case.tree, (
        f"[seed={case.seed}] load(save(x)) != x:\n{_udiff(case.text, after)}")
    touched, inserted = _line_diff(case.text, after)  # byte-identical modulo `k: null` -> `k:` (cosmetic)
    assert not touched and not inserted, f"[seed={case.seed}] a no-op save changed bytes:\n{_udiff(case.text, after)}"
    for name, secret in case.env.items():
        assert secret not in after, f"[seed={case.seed}] expanded ${{{name}}} leaked into config.yaml"


_XFAIL_NULL_STRIP = pytest.mark.xfail(
    strict=True,
    reason="NEW: save_config's _strip_default_values uses None as its 'drop me' sentinel, so an explicit "
           "`key: null` is deleted on every save even when preserved as user-set (display.skin: null "
           "silently becomes 'default'), and a section emptied that way takes its comments with it.")


@pytest.mark.parametrize("seed", P1_SEEDS)
def test_p1_noop_save_is_byte_identical(seed, home, monkeypatch):
    _noop_save_roundtrip(gen_case(seed, null_leaves=False), monkeypatch)


@pytest.mark.parametrize("seed", [pytest.param(s, marks=_XFAIL_NULL_STRIP) for s in (101, 102)])
def test_p1_explicit_null_leaves_survive_a_noop_save(seed, home, monkeypatch):
    case = gen_case(seed)
    assert any(v is None for v in _leaves(case.tree).values()), f"seed {seed} generated no null leaf"
    _noop_save_roundtrip(case, monkeypatch)


@pytest.mark.parametrize("seed", [pytest.param(s, marks=_XFAIL_119844) for s in P1_LONG_SEEDS])
def test_p1_long_scalars_survive_a_noop_save(seed, home, monkeypatch):
    _noop_save_roundtrip(gen_case(seed, long=True, null_leaves=False), monkeypatch)


@pytest.mark.parametrize("seed", P1_SEEDS)
def test_p1_save_of_one_changed_key_touches_only_that_key(seed, home, monkeypatch):
    case = gen_case(seed, null_leaves=False)
    rng = random.Random(seed * 7919)
    present = [p for p in case.leaf_lines if p[0] not in ("_config_version", "providers")]
    absent = [p for p, _d in _GEN_LEAVES if _get(case.tree, p) is _MISSING
              and not _under(p, [q for q in _leaves(case.tree) if not isinstance(_get(case.tree, q), dict)])]
    for round_ in range(4):
        path = _install(case, monkeypatch)
        target = rng.choice(present) if round_ % 2 == 0 else rng.choice(absent)
        old = _get(case.tree, target)
        new = old
        # save_config by contract never writes a default-valued key the user did not already have.
        dflt = _get(DEFAULT_CONFIG, target) if old is _MISSING else _MISSING
        while new == old or new == dflt or (isinstance(new, str) and new.startswith("${")):
            new = _value_for(_get(DEFAULT_CONFIG, target), rng, long=False, env={}, seed=seed)
            if new is None:
                new = f"c18-{rng.randrange(10**6)}"
        cfg = C.load_config()
        _set(cfg, target, new)
        with _quiet():
            C.save_config(cfg)
        after = _read(path)
        ctx = f"seed={seed} round={round_} target={target} new={new!r}"
        expect = copy.deepcopy(case.tree)
        _set(expect, target, new)
        got = yaml.safe_load(after)
        assert got == expect, (
            f"[{ctx}] changed paths {sorted(_changed_paths(expect, got), key=str)}:\n{_udiff(case.text, after)}")
        _assert_one_key_write(ctx, case.text, after, {target}, case.lines_of(target),
                              allow_insert=target not in case.leaf_lines)


# ─────────────────────────────────────────────────────────────────────────────
# P2 — "set one key" changes exactly one key, on every writer
# ─────────────────────────────────────────────────────────────────────────────

# Documented side effects that legitimately touch a second key or another file.
_SET_EXCLUDE = {
    ("model",),                       # bare `model` is redirected to model.default by design
    ("model", "provider"),            # provider switch drops the old provider's base_url/api_mode
    ("model", "api_base"),            # alias rewritten to model.base_url
    ("_config_version",),
}


def _cli_key(path: tuple) -> str:
    return ".".join(seg.replace(".", "\\.") for seg in path)


def _set_value_for(path: tuple, default: Any, current: Any, rng: random.Random) -> tuple[str, Any]:
    """(CLI string, the value it must persist as). Only unambiguous encodings."""
    if isinstance(default, bool):
        v = not current if isinstance(current, bool) else not default
        return ("true" if v else "false"), v
    if isinstance(default, int):
        v = rng.randint(1, 10**6)
        while v == current:
            v += 1
        return str(v), v
    if isinstance(default, float):
        v = round(rng.uniform(1, 999), 2) + 0.25
        return repr(v), v
    v = f"c18-{rng.choice(_WORDS)}-{rng.randrange(10**6)}"
    return v, v


_SET_LEAVES = [(p, d) for p, d in DOCUMENTED if p not in _SET_EXCLUDE]
_N_SET_CHUNKS = 8


@pytest.mark.parametrize("chunk", range(_N_SET_CHUNKS))
def test_p2_every_documented_key_is_settable_and_changes_exactly_one_key(chunk, home, monkeypatch):
    """`config set` over EVERY documented scalar leaf (chunked), on a rich generated config."""
    case = gen_case(300 + chunk)
    rng = random.Random(300 + chunk)
    failures = []
    for path, default in _SET_LEAVES[chunk::_N_SET_CHUNKS]:
        if C._is_env_config_key(_cli_key(path)):
            continue
        cfg_path = _install(case, monkeypatch)
        current = _get(case.tree, path, None)
        if isinstance(current, (dict, list)):
            continue
        blocked = [q for q in _leaves(case.tree) if len(q) < len(path) and path[:len(q)] == q]
        if blocked:  # a scalar sits where this key needs a mapping: the guard refuses by design
            continue
        s, want = _set_value_for(path, default, current, rng)
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                C.set_config_value(_cli_key(path), s)
        except SystemExit:
            failures.append(f"{path}: refused: {out.getvalue().strip()[:200]}")
            continue
        after = _read(cfg_path)
        got = yaml.safe_load(after)
        changed = _changed_paths(case.tree, got)
        if changed != {path} or _get(got, path) != want or type(_get(got, path)) is not type(want):
            failures.append(f"{path}={s!r}: changed {sorted(changed, key=str)[:6]}, stored {_get(got, path)!r}")
            continue
        try:
            _assert_one_key_write(f"chunk={chunk} {path}", case.text, after, {path}, case.lines_of(path),
                                  allow_insert=path not in case.leaf_lines)
        except AssertionError as exc:
            failures.append(str(exc)[:600])
    assert not failures, f"[chunk={chunk} seed={case.seed}] {len(failures)} documented key(s) failed:\n" + "\n".join(failures)


def _dotted_provider(case: Case) -> str:
    return next(iter(case.tree["providers"]))


def _cli(env: dict, *args: str, timeout: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "hermes_cli.main", *args], cwd=str(WORKTREE), env=env,
                          capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)


def _cli_home(tmp_path: Path, case: Case) -> tuple[dict, Path]:
    env = isolated_env(tmp_path / "cli", pythonpath=WORKTREE, extra=case.env)
    cfg = Path(env["HERMES_HOME"]) / "config.yaml"
    _write_file(cfg, case.text)
    return env, cfg


_CLI_CASES = [
    # (id, seed, key-builder, value, expected persisted value)
    ("documented-existing", 401, lambda c: _cli_key(next(p for p in c.leaf_lines if p[0] != "providers" and p[0] != "_config_version")), "c18-cli-value", "c18-cli-value"),
    ("documented-absent-float", 402, lambda c: "compression.threshold", "0.61", 0.61),
    ("documented-absent-bool", 408, lambda c: "compression.enabled", "false", False),
    ("dotted-provider-name", 403, lambda c: f"providers.{_dotted_provider(c)}.api_mode", "codex_responses", "codex_responses"),
    ("yaml11-word-stays-a-string", 404, lambda c: "approvals.mode", "off", "off"),
    ("unicode-value", 405, lambda c: "agent.system_prompt", "Réponds en français — 日本語 😀", "Réponds en français — 日本語 😀"),
    ("custom-top-level-key", 406, lambda c: "c18_custom_root", "changed ü", "changed ü"),
]


@pytest.mark.parametrize("cid,seed,keyf,value,want", _CLI_CASES, ids=[c[0] for c in _CLI_CASES])
def test_p2_real_cli_config_set_changes_exactly_one_key(cid, seed, keyf, value, want, tmp_path):
    case = gen_case(seed, exclude_top={"compression"})
    env, cfg = _cli_home(tmp_path, case)
    key = keyf(case)
    path = tuple(C._split_key_path(key))
    if cid == "dotted-provider-name":  # greedy literal match: the existing "acme.vN" entry
        path = ("providers", _dotted_provider(case), "api_mode")
    r = _cli(env, "config", "set", key, value)
    assert r.returncode == 0, f"[{cid} seed={seed}] `config set {key} {value}` exit {r.returncode}:\n{r.stdout}\n{r.stderr}"
    after = _read(cfg)
    got = yaml.safe_load(after)
    changed = _changed_paths(case.tree, got)
    assert changed == {path}, f"[{cid} seed={seed}] changed {sorted(changed, key=str)}:\n{_udiff(case.text, after)}"
    assert _get(got, path) == want and type(_get(got, path)) is type(want), (
        f"[{cid}] stored {_get(got, path)!r}, want {want!r}")
    _assert_one_key_write(f"{cid} seed={seed}", case.text, after, {path}, case.lines_of(path),
                          allow_insert=path not in case.leaf_lines)
    # A dotted provider name must stay ONE literal key, never a phantom nested sibling.
    assert list(got["providers"]) == list(case.tree["providers"]), f"[{cid}] providers keys: {list(got['providers'])}"


def test_p2_real_cli_env_setting_changes_exactly_one_env_line_and_no_config_bytes(tmp_path):
    case = gen_case(407)
    env, cfg = _cli_home(tmp_path, case)
    env_file = cfg.with_name(".env")
    env_before = "# hand-written\nC18_KEEP=one\nexport C18_OTHER=\"two # not comment\"\nTELEGRAM_HOME_CHANNEL=5\n"
    env_file.write_text(env_before, encoding="utf-8")
    r = _cli(env, "config", "set", "TELEGRAM_HOME_CHANNEL", "4242")
    assert r.returncode == 0, r.stdout + r.stderr
    assert _read(cfg) == case.text, "an env-routed `config set` rewrote config.yaml"
    touched, inserted = _line_diff(env_before, _read(env_file))
    assert touched == {3} and sum(map(len, inserted)) == 1, _udiff(env_before, _read(env_file))
    assert "TELEGRAM_HOME_CHANNEL=4242" in _read(env_file).splitlines()


_XFAIL_119928 = pytest.mark.xfail(
    strict=True,
    reason="#119928 (open): when the managed-scope .env pins a key, `hermes config set` prints the "
           "refusal, then '✓ Set', exits 0 — and the credential route still rewrites config.yaml.")


@pytest.mark.parametrize("key", [pytest.param("DEEPSEEK_API_KEY", marks=_XFAIL_119928),
                                 pytest.param("TELEGRAM_HOME_CHANNEL", marks=_XFAIL_119928)])
def test_p2_env_lock_refusal_is_not_reported_as_success(key, tmp_path):
    """exit 0 ⇔ the write happened; a refused key leaves .env AND config.yaml byte-identical."""
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / ".env").write_text("DEEPSEEK_API_KEY=sk-admin\nTELEGRAM_HOME_CHANNEL=111\n", encoding="utf-8")
    env = isolated_env(tmp_path / "cli", pythonpath=WORKTREE, extra={"HERMES_MANAGED_DIR": str(managed)})
    hh = Path(env["HERMES_HOME"])
    (hh / ".env").write_text("DEEPSEEK_API_KEY=sk-old\nTELEGRAM_HOME_CHANNEL=5\n", encoding="utf-8")
    cfg_text = "model:\n  default: deepseek-chat\n  api_key: sk-old\n"
    _write_file(hh / "config.yaml", cfg_text)
    env_before = _read(hh / ".env")
    r = _cli(env, "config", "set", key, "c18-new")
    wrote = _read(hh / ".env") != env_before
    assert (r.returncode == 0) == wrote, (
        f"`config set {key}` exit={r.returncode} but .env {'changed' if wrote else 'unchanged'}:\n{r.stdout}{r.stderr}")
    if not wrote:
        assert _read(hh / "config.yaml") == cfg_text, "a refused env write still rewrote config.yaml"


# ── real tui_gateway stdio JSON-RPC process ──────────────────────────────────

class _RpcProc:
    def __init__(self, root: Path):
        self.root = root
        self.env = isolated_env(root, pythonpath=WORKTREE)
        self.home = Path(self.env["HERMES_HOME"])
        _write_file(self.home / "config.yaml", "model_catalog:\n  enabled: false\n")
        self.stderr = open(root / "tui_gateway.stderr", "w", encoding="utf-8")
        self.proc = subprocess.Popen([sys.executable, "-m", "tui_gateway.entry"], cwd=str(WORKTREE), env=self.env,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
                                     text=True, encoding="utf-8", bufsize=1)
        self.q: queue.Queue = queue.Queue()
        self._rid = 100
        threading.Thread(target=self._pump, daemon=True).start()
        self.wait_event("gateway.ready", timeout=120)

    def _pump(self) -> None:
        for line in self.proc.stdout:
            with contextlib.suppress(ValueError):
                self.q.put(json.loads(line))
        self.q.put(None)

    def _next(self, deadline: float) -> dict:
        try:
            msg = self.q.get(timeout=max(0.05, deadline - time.monotonic()))
        except queue.Empty:
            raise AssertionError(f"tui_gateway timed out; stderr:\n{(self.root / 'tui_gateway.stderr').read_text()[-3000:]}")
        if msg is None:
            raise AssertionError(f"tui_gateway exited rc={self.proc.poll()}; stderr:\n"
                                 f"{(self.root / 'tui_gateway.stderr').read_text()[-3000:]}")
        return msg

    def wait_event(self, etype: str, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            msg = self._next(deadline)
            if msg.get("method") == "event" and (msg.get("params") or {}).get("type") == etype:
                return msg

    def call(self, method: str, params: dict, timeout: float = 60) -> dict:
        self._rid += 1
        rid = self._rid
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params}) + "\n")
        self.proc.stdin.flush()
        deadline = time.monotonic() + timeout
        while True:
            msg = self._next(deadline)
            if msg.get("id") == rid:
                return msg

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=15)
        self.stderr.close()


@pytest.fixture(scope="module")
def rpc(tmp_path_factory):
    p = _RpcProc(tmp_path_factory.mktemp("c18_tui_gateway"))
    yield p
    p.close()


# (rpc key, candidate values, config path it owns, value transform to the persisted form)
_RPC_CASES = [
    ("display.in_app_tips", ["false", "true"], ("display", "in_app_tips"), {"false": False, "true": True}),
    ("skin", ["mono", "ares", "slate"], ("display", "skin"), None),
    ("approvals.mode", ["off", "smart", "manual"], ("approvals", "mode"), None),
    ("theme", ["dark", "light", "auto"], ("display", "tui_theme"), None),
    ("busy", ["steer", "queue", "interrupt"], ("display", "busy_input_mode"), None),
    ("voice.voice_chat_mode", ["gpt-live", "chained"], ("voice", "voice_chat_mode"), None),
    ("density", ["on", "off"], ("display", "tui_compact"), {"on": True, "off": False}),
    ("statusbar", ["bottom", "top", "off"], ("display", "tui_statusbar"), None),
    ("mouse", ["wheel", "buttons", "off"], ("display", "mouse_tracking"), None),
    ("verbose", ["all", "new", "off"], ("display", "tool_progress"), None),
    ("prompt", ["Réponds brièvement 😀", "be terse"], ("custom_prompt",), None),
]


@pytest.mark.parametrize("key,values,path,xf", _RPC_CASES, ids=[c[0] for c in _RPC_CASES])
def test_p2_real_tui_gateway_config_set_changes_exactly_one_key(key, values, path, xf, rpc):
    seed = 500 + _RPC_CASES.index(next(c for c in _RPC_CASES if c[0] == key))
    case = gen_case(seed, exclude_top={"model_catalog"})
    text = _before_version(case.text, "model_catalog:\n  enabled: false\n")
    tree = yaml.safe_load(text)
    cfg = rpc.home / "config.yaml"
    _write_file(cfg, text)
    current = _get(tree, path, None)
    value = next(v for v in values if (xf or {}).get(v, v) != current)
    want = (xf or {}).get(value, value)
    resp = rpc.call("config.set", {"key": key, "value": value})
    assert "result" in resp, f"[{key} seed={seed}] config.set failed: {resp}"
    after = _read(cfg)
    got = yaml.safe_load(after)
    changed = _changed_paths(tree, got)
    assert changed == {path}, f"[{key} seed={seed}] changed {sorted(changed, key=str)}:\n{_udiff(text, after)}"
    assert _get(got, path) == want and type(_get(got, path)) is type(want), f"[{key}] stored {_get(got, path)!r}"
    tl = {i for i, ln in enumerate(text.splitlines()) if path in case.leaf_lines and i == case.leaf_lines[path]}
    _assert_one_key_write(f"rpc {key} seed={seed}", text, after, {path}, tl, allow_insert=not tl)


# ── dashboard / Desktop PUT /api/config ──────────────────────────────────────

@pytest.fixture(scope="module")
def web_app():
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("starlette not installed")
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app
    client = TestClient(app)  # no lifespan: requests only; profile/home resolved per request
    client.headers.update({_SESSION_HEADER_NAME: _SESSION_TOKEN})
    return client


def _client(web_app):
    return web_app


@pytest.mark.parametrize("seed", [601, 602, 603, 604])
def test_p2_dashboard_get_put_roundtrip_and_partial_put(seed, web_app, home, monkeypatch):
    """Settings page open→save with no edits leaves the file byte-identical; a Desktop-style
    partial PUT (``setNested({}, key, value)``) changes exactly that key."""
    _dashboard_roundtrip(gen_case(seed, null_leaves=False), web_app, monkeypatch)


@pytest.mark.xfail(strict=True, reason=(
    "NEW: save_config -> _normalize_max_turns_config always sets config['agent'] (even to {}), so "
    "any save of a sparse raw document lacking `agent:` (Desktop partial PUT, every migration "
    "persist) appends a phantom `agent: {}` section to config.yaml."))
def test_p2_dashboard_partial_put_adds_no_phantom_section(web_app, home, monkeypatch):
    case = gen_case(606, null_leaves=False, exclude_top={"agent", "display"})
    cfg = _install(case, monkeypatch)
    r = _client(web_app).put("/api/config", json={"config": {"display": {"skin": "ares"}}})
    assert r.status_code == 200, r.text
    got = yaml.safe_load(_read(cfg))
    assert set(got) == set(case.tree) | {"display"}, (
        f"PUT display.skin added sections {sorted(set(got) - set(case.tree) - {'display'})}:\n{_udiff(case.text, _read(cfg))}")


@pytest.mark.parametrize("seed", [pytest.param(605, marks=_XFAIL_NULL_STRIP)])
def test_p2_dashboard_noop_put_keeps_explicit_nulls(seed, web_app, home, monkeypatch):
    case = gen_case(seed)
    assert any(v is None for v in _leaves(case.tree).values())
    _dashboard_roundtrip(case, web_app, monkeypatch)


def _dashboard_roundtrip(case: Case, web_app, monkeypatch) -> None:
    seed = case.seed
    if "agent" not in case.tree:  # see test_p2_dashboard_partial_put_adds_no_phantom_section
        text = _before_version(case.text, "agent:\n  verify_on_stop: false\n")
        case = Case(seed, text, yaml.safe_load(text), case.leaf_lines, case.env)
    cfg = _install(case, monkeypatch)
    client = _client(web_app)
    body = client.get("/api/config").json()
    r = client.put("/api/config", json={"config": body})
    assert r.status_code == 200, r.text
    touched, inserted = _line_diff(case.text, _read(cfg))  # modulo cosmetic `k: null` -> `k:`
    assert not touched and not inserted, (
        f"[seed={seed}] GET→PUT with no edits changed the file:\n{_udiff(case.text, _read(cfg))}")
    rng = random.Random(seed)
    for target in rng.sample([p for p in case.leaf_lines if p[0] not in ("_config_version", "providers", "c18_custom_root")], 2) + [("display", "resume_last_session")]:
        _install(case, monkeypatch)
        cur = _get(case.tree, target, _get(DEFAULT_CONFIG, target, None))
        # a partial PUT of a key the user never set writes it only when it differs from the default
        new = (not cur) if isinstance(cur, bool) else f"c18-dash-{rng.randrange(10**6)}"
        partial: dict = {}
        _set(partial, target, new)
        r = client.put("/api/config", json={"config": partial})
        assert r.status_code == 200, r.text
        after = _read(cfg)
        got = yaml.safe_load(after)
        changed = _changed_paths(case.tree, got)
        assert changed == {target}, f"[seed={seed} {target}] changed {sorted(changed, key=str)}:\n{_udiff(case.text, after)}"
        _assert_one_key_write(f"dashboard seed={seed} {target}", case.text, after, {target}, case.lines_of(target),
                              allow_insert=target not in case.leaf_lines)


# ─────────────────────────────────────────────────────────────────────────────
# P3 — a failed read never results in a clobbering write
# ─────────────────────────────────────────────────────────────────────────────

_P3_TARGET = ("display", "skin")


def _op_cli_set(ctx: dict) -> None:
    C.set_config_value("display.skin", ctx["new"])


def _op_save(ctx: dict) -> None:
    cfg = C.load_config()
    _set(cfg, _P3_TARGET, ctx["new"])
    C.save_config(cfg)


def _op_tui(ctx: dict) -> None:
    from tui_gateway import server
    resp = server._methods["config.set"](1, {"key": "skin", "value": ctx["new"]})
    if "error" in resp:
        raise RuntimeError(resp["error"])


def _op_dashboard(ctx: dict) -> None:
    r = ctx["client"].put("/api/config", json={"config": {"display": {"skin": ctx["new"]}}})
    if r.status_code != 200:
        raise RuntimeError(r.text)


def _op_migrate(ctx: dict) -> None:
    C.migrate_config(interactive=False, quiet=True)


_P3_OPS: dict[str, tuple[Callable[[dict], None], set]] = {
    "cli_set": (_op_cli_set, {_P3_TARGET}),
    "save_config": (_op_save, {_P3_TARGET}),
    "tui_rpc": (_op_tui, {_P3_TARGET}),
    "dashboard_put": (_op_dashboard, {_P3_TARGET}),
    # v35 → latest: delegation.max_iterations 50 is the old default the v36 step rewrites.
    "migrate": (_op_migrate, {("_config_version",), ("delegation", "max_iterations")}),
}

_XFAIL_TRANSIENT = {
    "tui_rpc": "NEW (#113301 class, TUI/Desktop surface): tui_gateway.server._load_cfg_raw swallows any "
               "read exception into {} and _write_config_key saves that through the full-document "
               "writer, which deletes every section the caller never saw.",
    "dashboard_put": "NEW (#113301 class, dashboard/Desktop surface): PUT /api/config merges the body "
                     "over read_raw_config(), whose swallowed-error path returns {}; Desktop PUTs "
                     "single-key bodies, so one transient read error collapses config.yaml to that key.",
}


def _p3_case(seed: int) -> Case:
    case = gen_case(seed, null_leaves=False, exclude_top={"display", "delegation", "agent"})
    text = _before_version(case.text, "display:\n  skin: mono\n  compact: true\ndelegation:\n  max_iterations: 50\n"
                                      "agent:\n  verify_on_stop: false\n")
    tree = yaml.safe_load(text)
    return Case(seed, text, tree, {}, case.env)


def _p3_run(op_name: str, ctx: dict, cfg: Path, text: str, version: int | None) -> tuple[str, str]:
    body = text if version is None else re.sub(r"^_config_version: \d+$", f"_config_version: {version}", text, flags=re.M)
    _write_file(cfg, body)
    _reset_config_caches(keep_lkg=False)
    outcome = "ok"
    try:
        with _quiet():
            _P3_OPS[op_name][0](ctx)
    except (Exception, SystemExit) as exc:  # a refusal is fine; a clobber is not
        outcome = f"refused: {type(exc).__name__}"
    return body, outcome


_MULTI_STEP_OPS = {"migrate"}  # each migration step persists on its own; a later refusal keeps earlier steps


def _p3_check(label: str, before_text: str, after_text: str, targets: set, outcome: str) -> str | None:
    before, after = yaml.safe_load(before_text), yaml.safe_load(after_text)
    if not isinstance(after, dict):
        return f"{label}: file no longer a mapping ({outcome})"
    stray = {p for p in _changed_paths(before, after) if not _under(p, targets)}
    if stray:
        lost = sorted(set(before) - set(after))
        return (f"{label} ({outcome}): clobbered {len(stray)} untargeted path(s) "
                f"{sorted(stray, key=str)[:5]}; lost sections {lost}")
    if outcome != "ok" and after_text != before_text and label.split()[0] not in _MULTI_STEP_OPS:
        return f"{label}: refused ({outcome}) but still rewrote the file"
    return None


def _p3_params():
    out = []
    for op in _P3_OPS:
        marks = [pytest.mark.xfail(strict=True, reason=_XFAIL_TRANSIENT[op])] if op in _XFAIL_TRANSIENT else []
        out.append(pytest.param(op, marks=marks, id=op))
    return out


@pytest.mark.parametrize("op_name", _p3_params())
def test_p3_a_transient_error_on_any_single_read_never_clobbers(op_name, web_app, home, monkeypatch):
    """Dry-run the operation to count its config reads R, then for k in 1..R fail exactly read k."""
    case = _p3_case(700)
    for k, v in case.env.items():
        monkeypatch.setenv(k, v)
    cfg = _cfg_path()
    version = 35 if op_name == "migrate" else None
    ctx = {"new": "ares", "client": _client(web_app)}
    # A normal install has loaded its config before: prime the last-known-good backup.
    _write_file(cfg, case.text)
    _reset_config_caches()
    C.load_config()
    faults = ReadFaults(monkeypatch, cfg)
    faults.arm()
    before, outcome = _p3_run(op_name, ctx, cfg, case.text, version)
    reads = faults.count
    clean_err = _p3_check(f"{op_name} clean run", before, _read(cfg), _P3_OPS[op_name][1], outcome)
    assert clean_err is None and outcome == "ok", clean_err or f"{op_name} clean run {outcome}"
    assert reads >= 1, f"{op_name} performed no config read through the read seam"
    problems = []
    for k in range(1, reads + 1):
        faults.arm(k)
        before, outcome = _p3_run(op_name, ctx, cfg, case.text, version)
        err = _p3_check(f"{op_name} EMFILE on read {k}/{reads}", before, _read(cfg), _P3_OPS[op_name][1], outcome)
        if err:
            problems.append(err)
    assert not problems, "\n".join(problems)


def _truncate_mid_scalar(text: str) -> str:
    idx = text.index('"', text.index("\n", 10))  # first double-quoted scalar after the header
    return text[: idx + 5]


_PERSISTENT_FAULTS = {
    "unreadable": None,
    "truncated_mid_write": _truncate_mid_scalar,
    "non_mapping_root": lambda _t: "- c18\n- not a mapping\n",
}


@pytest.mark.parametrize("fault", sorted(_PERSISTENT_FAULTS))
@pytest.mark.parametrize("op_name", list(_P3_OPS))
def test_p3_unreadable_or_partial_file_is_never_rewritten(op_name, fault, web_app, home, monkeypatch):
    if fault == "unreadable" and os.geteuid() == 0:
        pytest.skip("root reads mode-000 files")
    case = gen_case(710, null_leaves=False, exclude_top={"display"})
    cfg = _cfg_path()
    ctx = {"new": "ares", "client": _client(web_app)}
    for k, v in case.env.items():
        monkeypatch.setenv(k, v)
    text = case.text if fault == "unreadable" else _PERSISTENT_FAULTS[fault](case.text)
    if fault == "truncated_mid_write":
        with pytest.raises(yaml.YAMLError):
            yaml.safe_load(text)
    _write_file(cfg, text)
    _reset_config_caches()
    if fault == "unreadable":
        cfg.chmod(0)
    try:
        with _quiet():
            with contextlib.suppress(Exception, SystemExit):
                _P3_OPS[op_name][0](ctx)
    finally:
        cfg.chmod(0o600)
    assert _read(cfg) == text, f"[{op_name}/{fault}] an unreadable/partial config.yaml was rewritten:\n{_udiff(text, _read(cfg))}"


# ─────────────────────────────────────────────────────────────────────────────
# P4 — .env loaders are idempotent; the .env writer changes exactly one line
# ─────────────────────────────────────────────────────────────────────────────

P4_SEEDS = [801, 802, 803, 804, 805, 806]
_BASE_PATH = "/usr/bin:/bin"


def gen_dotenv(seed: int) -> tuple[str, list[str], dict[str, str]]:
    """(.env text, keys it defines, shell exports present before the first load)."""
    rng = random.Random(seed)
    p = f"C18_{seed}_"
    shell = {f"{p}SELF": "shellval", f"{p}SHADOW": "from-shell"}
    entries = [
        f"{p}A=plain-{rng.choice(_WORDS)}",
        f'export {p}B="double quoted # not a comment {rng.choice(_UNICODE)}"',
        f"{p}C='single ${{NOT_EXPANDED}} $HOME'",
        f"{p}D=${{{p}A}}-chained",
        f"{p}E=${{C18_UNDEFINED_{seed}}}",
        f"PATH=/c18/{seed}/bin:${{PATH}}",
        f"{p}SELF=pre-${{{p}SELF}}",
        f"{p}SP = spaced value",
        f"{p}U={rng.choice(_UNICODE)}",
        f"{p}EQ=a=b=c",
        f"{p}SHADOW=from-dotenv",
        f'{p}ESC="tab\\tnewline\\n"',
        f"{p}CHAIN2=${{{p}D}}/${{{p}U}}",
    ]
    rng.shuffle(entries)
    # dependency order for references (dotenv resolves top-down)
    order = {f"{p}A": 0, f"{p}U": 0, f"{p}D": 1, f"{p}CHAIN2": 2}
    entries.sort(key=lambda e: order.get(e.split("=", 1)[0].replace("export ", "").strip(), 0))
    lines = ["# C18 generated .env", ""]
    for e in entries:
        if rng.random() < 0.2:
            lines.append("# note")
        lines.append(e)
    keys = sorted({e.split("=", 1)[0].replace("export ", "").strip() for e in entries})
    return "\n".join(lines) + "\n", keys, shell


@pytest.fixture
def env_restore(monkeypatch):
    saved = dict(os.environ)
    monkeypatch.setenv("PATH", _BASE_PATH)
    monkeypatch.setenv("HERMES_MULTIPLEX_PROFILES", "0")
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.mark.parametrize("seed", P4_SEEDS)
def test_p4_load_hermes_dotenv_is_idempotent(seed, home, env_restore, monkeypatch):
    from hermes_cli.env_loader import load_hermes_dotenv
    text, keys, shell = gen_dotenv(seed)
    for k, v in shell.items():
        monkeypatch.setenv(k, v)
    (home / ".env").write_text(text, encoding="utf-8")
    snaps = []
    for _ in range(4):
        load_hermes_dotenv(hermes_home=home, load_external_secrets=False)
        snaps.append({k: os.environ.get(k) for k in keys})
    assert all(s == snaps[0] for s in snaps), f"[seed={seed}] env drifted across reloads:\n" + "\n".join(
        f"{k}: {[s[k] for s in snaps]}" for k in keys if len({s[k] for s in snaps}) > 1)
    assert snaps[0]["PATH"].count(f"/c18/{seed}/bin") == 1, f"[seed={seed}] PATH={snaps[0]['PATH']}"
    assert snaps[0]["PATH"].endswith(_BASE_PATH), f"[seed={seed}] PATH lost the boot value: {snaps[0]['PATH']}"
    assert snaps[0][f"C18_{seed}_SELF"] == "pre-shellval", f"[seed={seed}] self-ref: {snaps[0][f'C18_{seed}_SELF']}"
    assert snaps[0][f"C18_{seed}_D"] == snaps[0][f"C18_{seed}_A"] + "-chained"


@pytest.mark.parametrize("seed", P4_SEEDS)
def test_p4_env_parser_sanitizer_and_writer_round_trip(seed, home, env_restore):
    text, keys, _shell = gen_dotenv(seed)
    env_path = home / ".env"
    env_path.write_text(text, encoding="utf-8")
    C.invalidate_env_cache()
    first = C.load_env()
    C.invalidate_env_cache()
    assert C.load_env() == first, f"[seed={seed}] load_env() is not a pure function of the file"
    with _quiet():
        C.sanitize_env_file()
        sanitized = _read(env_path)
        assert C.sanitize_env_file() == 0 and _read(env_path) == sanitized, f"[seed={seed}] sanitize is not a fixed point"
    C.invalidate_env_cache()
    parsed = C.load_env()
    target = f"C18_{seed}_A"
    with _quiet():
        C.save_env_value(target, "rewritten value # with hash")
    C.invalidate_env_cache()
    after = C.load_env()
    changed = {k for k in parsed.keys() | after.keys() if parsed.get(k) != after.get(k)}
    assert changed == {target}, f"[seed={seed}] save_env_value({target}) changed {sorted(changed)}"
    assert after[target] == "rewritten value # with hash"
    touched, inserted = _line_diff(sanitized, _read(env_path))
    assert len(touched) == 1 and sum(map(len, inserted)) == 1, f"[seed={seed}]\n{_udiff(sanitized, _read(env_path))}"


# ─────────────────────────────────────────────────────────────────────────────
# P5 — migrations are idempotent from every historical version
# ─────────────────────────────────────────────────────────────────────────────

def _migration_start_versions() -> list[int | None]:
    from hermes_cli.config_migrations import MIGRATIONS, SUPPORT_FLOOR_VERSION
    starts = sorted({v - 1 for v, _fn in MIGRATIONS if v - 1 >= SUPPORT_FLOOR_VERSION})
    return [None, *starts]  # None = a hand-written config with no _config_version (full ladder)


# Stale-default rewrites: (path, old default the migration rewrites).
_STALE = {
    ("model_catalog", "ttl_hours"): 24,
    ("agent", "verify_on_stop"): "auto",
    ("display", "background_process_notifications"): "all",
    ("delegation", "max_iterations"): 50,
    ("delegation", "max_concurrent_children"): 3,
    ("curator", "stale_after_days"): 30,
    ("curator", "archive_after_days"): 90,
}
_CANARIES = {
    ("model", "default"): "c18/canary-model",
    ("terminal", "timeout"): 4321,
    ("c18_custom_root",): "keep me — ü 😀",
    ("approvals", "mode"): "off",
    ("agent", "system_prompt"): "Réponds brièvement.",
}


def gen_legacy(version: int | None, seed: int) -> tuple[str, dict[tuple, Any]]:
    rng = random.Random(seed)
    tree: dict = {}
    customized: dict[tuple, Any] = {}
    for path, old in _STALE.items():
        if rng.random() < 0.5:
            _set(tree, path, old)
        else:
            val = {int: 77, str: "c18-custom"}[type(old)]
            _set(tree, path, val)
            customized[path] = val
    for path, val in _CANARIES.items():
        _set(tree, path, val)
    structural = [
        (("stt",), {"provider": "local", "model": "base"}),
        (("display", "tool_progress_overrides"), {"telegram": "all"}),
        (("compression", "summary_model"), "c18/summary"),
        (("plugins", "disabled"), ["c18-off"]),
        (("memory", "write_mode"), "approve"),
        (("delegation", "max_async_children"), 5),
        (("display", "personality"), "kawaii"),
        (("platform_toolsets",), {"cli": ["terminal", "bfl"]}),
        (("mcp_servers",), {"c18srv": {"command": "c18-mcp-server", "args": ["--stdio"], "disabled": True}}),
        (("cron", "model_drift_guard"), True),
        (("gateway", "multiplex_profile_allowlist"), ["c18"]),
        (("providers",), {"acme.v1": {"base_url": "http://127.0.0.1:9/v1", "api_mode": "chat_completions"}}),
    ]
    for path, val in structural:
        if rng.random() < 0.8:
            _set(tree, path, copy.deepcopy(val))
    if version is None:
        tree["custom_providers"] = [{"name": "C18 Local", "base_url": "http://127.0.0.1:9/v1"}]
    else:
        tree["_config_version"] = version
    text = "# C18 legacy config — this comment must survive migration\n" + yaml.safe_dump(
        tree, sort_keys=False, allow_unicode=True, default_flow_style=False, width=4096)
    text = text.replace("\nterminal:\n", "\n# terminal: user note\nterminal:\n", 1)
    return text, customized


@pytest.mark.parametrize("version", _migration_start_versions(), ids=lambda v: f"from_v{v}" if v is not None else "unversioned")
def test_p5_migration_is_idempotent_and_keeps_user_values(version, home, monkeypatch):
    seed = 900 + (version or 0)
    text, customized = gen_legacy(version, seed)
    cfg = _cfg_path()
    _write_file(cfg, text)
    (home / ".env").write_text("LLM_MODEL=legacy-model\nC18_KEEP=1\n", encoding="utf-8")
    _reset_config_caches()
    with _quiet():
        C.migrate_config(interactive=False, quiet=True)
    once = _read(cfg)
    env_once = _read(home / ".env")
    parsed = yaml.safe_load(once)
    ctx = f"from v{version} seed={seed}"
    assert parsed.get("_config_version") == LATEST, f"[{ctx}] not stamped latest:\n{once}"
    for path, val in {**_CANARIES, **customized}.items():
        assert _get(parsed, path) == val, f"[{ctx}] migration clobbered user value {path}: {_get(parsed, path)!r} != {val!r}\n{once}"
    for comment in ("# C18 legacy config", "# terminal: user note"):
        assert comment in once, f"[{ctx}] migration dropped the comment {comment!r}"
    _assert_no_duplicate_top_level(once, ctx)
    # twice == once (bytes, both files)
    _reset_config_caches()
    with _quiet():
        C.migrate_config(interactive=False, quiet=True)
    assert _read(cfg) == once, f"[{ctx}] a second migrate changed the file:\n{_udiff(once, _read(cfg))}"
    assert _read(home / ".env") == env_once, f"[{ctx}] a second migrate changed .env"
    # Re-applying every step N→latest to its own output is a no-op (step idempotence).
    if version is not None:
        rewound = re.sub(r"^_config_version: \d+$", f"_config_version: {version}", once, flags=re.M)
        _write_file(cfg, rewound)
        _reset_config_caches()
        with _quiet():
            C.migrate_config(interactive=False, quiet=True)
        again = yaml.safe_load(_read(cfg))
        assert again == parsed, (
            f"[{ctx}] re-running the steps on migrated output changed "
            f"{sorted(_changed_paths(parsed, again), key=str)}:\n{_udiff(once, _read(cfg))}")


# ─────────────────────────────────────────────────────────────────────────────
# P6 — every top-level section tolerates `section: null`
# ─────────────────────────────────────────────────────────────────────────────

_TOP_LEVEL = [k for k in DEFAULT_CONFIG if k != "_config_version"]


def _p6_setup(section: str, monkeypatch) -> tuple[str, dict, Path, int]:
    case = gen_case(1000 + _TOP_LEVEL.index(section), n_sections=(3, 5), null_leaves=False,
                    exclude_top={section, "display", "agent"})
    block = f"{section}: null\n"
    if section != "agent":  # keep the separately-xfailed phantom `agent: {}` out of these cells
        block = "agent:\n  verify_on_stop: false\n" + block
    text = _before_version(case.text, block)
    for k, v in case.env.items():
        monkeypatch.setenv(k, v)
    cfg = _cfg_path()
    _write_file(cfg, text)
    _reset_config_caches()
    return text, yaml.safe_load(text), cfg, text.splitlines().index(f"{section}: null")


@pytest.mark.parametrize("section", _TOP_LEVEL)
def test_p6_null_section_survives_every_surface(section, web_app, home, monkeypatch):
    """No crash and no clobber of any OTHER section on load, effective resolution, gateway display
    resolvers, save, CLI set, TUI RPC and Desktop PUT (the #105674 `display: null` class)."""
    from gateway.display_config import resolve_display_setting, resolve_tool_progress
    from hermes_cli.config_effective import load_user_config_effective
    from tui_gateway import server

    text, tree, cfg, null_line = _p6_setup(section, monkeypatch)
    ctx = f"{section}: null"

    def reset(body: str = text) -> None:
        _write_file(cfg, body)
        _reset_config_caches()

    effective = C.load_config()
    if isinstance(DEFAULT_CONFIG[section], dict):
        assert isinstance(effective.get(section), dict), f"[{ctx}] load_config()[{section!r}] = {effective.get(section)!r}"
    load_user_config_effective(cfg)
    raw = C.read_user_config_raw(cfg)
    for platform in ("telegram", "discord", "cli"):
        resolve_display_setting(raw, platform, "tool_progress")
        resolve_tool_progress(raw, platform)

    # save_config(load_config()): only the null key's own line may change (dropped or expanded in place)
    with _quiet():
        C.save_config(C.load_config())
    after = _read(cfg)
    touched, inserted = _line_diff(text, after)
    assert touched <= {null_line} and len(inserted) <= 1, f"[{ctx}] save touched other lines:\n{_udiff(text, after)}"
    _assert_no_duplicate_top_level(after, ctx)

    # (section,) itself is allowed to change here: dropping `section: null` on a save is the
    # separately-xfailed null-strip bug (test_p1_explicit_null_leaves_survive_a_noop_save).
    writers = {
        "cli_set": (lambda: C.set_config_value("terminal.timeout", "4242"), ("terminal", "timeout")),
        "tui_rpc": (lambda: server._methods["config.set"](1, {"key": "skin", "value": "ares"}), ("display", "skin")),
        "dashboard_put": (lambda: _client(web_app).put("/api/config", json={"config": {"display": {"skin": "ares"}}}),
                          ("display", "skin")),
    }
    for name, (fn, target) in writers.items():
        reset()
        with _quiet():
            res = fn()
        if name == "tui_rpc":
            assert "result" in res, f"[{ctx}] tui config.set: {res}"
        if name == "dashboard_put":
            assert res.status_code == 200, f"[{ctx}] PUT /api/config: {res.text}"
        got = yaml.safe_load(_read(cfg))
        assert _get(got, target) not in (_MISSING, None), f"[{ctx}] {name} did not write {target}"
        stray = {p for p in _changed_paths(tree, got) if not _under(p, {target, (section,)})}
        assert not stray, f"[{ctx}] {name} changed untargeted paths {sorted(stray, key=str)}:\n{_udiff(text, _read(cfg))}"



def test_p6_all_sections_null_at_once_through_gateway_loader_and_migration(home, monkeypatch):
    """Every top-level key null simultaneously: the real gateway config loader, the effective
    resolver and a migration from the previous schema version run without crashing; the
    migration stamps the latest version and keeps the file's comments."""
    from gateway.config import load_gateway_config
    from hermes_cli.config_effective import load_user_config_effective

    text = "# every section null\n" + "".join(f"{k}: null\n" for k in _TOP_LEVEL) + f"_config_version: {LATEST - 1}\n"
    cfg = _cfg_path()
    _write_file(cfg, text)
    _reset_config_caches()
    load_gateway_config()
    load_user_config_effective(cfg)
    effective = C.load_config()
    for k in _TOP_LEVEL:
        if isinstance(DEFAULT_CONFIG[k], dict):
            assert isinstance(effective.get(k), dict), f"load_config()[{k!r}] = {effective.get(k)!r}"
    with _quiet():
        C.migrate_config(interactive=False, quiet=True)
    after = _read(cfg)
    got = yaml.safe_load(after)
    assert got.get("_config_version") == LATEST, after
    assert "# every section null" in after
    _assert_no_duplicate_top_level(after, "all-null migrate")
    _reset_config_caches()
    load_gateway_config()


def _p6_effective_params():
    out = []
    for k in _TOP_LEVEL:
        d = DEFAULT_CONFIG[k]
        marks = []
        if not isinstance(d, dict) and d is not None:
            marks = [pytest.mark.xfail(strict=True, reason=(
                "NEW (null-strip): `key: null` for a key with a non-null default loads as None but "
                "save_config deletes it, so after any save the default silently takes effect."))]
        out.append(pytest.param(k, marks=marks, id=k))
    return out


@pytest.mark.parametrize("section", _p6_effective_params())
def test_p6_noop_save_keeps_the_effective_value_of_a_null_section(section, home, monkeypatch):
    _p6_setup(section, monkeypatch)
    effective = C.load_config()
    with _quiet():
        C.save_config(copy.deepcopy(effective))
    _reset_config_caches()
    after = C.load_config()
    assert after.get(section) == effective.get(section), (
        f"[{section}: null] save_config(load_config()) changed {section!r}: {effective.get(section)!r} -> {after.get(section)!r}")

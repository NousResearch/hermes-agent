"""Three-way merge of the Nix base into user-owned Hermes state (``configMode = "imperative"``).

Activation runs this instead of the overwrite-style merge so that Hermes keeps full imperative
control of ``config.yaml`` and ``.env`` while Nix still declares a base:

    N  the new Nix base (this generation's settings / environment)
    O  the base applied last time, recorded under the state directory
    C  the file on disk, which Hermes and the user write freely

Per leaf (mappings recurse; anything else, including lists, is a leaf):

    C missing                 -> take N (first apply, and `hermes config unset` reverts to Nix)
    C == O                    -> the user never touched it: follow N (update, or drop if N dropped it)
    C != O, or O lacks it     -> the user's value: keep it (reported when it shadows N)
    absent from both O and N  -> a pure user key: never touched

``plugins.enabled`` / ``plugins.disabled`` merge as sets because ``hermes plugins enable`` rewrites
the whole list: result = (N - user removals) | user additions.

When there is no record (first imperative activation, or a switch from declarative mode) O := N.
Declarative activation forced every Nix leaf to N, so a leaf still equal to N is Nix's, and a leaf
that differs was written at runtime and is the user's.

This file is standalone on purpose (stdlib + ruamel): activation must not import the Hermes package.
The YAML dialect and the .env tokenizer mirror ``hermes_yaml.roundtrip_yaml`` and
``agent.secret_scope._parse_env_text`` so the files read back identically in Hermes.
"""

from __future__ import annotations

import argparse
import fcntl
import io
import json
import os
import re
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.resolver import VersionedResolver

MISSING: Any = object()
SET_MERGED = {("plugins", "enabled"), ("plugins", "disabled")}
# Hermes migrates its own file now; the base's stamp only seeds a file that has none.
UNMANAGED_ROOT_KEYS = {"_config_version"}
ENV_HEADER = "# Added by the Nix base (services.hermes-agent.environment / environmentFiles)"
RENDER_ATTEMPTS = 3


class _Yaml11Resolver(VersionedResolver):
    @property
    def processing_version(self) -> tuple[int, int]:
        return (1, 1)


def _roundtrip_yaml() -> YAML:
    yaml = YAML(typ="rt")
    yaml.width = 2**31 - 1
    yaml.Resolver = _Yaml11Resolver
    yaml.preserve_quotes = True
    yaml.allow_unicode = True
    yaml.default_flow_style = False
    yaml.indent(mapping=2, sequence=4, offset=2)
    return yaml


def _plain(value: Any) -> Any:
    """Comparable form: ruamel node types -> JSON types (keeps bool distinct from int)."""
    return json.loads(json.dumps(value, default=str))


def _same(a: Any, b: Any) -> bool:
    return _plain(a) == _plain(b) and isinstance(a, bool) is isinstance(b, bool)


class Report:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.applied: list[str] = []
        self.removed: list[str] = []
        self.shadowed: list[str] = []

    def emit(self, label: str) -> None:
        for title, items in (("set from Nix", self.applied), ("removed with Nix", self.removed),
                             ("user value overrides Nix", self.shadowed)):
            if items:
                print(f"hermes-agent: {label}: {title}: {', '.join(items)}")


# ── config.yaml ─────────────────────────────────────────────────────────────


def _merge_set(c: dict, key: str, cv: Any, ov: Any, nv: Any, dotted: str, report: Report) -> None:
    user = set(cv) if isinstance(cv, list) else set()
    old = set(ov) if isinstance(ov, list) else set()
    new = set(nv) if isinstance(nv, list) else set()
    if cv is MISSING:
        user = old  # the whole list was unset: no user additions or removals remain
    result = sorted((new - (old - user)) | (user - old))
    if cv is MISSING and nv is MISSING:
        return
    if cv is MISSING or sorted(user) != result:
        c[key] = result
        report.applied.append(dotted)


def _merge_leaf(c: dict, key: str, cv: Any, ov: Any, nv: Any, dotted: str, report: Report) -> None:
    if cv is MISSING:
        if nv is not MISSING:
            c[key] = nv
            report.applied.append(dotted)
    elif ov is not MISSING and _same(cv, ov):
        if nv is MISSING:
            del c[key]
            report.removed.append(dotted)
        elif not _same(cv, nv):
            c[key] = nv
            report.applied.append(dotted)
    elif nv is not MISSING and not _same(cv, nv):
        report.shadowed.append(dotted)


def merge_mapping(c: dict, o: dict, n: dict, path: tuple[str, ...], report: Report) -> None:
    for key in dict.fromkeys([*n, *o]):
        if not path and key in UNMANAGED_ROOT_KEYS:
            continue
        here = (*path, key)
        dotted = ".".join(map(str, here))
        cv, ov, nv = c.get(key, MISSING), o.get(key, MISSING), n.get(key, MISSING)
        if here in SET_MERGED:
            _merge_set(c, key, cv, ov, nv, dotted, report)
        elif (cv is MISSING or isinstance(cv, dict)) and (
                isinstance(nv, dict) or (nv is MISSING and isinstance(ov, dict))):
            if cv is MISSING and nv is MISSING:
                continue  # Nix dropped a section the user had already removed
            created = cv is MISSING
            child = CommentedMap() if created else cv
            merge_mapping(child, ov if isinstance(ov, dict) else {}, nv if isinstance(nv, dict) else {},
                          here, report)
            if child:
                c[key] = child
            elif not created and nv is MISSING:
                del c[key]  # the merge emptied a section Nix no longer declares
        else:
            _merge_leaf(c, key, cv, ov, nv, dotted, report)


def _write_atomic(path: Path, data: bytes, mode: int | None) -> None:
    """Replace *path* with *data* through a temp file and a rename. Keeps the owner, group and mode
    of the file it replaces; a new file gets *mode*."""
    try:
        st = path.stat()
    except FileNotFoundError:
        st = None
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if st is not None:
            if os.geteuid() == 0:
                os.chown(tmp, st.st_uid, st.st_gid)
            os.chmod(tmp, st.st_mode & 0o7777)
        elif mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _signature(st: os.stat_result | None) -> tuple | None:
    return None if st is None else (st.st_ino, st.st_size, st.st_mtime_ns)


def _read_with_signature(path: Path) -> tuple[bytes | None, tuple | None]:
    try:
        st = path.stat()
        return path.read_bytes(), _signature(st)
    except FileNotFoundError:
        return None, None


def _current_signature(path: Path) -> tuple | None:
    try:
        return _signature(path.stat())
    except FileNotFoundError:
        return None


def _apply_unlocked(path: Path, render: Callable[[bytes | None], bytes | None], mode: int | None) -> None:
    """Read, merge and replace a file that its other writers do not lock.

    Hermes takes no cross-process lock when it saves config.yaml (its own processes can lose each
    other's saves the same way), so there is nothing to synchronize with. This NARROWS the window
    and does not close it: a save seen while the merge was rendered makes it start again from that
    save, but a save that lands between the last check and the rename is overwritten. Activation is
    rare and a save is a short read-modify-write, so the window is small, but it is not zero.
    """
    for _ in range(RENDER_ATTEMPTS):
        current, signature = _read_with_signature(path)
        rendered = render(current)
        if rendered is None or rendered == current:
            return
        if _current_signature(path) == signature:
            _write_atomic(path, rendered, mode)
            return
    sys.exit(f"hermes-agent: {path} kept changing during activation; run activation again")


@contextmanager
def _env_lock(env_path: Path, timeout: float = 30.0) -> Iterator[None]:
    """Hold the lock that Hermes's own .env writers hold (hermes_cli.config._env_write_lock):
    an exclusive flock on ``<.env>.lock`` around the whole read-modify-write. A Hermes save then
    runs entirely before or after the merge, so neither can lose the other's keys.

    Hermes opens the lock file for append as the service user. If root creates it here, it must
    belong to the owner of the directory, or each later Hermes save of .env would fail."""
    lock_path = env_path.with_name(env_path.name + ".lock")
    created = not lock_path.exists()
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o660)
    try:
        if created:
            if os.geteuid() == 0:
                home = env_path.parent.stat()
                os.fchown(fd, home.st_uid, home.st_gid)
            os.fchmod(fd, 0o660)  # the umask must not take the group's write bit away
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    sys.exit(f"hermes-agent: timed out waiting for {lock_path}; run activation again")
                time.sleep(0.1)
        yield
    finally:
        os.close(fd)  # closing the descriptor releases the flock


def _read_record(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def _write_record(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def run_config(state: Path, config_path: Path, base_path: Path, mode: int) -> None:
    safe = YAML(typ="safe", pure=True)
    safe.version = (1, 1)
    new = _plain(safe.load(base_path.read_text(encoding="utf-8")) or {})
    record = state / "config.json"
    old = _read_record(record)
    old = new if not isinstance(old, dict) else old
    report = Report()

    def render(current: bytes | None) -> bytes | None:
        report.reset()
        yaml = _roundtrip_yaml()
        doc = yaml.load(current.decode("utf-8-sig")) if current else None
        if doc is None:
            doc = CommentedMap()
        if not isinstance(doc, dict):
            sys.exit(f"hermes-agent: {config_path} is not a mapping; fix it and run activation again")
        seeded = [key for key in UNMANAGED_ROOT_KEYS if key not in doc and key in new]
        for key in seeded:
            doc[key] = new[key]
        merge_mapping(doc, old, new, (), report)
        if current is not None and not (seeded or report.applied or report.removed):
            return None  # nothing changed: leave the user's bytes (and YAML spelling) alone
        out = io.StringIO()
        yaml.dump(doc, out)
        return out.getvalue().encode("utf-8")

    _apply_unlocked(config_path, render, mode)
    _write_record(record, new)
    report.emit("config.yaml")


# ── .env ────────────────────────────────────────────────────────────────────
# Mirrors agent.secret_scope (_parse_env_text, _strip_inline_comment, _parse_env_value) and
# hermes_cli.config._quote_env_value.


def _line_key(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export "):].lstrip()
    key, sep, _ = stripped.partition("=")
    key = key.strip()
    return key if sep and key else None


def _strip_inline_comment(value: str) -> str:
    value = value.strip()
    if not value:
        return value
    quote = value[0]
    if quote in ("'", '"'):
        i = 1
        while i < len(value):
            ch = value[i]
            if quote == '"' and ch == "\\":
                i += 2
                continue
            if ch == quote:
                return value[: i + 1] if value[i + 1:].lstrip().startswith("#") else value
            i += 1
        return value
    return re.split(r"\s+#", value, maxsplit=1)[0].strip()


def _parse_value(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        quoted, parsed, i = value[1:-1], [], 0
        while i < len(quoted):
            escaped = quoted[i] == "\\" and quoted[i + 1:i + 2] in ('"', "\\")
            parsed.append(quoted[i + 1] if escaped else quoted[i])
            i += 2 if escaped else 1
        return "".join(parsed)
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1]
    return value


def parse_env(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        key = _line_key(line)
        if key is not None:
            values[key] = _parse_value(_strip_inline_comment(line.split("=", 1)[1]))
    return values


def _quote(value: str) -> str:
    if value == "" or not ("#" in value or '"' in value or "'" in value or any(c.isspace() for c in value)):
        return value
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _replace_value(line: str, value: str) -> str:
    """Swap only the value token of an assignment line: an ``export`` prefix, the spacing around
    ``=``, a trailing ``# comment`` and the line ending all stay as the user wrote them."""
    body = line.rstrip("\r\n")
    ending = line[len(body):]
    head, _, tail = body.partition("=")
    stripped = tail.strip()
    token = _strip_inline_comment(tail)  # the value as written, always a prefix of ``stripped``
    lead = tail[: len(tail) - len(tail.lstrip())]
    return f"{head}={lead}{_quote(value)}{stripped[len(token):]}{ending or chr(10)}"


def merge_env(text: str, old: dict[str, str], new: dict[str, str], report: Report) -> str:
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    current = parse_env(text)
    for key in dict.fromkeys([*new, *old]):
        cv, ov, nv = current.get(key, MISSING), old.get(key, MISSING), new.get(key, MISSING)
        defining = [i for i, line in enumerate(lines) if _line_key(line) == key]
        if cv is MISSING:
            if nv is MISSING:
                continue
            if ENV_HEADER + "\n" not in lines:
                lines.extend(["\n", ENV_HEADER + "\n"] if lines else [ENV_HEADER + "\n"])
            lines.append(f"{key}={_quote(nv)}\n")
            report.applied.append(key)
        elif ov is not MISSING and cv == ov:
            if nv is MISSING:
                lines = [line for i, line in enumerate(lines) if i not in defining]
                report.removed.append(key)
            elif cv != nv:
                lines[defining[-1]] = _replace_value(lines[defining[-1]], nv)
                report.applied.append(key)
        elif nv is not MISSING and cv != nv:
            report.shadowed.append(key)
    return "".join(lines)


def run_env(state: Path, env_path: Path, mode: int, base_env: Path, env_files: list[str]) -> None:
    new = parse_env(base_env.read_text(encoding="utf-8-sig"))
    record = state / "env.json"
    recorded = _read_record(record)
    old = recorded if isinstance(recorded, dict) else None
    incomplete = False
    for name in env_files:
        try:
            new.update(parse_env(Path(name).read_text(encoding="utf-8-sig")))
        except OSError as exc:
            incomplete = True
            print(f"hermes-agent: WARNING cannot read environmentFile {name}: {exc}", file=sys.stderr)
    bootstrap = old is None
    if bootstrap:
        old = dict(new)
    if incomplete:
        # A secret that is not readable yet must not count as dropped by Nix.
        new = {**{k: v for k, v in old.items() if k not in new}, **new}
    report = Report()

    def render(current: bytes | None) -> bytes | None:
        report.reset()
        merged = merge_env(current.decode("utf-8-sig") if current else "", old, new, report)
        return None if current is None and not merged else merged.encode("utf-8")

    with _env_lock(env_path):
        current = env_path.read_bytes() if env_path.exists() else None
        rendered = render(current)
        if rendered is not None and rendered != current:
            _write_atomic(env_path, rendered, mode)
    if not (bootstrap and incomplete):  # a partial first record would orphan the unread keys
        _write_record(record, new)
    report.emit(".env")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="kind", required=True)
    cfg = sub.add_parser("config")
    cfg.add_argument("state", type=Path)
    cfg.add_argument("target", type=Path)
    cfg.add_argument("mode", type=lambda s: int(s, 8))
    cfg.add_argument("base", type=Path)
    env = sub.add_parser("env")
    env.add_argument("state", type=Path)
    env.add_argument("target", type=Path)
    env.add_argument("mode", type=lambda s: int(s, 8))
    env.add_argument("base", type=Path)
    env.add_argument("env_files", nargs="*")
    args = parser.parse_args(argv)
    args.state.mkdir(mode=0o700, parents=True, exist_ok=True)
    if args.kind == "config":
        run_config(args.state, args.target, args.base, args.mode)
    else:
        run_env(args.state, args.target, args.mode, args.base, args.env_files)


if __name__ == "__main__":
    main()

"""What the conversation backend's ``_store`` modules copy from the legacy store, and the text each copy must have.

One spec module per backend module (``<target>.py`` beside this file) defines ``COPY``, one ``Copy`` or a tuple of
them when the module copies from several legacy modules: the legacy module, the module-level functions and
assignments it copies, the legacy class whose members it copies as module-level functions and assignments, and the
explicit substitutions. ``expected_text`` gives the text a copied definition must have: the
legacy text with every legacy import of a copied name pointed at its copy, then the substitutions, each found exactly
once, then ``receiver_form``, which puts a class member at module level and changes the receivers and nothing else.
The legacy text is main's, read from a checkout of it: a spec's legacy module is read under main's name for it.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import io
import re
import tokenize
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

STORE_PACKAGE = "persistence.sqlite.conversation._store"
_SPEC_DIR = Path(__file__).parent

# Legacy one-line methods that call the module function of the same name with ``self``: they have no copy, and a
# call of one is a call of that function.
DELEGATES = {
    "update_conversation_model_config_value": "model_config",
    "session_ids_with_model_config_key": "model_config",
}


@dataclass(frozen=True)
class Copy:
    """One backend module: ``names`` are module-level definitions copied in legacy order, ``members`` the members of
    ``legacy_class`` copied after them as module-level definitions, in legacy order; ``substitutions`` maps a copied
    name to its ``(old, new)`` edits."""

    legacy: str
    doc: str = ""
    names: tuple[str, ...] = ()
    legacy_class: Optional[str] = None
    members: tuple[str, ...] = ()
    substitutions: dict[str, tuple[tuple[str, str], ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class Member:
    """A copied class member: ``kind`` is ``instance``, ``static``, ``class`` or ``value``; ``target`` its module."""

    kind: str
    target: str


@dataclass(frozen=True)
class Catalog:
    """Everything the copies are: ``index`` maps ``(legacy module, name)`` of each copied module-level definition to
    its backend module, ``members`` each copied class member by name, ``classes`` the legacy class names."""

    index: dict[tuple[str, str], str]
    members: dict[str, Member]
    classes: frozenset[str]


def load_specs() -> dict[str, tuple[Copy, ...]]:
    """Every spec beside this module, keyed by its backend module name; the first ``Copy`` holds the module's doc."""
    specs = {}
    for path in sorted(_SPEC_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{path.stem}")
        specs[path.stem] = module.COPY if isinstance(module.COPY, tuple) else (module.COPY,)
    return specs


# Spec legacy modules that main holds under another name.
MAIN_MODULES = {
    "hermes_state_shared": "hermes_state_common",
    "hermes_state_common": "hermes_state_common",
    "hermes_state_transcript_repair": "agent.transcript_repair",
}

# Main's legacy modules the backend holds whole: any name imported from one comes from the backend's module.
MODULE_COPIES = {
    "hermes_state_errors": "hermes_state_errors",
    "hermes_state_keys": "hermes_state_keys",
}


def main_module(legacy: str) -> str:
    """Main's name for a spec's legacy module."""
    return MAIN_MODULES.get(legacy, legacy)


# Legacy module-level definitions the backend core already holds a copy of.
CORE_COPIES = {
    ("hermes_state_search", "_meta_row"): "persistence.sqlite._schema",
    ("hermes_state_search", "_delete_meta"): "persistence.sqlite._schema",
    ("hermes_state_titles", "next_title_in_lineage"): "persistence.sqlite._schema",
}


# Names a legacy module only re-exports, imported from the module that defines them; ``module:name`` imports a
# differently named definition under the legacy name.
REEXPORTS = {
    ("hermes_state", "SessionCompressionInProgressError"): "hermes_state_errors",
    ("hermes_state_messages", "_placeholders"): f"{STORE_PACKAGE}.sql_common",
    ("hermes_state_runtime", "admission_fingerprint"): "hermes_state_keys",
    ("hermes_state_runtime", "_json"): "persistence.core.canonical:canonical_json",
}


# Application names the neutral session_rules modules hold a copy of.
TURNS = "session_rules.turns"
ACTIVITY = "session_rules.activity"
MEDIA = "session_rules.media"
SURFACES = "session_rules.surfaces"
NEUTRAL = {
    ("agent.agent_runtime_helpers", "repair_message_sequence"): TURNS,
    ("agent.compaction_display", "project_compaction_message_for_display"): TURNS,
    **{("agent.context_compressor", name): TURNS for name in (
        "MODEL_ONLY_DISPLAY_METADATA_KEY", "_DB_PERSISTED_MARKER", "history_before_user_originated_turn",
        "retryable_user_text", "split_user_originated_turn", "user_originated_turn_view")},
    **{("agent.conversation_compression_archive", name): TURNS for name in (
        "MERGED_DURABLE_ROWS", "OWN_ROW", "RETIRED_DURABLE_ROWS", "RETIRED_ROW", "UNNAMED_DURABLE_ROWS")},
    ("agent.memory_context", "sanitize_context"): TURNS,
    ("agent.message_content", "flatten_message_text"): TURNS,
    **{("agent.message_metadata", name): TURNS for name in (
        "ABSORBED_MESSAGE_UIDS", "MESSAGE_UID", "TOOL_CALL_UIDS", "index_tool_call_uids", "message_uid_or_none",
        "mint_uid", "resolve_tool_call_uid", "uid_list")},
    ("agent.message_sanitization", "coalesce_tool_call_id"): TURNS,
    ("agent.session_persistence", "_durable_content"): TURNS,
    ("agent.session_persistence", "_is_ephemeral_scaffolding"): TURNS,
    ("agent.skill_commands", "_AUTO_LOAD_PREFIX"): TURNS,
    ("agent.skill_commands", "_SKILL_INVOCATION_PREFIX"): TURNS,
    ("hermes_state_transcript_repair", "is_content_blank"): TURNS,
    **{("agent.session_activity", name): ACTIVITY for name in (
        "ActivityProvenance", "bound_activity_description", "normalize_activity_provenance")},
    **{("gateway.session_ingress_media", name): MEDIA for name in (
        "admission_media_references", "hosted_document_references")},
    ("gateway.session_policy", "created_surface"): SURFACES,
}


def core_members() -> dict[str, Member]:
    """The legacy store's members the backend core holds as module-level names of ``persistence.sqlite._schema``: a
    function taking ``holder`` first is an instance member, any other function a static one, an assignment a value."""
    import hermes_state
    from persistence.sqlite import _schema

    legacy = {name for klass in hermes_state.SessionDB.__mro__ for name in vars(klass)}
    tree = ast.parse(Path(_schema.__file__).read_text(encoding="utf-8"))
    members = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in legacy:
            params = [a.arg for a in node.args.args]
            members[node.name] = Member("instance" if params[:1] == ["holder"] else "static", _schema.__name__)
        elif isinstance(node, ast.Assign):
            for name in (t.id for t in node.targets if isinstance(t, ast.Name) and t.id in legacy):
                members[name] = Member("value", _schema.__name__)
    return members


def catalog(specs: dict[str, tuple[Copy, ...]], legacy_root: Path) -> Catalog:
    index = {(main_module(module), name): copy
             for (module, name), copy in {**CORE_COPIES, **REEXPORTS, **NEUTRAL}.items()}
    index.update({(main_module(spec.legacy), name): f"{STORE_PACKAGE}.{target}" for target, copies in specs.items()
                  for spec in copies for name in spec.names})
    members = {**core_members(), **{name: Member("instance", target) for name, target in DELEGATES.items()}}
    classes = {"SessionDB"}
    for target, spec in ((target, spec) for target, copies in specs.items() for spec in copies):
        if not spec.legacy_class:
            continue
        classes.add(spec.legacy_class)
        tree = ast.parse(legacy_source(spec.legacy, legacy_root))
        for name in spec.members:
            node = definition(tree, name, spec.legacy_class)
            # A class-level alias of a method (``alias = method``) is called as that method is.
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name) and node.value.id in spec.members:
                node = definition(tree, node.value.id, spec.legacy_class)
            members[name] = Member(member_kind(node), target)
    return Catalog(index, members, frozenset(classes))


def member_kind(node: ast.stmt) -> str:
    """``instance``, ``static`` or ``class`` for a method, ``value`` for an assignment."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return "value"
    decorators = {ast.unparse(d) for d in node.decorator_list}
    assert decorators <= {"staticmethod", "classmethod"}, (node.name, decorators)
    return "static" if "staticmethod" in decorators else "class" if "classmethod" in decorators else "instance"


def definition(tree: ast.Module, name: str, cls: Optional[str] = None) -> ast.stmt:
    """The module-level (or ``cls`` class-level) statement defining ``name``."""
    body = tree.body
    if cls is not None:
        body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            return node
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        for target in targets:
            names = [t.id for t in ast.walk(target) if isinstance(t, ast.Name)]
            if name in names:
                return node
    raise KeyError(f"{cls + '.' if cls else ''}{name}")


def segment(source: str, node: ast.stmt) -> str:
    """Whole lines of ``node``, from its first decorator to its last line."""
    lines = source.split("\n")
    start = min([node.lineno, *(d.lineno for d in getattr(node, "decorator_list", ()))])
    return "\n".join(lines[start - 1:node.end_lineno])


_FROM_IMPORT = re.compile(
    r"^(?P<indent>[ \t]*)from (?P<module>[\w.]+) import "
    r"(?:\((?P<group>[^)]*)\)|(?P<line>[^\n(#]+?))[ \t]*(?:#[^\n]*)?$",
    re.MULTILINE,
)


def _alias_parts(text: str) -> list[tuple[str, str]]:
    parts = []
    for raw in text.replace("\n", " ").split(","):
        raw = raw.strip()
        if raw:
            name = raw.partition(" as ")[0]
            parts.append((name.strip(), raw))
    return parts


def legacy_file(legacy: str) -> str:
    """The path of a legacy module, relative to the tree root."""
    return legacy.replace(".", "/") + ".py"


def legacy_source(legacy: str, legacy_root: Path) -> str:
    """The text of a spec's legacy module, read under main's name from the checkout at ``legacy_root``."""
    return (legacy_root / legacy_file(main_module(legacy))).read_text(encoding="utf-8")


def point_imports(text: str, index: dict[tuple[str, str], str]) -> str:
    """``text`` with each ``from <legacy module> import ...`` of a copied name importing it from its copy
    instead (one line per backend module, then the rest from the legacy module, in their original order; a
    ``module:name`` copy is imported as the legacy name), and any name of a module in ``MODULE_COPIES`` from the
    backend's module; a trailing comment on a rewritten import is dropped."""

    def replace(match: re.Match) -> str:
        module = match["module"]
        parts = _alias_parts(match["group"] if match["group"] is not None else match["line"])
        groups: dict[str, list[str]] = {}
        for name, raw in parts:
            groups.setdefault(index.get((module, name), MODULE_COPIES.get(module, "")), []).append(raw)
        if set(groups) == {""}:
            return match.group(0)
        lines = []
        for target, raws in groups.items():
            source, _, defined = (module if target == "" else target).partition(":")
            if defined:
                raws = [f"{defined} as {raw.partition(' as ')[2] or raw}" for raw in raws]
            lines.append(f"{match['indent']}from {source} import {', '.join(raws)}")
        return "\n".join(lines)

    return _FROM_IMPORT.sub(replace, text)


def substitute(text: str, edits, where: str) -> str:
    """Apply ``(old, new)`` edits, each of which must occur exactly once."""
    for old, new in edits:
        count = text.count(old)
        if count != 1:
            raise AssertionError(f"{where}: {old!r} occurs {count} times")
        text = text.replace(old, new)
    return text


_SKIPPED = (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER)


def receiver_form(text: str, cat: Catalog, kind: Optional[str] = None) -> str:
    """``text``, a copied definition, with its receivers changed and nothing else.

    A class member (``kind`` is its member kind) moves to module level: its ``@staticmethod`` or ``@classmethod``
    line is dropped, every line outside a non-docstring string literal loses one indentation level, an instance
    method's ``self`` parameter becomes ``holder: StoreHolder`` and a classmethod's ``cls`` parameter is dropped.

    In any definition, a copied member reached through ``self``, ``cls``, a legacy class name or (in a module-level
    function) the function's first parameter is the copy itself: ``r.m(args)`` of an instance member becomes
    ``m(R, args)``, where ``R`` is ``holder`` for ``self`` and the receiver otherwise; an uncalled ``r.m`` of an
    instance member and ``getattr(self, "m", None)`` become ``partial(m, R)``; ``r.m`` of any other member becomes
    ``m``. Every other ``self`` becomes ``holder``, and every other ``cls.<name>`` becomes ``StoreHolder.<name>``.
    A member the definition assigns through a receiver (a class-level cache) is a module global of the copy: a
    ``global <name>`` line opens its body, after the docstring. Otherwise the line count is unchanged apart from
    the dropped decorator line. A module-level class keeps its own ``self``: only legacy class names are receivers
    in it.
    """
    node = _parse(text).body[0]
    own_self = kind is None and isinstance(node, ast.ClassDef)
    receivers = set(cat.classes) if own_self else {"self"} | cat.classes
    if kind == "class":
        receivers.add("cls")
    if kind is None and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.args.args:
        receivers.add(node.args.args[0].arg)
    code = [t for t in tokenize.generate_tokens(io.StringIO(text).readline) if t.type not in _SKIPPED]
    edits: list[tuple[tuple[int, int], tuple[int, int], str]] = []
    header = kind in ("instance", "class")
    i = 0
    while i < len(code):
        tok = code[i]
        after_dot = i > 0 and code[i - 1].string == "."
        if header and tok.string == "def":
            first = code[i + 3]
            assert code[i + 2].string == "(" and first.string == ("self" if kind == "instance" else "cls"), text[:80]
            if kind == "instance":
                edits.append((first.start, first.end, "holder: StoreHolder"))
            else:
                end = code[i + 5].start if code[i + 4].string == "," else first.end
                assert end[0] == first.start[0], text[:80]
                edits.append((first.start, end, ""))
            header = False
            i += 4
            continue
        if (tok.string == "getattr" and not after_dot and [t.string for t in code[i + 1:i + 4]] == ["(", "self", ","]
                and code[i + 4].type == tokenize.STRING and [t.string for t in code[i + 5:i + 8]] == [",", "None", ")"]
                and getattr(cat.members.get(ast.literal_eval(code[i + 4].string)), "kind", None) == "instance"):
            edits.append((tok.start, code[i + 7].end, f"partial({ast.literal_eval(code[i + 4].string)}, holder)"))
            i += 8
            continue
        if (tok.type == tokenize.NAME and tok.string in receivers and not after_dot and i + 2 < len(code)
                and code[i + 1].string == "." and code[i + 2].string in cat.members):
            name = code[i + 2].string
            if cat.members[name].kind != "instance":
                edits.append((tok.start, code[i + 2].end, name))
                i += 3
                continue
            assert tok.string not in cat.classes | {"cls"}, text[:80]
            arg = "holder" if tok.string == "self" else tok.string
            paren = code[i + 3] if i + 3 < len(code) else None
            if paren is None or paren.string != "(":
                edits.append((tok.start, code[i + 2].end, f"partial({name}, {arg})"))
                i += 3
                continue
            inner = code[i + 4]
            if inner.string == ")":
                edits.append((tok.start, paren.end, f"{name}({arg}"))
            elif inner.start[0] > paren.end[0]:
                edits.append((tok.start, paren.end, f"{name}({arg},"))
            else:
                edits.append((tok.start, inner.start, f"{name}({arg}, "))
            i += 4
            continue
        if tok.type == tokenize.NAME and not after_dot and tok.string == "self" and kind != "class" and not own_self:
            edits.append((tok.start, tok.end, "holder"))
        elif tok.type == tokenize.NAME and not after_dot and tok.string == "cls" and kind == "class":
            assert code[i + 1].string == ".", text[:80]
            edits.append((tok.start, tok.end, "StoreHolder"))
        i += 1
    lines = text.split("\n")
    for (row, start), (end_row, end), new in sorted(edits, reverse=True):
        assert row == end_row, text[:80]
        lines[row - 1] = lines[row - 1][:start] + new + lines[row - 1][end:]
    assigned = sorted({n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)
                       and isinstance(n.value, ast.Name) and n.value.id in receivers and n.attr in cat.members})
    if assigned:
        body = node.body
        first = body[1] if _has_docstring(node) and len(body) > 1 else body[0]
        indent = " " * (len(lines[first.lineno - 1]) - len(lines[first.lineno - 1].lstrip(" ")))
        lines.insert(first.lineno - 1, f"{indent}global {', '.join(assigned)}")
    return "\n".join(lines) if kind is None else _module_level("\n".join(lines))


def _parse(text: str) -> ast.Module:
    """``text`` parsed with its first line's indentation removed from every line that has it, so a definition whose
    string literal holds lines indented less than the definition parses, its line numbers unchanged."""
    indent = text[:len(text) - len(text.lstrip(" "))]
    return ast.parse("\n".join(line.removeprefix(indent) for line in text.split("\n")))


def _has_docstring(node: ast.AST) -> bool:
    first = node.body[0] if getattr(node, "body", None) else None
    return isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)


def _module_level(text: str) -> str:
    """A class member's text at module level: its ``@staticmethod``/``@classmethod`` line dropped and one
    indentation level removed from every line outside a non-docstring string literal."""
    inside = string_rows(text)
    out = []
    for row, line in enumerate(text.split("\n"), start=1):
        if row in inside:
            out.append(line)
        elif line.strip() not in ("@staticmethod", "@classmethod"):
            out.append(line[4:] if line.startswith("    ") else line.lstrip(" "))
    return "\n".join(out)


def string_rows(text: str) -> set[int]:
    """Rows (1-based) inside a multi-line string literal of ``text`` other than a docstring: every row of the literal
    after its first."""
    docstrings = set()
    for node in ast.walk(_parse(text)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                docstrings.add(first.lineno)
    rows: set[int] = set()
    fstring_starts: list[int] = []
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.STRING and tok.end[0] > tok.start[0] and tok.start[0] not in docstrings:
            rows.update(range(tok.start[0] + 1, tok.end[0] + 1))
        elif tok.type == tokenize.FSTRING_START:
            fstring_starts.append(tok.start[0])
        elif tok.type == tokenize.FSTRING_END:
            start = fstring_starts.pop()
            if not fstring_starts:
                rows.update(range(start + 1, tok.end[0] + 1))
    return rows


def expected_text(spec: Copy, name: str, legacy_root: Path, cat: Catalog, *, member: bool = False) -> str:
    """The text the copy of ``name`` (a member of the spec's legacy class when ``member``) must have."""
    source = legacy_source(spec.legacy, legacy_root)
    node = definition(ast.parse(source), name, spec.legacy_class if member else None)
    text = point_imports(segment(source, node), cat.index)
    # Main raises ``RuntimeStoreError`` where the copies raise the contract's ``AuthorityRefusal``, same reason.
    text = re.sub(r"\bRuntimeStoreError\b", "AuthorityRefusal", text)
    text = substitute(text, spec.substitutions.get(name, ()), f"{spec.legacy}.{name}")
    return receiver_form(text, cat, member_kind(node) if member else None)

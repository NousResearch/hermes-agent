"""ts-type extensions for documented Python gateway contracts.

Python keeps PEP 257 attribute docstrings in source ASTs but discards them at
runtime.  This module restores those docs while `ts-type` renders our stdlib
``TypedDict`` contracts, making Python the canonical source for both wire
shape and TypeScript documentation.
"""

from __future__ import annotations

import ast
from functools import lru_cache
import inspect
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints, NotRequired, Required

import ts_type as ts


class OpaqueValue:
    """A JSON value whose structure is intentionally not part of the contract."""


def _unwrap_typed_dict_marker(value: Any) -> Any:
    """Strip Python-only requiredness markers before ts-type renders a field."""
    if get_origin(value) in {NotRequired, Required}:
        return get_args(value)[0]
    return value


@lru_cache(maxsize=None)
def _typed_dict_docs(contract: type) -> tuple[str | None, dict[str, str]]:
    """Return the PEP 257 class and attribute docs for a contract source class."""
    source_path = inspect.getsourcefile(contract)
    if source_path is None:
        return None, {}

    module = ast.parse(Path(source_path).read_text(encoding="utf-8"))
    node: ast.ClassDef | ast.Module = module
    for name in contract.__qualname__.split("."):
        if name == "<locals>":
            continue
        child = next(
            (
                candidate
                for candidate in node.body
                if isinstance(candidate, ast.ClassDef) and candidate.name == name
            ),
            None,
        )
        if child is None:
            return None, {}
        node = child

    class_doc = ast.get_docstring(node)
    field_docs: dict[str, str] = {}
    for index, child in enumerate(node.body[:-1]):
        next_child = node.body[index + 1]
        if not (
            isinstance(child, ast.AnnAssign)
            and isinstance(child.target, ast.Name)
            and isinstance(next_child, ast.Expr)
            and isinstance(next_child.value, ast.Constant)
            and isinstance(next_child.value.value, str)
        ):
            continue
        field_docs[child.target.id] = inspect.cleandoc(next_child.value.value)
    return class_doc, field_docs


def _render_jsdoc(doc: str, indent: str = "") -> str:
    """Render a Python docstring as an escaped TypeScript JSDoc block."""
    lines = inspect.cleandoc(doc).replace("*/", "*\\/").splitlines() or [""]
    if len(lines) == 1:
        return f"{indent}/** {lines[0]} */"
    return "\n".join(
        [f"{indent}/**", *[f"{indent} * {line}" for line in lines], f"{indent} */"]
    )


class DocumentedObject(ts.Object):
    """A TypeScript object node which retains PEP 257 field docs."""

    def __init__(
        self,
        *args: Any,
        class_doc: str | None,
        field_docs: dict[str, str],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.class_doc = class_doc
        self.field_docs = field_docs

    def render(self, context: ts.RenderContext) -> str:
        child_context = context.clone(indent_level=context.indent_level + 1)
        fields: list[str] = []
        for key, value in self.attrs.items():
            if doc := self.field_docs.get(key):
                fields.append(_render_jsdoc(doc, child_context.indent))
            fields.append(
                "".join(
                    [
                        child_context.indent,
                        "readonly " if key in self.readonly else "",
                        f'"{key}"',
                        "?" if key in self.omissible else "",
                        ": ",
                        value.render(child_context),
                        ";",
                    ]
                )
            )
        return "\n".join(["{", *fields, f"{context.indent}}}"])


class DocumentedContractBuilder(ts.NodeBuilder):
    """Render ``TypedDict`` contracts with PEP 257 class and field docs."""

    def handle_unknown_type(self, value: Any) -> ts.TypeNode:
        if value is OpaqueValue:
            return ts.Unknown()
        return super().handle_unknown_type(value)

    def typeddict_to_node(self, contract: type) -> ts.TypeNode:
        class_doc, field_docs = _typed_dict_docs(contract)
        annotations = get_type_hints(contract, include_extras=True)
        omissible = contract.__optional_keys__ | {
            key
            for key, value in annotations.items()
            if get_origin(value) is NotRequired
        }
        return self.define_ref_node(
            contract,
            lambda: DocumentedObject(
                attrs={
                    key: self.type_to_node(_unwrap_typed_dict_marker(value))
                    for key, value in annotations.items()
                },
                omissible=omissible,
                class_doc=class_doc,
                field_docs=field_docs,
            ),
        )

    def render(self, refs_to_export: set[str] | None = None) -> str:
        context = ts.RenderContext(self.definitions)
        to_export = refs_to_export or set(self.definitions)
        names = [name for name in self.definitions if name in to_export] + [
            name for name in self.definitions if name not in to_export
        ]

        def render(name: str) -> str:
            node = context.definitions[name]
            prefix = ""
            if name in to_export and isinstance(node, ts.Reference):
                resolved = context.resolve_ref(node)
                if isinstance(resolved, DocumentedObject) and resolved.class_doc:
                    prefix = _render_jsdoc(resolved.class_doc) + "\n"
            export = "export " if name in to_export else ""
            return f"{prefix}{export}type {ts.Reference(name).render(context)} = {node.render(context)};"

        return "\n\n".join(render(name) for name in names)

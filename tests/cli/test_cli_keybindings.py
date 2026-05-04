import ast
from pathlib import Path

from prompt_toolkit.key_binding import KeyBindings


def _literal_kb_add_sequences(source: str) -> list[tuple[str, ...]]:
    tree = ast.parse(source)
    sequences: list[tuple[str, ...]] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            func = decorator.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == "add"
                and isinstance(func.value, ast.Name)
                and func.value.id == "kb"
            ):
                continue

            keys = tuple(
                arg.value
                for arg in decorator.args
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            )
            if keys:
                sequences.append(keys)

    return sequences


def test_cli_literal_prompt_toolkit_keybindings_are_valid():
    source = Path("cli.py").read_text(encoding="utf-8")
    sequences = _literal_kb_add_sequences(source)

    assert sequences

    invalid: list[tuple[tuple[str, ...], str]] = []
    for keys in sequences:
        kb = KeyBindings()
        try:
            kb.add(*keys)(lambda event: None)
        except Exception as exc:  # pragma: no cover - assertion reports details
            invalid.append((keys, str(exc)))

    assert invalid == []

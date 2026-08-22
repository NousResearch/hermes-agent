from __future__ import annotations

import ast
import inspect
from pathlib import Path

from agent import conversation_loop
from agent.conversation_loop_image_recovery import _IMAGE_REJECTION_PHRASES


_EXPECTED_IMAGE_REJECTION_PHRASES = (
    "only 'text' content type is supported",
    "only text content type is supported",
    "image_url is not supported",
    "image content is not supported",
    "multimodal is not supported",
    "multimodal content is not supported",
    "multimodal input is not supported",
    "vision is not supported",
    "vision input is not supported",
    "does not support images",
    "does not support image input",
    "does not support multimodal",
    "does not support vision",
    "model does not support image",
    "image_url'. expected",
    "unknown variant `image_url`, expected `text`",
    "unknown variant image_url, expected text",
    "no endpoints found that support image input",
)


_MODULE_PATH = Path(__file__).parents[2] / "agent" / "conversation_loop_image_recovery.py"
_LOOP_PATH = Path(conversation_loop.__file__)


def test_image_rejection_phrase_tuple_preserves_value_order_and_identity():
    assert _IMAGE_REJECTION_PHRASES == _EXPECTED_IMAGE_REJECTION_PHRASES
    assert conversation_loop._IMAGE_REJECTION_PHRASES is _IMAGE_REJECTION_PHRASES


def test_original_module_resolves_private_compatibility_reference():
    source = inspect.getsource(conversation_loop.run_conversation)
    assert "_IMAGE_REJECTION_PHRASES" in source
    assert conversation_loop._IMAGE_REJECTION_PHRASES is _IMAGE_REJECTION_PHRASES


def test_original_module_has_no_duplicate_local_tuple_definition():
    tree = ast.parse(_LOOP_PATH.read_text(encoding="utf-8"))
    local_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr))
        and any(
            isinstance(target, ast.Name)
            and target.id == "_IMAGE_REJECTION_PHRASES"
            for target in (
                list(node.targets)
                if isinstance(node, ast.Assign)
                else [node.target]
            )
        )
    ]
    assert local_assignments == []
    assert "from agent.conversation_loop_image_recovery import _IMAGE_REJECTION_PHRASES" in (
        _LOOP_PATH.read_text(encoding="utf-8")
    )


def test_image_recovery_module_has_no_import_time_side_effect_nodes():
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and not (
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
        )
        for node in tree.body
    )
    assert not any(
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        for node in tree.body
    )
    assert [
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    ] == ["_IMAGE_REJECTION_PHRASES"]

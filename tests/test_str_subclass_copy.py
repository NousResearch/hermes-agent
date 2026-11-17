"""A ``str`` subclass whose ``__new__`` needs more than the text cannot be rebuilt by the default
copy/pickle path (``cls.__new__(cls, text)``): ``copy.deepcopy`` of any payload holding one raises
``TypeError``. The gateway deep-copies every published event payload, so a footered reply lost
its ``message.complete`` (#156709) the same way an active-session refusal did before it."""
import ast
import copy
import pickle
from pathlib import Path

from gateway.runtime_footer import FooteredReply

REPO = Path(__file__).resolve().parents[1]
_SKIP = {"tests", ".venv", "node_modules", "apps", "web", "website", ".worktrees", "evals"}
_HOOKS = {"__getnewargs__", "__getnewargs_ex__", "__reduce__", "__reduce_ex__"}


def _needs_extra_args(fn: ast.FunctionDef) -> bool:
    a = fn.args
    positional = len(a.posonlyargs) + len(a.args) - len(a.defaults) - 2  # cls + text
    return positional > 0 or any(d is None for d in a.kw_defaults)


def test_every_str_subclass_with_a_wider_new_defines_its_copy_args():
    offenders = []
    for path in REPO.rglob("*.py"):
        if _SKIP & set(path.relative_to(REPO).parts):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.ClassDef) and any(getattr(b, "id", None) == "str" for b in node.bases)):
                continue
            methods = {f.name: f for f in node.body if isinstance(f, ast.FunctionDef)}
            if "__new__" in methods and _needs_extra_args(methods["__new__"]) and not _HOOKS & set(methods):
                offenders.append(f"{path.relative_to(REPO)}::{node.name}")
    assert offenders == [], offenders


def test_footered_reply_survives_deepcopy_and_pickle_with_its_footer():
    reply = FooteredReply("Here it is", "gpt-5.4 · 68% · ~/projects/hermes")
    for clone in (copy.deepcopy({"response": reply})["response"], pickle.loads(pickle.dumps(reply))):
        assert str(clone) == "Here it is\n\ngpt-5.4 · 68% · ~/projects/hermes"
        assert clone.runtime_footer == reply.runtime_footer

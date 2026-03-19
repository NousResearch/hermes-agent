"""
Cosmos Safe Expression Evaluator

Provides sandboxed eval/exec replacements that prevent code injection.
Used by workflow_builder, tool_router, runbook_executor, and HermesAgent_adapter.

AGI v1.9.1: Eliminates RCE vectors while preserving workflow functionality.
"""

import ast
import operator
import re
from typing import Dict, List, Any, Set, Optional


# Allowed operators for safe expression evaluation
_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.And: lambda a, b: a and b,
    ast.Or: lambda a, b: a or b,
    ast.Not: operator.not_,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}

# Safe builtins for restricted code execution
SAFE_BUILTINS = {
    "True": True,
    "False": False,
    "None": None,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
}

# Patterns that indicate dangerous code
_DANGEROUS_PATTERNS = [
    r'__import__',
    r'__builtins__',
    r'__subclasses__',
    r'__bases__',
    r'__class__',
    r'__globals__',
    r'__code__',
    r'__reduce__',
    r'exec\s*\(',
    r'eval\s*\(',
    r'compile\s*\(',
    r'globals\s*\(',
    r'locals\s*\(',
    r'getattr\s*\(',
    r'setattr\s*\(',
    r'delattr\s*\(',
    r'open\s*\(',
    r'input\s*\(',
    r'breakpoint\s*\(',
    r'__import__',
    r'os\.',
    r'sys\.',
    r'subprocess',
    r'importlib',
    r'shutil',
    r'pathlib',
]


def _check_dangerous(code: str) -> Optional[str]:
    """Check if code contains dangerous patterns. Returns the pattern found or None."""
    for pattern in _DANGEROUS_PATTERNS:
        if re.search(pattern, code):
            return pattern
    return None


class SafeExprEvaluator:
    """
    Evaluates simple expressions safely using AST parsing.

    Supports: arithmetic, comparisons, boolean logic, attribute access on
    provided variables, subscript access, and function calls on whitelisted functions.
    """

    def __init__(self, variables: dict = None):
        self.variables = variables or {}

    def eval(self, expression: str) -> dict:
        """Safely evaluate an expression string."""
        # Check for dangerous patterns
        danger = _check_dangerous(expression)
        if danger:
            raise ValueError(f"Expression contains forbidden pattern: {danger}")

        try:
            tree = ast.parse(expression, mode='eval')
            return self._eval_node(tree.body)
        except (ValueError, TypeError, KeyError) as e:
            raise
        except Exception as e:
            raise ValueError(f"Cannot safely evaluate: {expression!r}: {e}")

    def _eval_node(self, node: ast.AST) -> dict:
        if isinstance(node, ast.Constant):
            return node.value

        elif isinstance(node, ast.Name):
            if node.id in self.variables:
                return self.variables[node.id]
            if node.id in ('True', 'False', 'None'):
                return {'True': True, 'False': False, 'None': None}[node.id]
            raise ValueError(f"Unknown variable: {node.id}")

        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _SAFE_OPERATORS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return _SAFE_OPERATORS[op_type](left, right)

        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _SAFE_OPERATORS:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            operand = self._eval_node(node.operand)
            return _SAFE_OPERATORS[op_type](operand)

        elif isinstance(node, ast.BoolOp):
            op_type = type(node.op)
            if op_type not in _SAFE_OPERATORS:
                raise ValueError(f"Unsupported boolean operator: {op_type.__name__}")
            values = [self._eval_node(v) for v in node.values]
            result = values[0]
            for v in values[1:]:
                result = _SAFE_OPERATORS[op_type](result, v)
            return result

        elif isinstance(node, ast.Compare):
            left = self._eval_node(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                op_type = type(op)
                if op_type not in _SAFE_OPERATORS:
                    raise ValueError(f"Unsupported comparison: {op_type.__name__}")
                right = self._eval_node(comparator)
                if not _SAFE_OPERATORS[op_type](left, right):
                    return False
                left = right
            return True

        elif isinstance(node, ast.Subscript):
            value = self._eval_node(node.value)
            if isinstance(node.slice, ast.Constant):
                return value[node.slice.value]
            elif isinstance(node.slice, ast.Name):
                key = self._eval_node(node.slice)
                return value[key]
            else:
                key = self._eval_node(node.slice)
                return value[key]

        elif isinstance(node, ast.Attribute):
            value = self._eval_node(node.value)
            attr = node.attr
            if attr.startswith('_'):
                raise ValueError(f"Access to private attribute '{attr}' is forbidden")
            return getattr(value, attr)

        elif isinstance(node, ast.Call):
            func = self._eval_node(node.func)
            if not callable(func):
                raise ValueError(f"Not callable: {func}")
            args = [self._eval_node(arg) for arg in node.args]
            kwargs = {kw.arg: self._eval_node(kw.value) for kw in node.keywords}
            return func(*args, **kwargs)

        elif isinstance(node, ast.IfExp):
            test = self._eval_node(node.test)
            return self._eval_node(node.body) if test else self._eval_node(node.orelse)

        elif isinstance(node, ast.list):
            return [self._eval_node(el) for el in node.elts]

        elif isinstance(node, ast.Tuple):
            return tuple(self._eval_node(el) for el in node.elts)

        elif isinstance(node, ast.dict):
            keys = [self._eval_node(k) for k in node.keys]
            values = [self._eval_node(v) for v in node.values]
            return dict(zip(keys, values))

        elif isinstance(node, ast.JoinedStr):
            # f-string
            parts = []
            for value in node.values:
                if isinstance(value, ast.Constant):
                    parts.append(str(value.value))
                else:
                    parts.append(str(self._eval_node(value)))
            return ''.join(parts)

        elif isinstance(node, ast.FormattedValue):
            return self._eval_node(node.value)

        elif isinstance(node, ast.ListComp):
            return self._eval_comprehension(node)

        elif isinstance(node, ast.GeneratorExp):
            return list(self._eval_comprehension_gen(node))

        else:
            raise ValueError(f"Unsupported expression type: {type(node).__name__}")

    def _eval_comprehension(self, node: ast.ListComp) -> list:
        """Evaluate a list comprehension safely."""
        result = []
        self._eval_generators(node.generators, 0, node.elt, result)
        return result

    def _eval_comprehension_gen(self, node: ast.GeneratorExp):
        """Evaluate a generator expression safely."""
        result = []
        self._eval_generators(node.generators, 0, node.elt, result)
        return result

    def _eval_generators(self, generators, idx, elt, result):
        """Recursively evaluate comprehension generators."""
        if idx >= len(generators):
            result.append(self._eval_node(elt))
            return

        gen = generators[idx]
        iterable = self._eval_node(gen.iter)

        for item in iterable:
            # Bind target variable
            if isinstance(gen.target, ast.Name):
                old_val = self.variables.get(gen.target.id)
                self.variables[gen.target.id] = item
            else:
                raise ValueError("Only simple loop variables supported in comprehensions")

            # Check all if-conditions
            if all(self._eval_node(if_clause) for if_clause in gen.ifs):
                self._eval_generators(generators, idx + 1, elt, result)

            # Restore old value
            if isinstance(gen.target, ast.Name):
                if old_val is not None:
                    self.variables[gen.target.id] = old_val
                else:
                    self.variables.pop(gen.target.id, None)


def safe_eval(expression: str, variables: dict = None) -> dict:
    """
    Safely evaluate a simple expression.

    Supports: arithmetic, comparisons, boolean logic, attribute/subscript access,
    and function calls on whitelisted builtins.
    """
    # Merge safe builtins as default variables (user vars take precedence)
    merged = dict(SAFE_BUILTINS)
    if variables:
        merged.update(variables)
    evaluator = SafeExprEvaluator(merged)
    return evaluator.eval(expression)


def safe_exec(code: str, variables: dict = None, max_lines: int = 100) -> dict:
    """
    Execute code with restricted builtins and dangerous pattern checks.

    Returns the local variables after execution.
    """
    # Check for dangerous patterns
    danger = _check_dangerous(code)
    if danger:
        raise ValueError(f"Code contains forbidden pattern: {danger}")

    # Limit code size
    lines = code.split('\n')
    if len(lines) > max_lines:
        raise ValueError(f"Code too long: {len(lines)} lines (max {max_lines})")

    local_vars = dict(variables) if variables else {}
    local_vars.setdefault("result", None)

    # Execute with restricted builtins
    exec(code, {"__builtins__": SAFE_BUILTINS}, local_vars)

    return local_vars

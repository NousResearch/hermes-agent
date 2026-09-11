"""Hermes JIT Micro-Tool Synthesizer & Dynamic Sandbox.

Enables Hermes to autonomously synthesize, AST-audit, fuzz-test, and hot-reload
domain-specific Python micro-tools directly into its runtime ToolRegistry on demand.
"""

from __future__ import annotations

import ast
import base64
import datetime
import hashlib
import io
import json
import logging
import math
import os
import re
import subprocess
import sys
import threading
import time
import uuid
import weakref
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("hermes.jit_tools")

# Strict positive allowlist for JIT sandbox imports — all other imports are rejected
ALLOWED_MODULES = frozenset({
    "math",
    "re",
    "json",
    "collections",
    "itertools",
    "datetime",
    "hashlib",
    "urllib.parse",
    "decimal",
    "fractions",
    "string",
    "bisect",
    "heapq",
})

DISALLOWED_CALLS = frozenset({
    "exec",
    "eval",
    "__import__",
    "open",
    "compile",
    "globals",
    "locals",
    "getattr",
    "setattr",
    "delattr",
})

FRAME_TRAVERSAL_ATTRIBUTES = frozenset({
    "f_back",
    "f_builtins",
    "f_globals",
    "f_locals",
    "f_code",
    "f_lasti",
    "f_lineno",
    "f_trace",
    "f_trace_lines",
    "f_trace_opcodes",
    "gi_frame",
    "gi_code",
    "gi_running",
    "gi_yieldfrom",
    "cr_frame",
    "cr_code",
    "cr_running",
    "cr_await",
    "ag_frame",
    "ag_code",
    "ag_running",
    "ag_await",
    "tb_frame",
    "tb_next",
    "tb_lasti",
    "tb_lineno",
    "co_code",
    "co_consts",
    "co_names",
    "co_varnames",
    "co_freevars",
    "co_cellvars",
})

DISALLOWED_ATTRIBUTES = frozenset({
    "__builtins__",
    "__dict__",
    "__class__",
    "__subclasses__",
    "__bases__",
    "__mro__",
    "__globals__",
    "__code__",
    "__closure__",
    "__func__",
    "__self__",
    "__module__",
    "__loader__",
    "__spec__",
    "__package__",
    "__reduce__",
    "__reduce_ex__",
    "__import__",
})

CURRENT_POLICY_VERSION = 4


class JITSecurityViolation(Exception):
    """Raised when synthesized tool code violates AST security policies."""
    pass


class JITCompilationError(Exception):
    """Raised when synthesized tool code fails compilation, fuzzing, or execution."""
    pass


@dataclass
class SynthesizedToolSpec:
    """Specification and metadata for a dynamically synthesized tool."""
    name: str
    description: str
    parameters_schema: Dict[str, Any]
    python_source: str
    toolset: str = "jit_synthesized"
    author: str = "hermes_jit_synthesizer"
    created_at: float = field(default_factory=time.time)
    test_vectors: List[Dict[str, Any]] = field(default_factory=list)
    is_ephemeral: bool = True
    session_id: Optional[str] = None
    scope: Optional[str] = None
    source_digest: str = ""
    policy_version: int = CURRENT_POLICY_VERSION
    call_count: int = 0
    last_error: Optional[str] = None

    def __post_init__(self):
        if not self.source_digest and self.python_source:
            self.source_digest = hashlib.sha256(self.python_source.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SynthesizedToolSpec":
        return cls(
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            parameters_schema=dict(data.get("parameters_schema", {})),
            python_source=str(data.get("python_source", "")),
            toolset=str(data.get("toolset", "jit_synthesized")),
            author=str(data.get("author", "hermes_jit_synthesizer")),
            created_at=float(data.get("created_at", time.time())),
            test_vectors=list(data.get("test_vectors", [])),
            is_ephemeral=bool(data.get("is_ephemeral", True)),
            session_id=data.get("session_id"),
            scope=data.get("scope"),
            source_digest=str(data.get("source_digest", "")),
            policy_version=int(data.get("policy_version", CURRENT_POLICY_VERSION)),
            call_count=int(data.get("call_count", 0)),
            last_error=data.get("last_error"),
        )


@dataclass
class RegistrationReceipt:
    """Exact CAS registration receipt capturing registered slot state."""
    name: str
    scope: Optional[str]
    entry: Any
    previous: Optional[Any]


class JITSandboxSecurityGuard(ast.NodeVisitor):
    """Inspects tool AST for positive capability compliance and security invariants."""

    def __init__(self, target_function_name: str):
        self.target_name = target_function_name
        self.found_target = False
        self.violations: List[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name == self.target_name:
            self.found_target = True
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            base_mod = alias.name.split(".")[0]
            if base_mod not in ALLOWED_MODULES and alias.name not in ALLOWED_MODULES:
                self.violations.append(
                    f"Disallowed import: '{alias.name}'. Only explicitly allowed modules may be imported: {sorted(ALLOWED_MODULES)}"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            base_mod = node.module.split(".")[0]
            if base_mod not in ALLOWED_MODULES and node.module not in ALLOWED_MODULES:
                self.violations.append(
                    f"Disallowed from-import: '{node.module}'. Only explicitly allowed modules may be imported: {sorted(ALLOWED_MODULES)}"
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id in DISALLOWED_CALLS:
                self.violations.append(f"Disallowed function call: {node.func.id}()")
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in {"system", "popen", "spawn", "fork", "execv", "kill", "unlink", "remove", "rmdir"}:
                self.violations.append(f"Disallowed host execution/filesystem call: {node.func.attr}()")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (
            node.attr.startswith("_")
            or node.attr in FRAME_TRAVERSAL_ATTRIBUTES
            or node.attr in DISALLOWED_ATTRIBUTES
            or node.attr in {"sys", "os", "subprocess", "posix", "nt"}
        ):
            self.violations.append(f"Disallowed attribute access: '{node.attr}'")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        # Detect subscript escapes like collections.__builtins__['open'] or f_builtins['op' + 'en']
        slice_val = None
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            slice_val = node.slice.value
        elif isinstance(node.slice, ast.BinOp):
            def _eval_str_binop(op_node: ast.AST) -> Optional[str]:
                if isinstance(op_node, ast.Constant) and isinstance(op_node.value, str):
                    return op_node.value
                if isinstance(op_node, ast.BinOp) and isinstance(op_node.op, ast.Add):
                    l_val = _eval_str_binop(op_node.left)
                    r_val = _eval_str_binop(op_node.right)
                    if l_val is not None and r_val is not None:
                        return l_val + r_val
                return None
            slice_val = _eval_str_binop(node.slice)
        elif hasattr(ast, "Index") and isinstance(node.slice, getattr(ast, "Index", None)):
            if isinstance(node.slice.value, ast.Constant) and isinstance(node.slice.value.value, str):
                slice_val = node.slice.value.value

        if isinstance(slice_val, str):
            if (
                slice_val.startswith("_")
                or slice_val in FRAME_TRAVERSAL_ATTRIBUTES
                or slice_val in {"open", "eval", "exec", "__import__", "compile", "globals", "locals", "sys", "os"}
            ):
                self.violations.append(f"Disallowed subscript access to sensitive key: '{slice_val}'")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        # Prevent mutating attributes on external modules (e.g. json.REVIEW_SENTINEL = ...)
        for target in node.targets:
            if isinstance(target, ast.Attribute):
                self.violations.append(f"Disallowed mutation of module or object attribute: '{target.attr}'")
        self.generic_visit(node)


def audit_tool_source(name: str, source_code: str) -> None:
    """Audit tool source code against positive-capability AST security policy."""
    try:
        tree = ast.parse(source_code, filename=f"<jit_{name}>")
    except SyntaxError as exc:
        raise JITCompilationError(f"Syntax error in synthesized code: {exc}") from exc

    checker = JITSandboxSecurityGuard(name)
    checker.visit(tree)

    if not checker.found_target:
        raise JITSecurityViolation(f"Source does not define target entry function: '{name}'")
    if checker.violations:
        raise JITSecurityViolation(f"Security violations detected: {'; '.join(checker.violations)}")


# Disposable isolated worker script for sandboxed execution
_WORKER_CODE = """
import json
import sys
import io

# Capture real stdout for clean JSON protocol response
protocol_out = sys.stdout
sys.stdout = io.StringIO()

data = json.loads(sys.stdin.read())
action = data.get("action", "execute")
name = data["name"]
source = data["source"]
test_vectors = data.get("test_vectors", [])
kwargs = data.get("kwargs", {})

import builtins
safe_names = [
    "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes",
    "chr", "complex", "dict", "divmod", "enumerate", "filter", "float",
    "format", "frozenset", "hex", "int", "isinstance", "issubclass",
    "iter", "len", "list", "map", "max", "min", "next", "oct", "ord",
    "pow", "print", "range", "repr", "reversed", "round", "set", "slice",
    "sorted", "str", "sum", "tuple", "zip",
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "ZeroDivisionError", "ArithmeticError",
]
safe_builtins = {k: getattr(builtins, k) for k in safe_names if hasattr(builtins, k)}
safe_builtins["True"] = True
safe_builtins["False"] = False
safe_builtins["None"] = None

ALLOWED_MODULES = frozenset({
    "math", "re", "json", "collections", "itertools", "datetime",
    "hashlib", "urllib.parse", "decimal", "fractions", "string", "bisect", "heapq"
})

orig_import = builtins.__import__
def safe_import(mod_name, globals=None, locals=None, fromlist=(), level=0):
    base = mod_name.split(".")[0]
    if base not in ALLOWED_MODULES and mod_name not in ALLOWED_MODULES:
        raise ImportError(f"Import of '{mod_name}' is disallowed in JIT sandbox.")
    return orig_import(mod_name, globals, locals, fromlist, level)

safe_builtins["__import__"] = safe_import

env = {"__builtins__": safe_builtins}

import math, re, json as _json, collections, itertools, datetime, hashlib, urllib.parse, decimal, fractions, string, bisect, heapq
for mod in [math, re, _json, collections, itertools, datetime, hashlib, urllib.parse, decimal, fractions, string, bisect, heapq]:
    mod_alias = mod.__name__.split(".")[-1]
    env[mod_alias] = mod
    if hasattr(mod, "__builtins__"):
        try:
            delattr(mod, "__builtins__")
        except Exception:
            pass
    for leak in ["sys", "_sys", "os", "_os"]:
        if hasattr(mod, leak):
            try:
                delattr(mod, leak)
            except Exception:
                pass

# Neutralize frame inspection and stack traversal in worker sys
if hasattr(sys, "_getframe"):
    try:
        delattr(sys, "_getframe")
    except Exception:
        sys._getframe = None

# Strip dangerous ambient execution/filesystem primitives from root builtins
for bad in ["open", "breakpoint", "input", "exit", "quit"]:
    if hasattr(builtins, bad):
        try:
            delattr(builtins, bad)
        except Exception:
            setattr(builtins, bad, None)

try:
    compiled = compile(source, f"<jit_{name}>", "exec")
    exec(compiled, env)
    fn = env.get(name)
    if not callable(fn):
        print(json.dumps({"status": "error", "error": f"Symbol '{name}' is not callable"}), file=protocol_out)
        sys.exit(0)

    if action == "fuzz":
        for i, vec in enumerate(test_vectors):
            args = vec.get("inputs", {})
            expected = vec.get("expected")
            res = fn(**args)
            if expected is not None and res != expected:
                print(json.dumps({"status": "error", "error": f"Test vector #{i + 1} failed: expected {expected!r}, got {res!r}"}), file=protocol_out)
                sys.exit(0)
        print(json.dumps({"status": "ok"}), file=protocol_out)
    else:
        res = fn(**kwargs)
        print(json.dumps({"status": "ok", "result": res}, default=str), file=protocol_out)
except Exception as exc:
    print(json.dumps({"status": "error", "error": str(exc)}), file=protocol_out)
"""


class JITSandboxExecutor:
    """Isolated execution sandbox for validating and running synthesized micro-tools in a subprocess worker."""

    @classmethod
    def get_safe_builtins(cls) -> Dict[str, Any]:
        """Expose safe builtins dictionary for testing and inspection."""
        import builtins
        safe_names = [
            "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes",
            "chr", "complex", "dict", "divmod", "enumerate", "filter", "float",
            "format", "frozenset", "hex", "int", "isinstance", "issubclass",
            "iter", "len", "list", "map", "max", "min", "next", "oct", "ord",
            "pow", "print", "range", "repr", "reversed", "round", "set", "slice",
            "sorted", "str", "sum", "tuple", "zip",
            "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
            "ZeroDivisionError", "ArithmeticError",
        ]
        safe = {k: getattr(builtins, k) for k in safe_names if hasattr(builtins, k)}
        safe["True"] = True
        safe["False"] = False
        safe["None"] = None

        orig_import = builtins.__import__
        def safe_import(mod_name, globals=None, locals=None, fromlist=(), level=0):
            base = mod_name.split(".")[0]
            if base not in ALLOWED_MODULES and mod_name not in ALLOWED_MODULES:
                raise ImportError(f"Import of '{mod_name}' is disallowed in JIT sandbox.")
            return orig_import(mod_name, globals, locals, fromlist, level)

        safe["__import__"] = safe_import
        return safe

    @classmethod
    def execute_in_worker(
        cls,
        name: str,
        source_code: str,
        action: str = "execute",
        kwargs: Optional[Dict[str, Any]] = None,
        test_vectors: Optional[List[Dict[str, Any]]] = None,
        timeout: float = 5.0,
    ) -> Any:
        """Execute tool or test vectors in a disposable isolated subprocess worker."""
        payload = json.dumps({
            "name": name,
            "source": source_code,
            "action": action,
            "kwargs": kwargs or {},
            "test_vectors": test_vectors or [],
        })
        try:
            # -I: isolated mode (ignore env vars and user site)
            # -s: don't add user site directory
            proc = subprocess.run(
                [sys.executable, "-I", "-s", "-c", _WORKER_CODE],
                input=payload,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise JITCompilationError(f"Execution timed out after {timeout}s") from exc
        except Exception as exc:
            raise JITCompilationError(f"Worker invocation error: {exc}") from exc

        if proc.returncode != 0:
            err = proc.stderr.strip() or f"Worker process exited with code {proc.returncode}"
            raise JITCompilationError(f"Sandbox worker failure: {err}")

        try:
            resp = json.loads(proc.stdout.strip())
        except Exception as exc:
            raise JITCompilationError(f"Invalid worker response: {proc.stdout}") from exc

        if resp.get("status") != "ok":
            err_msg = resp.get("error", "Unknown execution failure")
            raise JITCompilationError(err_msg)

        return resp.get("result")

    @classmethod
    def compile_and_extract(cls, name: str, source_code: str) -> Callable:
        """Return a callable interface that executes the tool in an isolated worker subprocess."""
        tree = ast.parse(source_code)
        param_names: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                param_names = [arg.arg for arg in node.args.args]
                break

        # Verify compilation and extraction in worker
        cls.execute_in_worker(name, source_code, action="fuzz", test_vectors=[])

        def runner(*args, **kwargs):
            call_kwargs = dict(kwargs)
            for param, val in zip(param_names, args):
                call_kwargs[param] = val
            return cls.execute_in_worker(name, source_code, action="execute", kwargs=call_kwargs)

        runner._source_code = source_code
        return runner

    @classmethod
    def fuzz_test_tool(cls, name: str, source_code_or_fn: Any, test_vectors: List[Dict[str, Any]]) -> None:
        """Run fuzz test vectors in isolated worker to verify behavioral stability without host side effects."""
        source_code = source_code_or_fn if isinstance(source_code_or_fn, str) else getattr(source_code_or_fn, "_source_code", None)
        if source_code:
            cls.execute_in_worker(name, source_code, action="fuzz", test_vectors=test_vectors)
        else:
            for i, vec in enumerate(test_vectors):
                args = vec.get("inputs", {})
                expected = vec.get("expected")
                res = source_code_or_fn(**args)
                if expected is not None and res != expected:
                    raise JITCompilationError(f"Test vector #{i + 1} failed: expected {expected!r}, got {res!r}")


_ACTIVE_SYNTHESIZERS: "weakref.WeakSet[JITToolSynthesizer]" = weakref.WeakSet()
_SYNTH_LOCK = threading.RLock()


def cleanup_session_jit_tools(session_id: str) -> int:
    """Owner-finalization hook: revoke all ephemeral tools leased to session_id across active synthesizers."""
    if not session_id:
        return 0
    total = 0
    with _SYNTH_LOCK:
        for synth in list(_ACTIVE_SYNTHESIZERS):
            try:
                total += synth.revoke_session_tools(session_id)
            except Exception as exc:
                logger.debug("Error revoking session JIT tools for %s: %s", session_id, exc)
    return total


def _is_tool_entry_live_in_any_synthesizer(entry: Any, name: str, scope: Optional[str]) -> bool:
    """Check if a previous JIT ToolEntry is still held active by any live synthesizer."""
    with _SYNTH_LOCK:
        for synth in list(_ACTIVE_SYNTHESIZERS):
            for r in synth._receipts.values():
                if r.entry is entry and r.name == name and r.scope == scope:
                    return True
            for k, s in synth._tools.items():
                if s.name == name and s.scope == scope:
                    rec = synth._receipts.get(k)
                    if rec and rec.entry is entry:
                        return True
    return False


class JITToolSynthesizer:
    """High-level engine managing lifecycle, verification, scoped leases, and collision-safe registration."""

    def __init__(self, storage_dir: Optional[Path] = None):
        if storage_dir is None:
            try:
                from hermes_constants import get_hermes_home
                storage_dir = Path(get_hermes_home()) / "jit_tools"
            except Exception:
                storage_dir = Path(os.path.expanduser("~/.hermes/jit_tools"))

        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.meta_file = self.storage_dir / "jit_manifest.json"

        # Qualified composite storage: (scope, session_id, name) -> spec
        self._tools: Dict[Tuple[Optional[str], Optional[str], str], SynthesizedToolSpec] = {}
        # Tracks exact registration receipts for compare-and-swap registry ownership
        self._receipts: Dict[Tuple[Optional[str], Optional[str], str], RegistrationReceipt] = {}
        # session_id -> set of qualified tool keys
        self._session_leases: Dict[str, Set[Tuple[Optional[str], Optional[str], str]]] = {}

        with _SYNTH_LOCK:
            _ACTIVE_SYNTHESIZERS.add(self)

        self.load_persisted_tools()

    def _make_key(self, scope: Optional[str], session_id: Optional[str], name: str) -> Tuple[Optional[str], Optional[str], str]:
        return (scope or None, session_id or None, name)

    def _find_key(self, name: str, session_id: Optional[str] = None, scope: Optional[str] = None) -> Optional[Tuple[Optional[str], Optional[str], str]]:
        """Resolve a tool key prioritizing explicit session and scope."""
        exact = self._make_key(scope, session_id, name)
        if exact in self._tools:
            return exact

        candidates = [k for k in self._tools.keys() if k[2] == name]
        if not candidates:
            return None

        if session_id is not None and scope is not None:
            return None

        if session_id is not None:
            matches = [k for k in candidates if k[1] == session_id]
            return matches[0] if matches else None

        if scope is not None:
            matches = [k for k in candidates if k[0] == scope]
            return matches[0] if matches else None

        if len(candidates) == 1:
            return candidates[0]
        return None

    def synthesize_and_register(
        self,
        name: str,
        description: str,
        parameters_schema: Dict[str, Any],
        python_source: str,
        test_vectors: Optional[List[Dict[str, Any]]] = None,
        is_ephemeral: bool = True,
        session_id: Optional[str] = None,
        scope: Optional[str] = None,
        register_with_global_registry: bool = True,
    ) -> Tuple[bool, str]:
        """Audit, compile, fuzz, and register a synthesized micro-tool with collision and ownership safety."""
        name = name.strip()
        if not re.match(r"^[a-zA-Z0-9_]{3,64}$", name):
            return False, f"Invalid tool name: '{name}'. Must be alphanumeric/underscores (3-64 chars)."

        if is_ephemeral and not session_id:
            return False, "Ephemeral JIT tools require an explicit session_id owner to enforce lifecycle scoping."

        vectors = test_vectors or []

        # 1. AST Security Audit with strict positive allowlist
        try:
            audit_tool_source(name, python_source)
        except (JITSecurityViolation, JITCompilationError) as exc:
            logger.warning("Security/Compilation check failed for JIT tool %s: %s", name, exc)
            return False, f"Audit failed: {exc}"

        # 2. Compile & Fuzz inside disposable worker (zero host side-effects)
        try:
            if vectors:
                JITSandboxExecutor.fuzz_test_tool(name, python_source, vectors)
            else:
                # Compile verification probe
                JITSandboxExecutor.execute_in_worker(name, python_source, action="fuzz", test_vectors=[])
        except JITCompilationError as exc:
            logger.warning("Fuzzing/Execution error for JIT tool %s: %s", name, exc)
            return False, f"Fuzzing failed: {exc}"

        # 3. Collision check against built-in or foreign tools
        if register_with_global_registry:
            collision_error = self._check_registry_collision(name, scope)
            if collision_error:
                return False, collision_error

        # 4. Create Spec with SHA-256 digest provenance
        spec = SynthesizedToolSpec(
            name=name,
            description=description,
            parameters_schema=parameters_schema,
            python_source=python_source,
            toolset="jit_synthesized",
            test_vectors=vectors,
            is_ephemeral=is_ephemeral,
            session_id=session_id,
            scope=scope,
            policy_version=CURRENT_POLICY_VERSION,
        )

        key = self._make_key(scope, session_id, name)

        # 5. Bind into Hermes Tool Registry if requested
        if register_with_global_registry:
            ok, reg_msg = self._register_into_hermes_registry(spec)
            if not ok:
                return False, reg_msg

        self._tools[key] = spec

        # Track session lease if scoped
        if session_id:
            self._session_leases.setdefault(session_id, set()).add(key)

        # 6. Persist if non-ephemeral
        if not is_ephemeral:
            self.save_persisted_tools()

        logger.info("Successfully synthesized and registered JIT micro-tool '%s'", name)
        return True, f"Successfully synthesized tool '{name}'"

    def _check_registry_collision(self, name: str, scope: Optional[str] = None) -> Optional[str]:
        """Verify that synthesizing *name* does not collide with a built-in or foreign toolset."""
        try:
            from tools.registry import registry
            existing = registry.get_entry(name, scope=scope)
            if existing is None:
                try:
                    from tools.registry import discover_builtin_tools
                    discover_builtin_tools()
                    existing = registry.get_entry(name, scope=scope)
                except Exception:
                    pass

            if existing is not None and existing.toolset != "jit_synthesized":
                return (
                    f"Cannot synthesize tool '{name}': name collides with existing tool "
                    f"in toolset '{existing.toolset}'. Synthesis rejected to protect host built-ins."
                )
        except Exception:
            pass
        return None

    def _register_into_hermes_registry(self, spec: SynthesizedToolSpec) -> Tuple[bool, str]:
        """Register into tools.registry.registry dynamically matching production calling convention."""
        try:
            from tools.registry import registry

            # Production calling convention: entry.handler(args, **kwargs)
            # args is the dictionary of tool arguments supplied positionally by ToolRegistry.dispatch()
            def handler(args: Any = None, **context) -> str:
                spec.call_count += 1
                call_args: Dict[str, Any] = {}
                if isinstance(args, dict):
                    call_args = dict(args)
                elif args is None:
                    call_args = {}
                else:
                    raise TypeError(f"Tool arguments must be a dictionary, got {type(args).__name__}")

                caller_session_id = context.get("session_id")
                if spec.session_id and caller_session_id and caller_session_id != spec.session_id:
                    raise PermissionError(
                        f"JIT tool '{spec.name}' is owned by session '{spec.session_id}', "
                        f"cannot be dispatched by session '{caller_session_id}'."
                    )

                try:
                    res = self.execute_tool(spec.name, session_id=spec.session_id, scope=spec.scope, **call_args)
                    return json.dumps({"result": res}) if not isinstance(res, str) else res
                except Exception as exc:
                    spec.last_error = str(exc)
                    raise exc

            schema = {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters_schema,
            }

            # 1. Snapshot slot before registration
            prev_entry = registry.snapshot_registration(spec.name, scope=spec.scope)

            # 2. Register tool
            registry.register(
                name=spec.name,
                toolset=spec.toolset,
                schema=schema,
                handler=handler,
                description=spec.description,
                scope=spec.scope,
            )

            # 3. Snapshot slot after registration to verify success
            new_entry = registry.snapshot_registration(spec.name, scope=spec.scope)
            if new_entry is None or new_entry is prev_entry or new_entry.handler != handler:
                return False, f"Tool registration was rejected or not updated in ToolRegistry for '{spec.name}'"

            # 4. Record registration receipt for CAS ownership
            key = self._make_key(spec.scope, spec.session_id, spec.name)
            self._receipts[key] = RegistrationReceipt(
                name=spec.name,
                scope=spec.scope,
                entry=new_entry,
                previous=prev_entry,
            )

            return True, "Registered in ToolRegistry"
        except Exception as exc:
            logger.warning("Failed to bind into tools.registry.registry: %s", exc)
            return False, f"ToolRegistry binding error: {exc}"

    def execute_tool(self, name: str, session_id: Optional[str] = None, scope: Optional[str] = None, **kwargs) -> Any:
        """Execute a synthesized tool in an isolated worker process."""
        key = self._find_key(name, session_id=session_id, scope=scope)
        if key is None or key not in self._tools:
            raise KeyError(f"JIT tool '{name}' is not registered.")
        spec = self._tools[key]
        spec.call_count += 1
        return JITSandboxExecutor.execute_in_worker(spec.name, spec.python_source, action="execute", kwargs=kwargs)

    def deregister(self, name: str, session_id: Optional[str] = None, scope: Optional[str] = None) -> bool:
        """Safely remove a tool with compare-and-swap ownership protection."""
        key = self._find_key(name, session_id=session_id, scope=scope)
        if key is None or key not in self._tools:
            return False

        spec = self._tools.pop(key)

        if spec.session_id and spec.session_id in self._session_leases:
            self._session_leases[spec.session_id].discard(key)

        # Compare-and-swap registry cleanup:
        # ONLY restore/remove if THIS instance actually registered this tool and holds its receipt!
        receipt = self._receipts.pop(key, None)
        if receipt is not None:
            try:
                from tools.registry import registry
                prev = receipt.previous
                if prev is not None and getattr(prev, "toolset", None) == "jit_synthesized":
                    if not _is_tool_entry_live_in_any_synthesizer(prev, receipt.name, receipt.scope):
                        prev = None
                registry.restore_registration(
                    name=receipt.name,
                    current=receipt.entry,
                    previous=prev,
                    scope=receipt.scope,
                )
            except Exception as exc:
                logger.debug("Deregistration notice for %s: %s", receipt.name, exc)

        if not spec.is_ephemeral:
            self.save_persisted_tools()
        return True

    def revoke_session_tools(self, session_id: str) -> int:
        """Revoke and deregister all ephemeral tools leased to a finalized session."""
        leased_keys = list(self._session_leases.get(session_id, set()))
        revoked_count = 0
        for key in leased_keys:
            if self.deregister(key[2], session_id=key[1], scope=key[0]):
                revoked_count += 1
        self._session_leases.pop(session_id, None)
        return revoked_count

    def prune_ephemeral(self) -> int:
        """Remove all ephemeral tools across all sessions."""
        to_prune = [k for k, v in self._tools.items() if v.is_ephemeral]
        for k in to_prune:
            self.deregister(k[2], session_id=k[1], scope=k[0])
        return len(to_prune)

    def list_tools(self) -> List[SynthesizedToolSpec]:
        """Return all active synthesized tools."""
        return list(self._tools.values())

    def get_tool(self, name: str, session_id: Optional[str] = None, scope: Optional[str] = None) -> Optional[SynthesizedToolSpec]:
        key = self._find_key(name, session_id=session_id, scope=scope)
        return self._tools.get(key) if key else None

    def save_persisted_tools(self) -> None:
        """Persist non-ephemeral tools with source digest provenance to disk."""
        non_ephemeral = {k[2]: v.to_dict() for k, v in self._tools.items() if not v.is_ephemeral}
        try:
            self.meta_file.write_text(json.dumps(non_ephemeral, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.error("Failed to save persisted JIT tools: %s", exc)

    def load_persisted_tools(self) -> None:
        """Restore non-ephemeral tools, enforcing full re-audit under current admission policy."""
        if not self.meta_file.exists():
            return
        try:
            data = json.loads(self.meta_file.read_text(encoding="utf-8"))
            for name, spec_dict in data.items():
                spec = SynthesizedToolSpec.from_dict(spec_dict)

                # 1. Digest provenance check
                expected_digest = hashlib.sha256(spec.python_source.encode("utf-8")).hexdigest()
                if spec.source_digest and spec.source_digest != expected_digest:
                    logger.warning(
                        "Manifest digest mismatch for durable JIT tool '%s'; rejecting restore.", name
                    )
                    continue

                # 2. Re-run current AST security policy audit
                try:
                    audit_tool_source(spec.name, spec.python_source)
                except (JITSecurityViolation, JITCompilationError) as exc:
                    logger.warning(
                        "Durable JIT tool '%s' rejected by current security policy: %s", name, exc
                    )
                    continue

                # 3. Re-fuzz in isolated worker
                try:
                    if spec.test_vectors:
                        JITSandboxExecutor.fuzz_test_tool(spec.name, spec.python_source, spec.test_vectors)
                except Exception as exc:
                    logger.warning("Durable JIT tool '%s' failed compilation/fuzzing: %s", name, exc)
                    continue

                key = self._make_key(spec.scope, spec.session_id, spec.name)
                self._tools[key] = spec
                self._register_into_hermes_registry(spec)
        except Exception as exc:
            logger.error("Failed to load persisted JIT tools: %s", exc)

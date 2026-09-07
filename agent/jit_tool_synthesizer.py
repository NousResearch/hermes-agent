"""Hermes JIT Micro-Tool Synthesizer & Dynamic Sandbox.

Enables Hermes to autonomously synthesize, AST-audit, fuzz-test, and hot-reload
domain-specific Python micro-tools directly into its runtime ToolRegistry on demand.
"""

from __future__ import annotations

import ast
import base64
import datetime
import hashlib
import importlib
import json
import logging
import math
import os
import re
import sys
import time
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

DISALLOWED_ATTRIBUTES = frozenset({
    "__subclasses__",
    "__bases__",
    "__mro__",
    "__globals__",
    "__code__",
})

CURRENT_POLICY_VERSION = 2


class JITSecurityViolation(Exception):
    """Raised when synthesized tool code violates AST security policies."""
    pass


class JITCompilationError(Exception):
    """Raised when synthesized tool code fails compilation or fuzzing."""
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
        if node.attr in DISALLOWED_ATTRIBUTES:
            self.violations.append(f"Disallowed attribute access: {node.attr}")
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


class JITSandboxExecutor:
    """Isolated execution sandbox for validating and running synthesized micro-tools."""

    @staticmethod
    def get_safe_builtins() -> Dict[str, Any]:
        """Construct a restricted builtins dictionary with standard algorithmic utilities."""
        safe_names = [
            "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes",
            "chr", "complex", "dict", "divmod", "enumerate", "filter", "float",
            "format", "frozenset", "hex", "int", "isinstance", "issubclass",
            "iter", "len", "list", "map", "max", "min", "next", "oct", "ord",
            "pow", "range", "repr", "reversed", "round", "set", "slice",
            "sorted", "str", "sum", "tuple", "zip",
            "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
            "ZeroDivisionError", "ArithmeticError",
        ]
        import builtins
        orig_import = builtins.__import__

        def safe_import(name: str, globals=None, locals=None, fromlist=(), level=0):
            base = name.split(".")[0]
            if base not in ALLOWED_MODULES and name not in ALLOWED_MODULES:
                raise ImportError(
                    f"Import of '{name}' is disallowed in JIT sandbox. "
                    f"Only explicitly allowed modules are permitted: {sorted(ALLOWED_MODULES)}"
                )
            return orig_import(name, globals, locals, fromlist, level)

        safe_builtins = {k: getattr(builtins, k) for k in safe_names if hasattr(builtins, k)}
        safe_builtins["True"] = True
        safe_builtins["False"] = False
        safe_builtins["None"] = None
        safe_builtins["__import__"] = safe_import
        return safe_builtins

    @staticmethod
    def create_sandbox_env() -> Dict[str, Any]:
        """Create an execution environment with safe standard libraries."""
        import collections
        import datetime
        import hashlib
        import itertools
        import json
        import math
        import re
        import urllib.parse

        return {
            "__builtins__": JITSandboxExecutor.get_safe_builtins(),
            "math": math,
            "re": re,
            "json": json,
            "collections": collections,
            "itertools": itertools,
            "datetime": datetime,
            "hashlib": hashlib,
            "urllib_parse": urllib.parse,
        }

    @classmethod
    def compile_and_extract(cls, name: str, source_code: str) -> Callable:
        """Compile source within the sandbox and extract the entry function."""
        env = cls.create_sandbox_env()
        try:
            compiled = compile(source_code, f"<jit_tool_{name}>", "exec")
            exec(compiled, env)
        except Exception as exc:
            raise JITCompilationError(f"Compilation/execution error: {exc}") from exc

        fn = env.get(name)
        if not callable(fn):
            raise JITCompilationError(f"Extracted symbol '{name}' is not callable")
        return fn

    @classmethod
    def fuzz_test_tool(cls, name: str, fn: Callable, test_vectors: List[Dict[str, Any]]) -> None:
        """Execute test vectors against the compiled function to verify behavioral stability."""
        for i, vec in enumerate(test_vectors):
            args = vec.get("inputs", {})
            expected = vec.get("expected")
            try:
                result = fn(**args)
                if expected is not None and result != expected:
                    raise JITCompilationError(
                        f"Test vector #{i + 1} failed: expected {expected!r}, got {result!r}"
                    )
            except Exception as exc:
                if isinstance(exc, JITCompilationError):
                    raise
                raise JITCompilationError(f"Test vector #{i + 1} raised error: {exc}") from exc


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

        self._tools: Dict[str, SynthesizedToolSpec] = {}
        self._compiled_callables: Dict[str, Callable] = {}
        self._session_leases: Dict[str, Set[str]] = {}  # session_id -> set of tool names
        self.load_persisted_tools()

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

        vectors = test_vectors or []

        # 1. AST Security Audit with strict positive allowlist
        try:
            audit_tool_source(name, python_source)
        except (JITSecurityViolation, JITCompilationError) as exc:
            logger.warning("Security/Compilation check failed for JIT tool %s: %s", name, exc)
            return False, f"Audit failed: {exc}"

        # 2. Compile & Fuzz
        try:
            compiled_fn = JITSandboxExecutor.compile_and_extract(name, python_source)
            if vectors:
                JITSandboxExecutor.fuzz_test_tool(name, compiled_fn, vectors)
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

        # 5. Bind into Hermes Tool Registry if requested
        if register_with_global_registry:
            ok, reg_msg = self._register_into_hermes_registry(spec, compiled_fn)
            if not ok:
                return False, reg_msg

        self._tools[name] = spec
        self._compiled_callables[name] = compiled_fn

        # Track session lease if scoped
        if session_id:
            self._session_leases.setdefault(session_id, set()).add(name)

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

    def _register_into_hermes_registry(self, spec: SynthesizedToolSpec, compiled_fn: Callable) -> Tuple[bool, str]:
        """Register into tools.registry.registry dynamically with scope attribution."""
        try:
            from tools.registry import registry

            def handler(**kwargs) -> str:
                spec.call_count += 1
                try:
                    res = compiled_fn(**kwargs)
                    return json.dumps({"result": res}) if not isinstance(res, str) else res
                except Exception as exc:
                    spec.last_error = str(exc)
                    return json.dumps({"error": str(exc)})

            schema = {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters_schema,
            }

            registry.register(
                name=spec.name,
                toolset=spec.toolset,
                schema=schema,
                handler=handler,
                description=spec.description,
                scope=spec.scope,
            )
            return True, "Registered in ToolRegistry"
        except Exception as exc:
            logger.warning("Failed to bind into tools.registry.registry: %s", exc)
            return False, f"ToolRegistry binding error: {exc}"

    def execute_tool(self, name: str, **kwargs) -> Any:
        """Execute a synthesized tool with arguments."""
        if name not in self._compiled_callables:
            raise KeyError(f"JIT tool '{name}' is not registered.")
        spec = self._tools[name]
        spec.call_count += 1
        return self._compiled_callables[name](**kwargs)

    def deregister(self, name: str, scope: Optional[str] = None) -> bool:
        """Safely remove a tool from memory and registry, strictly guarding foreign tools."""
        if name not in self._tools:
            return False

        spec = self._tools[name]
        del self._tools[name]
        self._compiled_callables.pop(name, None)

        # Remove from session leases if tracked
        if spec.session_id and spec.session_id in self._session_leases:
            self._session_leases[spec.session_id].discard(name)

        target_scope = scope if scope is not None else spec.scope

        try:
            from tools.registry import registry
            entry = registry.get_entry(name, scope=target_scope)
            # CAS-style safety: only deregister if the entry belongs to jit_synthesized
            if entry is not None and entry.toolset == "jit_synthesized":
                if hasattr(registry, "deregister"):
                    registry.deregister(name, scope=target_scope)
        except Exception as exc:
            logger.debug("Deregistration notice: %s", exc)

        if not spec.is_ephemeral:
            self.save_persisted_tools()
        return True

    def revoke_session_tools(self, session_id: str) -> int:
        """Revoke and deregister all ephemeral tools leased to a finalized session."""
        leased_names = list(self._session_leases.get(session_id, set()))
        revoked_count = 0
        for name in leased_names:
            if self.deregister(name):
                revoked_count += 1
        self._session_leases.pop(session_id, None)
        return revoked_count

    def prune_ephemeral(self) -> int:
        """Remove all ephemeral tools across all sessions."""
        to_prune = [k for k, v in self._tools.items() if v.is_ephemeral]
        for k in to_prune:
            self.deregister(k)
        return len(to_prune)

    def list_tools(self) -> List[SynthesizedToolSpec]:
        """Return all active synthesized tools."""
        return list(self._tools.values())

    def get_tool(self, name: str) -> Optional[SynthesizedToolSpec]:
        return self._tools.get(name)

    def save_persisted_tools(self) -> None:
        """Persist non-ephemeral tools with source digest provenance to disk."""
        non_ephemeral = {k: v.to_dict() for k, v in self._tools.items() if not v.is_ephemeral}
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

                # 2. Re-run current AST security policy audit (fail closed on tightened rules)
                try:
                    audit_tool_source(spec.name, spec.python_source)
                except (JITSecurityViolation, JITCompilationError) as exc:
                    logger.warning(
                        "Durable JIT tool '%s' rejected by current security policy: %s", name, exc
                    )
                    continue

                # 3. Re-compile and fuzz against stored test vectors
                try:
                    compiled = JITSandboxExecutor.compile_and_extract(spec.name, spec.python_source)
                    if spec.test_vectors:
                        JITSandboxExecutor.fuzz_test_tool(spec.name, compiled, spec.test_vectors)
                except Exception as exc:
                    logger.warning("Durable JIT tool '%s' failed compilation/fuzzing: %s", name, exc)
                    continue

                self._tools[spec.name] = spec
                self._compiled_callables[spec.name] = compiled
                self._register_into_hermes_registry(spec, compiled)
        except Exception as exc:
            logger.error("Failed to load persisted JIT tools: %s", exc)

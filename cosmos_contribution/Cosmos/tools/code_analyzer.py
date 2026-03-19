"""
Code Analysis Tools - Autonomous Improvement #6 by Claude Sonnet 4.5

PROBLEM: No way to automatically analyze code quality, complexity, or patterns
SOLUTION: AST-based analysis with metrics and pattern detection

Provides:
- Complexity metrics (cyclomatic, cognitive)
- Dependency analysis
- Pattern detection
- Security scanning
- Performance hints
"""

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Set, Optional, Any
from collections import defaultdict
from loguru import logger


@dataclass
class FunctionMetrics:
    """Metrics for a function."""
    name: str
    lineno: int
    complexity: int = 0  # Cyclomatic complexity
    cognitive_complexity: int = 0
    num_params: int = 0
    num_returns: int = 0
    num_lines: int = 0
    calls: List[str] = field(default_factory=list)
    imports: Set[str] = field(default_factory=set)


@dataclass
class FileMetrics:
    """Metrics for a file."""
    path: str
    num_lines: int = 0
    num_functions: int = 0
    num_classes: int = 0
    imports: Set[str] = field(default_factory=set)
    functions: List[FunctionMetrics] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    todos: List[str] = field(default_factory=list)
    fixmes: List[str] = field(default_factory=list)


class CodeAnalyzer:
    """
    Analyzes Python code for quality, complexity, and patterns.

    Features:
    - AST parsing
    - Complexity metrics
    - Dependency analysis
    - Pattern detection
    - Security hints
    """

    def __init__(self):
        # Security patterns to check
        self.security_patterns = {
            "eval": "Dangerous: eval() can execute arbitrary code",
            "exec": "Dangerous: exec() can execute arbitrary code",
            "compile": "Risky: compile() can execute code",
            "__import__": "Risky: dynamic imports can be dangerous",
            "pickle": "Warning: pickle can execute code",
            "shell=True": "Dangerous: shell=True in subprocess",
        }

        # Performance anti-patterns
        self.performance_patterns = {
            "in_loop_append": "Performance: appending in loop, consider list comprehension",
            "nested_loops": "Performance: nested loops, complexity O(n²) or worse",
        }

    def analyze_file(self, filepath: str) -> Optional[FileMetrics]:
        """
        Analyze a Python file.

        Returns FileMetrics or None if can't parse.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            return self.analyze_code(content, filepath)

        except Exception as e:
            logger.error(f"Failed to analyze {filepath}: {e}")
            return None

    def analyze_code(self, code: str, filepath: str = "<string>") -> Optional[FileMetrics]:
        """Analyze Python code string."""
        try:
            tree = ast.parse(code)
            metrics = FileMetrics(path=filepath)

            # Count lines
            metrics.num_lines = len(code.splitlines())

            # Walk AST
            for node in ast.walk(tree):
                # Functions
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_metrics = self._analyze_function(node)
                    metrics.functions.append(func_metrics)
                    metrics.num_functions += 1

                # Classes
                elif isinstance(node, ast.ClassDef):
                    metrics.classes.append(node.name)
                    metrics.num_classes += 1

                # Imports
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        metrics.imports.add(alias.name)

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        metrics.imports.add(node.module)

            # Find TODOs and FIXMEs
            for i, line in enumerate(code.splitlines(), 1):
                if "TODO" in line.upper():
                    metrics.todos.append(f"{i}: {line.strip()}")
                if "FIXME" in line.upper():
                    metrics.fixmes.append(f"{i}: {line.strip()}")

            return metrics

        except SyntaxError as e:
            logger.error(f"Syntax error in {filepath}: {e}")
            return None

    def _analyze_function(self, node: ast.FunctionDef) -> FunctionMetrics:
        """Analyze a function node."""
        metrics = FunctionMetrics(
            name=node.name,
            lineno=node.lineno,
            num_params=len(node.args.args)
        )

        # Calculate complexity
        metrics.complexity = self._calculate_complexity(node)
        metrics.cognitive_complexity = self._calculate_cognitive_complexity(node)

        # Count returns
        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                metrics.num_returns += 1

            # Track function calls
            elif isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    metrics.calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    metrics.calls.append(child.func.attr)

        # Count lines (approximate)
        if hasattr(node, 'end_lineno') and node.end_lineno:
            metrics.num_lines = node.end_lineno - node.lineno

        return metrics

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """
        Calculate cyclomatic complexity.

        Complexity = 1 + number of decision points
        """
        complexity = 1

        for child in ast.walk(node):
            # Decision points
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def _calculate_cognitive_complexity(self, node: ast.FunctionDef) -> int:
        """
        Calculate cognitive complexity.

        More human-centric than cyclomatic.
        """
        complexity = 0
        nesting_level = 0

        def walk_with_nesting(n, level=0):
            nonlocal complexity

            # Nesting increases complexity
            if isinstance(n, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1 + level
                level += 1

            elif isinstance(n, ast.Try):
                complexity += 1 + level
                level += 1

            # Recursively walk children
            for child in ast.iter_child_nodes(n):
                walk_with_nesting(child, level)

        walk_with_nesting(node)
        return complexity

    def scan_security(self, code: str) -> List[Dict]:
        """
        Scan code for security issues.

        Returns list of findings.
        """
        findings = []

        for pattern, message in self.security_patterns.items():
            if pattern in code:
                findings.append({
                    "type": "security",
                    "pattern": pattern,
                    "message": message,
                    "severity": "high" if "Dangerous" in message else "medium"
                })

        return findings

    def find_complex_functions(
        self,
        metrics: FileMetrics,
        threshold: int = 10
    ) -> List[FunctionMetrics]:
        """Find functions with high complexity."""
        return [
            f for f in metrics.functions
            if f.complexity > threshold
        ]

    def analyze_directory(
        self,
        directory: str,
        pattern: str = "*.py",
        exclude_dirs: Optional[Set[str]] = None
    ) -> Dict[str, FileMetrics]:
        """
        Analyze all Python files in a directory.

        Returns dict of filepath -> FileMetrics.
        """
        if exclude_dirs is None:
            exclude_dirs = {"__pycache__", ".git", "venv", "node_modules"}

        results = {}
        path = Path(directory)

        for py_file in path.rglob(pattern):
            # Skip excluded directories
            if any(excluded in py_file.parts for excluded in exclude_dirs):
                continue

            metrics = self.analyze_file(str(py_file))
            if metrics:
                results[str(py_file)] = metrics

        return results

    def generate_report(self, metrics: Dict[str, FileMetrics]) -> Dict:
        """Generate summary report from metrics."""
        total_lines = sum(m.num_lines for m in metrics.values())
        total_functions = sum(m.num_functions for m in metrics.values())
        total_classes = sum(m.num_classes for m in metrics.values())

        # Find most complex functions
        all_functions = []
        for file_metrics in metrics.values():
            for func in file_metrics.functions:
                all_functions.append((file_metrics.path, func))

        all_functions.sort(key=lambda x: x[1].complexity, reverse=True)
        most_complex = all_functions[:10]

        # Find most imported modules
        all_imports = defaultdict(int)
        for file_metrics in metrics.values():
            for imp in file_metrics.imports:
                all_imports[imp] += 1

        top_imports = sorted(all_imports.items(), key=lambda x: x[1], reverse=True)[:10]

        # Collect TODOs and FIXMEs
        all_todos = []
        all_fixmes = []
        for file_metrics in metrics.values():
            all_todos.extend([(file_metrics.path, t) for t in file_metrics.todos])
            all_fixmes.extend([(file_metrics.path, f) for f in file_metrics.fixmes])

        return {
            "summary": {
                "total_files": len(metrics),
                "total_lines": total_lines,
                "total_functions": total_functions,
                "total_classes": total_classes,
                "avg_lines_per_file": total_lines / len(metrics) if metrics else 0
            },
            "most_complex_functions": [
                {
                    "file": path,
                    "function": func.name,
                    "complexity": func.complexity,
                    "lineno": func.lineno
                }
                for path, func in most_complex[:5]
            ],
            "top_imports": [
                {"module": mod, "count": count}
                for mod, count in top_imports[:10]
            ],
            "todos": len(all_todos),
            "fixmes": len(all_fixmes),
            "todo_list": all_todos[:10],
            "fixme_list": all_fixmes[:10]
        }


# Global instance
_code_analyzer: Optional[CodeAnalyzer] = None


def get_analyzer() -> CodeAnalyzer:
    """Get global code analyzer."""
    global _code_analyzer
    if _code_analyzer is None:
        _code_analyzer = CodeAnalyzer()
        logger.info("CodeAnalyzer initialized")
    return _code_analyzer


# Convenience functions
def analyze_python_file(filepath: str) -> Optional[FileMetrics]:
    """Analyze a Python file."""
    analyzer = get_analyzer()
    return analyzer.analyze_file(filepath)


def analyze_python_code(code: str) -> Optional[FileMetrics]:
    """Analyze Python code string."""
    analyzer = get_analyzer()
    return analyzer.analyze_code(code)


def scan_code_security(code: str) -> List[Dict]:
    """Scan code for security issues."""
    analyzer = get_analyzer()
    return analyzer.scan_security(code)


def analyze_project(directory: str) -> Dict:
    """Analyze entire project."""
    analyzer = get_analyzer()
    metrics = analyzer.analyze_directory(directory)
    return analyzer.generate_report(metrics)

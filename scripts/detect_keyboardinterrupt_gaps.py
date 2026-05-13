#!/usr/bin/env python3
"""
CI regression detector for KeyboardInterrupt shutdown cleanup — V2 (targeted).

Only flags `except Exception:` inside shutdown/cleanup functions that contain
a BLOCKING call vulnerable to KeyboardInterrupt:
  - future.result(timeout=N)
  - thread.join(timeout=N)
  - condition.wait(timeout=N)
  - lock.acquire(timeout=N)
  - queue.get(timeout=N)
  - event.wait(timeout=N)
  - subprocess.wait(timeout=N)
  - os.waitpid(pid, 0)
  - select.select(...)
  - asyncio.run_coroutine_threadsafe(...).result()
  - loop.run_until_complete(...)
  - loop.run_forever()
  - time.sleep(N)  [in shutdown context]

This avoids flagging harmless `except Exception:` in logging, metrics, etc.

Exit codes:
  0 = clean
  1 = violations found
  2 = internal error
"""
import ast
import os
import re
import sys
from pathlib import Path

SHUTDOWN_PATTERNS = re.compile(
    r'(shutdown|close|cleanup|finalize|stop|drain|disconnect|teardown|'
    r'_run_cleanup|_stop_mcp_loop|_atexit_|_cleanup_)',
    re.IGNORECASE,
)

SKIP_PATH_PATTERNS = [
    re.compile(r'/tests?/'),
    re.compile(r'/test_'),
    re.compile(r'/_test'),
    re.compile(r'/venv/'),
    re.compile(r'/\.venv/'),
    re.compile(r'/site-packages/'),
    re.compile(r'/node_modules/'),
    re.compile(r'\.egg-info/'),
    re.compile(r'/build/'),
    re.compile(r'/dist/'),
    re.compile(r'__pycache__'),
]

# Regex patterns for blocking calls that are vulnerable to KeyboardInterrupt
BLOCKING_CALL_PATTERNS = [
    re.compile(r'\.result\s*\('),           # future.result(timeout=...)
    re.compile(r'\.join\s*\('),             # thread.join(timeout=...)
    re.compile(r'\.wait\s*\('),             # condition.wait(timeout=...), event.wait()
    re.compile(r'\.acquire\s*\('),          # lock.acquire(timeout=...)
    re.compile(r'\.get\s*\('),              # queue.get(timeout=...)
    re.compile(r'os\.waitpid\s*\('),        # os.waitpid(pid, 0)
    re.compile(r'select\.select\s*\('),     # select.select(...)
    re.compile(r'run_until_complete\s*\('), # loop.run_until_complete(...)
    re.compile(r'run_forever\s*\('),        # loop.run_forever()
    re.compile(r'time\.sleep\s*\('),        # time.sleep(N) in shutdown
]


def should_skip_file(path: str) -> bool:
    for pat in SKIP_PATH_PATTERNS:
        if pat.search(path):
            return True
    return False


def has_blocking_call(source_lines: list[str], start_line: int, end_line: int) -> bool:
    """Check if any line in the range contains a blocking call pattern."""
    for i in range(start_line - 1, min(end_line, len(source_lines))):
        line = source_lines[i]
        for pat in BLOCKING_CALL_PATTERNS:
            if pat.search(line):
                return True
    return False


def find_shutdown_functions(tree: ast.AST) -> list[tuple[str, int, int]]:
    """Return (func_name, start_line, end_line) for shutdown-pattern functions."""
    results = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if SHUTDOWN_PATTERNS.search(node.name):
                # Estimate end line
                end_line = getattr(node, 'end_lineno', node.lineno + 50)
                results.append((node.name, node.lineno, end_line))
    return results


def find_except_exception_in_func(
    tree: ast.AST, func_name: str, func_lineno: int, source_lines: list[str]
) -> list[tuple[int, bool]]:
    """Find `except Exception:` (without KeyboardInterrupt) inside a specific function.
    Returns [(except_line, has_blocking_call_in_try_body), ...].
    """
    violations = []
    
    func_node = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == func_name and node.lineno == func_lineno:
                func_node = node
                break
    
    if func_node is None:
        return violations
    
    for node in ast.walk(func_node):
        if isinstance(node, ast.Try):
            for handler in node.handlers:
                if handler.type is None:
                    continue  # bare `except:` catches BaseException, fine
                
                is_violation = False
                if isinstance(handler.type, ast.Name) and handler.type.id == 'Exception':
                    is_violation = True
                elif isinstance(handler.type, ast.Tuple):
                    has_ki = any(
                        isinstance(elt, ast.Name) and elt.id == 'KeyboardInterrupt'
                        for elt in handler.type.elts
                    )
                    has_exc = any(
                        isinstance(elt, ast.Name) and elt.id == 'Exception'
                        for elt in handler.type.elts
                    )
                    if has_exc and not has_ki:
                        is_violation = True
                
                if is_violation:
                    # Check if the try body (not the handler) contains a blocking call
                    try_start = node.lineno
                    try_end = handler.lineno  # handler starts after try body
                    has_block = has_blocking_call(source_lines, try_start, try_end)
                    violations.append((handler.lineno, has_block))
    
    return violations


def scan_file(filepath: Path) -> list[tuple[str, int, int, bool]]:
    """Returns [(func_name, func_line, except_line, has_blocking_call), ...]."""
    try:
        source = filepath.read_text(encoding='utf-8', errors='replace')
    except Exception:
        return []
    
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    
    source_lines = source.split('\n')
    violations = []
    shutdown_funcs = find_shutdown_functions(tree)
    
    for func_name, func_lineno, func_end in shutdown_funcs:
        except_lines = find_except_exception_in_func(tree, func_name, func_lineno, source_lines)
        for except_line, has_block in except_lines:
            violations.append((func_name, func_lineno, except_line, has_block))
    
    return violations


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
    
    all_violations = []
    
    for py_file in root.rglob('*.py'):
        path_str = str(py_file)
        if should_skip_file(path_str):
            continue
        
        violations = scan_file(py_file)
        for func_name, func_line, except_line, has_block in violations:
            all_violations.append((path_str, func_name, func_line, except_line, has_block))
    
    # Only report violations where the try body contains a blocking call
    blocking_violations = [v for v in all_violations if v[4]]
    non_blocking_violations = [v for v in all_violations if not v[4]]
    
    if blocking_violations:
        print(f"FAIL: {len(blocking_violations)} CRITICAL violation(s) found")
        print("=" * 60)
        for path_str, func_name, func_line, except_line, has_block in sorted(blocking_violations):
            print(f"  {path_str}:{except_line}")
            print(f"    Function: {func_name} (line {func_line})")
            print(f"    Issue: `except Exception:` does not catch KeyboardInterrupt")
            print(f"    Fix:    `except (Exception, KeyboardInterrupt):`")
            print()
        print(f"INFO: {len(non_blocking_violations)} non-blocking violations also found (cosmetic)")
        sys.exit(1)
    elif non_blocking_violations:
        print(f"PASS: No critical blocking-call violations")
        print(f"INFO: {len(non_blocking_violations)} non-blocking `except Exception:` in shutdown paths")
        print("      (harmless — no blocking calls in try body)")
        sys.exit(0)
    else:
        print("PASS: No KeyboardInterrupt shutdown gaps found")
        sys.exit(0)


if __name__ == '__main__':
    main()

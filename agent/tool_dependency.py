import os
import re
import time
import ast
from typing import Callable, Optional, Any
import importlib.util


# Tool dependency rules
# Format: "tool_name": [list of tools that MUST run before it, or {"type": "param", "source": "tool", "param": "param_name"}]
TOOL_DEPENDENCIES: dict[str, list] = {
    "patch": [
        {"type": "file_read", "required": True, "tools": ["read_file", "search_files"]}
    ],
    "write_file": [
        {"type": "directory_read", "required": False, "tools": ["terminal"]}
    ],
    "browser_click": [
        {"type": "requires_snapshot", "tools": ["browser_snapshot", "browser_navigate"]}
    ],
    "browser_type": [
        {"type": "requires_snapshot", "tools": ["browser_snapshot", "browser_navigate"]}
    ],
}

# Tools that produce outputs consumed by other tools
TOOL_OUTPUT_REGISTRY: dict[str, str] = {
    "read_file": "file_content",
    "search_files": "matches",
    "browser_snapshot": "snapshot_html",
    "terminal": "output",
    "delegate_task": "results",
}


class ToolDependencyGraph:
    """
    Tracks tool call dependencies and automatically reorders or 
    injects missing prerequisite tool calls.
    """
    
    def __init__(self):
        self._tool_results: dict[str, dict] = {}  # tool_call_id -> result
        self._pending_deps: dict[str, list[str]] = {}  # tool_call_id -> required_tool_ids
        self._read_files_cache: list[str] = []  # Track recently read files (path -> timestamp)
        self._snapshot_sources: list[str] = []  # Track browser snapshot sources
        self._last_reads: dict[str, tuple[str, float]] = {}  # path -> (content, timestamp)
        
        # File definition/import graph
        self._file_definitions: dict[str, set[str]] = {}  # file path -> set of symbol names it defines
        self._file_imports: dict[str, set[str]] = {}  # file path -> set of module/function names it imports
        self._call_graph: dict[str, list[str]] = {}  # file -> list of files it depends on
        self._ast_cache: dict[str, tuple[float, Any]] = {}  # file path -> (mtime, parsed AST)
        
        # Project root for graph building
        self._project_root: Optional[str] = None
        
    def set_project_root(self, root_path: str) -> None:
        """Set the project root directory for graph building."""
        self._project_root = os.path.abspath(root_path)
        
    def _parse_file_for_graph(self, path: str, content: str) -> None:
        """
        Parse Python file using ast.parse() and update the definition/import graph.
        
        Args:
            path: Absolute or relative file path
            content: The file content to parse
        """
        try:
            tree = ast.parse(content)
        except SyntaxError:
            # If we can't parse the file, skip it
            return
            
        # Resolve to absolute path for consistent keys
        abs_path = os.path.abspath(path)
        
        # Initialize sets for this file
        if abs_path not in self._file_definitions:
            self._file_definitions[abs_path] = set()
        if abs_path not in self._file_imports:
            self._file_imports[abs_path] = set()
        if abs_path not in self._call_graph:
            self._call_graph[abs_path] = []
            
        definitions = self._file_definitions[abs_path]
        imports = self._file_imports[abs_path]
        
        # Walk the AST to collect definitions, imports, and calls
        for node in ast.walk(tree):
            # Collect function definitions
            if isinstance(node, ast.FunctionDef):
                definitions.add(node.name)
                
            # Collect class definitions
            if isinstance(node, ast.ClassDef):
                definitions.add(node.name)
                
            # Collect imports: import X -> X, import X as Y -> X, Y
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
                    
            # Collect from X import Y -> X, Y
            if isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
                for alias in node.names:
                    if alias.name != '*':
                        imports.add(alias.name)
                        
        # Store AST in cache with current mtime if file exists
        if os.path.exists(abs_path):
            mtime = os.path.getmtime(abs_path)
            self._ast_cache[abs_path] = (mtime, tree)
            
    def _resolve_import_to_file(self, import_name: str) -> Optional[str]:
        """
        Try to resolve an import name to an actual file path.
        
        Args:
            import_name: The module/import name to resolve
            
        Returns:
            Absolute file path if found, None otherwise
        """
        if self._project_root is None:
            return None
            
        # Handle relative imports within the project
        # Try common patterns: module.py, module/__init__.py
        possible_paths = [
            os.path.join(self._project_root, import_name.replace('.', os.sep) + ".py"),
            os.path.join(self._project_root, import_name.replace('.', os.sep), "__init__.py"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return os.path.abspath(path)
                
        return None
        
    def _build_impact_graph(self) -> None:
        """
        Scan the project directory for .py files and build the complete impact graph.
        
        This is an expensive operation that:
        1. Scans for all .py files in the project
        2. Parses each file to build _file_definitions and _file_imports
        3. Builds _call_graph with bidirectional dependencies
        """
        if self._project_root is None:
            return
            
        # Find all Python files in the project
        py_files = []
        for root, dirs, files in os.walk(self._project_root):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__', '.venv', 'venv', 'node_modules', '.pytest_cache')]
            for file in files:
                if file.endswith('.py'):
                    py_files.append(os.path.join(root, file))
                    
        # Process each file
        for file_path in py_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self._parse_file_for_graph(file_path, content)
            except (IOError, OSError):
                continue
                
        # Build call graph: map imports to actual files
        self._update_call_graph_from_imports()
        
    def _update_call_graph_from_imports(self) -> None:
        """
        Update the call graph by resolving imports to actual file dependencies.
        """
        for file_path, imports in list(self._file_imports.items()):
            dependencies = []
            for import_name in imports:
                resolved = self._resolve_import_to_file(import_name)
                if resolved and resolved != file_path:
                    dependencies.append(resolved)
            self._call_graph[file_path] = list(set(dependencies))
            
    def get_downstream_impact(self, file_path: str) -> list[str]:
        """
        Find ALL files that import from the given file or call its functions.
        
        Args:
            file_path: The file to find downstream impact for
            
        Returns:
            List of affected file paths (absolute paths)
        """
        abs_path = os.path.abspath(file_path)
        impacted = []
        visited = set()
        
        # Also check by symbol: find files that define symbols that this file imports
        # or files that import symbols this file defines
        target_symbols = self._file_definitions.get(abs_path, set())
        
        def _find_impact(path: str) -> None:
            """Recursively find all files impacted by the given file."""
            if path in visited:
                return
            visited.add(path)
            
            # Find all files that import from this one
            for other_file, imports in self._file_imports.items():
                if other_file == path or other_file in visited:
                    continue
                # Check if any import matches our definitions
                if any(sym in self._file_definitions.get(path, set()) for sym in imports):
                    impacted.append(other_file)
                    _find_impact(other_file)
                    
        # Direct dependents: files that import from us
        for other_file, imports in self._file_imports.items():
            if other_file == abs_path:
                continue
            # If other_file imports something we define
            our_symbols = self._file_definitions.get(abs_path, set())
            if our_symbols & imports:  # intersection
                if other_file not in impacted:
                    impacted.append(other_file)
                    
        # Build reverse graph: files that depend on us
        for other_file, deps in self._call_graph.items():
            if abs_path in deps:
                if other_file not in impacted:
                    impacted.append(other_file)
                    
        return impacted
        
    def register_result(self, tool_call_id: str, tool_name: str, 
                       result: str, args: dict):
        """Register a tool execution result for dependency tracking."""
        self._tool_results[tool_call_id] = {
            "tool_name": tool_name,
            "result": result,
            "args": args,
            "timestamp": time.time(),
        }
        
        # Track file reads for patch/write_file dependency checking
        if tool_name == "read_file" and "path" in args:
            path = args["path"]
            self._last_reads[path] = (result, time.time())
            self._read_files_cache.append(path)
            # Keep cache bounded
            if len(self._read_files_cache) > 100:
                self._read_files_cache = self._read_files_cache[-50:]
            
        # Parse file for graph when reading Python files
        if tool_name == "read_file" and "path" in args:
            path = args["path"]
            if path.endswith('.py'):
                self._parse_file_for_graph(path, result)
                
        # Track browser snapshots
        if tool_name in ("browser_snapshot", "browser_navigate"):
            self._snapshot_sources.append(tool_call_id)
            if len(self._snapshot_sources) > 50:
                self._snapshot_sources = self._snapshot_sources[-25:]
                
    def analyze_sequence(self, tool_calls: list[dict]) -> dict:
        """
        Analyze a sequence of tool calls and:
        1. Detect missing dependencies (e.g., patch without prior read)
        2. Reorder to respect dependencies
        3. Return analysis with warnings and reordered list
        4. Detect downstream impact for patch operations
        
        Returns:
          {
            "reordered": list[dict],  # tool calls in execution order
            "missing_deps": list[dict],  # {needed: tool_name, by: tool_name}
            "warnings": list[str],
            "parallel_groups": list[list[dict]],  # groups that can run in parallel
            "downstream_impact": dict[str, list[str]],  # file_path -> [affected_files]
          }
        """
        if not tool_calls:
            return {
                "reordered": [],
                "missing_deps": [],
                "warnings": [],
                "parallel_groups": [],
                "downstream_impact": {},
            }
            
        missing_deps = []
        warnings = []
        reordered = list(tool_calls)  # Start with original order
        parallel_groups = []
        downstream_impact: dict[str, list[str]] = {}
        
        # Check each tool for missing dependencies
        for i, tool_call in enumerate(tool_calls):
            tool_name = tool_call.get("name") or tool_call.get("function", {}).get("name", "")
            args = tool_call.get("args") or tool_call.get("function", {}).get("arguments", {})
            
            if isinstance(args, str):
                import json
                try:
                    args = json.loads(args)
                except:
                    args = {}
            
            # Check file_read dependency for patch/write_file
            if tool_name in ("patch", "write_file"):
                path = args.get("path", "")
                if path and path not in self._last_reads:
                    missing_deps.append({
                        "needed": "read_file",
                        "by": tool_name,
                        "for_path": path,
                        "tool_call_id": tool_call.get("id"),
                    })
                    warnings.append(
                        f"Tool '{tool_name}' targets '{path}' which was not recently read. "
                        f"Consider reading the file first to ensure proper context."
                    )
                    
            # Check browser_snapshot dependency
            if tool_name in ("browser_click", "browser_type"):
                if not self._snapshot_sources:
                    missing_deps.append({
                        "needed": "browser_snapshot",
                        "by": tool_name,
                        "tool_call_id": tool_call.get("id"),
                    })
                    warnings.append(
                        f"Tool '{tool_name}' requires a browser snapshot, but none was found. "
                        f"Consider calling browser_navigate or browser_snapshot first."
                    )
                    
            # NEW: Detect downstream impact for patch operations
            if tool_name == "patch" and "path" in args:
                path = args["path"]
                impacted = self.get_downstream_impact(path)
                if impacted:
                    downstream_impact[path] = impacted
                    warnings.append(
                        f"Tool 'patch' to '{path}' will affect {len(impacted)} downstream file(s): "
                        f"{', '.join(os.path.basename(f) for f in impacted)}"
                    )
                    
        # Group tools for parallel execution
        # Tools that don't depend on each other can run in parallel
        parallel_groups = self._compute_parallel_groups(tool_calls)
        
        return {
            "reordered": reordered,
            "missing_deps": missing_deps,
            "warnings": warnings,
            "parallel_groups": parallel_groups,
            "downstream_impact": downstream_impact,
        }
    
    def _compute_parallel_groups(self, tool_calls: list[dict]) -> list[list[dict]]:
        """
        Compute groups of tool calls that can run in parallel.
        Tools in the same group have no dependencies between them.
        """
        if not tool_calls:
            return []
            
        groups = []
        remaining = list(tool_calls)
        
        while remaining:
            group = []
            group_paths = set()
            group_snapshot_dependent = False
            
            for tool in remaining[:]:
                tool_name = tool.get("name") or tool.get("function", {}).get("name", "")
                args = tool.get("args") or tool.get("function", {}).get("arguments", {})
                
                if isinstance(args, str):
                    import json
                    try:
                        args = json.loads(args)
                    except:
                        args = {}
                
                # Check if this tool can be added to the current group
                can_add = True
                
                # Edit tools (patch, write_file) need exclusive file access
                if tool_name in ("patch", "write_file"):
                    path = args.get("path", "")
                    if path in group_paths or group_snapshot_dependent:
                        can_add = False
                    else:
                        group_paths.add(path)
                        
                # Browser tools that need snapshots are sequential
                if tool_name in ("browser_click", "browser_type") and group:
                    can_add = False
                    
                # Snapshot tools enable subsequent browser tools
                if tool_name in ("browser_snapshot", "browser_navigate") and group:
                    can_add = False
                    
                if can_add:
                    group.append(tool)
                    remaining.remove(tool)
            
            if not group and remaining:
                # Safety: add first remaining tool if stuck
                group = [remaining.pop(0)]
                
            if group:
                groups.append(group)
                
        return groups
        
    def check_read_before_edit(self, tool_name: str, args: dict) -> dict | None:
        """
        For edit tools (patch, write_file), check if the target file 
        was recently read. If not, return a synthetic 'read_file' call
        that should be injected first.
        
        Returns synthetic tool call dict or None.
        """
        if tool_name not in ("patch", "write_file"):
            return None
            
        path = args.get("path", "")
        if not path:
            return None
            
        # Check if file was recently read
        if path in self._last_reads:
            return None
            
        # Generate synthetic read_file call
        import uuid
        return {
            "id": f"dep_read_{uuid.uuid4().hex[:8]}",
            "name": "read_file",
            "args": {"path": path},
            "synthetic": True,
        }
        
    def get_cached_result(self, tool_name: str, 
                          match_fn: Callable[[dict], bool] = None) -> str | None:
        """Get a cached tool result by tool name, optionally filtered."""
        results = []
        
        for tool_call_id, info in self._tool_results.items():
            if info["tool_name"] == tool_name:
                if match_fn is None or match_fn(info):
                    results.append((info, tool_call_id))
        
        if not results:
            return None
            
        # Return most recent result
        results.sort(key=lambda x: x[0].get("timestamp", 0), reverse=True)
        return results[0][0].get("result")
    
    def get_last_read_content(self, path: str) -> tuple[str, float] | None:
        """Get the content and timestamp of the last read for a path."""
        if path in self._last_reads:
            return self._last_reads[path]
        return None
        
    def has_recent_read(self, path: str, max_age_seconds: float = 300) -> bool:
        """Check if a file was read within the last max_age_seconds."""
        if path not in self._last_reads:
            return False
        _, timestamp = self._last_reads[path]
        return (time.time() - timestamp) < max_age_seconds
        
    def propagate_rename(self, old_path: str, new_path: str, 
                        old_symbol: str, new_symbol: str) -> dict[str, list[str]]:
        """
        When a function/class is renamed, propagate the rename info to all dependent files.
        
        Args:
            old_path: The original file path
            new_path: The new file path (if file was renamed) or same path
            old_symbol: The original symbol name
            new_symbol: The new symbol name
            
        Returns:
            Dictionary with 'updated_files' list and 'errors' list
        """
        results = {
            "updated_files": [],
            "errors": [],
            "propagation_details": [],
        }
        
        abs_old_path = os.path.abspath(old_path)
        abs_new_path = os.path.abspath(new_path)
        
        # Find all files that define or import the old symbol
        affected_files = []
        
        # Files that define the old symbol
        if abs_old_path in self._file_definitions:
            if old_symbol in self._file_definitions[abs_old_path]:
                # This file defines the symbol - update its definitions
                self._file_definitions[abs_old_path].discard(old_symbol)
                self._file_definitions[abs_old_path].add(new_symbol)
                
        # Files that import the old symbol
        for file_path, imports in list(self._file_imports.items()):
            if old_symbol in imports:
                affected_files.append(file_path)
                
        # If the file itself was renamed, update the data structures
        if old_path != new_path:
            if abs_old_path in self._file_definitions:
                self._file_definitions[abs_new_path] = self._file_definitions.pop(abs_old_path)
            if abs_old_path in self._file_imports:
                self._file_imports[abs_new_path] = self._file_imports.pop(abs_old_path)
            if abs_old_path in self._call_graph:
                self._call_graph[abs_new_path] = self._call_graph.pop(abs_old_path)
            if abs_old_path in self._ast_cache:
                self._ast_cache[abs_new_path] = self._ast_cache.pop(abs_old_path)
                
        # Record propagation details
        results["propagation_details"].append({
            "type": "rename",
            "old_path": old_path,
            "new_path": new_path,
            "old_symbol": old_symbol,
            "new_symbol": new_symbol,
            "affected_importers": affected_files,
        })
        
        results["updated_files"] = affected_files
        
        return results
        
    def get_file_definitions(self, file_path: str) -> set[str]:
        """Get the set of symbols defined in a file."""
        abs_path = os.path.abspath(file_path)
        return self._file_definitions.get(abs_path, set()).copy()
        
    def get_file_imports(self, file_path: str) -> set[str]:
        """Get the set of symbols imported by a file."""
        abs_path = os.path.abspath(file_path)
        return self._file_imports.get(abs_path, set()).copy()
        
    def get_call_graph(self, file_path: str) -> list[str]:
        """Get the list of files that the given file depends on."""
        abs_path = os.path.abspath(file_path)
        return self._call_graph.get(abs_path, []).copy()
        
    def invalidate_cache(self, file_path: str) -> None:
        """Invalidate the AST cache for a file when it changes."""
        abs_path = os.path.abspath(file_path)
        if abs_path in self._ast_cache:
            del self._ast_cache[abs_path]
        if abs_path in self._file_definitions:
            del self._file_definitions[abs_path]
        if abs_path in self._file_imports:
            del self._file_imports[abs_path]
        if abs_path in self._call_graph:
            del self._call_graph[abs_path]

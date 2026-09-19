"""Import-free channel ownership for offline profile operations.

Runtime discovery is not a metadata API: it writes config backups, activates
plugins and refreshes secret providers. Read installed manifests and literal
registration declarations instead. Dynamic declarations are refused rather than
silently retaining a custom bot's credentials in a supposedly channel-less clone.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ChannelDeclaration:
    name: str
    required_env: list[str] = field(default_factory=list)
    allowed_users_env: str = ""
    allow_all_env: str = ""
    cron_deliver_env_var: str = ""


_FIELDS = ("name", "required_env", "allowed_users_env", "allow_all_env", "cron_deliver_env_var")


def _literal_metadata(node: ast.AST, assignments: dict[str, list[ast.AST]], seen=()):
    """Resolve only unambiguous literal bindings, never execute plugin expressions."""
    if isinstance(node, ast.Name):
        candidates = assignments.get(node.id, [])
        if node.id in seen or len(candidates) != 1:
            raise ValueError("ambiguous metadata binding")
        return _literal_metadata(candidates[0], assignments, (*seen, node.id))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict":
        if node.args:
            raise ValueError("dynamic dictionary")
        result = {}
        for keyword in node.keywords:
            value = _literal_metadata(keyword.value, assignments, seen)
            if keyword.arg is None:
                if not isinstance(value, dict):
                    raise ValueError("dynamic dictionary expansion")
                result.update(value)
            else:
                result[keyword.arg] = value
        return result
    return ast.literal_eval(node)


def _registration_symbols(tree: ast.AST) -> dict[str, str]:
    """Imports may rename a declaration, but runtime callable aliases are opaque."""
    symbols = {name: name for name in ("register_platform", "PlatformEntry")}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in symbols:
                    symbols[alias.asname or alias.name] = symbols[alias.name]
    return symbols


def _registration_name(node: ast.AST, symbols: dict[str, str]) -> str:
    if isinstance(node, ast.Name):
        return symbols.get(node.id, "")
    if isinstance(node, ast.Attribute):
        return symbols.get(node.attr, "")
    return ""


_CONTEXT_METHODS = {"register_platform", "register_tool", "register_hook", "register_cli_command"}
_LOG_METHODS = {"debug", "info", "warning", "error", "exception", "critical"}
# Reviewed read-only leaves used by bundled declaration helpers. Never allow an
# arbitrary import merely because its local name resembles one of these.
_READ_ONLY_CALLS = {
    "gateway.platforms._shared.profile_scoped",
    "agent.secret_scope.get_secret",
    "tools.lazy_deps.feature_install_command",
    "os.environ.get",
}


def _scope_bindings(tree: ast.Module, scope: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, list[object]]:
    """Collect possible bindings without guessing assignment/branch execution order."""
    def collect(statements):
        bindings: dict[str, list[object]] = {}
        pending = list(statements)
        while pending:
            node = pending.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bindings.setdefault(node.name, []).append(node)
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    local = alias.asname or (alias.name.split(".")[0] if isinstance(node, ast.Import) else alias.name)
                    value = (node, alias) if isinstance(node, ast.ImportFrom) and node.level else (
                        f"{node.module}.{alias.name}" if isinstance(node, ast.ImportFrom) else alias.name)
                    bindings.setdefault(local, []).append(value)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                bindings.setdefault(node.id, []).append(node)
            pending.extend(ast.iter_child_nodes(node))
        return bindings

    bindings = collect(tree.body)
    local = collect(scope.body)
    for arg in (*scope.args.posonlyargs, *scope.args.args, *scope.args.kwonlyargs,
                *([scope.args.vararg] if scope.args.vararg else []),
                *([scope.args.kwarg] if scope.args.kwarg else [])):
        local.setdefault(arg.arg, []).append(arg)
    bindings.update(local)
    # A method/subscript write invalidates the receiver's provenance too; otherwise
    # logger.info = opaque would retain the logging.getLogger allowlist identity.
    for node in ast.walk(scope):
        if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(node.ctx, (ast.Store, ast.Del)):
            root = node.value
            while isinstance(root, (ast.Attribute, ast.Subscript)):
                root = root.value
            if isinstance(root, ast.Name):
                bindings.setdefault(root.id, []).append(node)
    return bindings


def _validate_scope_calls(scope: ast.FunctionDef | ast.AsyncFunctionDef, tree: ast.Module, source: Path,
                          *, trees: dict[Path, ast.Module], context: str = "", seen=()) -> None:
    """Check every explicit call, not a list of statement/value positions.

    Function bodies containing declarations and register() are conservative may-
    execute scopes: ast.walk covers every expression position, including function
    defaults and annotations. Only lambda bodies passed directly as registration
    callbacks are deferred; their defaults still execute and are checked. Local
    and one-level relative helpers are recursively checked with cycle refusal.

    The call grammar is direct context registration, the real PlatformEntry import,
    unshadowed dict, provenance-checked logging/literal-dict items/string methods,
    reviewed read-only core leaves, and inspectable helpers. Ambiguous bindings,
    runtime callable aliases and decorators are refused. This is not a Python
    sandbox: imports/module initialization and implicit protocol calls (operators,
    descriptors, iteration) are not interpreted. The no-omission guarantee is for
    explicit registration calls in these scopes, not arbitrary plugin execution.
    No helper or plugin is imported.
    """
    identity = (source, scope.name)
    if identity in seen:
        raise ValueError("recursive registration helper")
    seen = (*seen, identity)
    bindings = _scope_bindings(tree, scope)
    parents = {id(child): parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}

    def binding(name):
        values = bindings.get(name, [])
        return values[0] if len(values) == 1 else None

    def qualified(node):
        if isinstance(node, ast.Name):
            value = binding(node.id)
            return value if isinstance(value, str) else ""
        if isinstance(node, ast.Attribute):
            root = qualified(node.value)
            return f"{root}.{node.attr}" if root else ""
        return ""

    def context_method(node):
        return (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == context and isinstance(binding(context), ast.arg)
                and node.attr in _CONTEXT_METHODS)

    def declaration(node):
        return context_method(node) or qualified(node) == "gateway.platform_registry.PlatformEntry"

    def assigned_value(node):
        if not isinstance(node, ast.Name):
            return None
        target = binding(node.id)
        parent = parents.get(id(target))
        if isinstance(parent, (ast.Assign, ast.AnnAssign)):
            return parent.value
        return None

    def helper(node):
        if not isinstance(node, ast.Name):
            return None
        value = binding(node.id)
        if isinstance(value, ast.FunctionDef):
            return value, tree, source
        if isinstance(value, tuple):
            imported, alias = value
            if imported.level != 1 or not imported.module:
                return None
            path = source.parent.joinpath(*imported.module.split(".")).with_suffix(".py").resolve()
            if path.is_file():
                if path not in trees:
                    trees[path] = ast.parse(path.read_text(encoding="utf-8-sig"))
                parsed = trees[path]
                candidates = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == alias.name]
                if len(candidates) == 1:
                    return candidates[0], parsed, path
        return None

    def string_value(node):
        if isinstance(node, ast.Constant):
            return isinstance(node.value, str)
        if isinstance(node, ast.BoolOp):
            return all(string_value(value) for value in node.values)
        return (isinstance(node, ast.Call) and qualified(node.func) in {
            "agent.secret_scope.get_secret", "os.environ.get",
        })

    deferred = set()
    for node in ast.walk(scope):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.decorator_list:
            # Even @name implies a call; a decorator factory's return value is
            # not a proven callable just because its factory was inspectable.
            raise ValueError("decorated registration scope")
        if not isinstance(node, ast.Lambda):
            continue
        parent = parents.get(id(node))
        owner = parents.get(id(parent))
        if (isinstance(parent, ast.keyword) and parent.arg not in _FIELDS
                and isinstance(owner, ast.Call) and declaration(owner.func)):
            deferred.update(id(child) for child in ast.walk(node.body))

    for node in ast.walk(scope):
        if id(node) in deferred:
            continue
        if isinstance(node, ast.Name) and node.id == context:
            parent = parents.get(id(node))
            owner = parents.get(id(parent))
            if (isinstance(owner, ast.Call) and owner.func is parent and context_method(parent)):
                continue
            if isinstance(parent, ast.Call) and node in parent.args and helper(parent.func):
                continue
            raise ValueError("plugin context escapes direct registration")
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if declaration(function):
            continue
        if isinstance(function, ast.Name) and function.id == "dict" and "dict" not in bindings:
            # Both dictionary construction and its children are checked; ownership
            # keywords additionally pass the literal metadata resolver.
            continue
        if qualified(function) in _READ_ONLY_CALLS:
            continue
        if isinstance(function, ast.Attribute):
            receiver = assigned_value(function.value)
            if (function.attr in _LOG_METHODS and isinstance(receiver, ast.Call)
                    and qualified(receiver.func) == "logging.getLogger"):
                continue
            if function.attr == "items" and isinstance(receiver, ast.Dict):
                continue
            if function.attr == "format" and isinstance(function.value, ast.Constant) and isinstance(function.value.value, str):
                continue
            if function.attr == "strip" and string_value(function.value):
                # Nested calls are independently validated by this same walk.
                continue
        target = helper(function)
        if target is not None:
            definition, helper_tree, helper_source = target
            helper_context = ""
            if context and any(isinstance(arg, ast.Name) and arg.id == context for arg in node.args):
                if not definition.args.args or not node.args or not isinstance(node.args[0], ast.Name) or node.args[0].id != context:
                    raise ValueError("unsupported helper context binding")
                helper_context = definition.args.args[0].arg
            _validate_scope_calls(definition, helper_tree, helper_source,
                                  trees=trees, context=helper_context, seen=seen)
            continue
        raise ValueError("opaque registration helper")


def _check_registration_subset(tree: ast.Module, symbols: dict[str, str], source: Path,
                               *, platform: bool, trees: dict[Path, ast.Module]) -> None:
    """Reject opaque registration even when a sibling declaration is inspectable.

    This is a declaration reader, not a Python interpreter: direct calls (including
    imported aliases) and literal bindings are supported; callable escapes,
    reflection on registration APIs, and generated code are not. All syntactic
    branches are inventoried, irrespective of runtime reachability.
    """
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    direct = {id(call.func) for call in calls}
    for node in ast.walk(tree):
        if _registration_name(node, symbols) and id(node) not in direct:
            raise ValueError("indirect channel registration")
        if isinstance(node, ast.Name) and node.id in {"exec", "eval", "compile"}:
            raise ValueError("generated Python code")
        if isinstance(node, ast.Attribute) and node.attr in {"exec", "eval"}:
            raise ValueError("generated Python code")
        if isinstance(node, ast.ImportFrom):
            if any(alias.name == "*" for alias in node.names):
                raise ValueError("wildcard imports hide registration bindings")
            if node.module == "builtins" and any(alias.name in {"exec", "eval", "compile"} for alias in node.names):
                raise ValueError("generated Python code")
    # Adapter methods may use reflection for ordinary message dispatch. Only the
    # module body and declaration/registration functions are registration surfaces;
    # deferred adapter callbacks are not executed by this inventory.
    registration_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            (platform and node.name == "register") or any(
                isinstance(child, ast.Call) and _registration_name(child.func, symbols)
                for child in ast.walk(node)
            )
        ):
            registration_nodes.update(id(child) for child in ast.walk(node))
            _validate_scope_calls(node, tree, source,
                                  trees=trees, context=node.args.args[0].arg if node.args.args else "")
    registration_nodes.update(id(node) for statement in tree.body
                              if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                              for node in ast.walk(statement))
    for call in calls:
        if id(call) not in registration_nodes:
            continue
        name = call.func.id if isinstance(call.func, ast.Name) else ""
        if name in {"getattr", "setattr", "delattr"} and len(call.args) > 1:
            attribute = call.args[1]
            if not isinstance(attribute, ast.Constant) or attribute.value in symbols:
                raise ValueError("dynamic registration attribute")
        if isinstance(call.func, (ast.Call, ast.Subscript)):
            raise ValueError("computed callable may hide channel registration")


def _declarations(path: Path, *, platform: bool = True) -> list[ChannelDeclaration]:
    """Read literal metadata only; never eval/import a registration or its helpers."""
    declarations = []
    trees: dict[Path, ast.Module] = {}
    for source in ([path] if path.is_file() else sorted(path.rglob("*.py"))):
        if any(part in {".git", ".venv", "venv", "__pycache__", "node_modules"}
               for part in source.relative_to(path).parts):
            continue
        source = source.resolve()
        if source not in trees:
            trees[source] = ast.parse(source.read_text(encoding="utf-8-sig"))
    # Validation adds only helpers it actually follows. Extract from that same
    # graph, once per source, including helpers outside a module entry point.
    checked = set()
    while len(checked) < len(trees):
        source = next(source for source in trees if source not in checked)
        tree = trees[source]
        _check_registration_subset(tree, _registration_symbols(tree), source,
                                   platform=platform, trees=trees)
        checked.add(source)
    for tree in trees.values():
        symbols = _registration_symbols(tree)
        keyword_values = {id(keyword.value) for call in ast.walk(tree) if isinstance(call, ast.Call)
                          for keyword in call.keywords if keyword.arg is None or keyword.arg in _FIELDS}
        assignments: dict[str, list[ast.AST]] = {}
        for candidate in ast.walk(tree):
            if isinstance(candidate, ast.Assign):
                for target in candidate.targets:
                    if isinstance(target, ast.Name):
                        assignments.setdefault(target.id, []).append(candidate.value)
            elif isinstance(candidate, ast.AnnAssign) and isinstance(candidate.target, ast.Name):
                assignments.setdefault(candidate.target.id, []).append(candidate.value or candidate)
            # Attribute/subscript access could mutate a dictionary after its literal
            # assignment. Refuse that binding rather than invent execution order.
            elif isinstance(candidate, ast.Name) and isinstance(candidate.ctx, ast.Load) and id(candidate) not in keyword_values:
                assignments.setdefault(candidate.id, []).append(candidate)
            elif isinstance(candidate, (ast.Attribute, ast.Subscript)) and isinstance(candidate.value, ast.Name):
                assignments.setdefault(candidate.value.id, []).append(candidate)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _registration_name(node.func, symbols)
            if not name:
                continue
            try:
                values = {}
                for keyword in node.keywords:
                    if keyword.arg is None:
                        expanded = _literal_metadata(keyword.value, assignments)
                        if not isinstance(expanded, dict) or not all(isinstance(key, str) for key in expanded):
                            raise ValueError("dynamic registration keywords")
                        values.update({key: value for key, value in expanded.items() if key in _FIELDS})
                    elif keyword.arg in _FIELDS:
                        values[keyword.arg] = _literal_metadata(keyword.value, assignments)
                if node.args:
                    # register_platform is keyword-only. PlatformEntry accepts a
                    # positional name, but later fields include credentials too.
                    if name != "PlatformEntry" or len(node.args) != 1:
                        raise ValueError("unsupported positional channel metadata")
                    values["name"] = _literal_metadata(node.args[0], assignments)
                platform_name = values.get("name")
                if not isinstance(platform_name, str) or not platform_name.strip():
                    raise ValueError("nonliteral platform id")
                if any(not isinstance(value, str) for key, value in values.items() if key != "required_env"):
                    raise ValueError("nonliteral env name")
                required = values.get("required_env", [])
                if not isinstance(required, (list, tuple)) or not all(isinstance(key, str) and key.strip() for key in required):
                    raise ValueError("nonliteral required env names")
                declarations.append(ChannelDeclaration(**values))
            except (ValueError, TypeError):
                raise ValueError(
                    "Cannot safely inventory channel ownership without activating a plugin. "
                    "Use literal platform names and environment-key declarations in register_platform, "
                    "then retry cloning."
                ) from None
    return declarations


def channel_declarations() -> list[ChannelDeclaration]:
    """Inventory installed ownership in the caller's source-home scope, without activation.

    Include disabled installations too: disabling an adapter does not make its
    stored token safe to inherit. No config loader or enablement callback runs.
    """
    from hermes_cli.plugins_discovery import collect_directory_manifests, discover_entrypoint_manifests
    from hermes_cli.plugins_manifest import manifest_key, resolve_module_origin

    manifests = {manifest_key(m): m for m in collect_directory_manifests(strict=True)}
    for manifest in discover_entrypoint_manifests(strict=True):
        manifests.setdefault(manifest_key(manifest), manifest)
    entries = {}
    for manifest in manifests.values():
        if manifest.portable or manifest.kind in {"exclusive", "model-provider"}:
            continue
        if not manifest.path:
            raise ValueError("Cannot safely inventory an installed plugin without its source path.")
        if manifest.source == "entrypoint":
            origin = resolve_module_origin(str(manifest.path).split(":", 1)[0])
            if not origin:
                raise ValueError("Cannot safely inventory an installed plugin; repair its installation and retry cloning.")
            origin_path = Path(origin)
            path = origin_path.parent if origin_path.name == "__init__.py" else origin_path
        else:
            path = Path(manifest.path)
        required_env: list[str] = []
        if manifest.kind == "platform":
            metadata_error = (
                f"Cannot safely inventory channel ownership for plugin {manifest.name!r}: "
                "requires_env must be a list of nonempty strings or mappings with a nonempty "
                "string name; repair the plugin metadata and retry cloning."
            )
            if not isinstance(manifest.requires_env, (list, tuple)):
                raise ValueError(metadata_error)
            for item in manifest.requires_env:
                key = item.get("name") if isinstance(item, dict) else item
                if not isinstance(key, str) or not key.strip():
                    raise ValueError(metadata_error)
                required_env.append(key)
        try:
            declarations = _declarations(path, platform=manifest.kind == "platform")
        except (ValueError, SyntaxError, OSError, UnicodeError):
            raise ValueError(
                f"Cannot safely inventory channel ownership for plugin {manifest.name!r}. "
                "Use direct literal register_platform/PlatformEntry declarations without "
                "generated code, reflection or callable aliases; repair the plugin and retry cloning."
            ) from None
        if not declarations and manifest.kind == "platform":
            raise ValueError(
                f"Cannot safely inventory channel ownership for plugin {manifest.name!r} "
                "without literal platform declarations; repair the plugin metadata and retry cloning."
            )
        for entry in declarations:
            # Manifests are an additional source for non-canonical credential names.
            if manifest.kind == "platform":
                entry.required_env = list(entry.required_env)
                entry.required_env.extend(required_env)
            previous = entries.get(entry.name)
            if previous is not None:
                # Ownership is a may-set, not the runtime registry's last-writer
                # winner. Never guess which conditional declaration will execute.
                for key in _FIELDS[2:]:
                    if getattr(previous, key) != getattr(entry, key):
                        raise ValueError(
                            f"Cannot safely inventory plugin {manifest.name!r}: conflicting "
                            f"literal policy metadata for platform {entry.name!r}; "
                            "make duplicate declarations coherent and retry cloning."
                        )
                entry.required_env = list(dict.fromkeys((*previous.required_env, *entry.required_env)))
            entries[entry.name] = entry
    return list(entries.values())

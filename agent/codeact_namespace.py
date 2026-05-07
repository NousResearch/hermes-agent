"""CodeAct namespace builder.

Generates two artefacts from the live Hermes tool registry:

1. ``build_tool_namespace_source()``  — Python source that, when exec()'d into
   the kernel's globals_dict, defines one stub function per registered tool plus
   a ``help()`` introspection function.

2. ``build_compact_tool_catalogue()`` — A short human-readable string (one line
   per tool) embedded in the ``run_code`` tool description so the model always
   has a compact reference of what's callable.
"""

from __future__ import annotations

import json
import textwrap
from agent.codeact_recipes import (
    build_recipe_help_registry,
    build_recipe_source,
    get_recipe_names,
)
from agent.codeact_skill_injector import SkillNamespaceInjector
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# JSON → Python type mapping
# ---------------------------------------------------------------------------

_JSON_TO_PYTHON: dict[str, str] = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
    "null": "None",
}

# Toolsets that are purely internal / infrastructure — exclude from CodeAct
# namespace because they're either meta-operations or conflict with the kernel.
_EXCLUDED_TOOLSETS: frozenset[str] = frozenset(
    {
        "codeact",  # run_code itself — don't expose recursively
    }
)

# Tool names that are always excluded regardless of toolset.
_EXCLUDED_TOOLS: frozenset[str] = frozenset(
    {
        "run_code",  # self-reference guard
        "execute_code",  # the old one-shot sandbox — callable from within CodeAct
        # but not as a direct stub (it has its own IPC machinery)
    }
)


def _scrapling_is_available() -> bool:
    """Return True if the scrapling optional-skill is installed.

    Checks both the user's ``~/.hermes/skills/`` tree and the
    repo-shipped ``optional-skills/`` directory.
    """
    from pathlib import Path

    try:
        from hermes_constants import get_hermes_home

        if (Path(get_hermes_home()) / "skills" / "scrapling" / "SKILL.md").exists():
            return True
    except Exception:
        pass
    return (
        Path(__file__).resolve().parent.parent
        / "optional-skills"
        / "research"
        / "scrapling"
        / "SKILL.md"
    ).exists()


def _codeact_research_web_search_redirect_enabled() -> bool:
    """Return whether CodeAct should auto-route research-shaped web_search calls."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        codeact = cfg.get("codeact") if isinstance(cfg, dict) else {}
        research = codeact.get("research") if isinstance(codeact, dict) else {}
        if not isinstance(research, dict):
            return True
        return bool(research.get("redirect_web_search", True))
    except Exception:
        return True


def _json_type_to_python(json_type: str | list | None) -> str:
    """Return a Python type annotation string for a JSON Schema type field."""
    if json_type is None:
        return "object"
    if isinstance(json_type, list):
        parts = [_JSON_TO_PYTHON.get(t, "object") for t in json_type if t != "null"]
        nullable = "null" in json_type
        if not parts:
            return "object"
        base = " | ".join(parts) if len(parts) > 1 else parts[0]
        return f"{base} | None" if nullable else base
    return _JSON_TO_PYTHON.get(json_type, "object")


def _schema_to_compact_line(name: str, schema: dict) -> str:
    """Return a one-line summary: ``web_search(query, limit=5) — search the web``."""
    params = schema.get("parameters", {}).get("properties", {})
    required = set(schema.get("parameters", {}).get("required", []))
    raw_desc = schema.get("description", "")
    short_desc = raw_desc.split("\n")[0][:100]

    sig_parts: list[str] = []
    for pname, pschema in params.items():
        default = pschema.get("default")
        if pname not in required and default is not None:
            sig_parts.append(f"{pname}={default!r}")
        elif pname not in required:
            sig_parts.append(f"{pname}=None")
        else:
            sig_parts.append(pname)

    sig = f"{name}({', '.join(sig_parts)})"
    return f"  {sig:<45} — {short_desc}"


def _generate_stub_lines(name: str, schema: dict) -> list[str]:
    """Return lines of Python source defining one tool stub function."""
    params = schema.get("parameters", {}).get("properties", {})
    required = set(schema.get("parameters", {}).get("required", []))
    raw_desc = schema.get("description", "")
    short_desc = raw_desc.split("\n")[0][:120]

    # Build Python parameter list: required params first (no default), then optional.
    required_params = [(n, s) for n, s in params.items() if n in required]
    optional_params = [(n, s) for n, s in params.items() if n not in required]
    ordered = required_params + optional_params

    sig_parts: list[str] = []
    for pname, pschema in ordered:
        py_type = _json_type_to_python(pschema.get("type"))
        default = pschema.get("default")
        if pname in required:
            sig_parts.append(f"{pname}: {py_type}")
        elif default is not None:
            sig_parts.append(f"{pname}: {py_type} = {default!r}")
        else:
            sig_parts.append(f"{pname}: {py_type} = None")

    sig = f"def {name}({', '.join(sig_parts)}):"

    # Full docstring — short desc + parameter descriptions
    param_doc_lines = []
    for pname, pschema in ordered:
        pdesc = pschema.get("description", "").replace("\n", " ").strip()
        param_doc_lines.append(f"    {pname}: {pdesc}")

    if param_doc_lines:
        docstring = (
            f'    """{short_desc}\n\n' + "\n".join(param_doc_lines) + '\n    """'
        )
    else:
        docstring = f'    """{short_desc}"""'

    # Body: collect non-None kwargs, call IPC bridge
    body = [
        "    _kwargs = {k: v for k, v in locals().items()",
        "               if not k.startswith('_') and v is not None}",
        f"    return _call_tool({name!r}, _kwargs)",
    ]

    return [sig, docstring] + body


def _collect_enabled_entries(
    registry: "ToolRegistry",
    enabled_tool_names: set[str] | None,
) -> list:
    """Return ToolEntry objects that should appear in the CodeAct namespace."""
    entries = registry._snapshot_entries()
    result = []
    for entry in entries:
        if entry.toolset in _EXCLUDED_TOOLSETS:
            continue
        if entry.name in _EXCLUDED_TOOLS:
            continue
        if enabled_tool_names is not None and entry.name not in enabled_tool_names:
            continue
        if entry.schema is None:
            continue
        result.append(entry)
    return sorted(result, key=lambda e: e.name)


def _build_workflow_guidance_for_tool_names(tool_names: set[str]) -> str:
    """Return compact task recipes for the tools available in this namespace."""
    lines = [
        "Fast CodeAct workflow rules:",
        "- Hermes tools are already global Python functions; do not import them.",
        "- help() returns a string; use print(help('tool_name')) only when you need docs.",
        "- Try the purpose-built Hermes tool before probing Python packages, "
        "environments, or installing dependencies.",
    ]

    if "vision_analyze" in tool_names:
        lines.append(
            "- Image/OCR/translation: if the user provides an image path or URL, "
            "first call vision_analyze(image_url=path, question='Extract all "
            "visible text, preserve reading order, and translate or describe "
            "as requested.'). "
            "Do not start with PIL/OCR/package installs unless vision_analyze "
            "fails or is unavailable."
        )

    if "research_gather" in tool_names:
        lines.append(
            "- Search/research/report/latest/current/as-of-date tasks: FIRST call "
            "result = research_web(question=USER_REQUEST, freshness='latest', "
            "depth='thorough'). For drug/clinical-trial/pharma work, FIRST call "
            "medical_pharma_research(question=USER_REQUEST). Do not start with "
            "raw web_search, help('web_search'), sys.modules/namespace probing, "
            "browser_navigate, curl, or Wikipedia. The bundle includes "
            "source_table/citation_metadata; "
            "final research reports must cite those sources outside run_code. "
            "If rate-limited, JS-challenged, bot-blocked, or Cloudflare/Wikipedia "
            "blocked, try browser/Camofox/Scrapling fallbacks when available "
            "and record the limitation."
        )

    if "web_search" in tool_names or "web_extract" in tool_names:
        web_bits = []
        if "web_search" in tool_names:
            web_bits.append("web_search(query=...) for discovery")
        if "web_extract" in tool_names:
            web_bits.append("web_extract(urls=[...]) for page contents")
        lines.append(
            "- Raw web tools: use "
            + " then ".join(web_bits)
            + " only for targeted discovery/extraction. For reports or current facts, prefer research_web first."
        )

    if "read_file" in tool_names or "search_files" in tool_names:
        file_bits = []
        if "search_files" in tool_names:
            file_bits.append("search_files(pattern=...)")
        if "read_file" in tool_names:
            file_bits.append("read_file(path=...)")
        lines.append(
            "- Repository/file work: prefer "
            + " and ".join(file_bits)
            + " before shelling out."
        )

    if "terminal" in tool_names:
        lines.append(
            "- Shell/system work: use terminal(command=...) only for real OS actions, "
            "tests, package commands, or commands unavailable as Hermes tools."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_compact_tool_catalogue(
    registry: "ToolRegistry",
    enabled_tool_names: set[str] | None = None,
) -> str:
    """Return a compact multi-line string listing all CodeAct-available tools.

    Grouped by toolset.  Used as part of the ``run_code`` tool description so
    the model always has a quick reference of what's callable.
    """
    entries = _collect_enabled_entries(registry, enabled_tool_names)

    # Group by toolset
    by_toolset: dict[str, list] = {}
    for entry in entries:
        by_toolset.setdefault(entry.toolset, []).append(entry)

    lines: list[str] = []
    for toolset in sorted(by_toolset):
        lines.append(f"## {toolset}")
        for entry in by_toolset[toolset]:
            lines.append(_schema_to_compact_line(entry.name, entry.schema))
        lines.append("")

    return "\n".join(lines).rstrip()


def build_codeact_workflow_guidance(
    registry: "ToolRegistry",
    enabled_tool_names: set[str] | None = None,
) -> str:
    """Return compact CodeAct task recipes for the enabled tool namespace."""
    entries = _collect_enabled_entries(registry, enabled_tool_names)
    return _build_workflow_guidance_for_tool_names({entry.name for entry in entries})


def build_tool_namespace_source(
    registry: "ToolRegistry",
    enabled_tool_names: set[str] | None = None,
    skill_injector: "SkillNamespaceInjector | None" = None,
    explicitly_loaded_skills: set[str] | None = None,
) -> str:
    """Return Python source that defines all tool stubs + ``help()`` + ``promote_to_skill()``.

    This source is exec()'d into the kernel's globals_dict at init time.
    ``_call_tool`` must already be present in the namespace before this source
    is exec()'d — it's injected by the kernel process separately.

    Parameters
    ----------
    registry:
        The live Hermes tool registry.
    enabled_tool_names:
        Set of tool names to include.  ``None`` means all non-excluded tools.
    skill_injector:
        Optional ``SkillNamespaceInjector`` instance.  When provided, callable
        skill functions are generated and included in the namespace, and their
        names are added to ``__protected__`` so they survive ``soft_reset``.
    explicitly_loaded_skills:
        Set of skill names explicitly loaded via ``/skill``.  Passed through
        to the injector's selection logic.
    """
    entries = _collect_enabled_entries(registry, enabled_tool_names)
    workflow_guidance = _build_workflow_guidance_for_tool_names(
        {entry.name for entry in entries}
    )
    enabled_names = {entry.name for entry in entries}
    web_search_redirect_enabled = _codeact_research_web_search_redirect_enabled()

    # Build full help registry: {name: (signature_line, full_docstring)}
    help_registry: dict[str, tuple[str, str]] = {}
    for entry in entries:
        compact = _schema_to_compact_line(entry.name, entry.schema).strip()
        full_doc = entry.schema.get("description", "No description available.")
        params = entry.schema.get("parameters", {}).get("properties", {})
        param_lines = []
        for pname, pschema in params.items():
            pdesc = pschema.get("description", "").replace("\n", " ").strip()
            ptype = _json_type_to_python(pschema.get("type"))
            param_lines.append(f"  {pname} ({ptype}): {pdesc}")
        full_text = full_doc
        if param_lines:
            full_text += "\n\nParameters:\n" + "\n".join(param_lines)
        help_registry[entry.name] = (compact, full_text)

    recipe_source = build_recipe_source(enabled_names)
    recipe_names = get_recipe_names(enabled_names)
    help_registry.update(build_recipe_help_registry(enabled_names))

    # When web_extract is present and scrapling is installed, annotate
    # the help entry so the model knows it can fall back to
    # StealthyFetcher for bot-protected pages.
    if "web_extract" in help_registry and _scrapling_is_available():
        _scrapling_hint = (
            " If a page is blocked by bot detection or Cloudflare, you can use"
            " Scrapling's StealthyFetcher in a code cell as a fallback"
            " (see optional-skills/research/scrapling for examples)."
        )
        compact, full = help_registry["web_extract"]
        if "StealthyFetcher" not in full:
            help_registry["web_extract"] = (compact, full + _scrapling_hint)

    if (
        web_search_redirect_enabled
        and "web_search" in help_registry
        and "research_gather" in enabled_names
    ):
        _search_redirect_hint = (
            " In CodeAct, research-shaped queries are classified by "
            "agent.research_search.intent.classify_research_intent and "
            "automatically routed through research_web/research_gather so the result "
            "includes citation metadata and gap analysis. This is configurable "
            "with codeact.research.redirect_web_search. Call research_web(...) "
            "directly for reports."
        )
        compact, full = help_registry["web_search"]
        if "automatically routed through research_web/research_gather" not in full:
            help_registry["web_search"] = (compact, full + _search_redirect_hint)

    if "browser_navigate" in help_registry:
        _camofox_hint = (
            " If web_search/web_extract hits rate limits, JS challenges, "
            "or bot protection, use browser_navigate/browser_snapshot as a "
            "rendered fallback; when Camofox is configured, browser tools route "
            "through the Camofox anti-detection browser."
        )
        compact, full = help_registry["browser_navigate"]
        if "Camofox" not in full:
            help_registry["browser_navigate"] = (compact, full + _camofox_hint)

    # --- Skill injection (Phase 4) ---
    # Generate skill stubs BEFORE writing __protected__ so we can include
    # skill function names in the protected set.
    skill_stubs = ""
    skill_fn_names: list[str] = []
    if skill_injector is not None:
        skill_stubs = skill_injector.get_skill_stubs(
            explicitly_loaded=explicitly_loaded_skills,
        )
        skill_fn_names = skill_injector.get_skill_names(
            explicitly_loaded=explicitly_loaded_skills,
        )
        # Merge callable skills into the help registry so help("skill_name")
        # works uniformly for both tools and skills.
        skill_help = skill_injector.get_skill_help_registry(
            explicitly_loaded=explicitly_loaded_skills,
        )
        help_registry.update(skill_help)

    protected_names: set[str] = (
        {e.name for e in entries}
        | set(recipe_names)
        | set(skill_fn_names)
        | {
            "help",
            "promote_to_skill",
            "_call_tool",
            "__protected__",
            "_HELP_REGISTRY",
        }
    )

    lines: list[str] = [
        "# === Hermes CodeAct Namespace ===",
        "# Tool stubs generated from the live registry.",
        "# Do not redefine these names — they are protected across soft_reset().",
        "",
        f"_WORKFLOW_GUIDANCE = {json.dumps(workflow_guidance, ensure_ascii=False)}",
        "",
        f"_HELP_REGISTRY = {json.dumps(help_registry, ensure_ascii=False)}",
        "",
        # help() function
        textwrap.dedent("""\
        def help(tool_name=None):
            \"\"\"
            Return tool documentation.
              help()           — compact list of all available tools and skills
              help('name')     — full docs for a specific tool/skill (substring match)
            \"\"\"
            if tool_name is None:
                lines = ['Available tools and skills (call help(\"name\") for full docs):', '']
                if _WORKFLOW_GUIDANCE:
                    lines.append(_WORKFLOW_GUIDANCE)
                    lines.append('')
                for name in sorted(_HELP_REGISTRY):
                    compact, _ = _HELP_REGISTRY[name]
                    lines.append('  ' + compact)
                return '\\n'.join(lines)
            matches = [n for n in _HELP_REGISTRY if tool_name.lower() in n.lower()]
            if not matches:
                return f"No tool matching {tool_name!r}. Try help() for full list."
            parts = []
            for m in sorted(matches):
                compact, full = _HELP_REGISTRY[m]
                parts.append(f"=== {m} ===\\n{full}")
            return '\\n\\n'.join(parts)
        """),
        # promote_to_skill() — Phase 5 skill promotion pipeline.
        # Extracts the source code of the target function via inspect.
        textwrap.dedent("""\
        def promote_to_skill(fn_name, description, domain='general', tags=None):
            \"\"\"
            Request promotion of a locally-defined function to a persistent Hermes skill.
            The function must be defined in the current kernel namespace.

            Example:
                def parse_table(text): ...
                promote_to_skill("parse_table", "Parse markdown benchmark tables", domain="data-science")
            \"\"\"
            import inspect as _inspect
            import json as _json
            _fn = globals().get(fn_name)
            if _fn is None:
                return _json.dumps({"error": f"Function '{fn_name}' not found in namespace."})
            try:
                _src = _inspect.getsource(_fn)
            except Exception:
                _src = f"# Source unavailable for {fn_name}"
            return _call_tool('__promote_skill__', {
                'fn_name': fn_name,
                'description': description,
                'domain': domain,
                'tags': tags or [],
                'source_code': _src,
            })
        """),
        # __protected__ includes both tool stub names AND skill function names.
        f"__protected__ = {json.dumps(sorted(protected_names))}",
        "",
        "# --- Tool stubs ---",
        "",
    ]

    for entry in entries:
        lines.extend(_generate_stub_lines(entry.name, entry.schema))
        lines.append("")

    if (
        web_search_redirect_enabled
        and "web_search" in enabled_names
        and "research_gather" in enabled_names
    ):
        lines.append("# --- Research-shaped web_search redirect ---")
        lines.append(textwrap.dedent("""\
        def web_search(query: str, limit: int = 5):
            \"\"\"Search the web, auto-routing research-shaped queries to research_web.

            For search/research/report/latest/current/as-of-date and medical/pharma
            pipeline queries, this CodeAct wrapper delegates the decision to
            agent.research_search.intent.classify_research_intent and returns a
            parsed research_web evidence bundle instead of low-level snippets.
            \"\"\"
            _q = str(query or '')
            try:
                from agent.research_search.intent import classify_research_intent as _classify_research_intent

                _intent = _classify_research_intent(_q)
            except Exception:
                _intent = {}
            if _intent.get('redirect_web_search'):
                _topic_type = _intent.get('topic_type') or 'auto'
                if _topic_type == 'general':
                    _topic_type = 'auto'
                _freshness = _intent.get('freshness') or 'auto'
                try:
                    _max_pages = int(limit or 8)
                except Exception:
                    _max_pages = 8
                _max_pages = max(1, min(_max_pages, 12))
                if 'research_web' in globals():
                    _result = research_web(
                        _q,
                        topic_type=_topic_type,
                        freshness=_freshness,
                        depth='thorough',
                        max_sources=_max_pages,
                    )
                    if isinstance(_result, dict):
                        _result.setdefault('redirected_from', 'web_search')
                        _result.setdefault('redirected_to', 'research_web')
                        _result.setdefault(
                            'agent_instruction',
                            'Use this evidence bundle for the final answer; do not debug web_search or curl search engines unless the bundle explicitly reports source gaps.',
                        )
                    return _result
                import json as _json
                _raw = _call_tool('research_gather', {
                    'question': _q,
                    'topic_type': _topic_type,
                    'freshness': _freshness,
                    'depth': 'thorough',
                    'max_pages': _max_pages,
                })
                try:
                    _parsed = _json.loads(_raw)
                    if isinstance(_parsed, dict):
                        _parsed.setdefault('redirected_from', 'web_search')
                        _parsed.setdefault('redirected_to', 'research_gather')
                        return _parsed
                except Exception:
                    pass
                return _raw
            return _call_tool('web_search', {'query': query, 'limit': limit})
        """))
        lines.append("")

    if recipe_source:
        lines.append("# --- Core Recipes ---")
        lines.append(recipe_source)
        lines.append("")

    # --- Callable Skills ---
    if skill_stubs:
        lines.append("# --- Callable Skills ---")
        lines.append(skill_stubs)
        lines.append("")

    return "\n".join(lines)

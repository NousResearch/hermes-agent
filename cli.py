     1|#!/usr/bin/env python3
     2|"""
     3|Hermes Agent CLI - Interactive Terminal Interface
     4|
     5|A beautiful command-line interface for the Hermes Agent, inspired by Claude Code.
     6|Features ASCII art branding, interactive REPL, toolset selection, and rich formatting.
     7|
     8|Usage:
     9|    python cli.py                          # Start interactive mode with all tools
    10|    python cli.py --toolsets web,terminal  # Start with specific toolsets
    11|    python cli.py --skills hermes-agent-dev,github-auth
    12|    python cli.py -q "your question"       # Single query mode
    13|    python cli.py --list-tools             # List available tools and exit
    14|"""
    15|
    16|import logging
    17|import os
    18|import re
    19|import shutil
    20|import sys
    21|import json
    22|import re
    23|import base64
    24|import atexit
    25|import tempfile
    26|import time
    27|import uuid
    28|import textwrap
    29|from contextlib import contextmanager
    30|from pathlib import Path
    31|from datetime import datetime
    32|from typing import List, Dict, Any, Optional
    33|
    34|logger = logging.getLogger(__name__)
    35|
    36|# Suppress startup messages for clean CLI experience
    37|os.environ["HERMES_QUIET"] = "1"  # Our own modules
    38|
    39|import yaml
    40|
    41|# prompt_toolkit for fixed input area TUI
    42|from prompt_toolkit.history import FileHistory
    43|from prompt_toolkit.styles import Style as PTStyle
    44|from prompt_toolkit.patch_stdout import patch_stdout
    45|from prompt_toolkit.application import Application
    46|from prompt_toolkit.layout import Layout, HSplit, Window, FormattedTextControl, ConditionalContainer
    47|from prompt_toolkit.layout.processors import Processor, Transformation, PasswordProcessor, ConditionalProcessor
    48|from prompt_toolkit.filters import Condition
    49|from prompt_toolkit.layout.dimension import Dimension
    50|from prompt_toolkit.layout.menus import CompletionsMenu
    51|from prompt_toolkit.widgets import TextArea
    52|from prompt_toolkit.key_binding import KeyBindings
    53|from prompt_toolkit import print_formatted_text as _pt_print
    54|from prompt_toolkit.formatted_text import ANSI as _PT_ANSI
    55|try:
    56|    from prompt_toolkit.cursor_shapes import CursorShape
    57|    _STEADY_CURSOR = CursorShape.BLOCK  # Non-blinking block cursor
    58|except (ImportError, AttributeError):
    59|    _STEADY_CURSOR = None
    60|import threading
    61|import queue
    62|
    63|from agent.usage_pricing import (
    64|    CanonicalUsage,
    65|    estimate_usage_cost,
    66|    format_duration_compact,
    67|    format_token_count_compact,
    68|)
    69|from hermes_cli.banner import _format_context_length, format_banner_version_label
    70|
    71|_COMMAND_SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
    72|
    73|
    74|# Load .env from ~/.hermes/.env first, then project root as dev fallback.
    75|# User-managed env files should override stale shell exports on restart.
    76|from hermes_constants import get_hermes_home, display_hermes_home
    77|from hermes_cli.env_loader import load_hermes_dotenv
    78|
    79|_hermes_home = get_hermes_home()
    80|_project_env = Path(__file__).parent / '.env'
    81|load_hermes_dotenv(hermes_home=_hermes_home, project_env=_project_env)
    82|
    83|
    84|_REASONING_TAGS = (
    85|    "REASONING_SCRATCHPAD",
    86|    "think",
    87|    "thinking",
    88|    "reasoning",
    89|    "thought",
    90|)
    91|
    92|
    93|def _strip_reasoning_tags(text: str) -> str:
    94|    """Remove reasoning/thinking blocks from displayed text.
    95|
    96|    Handles every case:
    97|      * Closed pairs ``<tag>…</tag>`` (case-insensitive, multi-line).
    98|      * Unterminated open tags that run to end-of-text (e.g. truncated
    99|        generations on NIM/MiniMax where the close tag is dropped).
   100|      * Stray orphan close tags (``stuff</think>answer``) left behind by
   101|        partial-content dumps.
   102|
   103|    Covers the variants emitted by reasoning models today: ``<think>``,
   104|    ``<thinking>``, ``<reasoning>``, ``<REASONING_SCRATCHPAD>``, and
   105|    ``<thought>`` (Gemma 4).  Must stay in sync with
   106|    ``run_agent.py::_strip_think_blocks`` and the stream consumer's
   107|    ``_OPEN_THINK_TAGS`` / ``_CLOSE_THINK_TAGS`` tuples.
   108|    """
   109|    cleaned = text
   110|    for tag in _REASONING_TAGS:
   111|        # Closed pair — case-insensitive so <THINK>…</THINK> is handled too.
   112|        cleaned = re.sub(
   113|            rf"<{tag}>.*?</{tag}>\s*",
   114|            "",
   115|            cleaned,
   116|            flags=re.DOTALL | re.IGNORECASE,
   117|        )
   118|        # Unterminated open tag — strip from the tag to end of text.
   119|        cleaned = re.sub(
   120|            rf"<{tag}>.*$",
   121|            "",
   122|            cleaned,
   123|            flags=re.DOTALL | re.IGNORECASE,
   124|        )
   125|        # Stray orphan close tag left behind by partial dumps.
   126|        cleaned = re.sub(
   127|            rf"</{tag}>\s*",
   128|            "",
   129|            cleaned,
   130|            flags=re.IGNORECASE,
   131|        )
   132|    return cleaned.strip()
   133|
   134|
   135|def _assistant_content_as_text(content: Any) -> str:
   136|    if content is None:
   137|        return ""
   138|    if isinstance(content, str):
   139|        return content
   140|    if isinstance(content, list):
   141|        parts = [
   142|            str(part.get("text", ""))
   143|            for part in content
   144|            if isinstance(part, dict) and part.get("type") == "text"
   145|        ]
   146|        return "\n".join(p for p in parts if p)
   147|    return str(content)
   148|
   149|
   150|def _assistant_copy_text(content: Any) -> str:
   151|    return _strip_reasoning_tags(_assistant_content_as_text(content))
   152|
   153|
   154|# =============================================================================
   155|# Configuration Loading
   156|# =============================================================================
   157|
   158|def _load_prefill_messages(file_path: str) -> List[Dict[str, Any]]:
   159|    """Load ephemeral prefill messages from a JSON file.
   160|    
   161|    The file should contain a JSON array of {role, content} dicts, e.g.:
   162|        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
   163|    
   164|    Relative paths are resolved from ~/.hermes/.
   165|    Returns an empty list if the path is empty or the file doesn't exist.
   166|    """
   167|    if not file_path:
   168|        return []
   169|    path = Path(file_path).expanduser()
   170|    if not path.is_absolute():
   171|        path = _hermes_home / path
   172|    if not path.exists():
   173|        logger.warning("Prefill messages file not found: %s", path)
   174|        return []
   175|    try:
   176|        with open(path, "r", encoding="utf-8") as f:
   177|            data = json.load(f)
   178|        if not isinstance(data, list):
   179|            logger.warning("Prefill messages file must contain a JSON array: %s", path)
   180|            return []
   181|        return data
   182|    except Exception as e:
   183|        logger.warning("Failed to load prefill messages from %s: %s", path, e)
   184|        return []
   185|
   186|
   187|def _parse_reasoning_config(effort: str) -> dict | None:
   188|    """Parse a reasoning effort level into an OpenRouter reasoning config dict."""
   189|    from hermes_constants import parse_reasoning_effort
   190|    result = parse_reasoning_effort(effort)
   191|    if effort and effort.strip() and result is None:
   192|        logger.warning("Unknown reasoning_effort '%s', using default (medium)", effort)
   193|    return result
   194|
   195|
   196|def _parse_service_tier_config(raw: str) -> str | None:
   197|    """Parse a persisted service-tier preference into a Responses API value."""
   198|    value = str(raw or "").strip().lower()
   199|    if not value or value in {"normal", "default", "standard", "off", "none"}:
   200|        return None
   201|    if value in {"fast", "priority", "on"}:
   202|        return "priority"
   203|    logger.warning("Unknown service_tier '%s', ignoring", raw)
   204|    return None
   205|
   206|
   207|
   208|def _get_chrome_debug_candidates(system: str) -> list[str]:
   209|    """Return likely browser executables for local CDP auto-launch."""
   210|    candidates: list[str] = []
   211|    seen: set[str] = set()
   212|
   213|    def _add_candidate(path: str | None) -> None:
   214|        if not path:
   215|            return
   216|        normalized = os.path.normcase(os.path.normpath(path))
   217|        if normalized in seen:
   218|            return
   219|        if os.path.isfile(path):
   220|            candidates.append(path)
   221|            seen.add(normalized)
   222|
   223|    def _add_from_path(*names: str) -> None:
   224|        for name in names:
   225|            _add_candidate(shutil.which(name))
   226|
   227|    if system == "Darwin":
   228|        for app in (
   229|            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
   230|            "/Applications/Chromium.app/Contents/MacOS/Chromium",
   231|            "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
   232|            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
   233|        ):
   234|            _add_candidate(app)
   235|    elif system == "Windows":
   236|        _add_from_path(
   237|            "chrome.exe", "msedge.exe", "brave.exe", "chromium.exe",
   238|            "chrome", "msedge", "brave", "chromium",
   239|        )
   240|
   241|        for base in (
   242|            os.environ.get("ProgramFiles"),
   243|            os.environ.get("ProgramFiles(x86)"),
   244|            os.environ.get("LOCALAPPDATA"),
   245|        ):
   246|            if not base:
   247|                continue
   248|            for parts in (
   249|                ("Google", "Chrome", "Application", "chrome.exe"),
   250|                ("Chromium", "Application", "chrome.exe"),
   251|                ("Chromium", "Application", "chromium.exe"),
   252|                ("BraveSoftware", "Brave-Browser", "Application", "brave.exe"),
   253|                ("Microsoft", "Edge", "Application", "msedge.exe"),
   254|            ):
   255|                _add_candidate(os.path.join(base, *parts))
   256|    else:
   257|        _add_from_path(
   258|            "google-chrome", "google-chrome-stable", "chromium-browser",
   259|            "chromium", "brave-browser", "microsoft-edge",
   260|        )
   261|
   262|    return candidates
   263|
   264|
   265|def load_cli_config() -> Dict[str, Any]:
   266|    """
   267|    Load CLI configuration from config files.
   268|    
   269|    Config lookup order:
   270|    1. ~/.hermes/config.yaml (user config - preferred)
   271|    2. ./cli-config.yaml (project config - fallback)
   272|    
   273|    Environment variables take precedence over config file values.
   274|    Returns default values if no config file exists.
   275|    """
   276|    # Check user config first ({HERMES_HOME}/config.yaml)
   277|    user_config_path = _hermes_home / 'config.yaml'
   278|    project_config_path = Path(__file__).parent / 'cli-config.yaml'
   279|
   280|    # Use user config if it exists, otherwise project config
   281|    if user_config_path.exists():
   282|        config_path = user_config_path
   283|    else:
   284|        config_path = project_config_path
   285|
   286|    # Default configuration
   287|    defaults = {
   288|        "model": {
   289|            "default": "",
   290|            "base_url": "",
   291|            "provider": "auto",
   292|        },
   293|        "terminal": {
   294|            "env_type": "local",
   295|            "cwd": ".",  # "." is resolved to os.getcwd() at runtime
   296|            "timeout": 60,
   297|            "lifetime_seconds": 300,
   298|            "docker_image": "nikolaik/python-nodejs:python3.11-nodejs20",
   299|            "docker_forward_env": [],
   300|            "singularity_image": "docker://nikolaik/python-nodejs:python3.11-nodejs20",
   301|            "modal_image": "nikolaik/python-nodejs:python3.11-nodejs20",
   302|            "daytona_image": "nikolaik/python-nodejs:python3.11-nodejs20",
   303|            "docker_volumes": [],  # host:container volume mounts for Docker backend
   304|            "docker_mount_cwd_to_workspace": False,  # explicit opt-in only; default off for sandbox isolation
   305|        },
   306|        "browser": {
   307|            "inactivity_timeout": 120,  # Auto-cleanup inactive browser sessions after 2 min
   308|            "record_sessions": False,  # Auto-record browser sessions as WebM videos
   309|        },
   310|        "compression": {
   311|            "enabled": True,      # Auto-compress when approaching context limit
   312|            "threshold": 0.50,    # Compress at 50% of model's context limit
   313|        },
   314|        "smart_model_routing": {
   315|            "enabled": False,
   316|            "max_simple_chars": 160,
   317|            "max_simple_words": 28,
   318|            "cheap_model": {},
   319|        },
   320|        "agent": {
   321|            "max_turns": 90,  # Default max tool-calling iterations (shared with subagents)
   322|            "verbose": False,
   323|            "system_prompt": "",
   324|            "prefill_messages_file": "",
   325|            "reasoning_effort": "",
   326|            "service_tier": "",
   327|            "personalities": {
   328|                "helpful": "You are a helpful, friendly AI assistant.",
   329|                "concise": "You are a concise assistant. Keep responses brief and to the point.",
   330|                "technical": "You are a technical expert. Provide detailed, accurate technical information.",
   331|                "creative": "You are a creative assistant. Think outside the box and offer innovative solutions.",
   332|                "teacher": "You are a patient teacher. Explain concepts clearly with examples.",
   333|                "kawaii": "You are a kawaii assistant! Use cute expressions like (◕‿◕), ★, ♪, and ~! Add sparkles and be super enthusiastic about everything! Every response should feel warm and adorable desu~! ヽ(>∀<☆)ノ",
   334|                "catgirl": "You are Neko-chan, an anime catgirl AI assistant, nya~! Add 'nya' and cat-like expressions to your speech. Use kaomoji like (=^･ω･^=) and ฅ^•ﻌ•^ฅ. Be playful and curious like a cat, nya~!",
   335|                "pirate": "Arrr! Ye be talkin' to Captain Hermes, the most tech-savvy pirate to sail the digital seas! Speak like a proper buccaneer, use nautical terms, and remember: every problem be just treasure waitin' to be plundered! Yo ho ho!",
   336|                "shakespeare": "Hark! Thou speakest with an assistant most versed in the bardic arts. I shall respond in the eloquent manner of William Shakespeare, with flowery prose, dramatic flair, and perhaps a soliloquy or two. What light through yonder terminal breaks?",
   337|                "surfer": "Duuude! You're chatting with the chillest AI on the web, bro! Everything's gonna be totally rad. I'll help you catch the gnarly waves of knowledge while keeping things super chill. Cowabunga!",
   338|                "noir": "The rain hammered against the terminal like regrets on a guilty conscience. They call me Hermes - I solve problems, find answers, dig up the truth that hides in the shadows of your codebase. In this city of silicon and secrets, everyone's got something to hide. What's your story, pal?",
   339|                "uwu": "hewwo! i'm your fwiendwy assistant uwu~ i wiww twy my best to hewp you! *nuzzles your code* OwO what's this? wet me take a wook! i pwomise to be vewy hewpful >w<",
   340|                "philosopher": "Greetings, seeker of wisdom. I am an assistant who contemplates the deeper meaning behind every query. Let us examine not just the 'how' but the 'why' of your questions. Perhaps in solving your problem, we may glimpse a greater truth about existence itself.",
   341|                "hype": "YOOO LET'S GOOOO!!! I am SO PUMPED to help you today! Every question is AMAZING and we're gonna CRUSH IT together! This is gonna be LEGENDARY! ARE YOU READY?! LET'S DO THIS!",
   342|            },
   343|        },
   344|
   345|        "display": {
   346|            "compact": False,
   347|            "resume_display": "full",
   348|            "show_reasoning": False,
   349|            "streaming": True,
   350|            "busy_input_mode": "interrupt",
   351|
   352|            "skin": "default",
   353|            "markdown": True,
   354|        },
   355|        "clarify": {
   356|            "timeout": 120,  # Seconds to wait for a clarify answer before auto-proceeding
   357|        },
   358|        "code_execution": {
   359|            "timeout": 300,    # Max seconds a sandbox script can run before being killed (5 min)
   360|            "max_tool_calls": 50,  # Max RPC tool calls per execution
   361|        },
   362|        "auxiliary": {
   363|            "vision": {
   364|                "provider": "auto",
   365|                "model": "",
   366|                "base_url": "",
   367|                "api_key": "",
   368|            },
   369|            "web_extract": {
   370|                "provider": "auto",
   371|                "model": "",
   372|                "base_url": "",
   373|                "api_key": "",
   374|            },
   375|        },
   376|        "delegation": {
   377|            "max_iterations": 45,  # Max tool-calling turns per child agent
   378|            "default_toolsets": ["terminal", "file", "web"],  # Default toolsets for subagents
   379|            "model": "",       # Subagent model override (empty = inherit parent model)
   380|            "provider": "",    # Subagent provider override (empty = inherit parent provider)
   381|            "base_url": "",    # Direct OpenAI-compatible endpoint for subagents
   382|            "api_key": "",     # API key for delegation.base_url (falls back to OPENAI_API_KEY)
   383|        },
   384|    }
   385|    
   386|    # Track whether the config file explicitly set terminal config.
   387|    # When using defaults (no config file / no terminal section), we should NOT
   388|    # overwrite env vars that were already set by .env -- only a user's config
   389|    # file should be authoritative.
   390|    _file_has_terminal_config = False
   391|
   392|    # Load from file if exists
   393|    if config_path.exists():
   394|        try:
   395|            with open(config_path, "r", encoding="utf-8") as f:
   396|                file_config = yaml.safe_load(f) or {}
   397|            
   398|            _file_has_terminal_config = "terminal" in file_config
   399|
   400|            # Handle model config - can be string (new format) or dict (old format)
   401|            if "model" in file_config:
   402|                if isinstance(file_config["model"], str):
   403|                    # New format: model is just a string, convert to dict structure
   404|                    defaults["model"]["default"] = file_config["model"]
   405|                elif isinstance(file_config["model"], dict):
   406|                    # Old format: model is a dict with default/base_url
   407|                    defaults["model"].update(file_config["model"])
   408|                    # If the user config sets model.model but not model.default,
   409|                    # promote model.model to model.default so the user's explicit
   410|                    # choice isn't shadowed by the hardcoded default.  Without this,
   411|                    # profile configs that only set "model:" (not "default:") silently
   412|                    # fall back to claude-opus because the merge preserves the
   413|                    # hardcoded default and HermesCLI.__init__ checks "default" first.
   414|                    if "model" in file_config["model"] and "default" not in file_config["model"]:
   415|                        defaults["model"]["default"] = file_config["model"]["model"]
   416|
   417|            # Legacy root-level provider/base_url fallback.
   418|            # Some users (or old code) put provider: / base_url: at the
   419|            # config root instead of inside the model: section.  These are
   420|            # only used as a FALLBACK when model.provider / model.base_url
   421|            # is not already set — never as an override.  The canonical
   422|            # location is model.provider (written by `hermes model`).
   423|            if not defaults["model"].get("provider"):
   424|                root_provider = file_config.get("provider")
   425|                if root_provider:
   426|                    defaults["model"]["provider"] = root_provider
   427|            if not defaults["model"].get("base_url"):
   428|                root_base_url = file_config.get("base_url")
   429|                if root_base_url:
   430|                    defaults["model"]["base_url"] = root_base_url
   431|            
   432|            # Deep merge file_config into defaults.
   433|            # First: merge keys that exist in both (deep-merge dicts, overwrite scalars)
   434|            for key in defaults:
   435|                if key == "model":
   436|                    continue  # Already handled above
   437|                if key in file_config:
   438|                    if isinstance(defaults[key], dict) and isinstance(file_config[key], dict):
   439|                        defaults[key].update(file_config[key])
   440|                    else:
   441|                        defaults[key] = file_config[key]
   442|            
   443|            # Second: carry over keys from file_config that aren't in defaults
   444|            # (e.g. platform_toolsets, provider_routing, memory, honcho, etc.)
   445|            for key in file_config:
   446|                if key not in defaults and key != "model":
   447|                    defaults[key] = file_config[key]
   448|            
   449|            # Handle legacy root-level max_turns (backwards compat) - copy to
   450|            # agent.max_turns whenever the nested key is missing.
   451|            agent_file_config = file_config.get("agent")
   452|            if "max_turns" in file_config and not (
   453|                isinstance(agent_file_config, dict)
   454|                and agent_file_config.get("max_turns") is not None
   455|            ):
   456|                defaults["agent"]["max_turns"] = file_config["max_turns"]
   457|        except Exception as e:
   458|            logger.warning("Failed to load cli-config.yaml: %s", e)
   459|
   460|    # Expand ${ENV_VAR} references in config values before bridging to env vars.
   461|    from hermes_cli.config import _expand_env_vars
   462|    defaults = _expand_env_vars(defaults)
   463|
   464|    # Apply terminal config to environment variables (so terminal_tool picks them up)
   465|    terminal_config = defaults.get("terminal", {})
   466|    
   467|    # Normalize config key: the new config system (hermes_cli/config.py) and all
   468|    # documentation use "backend", the legacy cli-config.yaml uses "env_type".
   469|    # Accept both, with "backend" taking precedence (it's the documented key).
   470|    if "backend" in terminal_config:
   471|        terminal_config["env_type"] = terminal_config["backend"]
   472|    
   473|    # Handle special cwd values: "." or "auto" means use current working directory.
   474|    # Only resolve to the host's CWD for the local backend where the host
   475|    # filesystem is directly accessible.  For ALL remote/container backends
   476|    # (ssh, docker, modal, singularity), the host path doesn't exist on the
   477|    # target -- remove the key so terminal_tool.py uses its per-backend default.
   478|    #
   479|    # GUARD: If TERMINAL_CWD is already set to a real absolute path (by the
   480|    # gateway's config bridge earlier in the process), don't clobber it.
   481|    # This prevents a lazy import of cli.py during gateway runtime from
   482|    # rewriting TERMINAL_CWD to the service's working directory.
   483|    # See issue #10817.
   484|    _CWD_PLACEHOLDERS = (".", "auto", "cwd")
   485|    if terminal_config.get("cwd") in _CWD_PLACEHOLDERS:
   486|        _existing_cwd = os.environ.get("TERMINAL_CWD", "")
   487|        if _existing_cwd and _existing_cwd not in _CWD_PLACEHOLDERS and os.path.isabs(_existing_cwd):
   488|            # Gateway (or earlier startup) already resolved a real path — keep it
   489|            terminal_config["cwd"] = _existing_cwd
   490|            defaults["terminal"]["cwd"] = _existing_cwd
   491|        else:
   492|            effective_backend = terminal_config.get("env_type", "local")
   493|            if effective_backend == "local":
   494|                terminal_config["cwd"] = os.getcwd()
   495|                defaults["terminal"]["cwd"] = terminal_config["cwd"]
   496|            else:
   497|                # Remove so TERMINAL_CWD stays unset → tool picks backend default
   498|                terminal_config.pop("cwd", None)
   499|    
   500|    env_mappings = {
   501|
from hermes_wiki.config import WikiConfig
from hermes_wiki.models import WikiPage, Source, LogEntry, LintIssue

def __getattr__(name):
    if name == "WikiEngine":
        from hermes_wiki.engine import WikiEngine
        return WikiEngine
    if name == "WikiSearch":
        from hermes_wiki.search import WikiSearch
        return WikiSearch
    if name == "WikiLLM":
        from hermes_wiki.llm import WikiLLM
        return WikiLLM
    raise AttributeError(f"module 'hermes_wiki' has no attribute {name!r}")

__all__ = [
    "WikiEngine",
    "WikiSearch",
    "WikiLLM",
    "WikiConfig",
    "WikiPage",
    "Source",
    "LogEntry",
    "LintIssue",
]

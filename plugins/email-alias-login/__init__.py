"""Opt-in Email gateway adapter with separate IMAP/SMTP login identity."""

if __package__:
    from .adapter import register
else:  # pytest imports a repository-root __init__.py as a top-level module
    from adapter import register

__all__ = ["register"]

try:
    from .adapter import register
except ImportError:
    # Stub register until Task 6 creates adapter.py
    def register(*args, **kwargs):
        pass

__all__ = ["register"]

import sys

__all__ = ['tomllib']


if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


if sys.version_info >= (3, 11):

    def add_note(ex, note):
        ex.add_note(note)

else:  # pragma: no cover

    def add_note(ex, note):
        vars(ex).setdefault('__notes__', []).append(note)

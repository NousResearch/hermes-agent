import io
import sys


def _text_encoding(encoding, stacklevel=2, /):  # pragma: no cover
    return encoding


text_encoding = (
    io.text_encoding  # type: ignore[unused-ignore, attr-defined]
    if sys.version_info > (3, 10)
    else _text_encoding
)

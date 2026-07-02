import sys

# Explicitly use the ``"locale"`` encoding in versions that support it,
# otherwise just rely on the implicit handling of ``encoding=None``.
# Since all platforms that support ``EncodingWarning`` also support
# ``encoding="locale"``, this can be used to suppress the warning.
# However, please try to use UTF-8 when possible
# (.pth files are the notorious exception: python/cpython#77102, pypa/setuptools#3937).
LOCALE_ENCODING = "locale" if sys.version_info >= (3, 10) else None

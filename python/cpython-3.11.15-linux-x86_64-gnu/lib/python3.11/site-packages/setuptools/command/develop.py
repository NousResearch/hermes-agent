import site
import subprocess
import sys
from typing import cast

from setuptools import Command
from setuptools.warnings import SetuptoolsDeprecationWarning


class develop(Command):
    """Set up package for development"""

    user_options = [
        ("install-dir=", "d", "install package to DIR"),
        ('no-deps', 'N', "don't install dependencies"),
        ('user', None, f"install in user site-package '{site.USER_SITE}'"),
        ('prefix=', None, "installation prefix"),
        ("index-url=", "i", "base URL of Python Package Index"),
    ]
    boolean_options = [
        'no-deps',
        'user',
    ]

    install_dir = None
    no_deps = False
    user = False
    prefix = None
    index_url = None

    def run(self) -> None:
        # Casting because mypy doesn't understand bool mult conditionals
        cmd = cast(
            list[str],
            [sys.executable, '-m', 'pip', 'install', '-e', '.', '--use-pep517']
            + ['--target', self.install_dir] * bool(self.install_dir)
            + ['--no-deps'] * self.no_deps
            + ['--user'] * self.user
            + ['--prefix', self.prefix] * bool(self.prefix)
            + ['--index-url', self.index_url] * bool(self.index_url),
        )
        subprocess.check_call(cmd)

    def initialize_options(self) -> None:
        DevelopDeprecationWarning.emit()

    def finalize_options(self) -> None:
        pass


class DevelopDeprecationWarning(SetuptoolsDeprecationWarning):
    _SUMMARY = "develop command is deprecated."
    _DETAILS = """
    Please avoid running ``setup.py`` and ``develop``.
    Instead, use standards-based tools like pip or uv.
    """
    _SEE_URL = "https://github.com/pypa/setuptools/issues/917"
    _DUE_DATE = 2025, 10, 31

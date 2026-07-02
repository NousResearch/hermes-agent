import itertools
import os

from .compat import py312

from distutils import log

flatten = itertools.chain.from_iterable


class Installer:
    nspkg_ext = '-nspkg.pth'

    def install_namespaces(self) -> None:
        nsp = self._get_all_ns_packages()
        if not nsp:
            return
        filename = self._get_nspkg_file()
        self.outputs.append(filename)
        log.info("Installing %s", filename)
        lines = map(self._gen_nspkg_line, nsp)

        with open(filename, 'wt', encoding=py312.PTH_ENCODING) as f:
            # Python<3.13 requires encoding="locale" instead of "utf-8"
            # See: python/cpython#77102
            f.writelines(lines)

    def uninstall_namespaces(self) -> None:
        filename = self._get_nspkg_file()
        if not os.path.exists(filename):
            return
        log.info("Removing %s", filename)
        os.remove(filename)

    def _get_nspkg_file(self):
        filename, _ = os.path.splitext(self._get_target())
        return filename + self.nspkg_ext

    def _get_target(self):
        return self.target

    _nspkg_tmpl = (
        "import sys, types, os",
        "p = os.path.join(%(root)s, *%(pth)r)",
        "importlib = __import__('importlib.util')",
        "__import__('importlib.machinery')",
        (
            "m = "
            "sys.modules.setdefault(%(pkg)r, "
            "importlib.util.module_from_spec("
            "importlib.machinery.PathFinder.find_spec(%(pkg)r, "
            "[os.path.dirname(p)])))"
        ),
        ("m = m or sys.modules.setdefault(%(pkg)r, types.ModuleType(%(pkg)r))"),
        "mp = (m or []) and m.__dict__.setdefault('__path__',[])",
        "(p not in mp) and mp.append(p)",
    )
    "lines for the namespace installer"

    _nspkg_tmpl_multi = ('m and setattr(sys.modules[%(parent)r], %(child)r, m)',)
    "additional line(s) when a parent package is indicated"

    def _get_root(self):
        return "sys._getframe(1).f_locals['sitedir']"

    def _gen_nspkg_line(self, pkg):
        pth = tuple(pkg.split('.'))
        root = self._get_root()
        tmpl_lines = self._nspkg_tmpl
        parent, sep, child = pkg.rpartition('.')
        if parent:
            tmpl_lines += self._nspkg_tmpl_multi
        return ';'.join(tmpl_lines) % locals() + '\n'

    def _get_all_ns_packages(self):
        """Return sorted list of all package namespaces"""
        pkgs = self.distribution.namespace_packages or []
        return sorted(set(flatten(map(self._pkg_names, pkgs))))

    @staticmethod
    def _pkg_names(pkg):
        """
        Given a namespace package, yield the components of that
        package.

        >>> names = Installer._pkg_names('a.b.c')
        >>> set(names) == set(['a', 'a.b', 'a.b.c'])
        True
        """
        parts = pkg.split('.')
        while parts:
            yield '.'.join(parts)
            parts.pop()


class DevelopInstaller(Installer):
    def _get_root(self):
        return repr(str(self.egg_path))

    def _get_target(self):
        return self.egg_link

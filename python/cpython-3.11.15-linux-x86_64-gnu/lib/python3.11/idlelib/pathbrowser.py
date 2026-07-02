import importlib.machinery
import os
import sys

from idlelib.browser import ModuleBrowser, ModuleBrowserTreeItem
from idlelib.tree import TreeItem


class PathBrowser(ModuleBrowser):

    def __init__(self, master, *, _htest=False, _utest=False):
        """
        _htest - bool, change box location when running htest
        """
        self.master = master
        self._htest = _htest
        self._utest = _utest
        self.init()

    def settitle(self):
        "Set window titles."
        self.top.wm_title("Path Browser")
        self.top.wm_iconname("Path Browser")

    def rootnode(self):
        return PathBrowserTreeItem()


class PathBrowserTreeItem(TreeItem):

    def GetText(self):
        return "sys.path"

    def GetSubList(self):
        sublist = []
        for dir in sys.path:
            item = DirBrowserTreeItem(dir)
            sublist.append(item)
        return sublist


class DirBrowserTreeItem(TreeItem):

    def __init__(self, dir, packages=[]):
        self.dir = dir
        self.packages = packages

    def GetText(self):
        if not self.packages:
            return self.dir
        else:
            return self.packages[-1] + ": package"

    def GetSubList(self):
        try:
            names = os.listdir(self.dir or os.curdir)
        except OSError:
            return []
        packages = []
        for name in names:
            file = os.path.join(self.dir, name)
            if self.ispackagedir(file):
                nn = os.path.normcase(name)
                packages.append((nn, name, file))
        packages.sort()
        sublist = []
        for nn, name, file in packages:
            item = DirBrowserTreeItem(file, self.packages + [name])
            sublist.append(item)
        for nn, name in self.listmodules(names):
            item = ModuleBrowserTreeItem(os.path.join(self.dir, name))
            sublist.append(item)
        return sublist

    def ispackagedir(self, file):
        " Return true for directories that are packages."
        if not os.path.isdir(file):
            return False
        init = os.path.join(file, "__init__.py")
        return os.path.exists(init)

    def listmodules(self, allnames):
        modules = {}
        suffixes = importlib.machinery.EXTENSION_SUFFIXES[:]
        suffixes += importlib.machinery.SOURCE_SUFFIXES
        suffixes += importlib.machinery.BYTECODE_SUFFIXES
        sorted = []
        for suff in suffixes:
            i = -len(suff)
            for name in allnames[:]:
                normed_name = os.path.normcase(name)
                if normed_name[i:] == suff:
                    mod_name = name[:i]
                    if mod_name not in modules:
                        modules[mod_name] = None
                        sorted.append((normed_name, name))
                        allnames.remove(name)
        sorted.sort()
        return sorted


if __name__ == "__main__":
    from unittest import main
    main('idlelib.idle_test.test_pathbrowser', verbosity=2, exit=False)

    from idlelib.idle_test.htest import run
    run(PathBrowser)

# Rename to stackbrowser or possibly consolidate with browser.

import linecache
import os

import tkinter as tk

from idlelib.debugobj import ObjectTreeItem, make_objecttreeitem
from idlelib.tree import TreeNode, TreeItem, ScrolledCanvas

def StackBrowser(root, exc, flist=None, top=None):
    global sc, item, node  # For testing.
    if top is None:
        top = tk.Toplevel(root)
    sc = ScrolledCanvas(top, bg="white", highlightthickness=0)
    sc.frame.pack(expand=1, fill="both")
    item = StackTreeItem(exc, flist)
    node = TreeNode(sc.canvas, None, item)
    node.expand()


class StackTreeItem(TreeItem):

    def __init__(self, exc, flist=None):
        self.flist = flist
        self.stack = self.get_stack(None if exc is None else exc.__traceback__)
        self.text = f"{type(exc).__name__}: {str(exc)}"

    def get_stack(self, tb):
        stack = []
        if tb and tb.tb_frame is None:
            tb = tb.tb_next
        while tb is not None:
            stack.append((tb.tb_frame, tb.tb_lineno))
            tb = tb.tb_next
        return stack

    def GetText(self):  # Titlecase names are overrides.
        return self.text

    def GetSubList(self):
        sublist = []
        for info in self.stack:
            item = FrameTreeItem(info, self.flist)
            sublist.append(item)
        return sublist


class FrameTreeItem(TreeItem):

    def __init__(self, info, flist):
        self.info = info
        self.flist = flist

    def GetText(self):
        frame, lineno = self.info
        try:
            modname = frame.f_globals["__name__"]
        except:
            modname = "?"
        code = frame.f_code
        filename = code.co_filename
        funcname = code.co_name
        sourceline = linecache.getline(filename, lineno)
        sourceline = sourceline.strip()
        if funcname in ("?", "", None):
            item = "%s, line %d: %s" % (modname, lineno, sourceline)
        else:
            item = "%s.%s(...), line %d: %s" % (modname, funcname,
                                             lineno, sourceline)
        return item

    def GetSubList(self):
        frame, lineno = self.info
        sublist = []
        if frame.f_globals is not frame.f_locals:
            item = VariablesTreeItem("<locals>", frame.f_locals, self.flist)
            sublist.append(item)
        item = VariablesTreeItem("<globals>", frame.f_globals, self.flist)
        sublist.append(item)
        return sublist

    def OnDoubleClick(self):
        if self.flist:
            frame, lineno = self.info
            filename = frame.f_code.co_filename
            if os.path.isfile(filename):
                self.flist.gotofileline(filename, lineno)


class VariablesTreeItem(ObjectTreeItem):

    def GetText(self):
        return self.labeltext

    def GetLabelText(self):
        return None

    def IsExpandable(self):
        return len(self.object) > 0

    def GetSubList(self):
        sublist = []
        for key in self.object.keys():  # self.object not necessarily dict.
            try:
                value = self.object[key]
            except KeyError:
                continue
            def setfunction(value, key=key, object_=self.object):
                object_[key] = value
            item = make_objecttreeitem(key + " =", value, setfunction)
            sublist.append(item)
        return sublist


def _stackbrowser(parent):  # htest #
    from idlelib.pyshell import PyShellFileList
    top = tk.Toplevel(parent)
    top.title("Test StackViewer")
    x, y = map(int, parent.geometry().split('+')[1:])
    top.geometry("+%d+%d" % (x + 50, y + 175))
    flist = PyShellFileList(top)
    try: # to obtain a traceback object
        intentional_name_error
    except NameError as e:
        StackBrowser(top, e, flist=flist, top=top)


if __name__ == '__main__':
    from unittest import main
    main('idlelib.idle_test.test_stackviewer', verbosity=2, exit=False)

    from idlelib.idle_test.htest import run
    run(_stackbrowser)

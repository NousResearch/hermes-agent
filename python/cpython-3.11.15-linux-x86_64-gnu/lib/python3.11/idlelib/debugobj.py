"""Define tree items for debug stackviewer, which is only user.
"""
# XXX TO DO:
# - popup menu
# - support partial or total redisplay
# - more doc strings
# - tooltips

# object browser

# XXX TO DO:
# - for classes/modules, add "open source" to object browser
from reprlib import Repr

from idlelib.tree import TreeItem, TreeNode, ScrolledCanvas

myrepr = Repr()
myrepr.maxstring = 100
myrepr.maxother = 100

class ObjectTreeItem(TreeItem):
    def __init__(self, labeltext, object_, setfunction=None):
        self.labeltext = labeltext
        self.object = object_
        self.setfunction = setfunction
    def GetLabelText(self):
        return self.labeltext
    def GetText(self):
        return myrepr.repr(self.object)
    def GetIconName(self):
        if not self.IsExpandable():
            return "python"
    def IsEditable(self):
        return self.setfunction is not None
    def SetText(self, text):
        try:
            value = eval(text)
            self.setfunction(value)
        except:
            pass
        else:
            self.object = value
    def IsExpandable(self):
        return not not dir(self.object)
    def GetSubList(self):
        keys = dir(self.object)
        sublist = []
        for key in keys:
            try:
                value = getattr(self.object, key)
            except AttributeError:
                continue
            item = make_objecttreeitem(
                str(key) + " =",
                value,
                lambda value, key=key, object_=self.object:
                    setattr(object_, key, value))
            sublist.append(item)
        return sublist

class ClassTreeItem(ObjectTreeItem):
    def IsExpandable(self):
        return True
    def GetSubList(self):
        sublist = ObjectTreeItem.GetSubList(self)
        if len(self.object.__bases__) == 1:
            item = make_objecttreeitem("__bases__[0] =",
                self.object.__bases__[0])
        else:
            item = make_objecttreeitem("__bases__ =", self.object.__bases__)
        sublist.insert(0, item)
        return sublist

class AtomicObjectTreeItem(ObjectTreeItem):
    def IsExpandable(self):
        return False

class SequenceTreeItem(ObjectTreeItem):
    def IsExpandable(self):
        return len(self.object) > 0
    def keys(self):
        return range(len(self.object))
    def GetSubList(self):
        sublist = []
        for key in self.keys():
            try:
                value = self.object[key]
            except KeyError:
                continue
            def setfunction(value, key=key, object_=self.object):
                object_[key] = value
            item = make_objecttreeitem(f"{key!r}:", value, setfunction)
            sublist.append(item)
        return sublist

class DictTreeItem(SequenceTreeItem):
    def keys(self):
        # TODO return sorted(self.object)
        keys = list(self.object)
        try:
            keys.sort()
        except:
            pass
        return keys

dispatch = {
    int: AtomicObjectTreeItem,
    float: AtomicObjectTreeItem,
    str: AtomicObjectTreeItem,
    tuple: SequenceTreeItem,
    list: SequenceTreeItem,
    dict: DictTreeItem,
    type: ClassTreeItem,
}

def make_objecttreeitem(labeltext, object_, setfunction=None):
    t = type(object_)
    if t in dispatch:
        c = dispatch[t]
    else:
        c = ObjectTreeItem
    return c(labeltext, object_, setfunction)


def _debug_object_browser(parent):  # htest #
    import sys
    from tkinter import Toplevel
    top = Toplevel(parent)
    top.title("Test debug object browser")
    x, y = map(int, parent.geometry().split('+')[1:])
    top.geometry("+%d+%d" % (x + 100, y + 175))
    top.configure(bd=0, bg="yellow")
    top.focus_set()
    sc = ScrolledCanvas(top, bg="white", highlightthickness=0, takefocus=1)
    sc.frame.pack(expand=1, fill="both")
    item = make_objecttreeitem("sys", sys)
    node = TreeNode(sc.canvas, None, item)
    node.update()


if __name__ == '__main__':
    from unittest import main
    main('idlelib.idle_test.test_debugobj', verbosity=2, exit=False)

    from idlelib.idle_test.htest import run
    run(_debug_object_browser)
